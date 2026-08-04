"""Tier 2 univariate: phylogenetic logistic regression, one defense system
at a time, with covariates and bidirectional framing.

Primary direction ("plasmid_given_defense"):
    plasmid_class ~ defense + genome_covariates [+ log(n_plasmids)]
    — for each defense system, a separate phyloglm fit.

Reverse direction ("defense_given_plasmid"):
    defense ~ plasmid_class + genome_covariates [+ log(n_plasmids)]
    — for each defense system (as outcome), a separate phyloglm fit reporting
    the plasmid_class coefficient. Answers "does plasmid carriage predict
    defense presence?", which the primary direction does not.

Both directions are run across every outcome stratum from outcome_spec.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import multiprocessing as mp

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import apply_fdr


def _run_one_direction(phylo_data: pd.DataFrame,
                       defense_cols: List[str],
                       outcome_col: str,
                       outcome_label: str,
                       direction: str,
                       tree_path: str,
                       config: Config,
                       logger: logging.Logger,
                       workdir: Path,
                       covariates: List[str],
                       covariate_mode: str) -> pd.DataFrame:
    """Run phyloglm in one direction for a single outcome column."""
    mode = "predictor" if direction == "plasmid_given_defense" else "response"
    if direction == "plasmid_given_defense":
        call_response = outcome_col
        call_predictors = defense_cols
    else:
        call_response = defense_cols     # iterate on responses
        call_predictors = [outcome_col]  # fixed predictor

    # Check all required columns exist
    missing = [c for c in defense_cols + [outcome_col] + covariates
               if c not in phylo_data.columns]
    if missing:
        logger.warning(
            f"phyloglm [{covariate_mode}/{outcome_label}/{direction}]: "
            f"missing columns ({len(missing)}) — skipping"
        )
        return pd.DataFrame()

    # One-time sanity log of what we're handing to R. Helps diagnose
    # "Too few matched tips (0)" — if the tip column is absent or empty
    # here, R was never going to find anything to intersect against the
    # tree. Logged at debug level after the first call; see r_bridge for
    # the full stderr dump on R-side failures.
    tip_col = "tip"
    if tip_col in phylo_data.columns and not getattr(_run_one_direction,
                                                     "_tip_sample_logged", False):
        sample = phylo_data[tip_col].head(3).tolist()
        logger.info(
            f"phyloglm input sample — rows={len(phylo_data)}, "
            f"has '{tip_col}' column: True, "
            f"first 3 tip values: {sample}"
        )
        _run_one_direction._tip_sample_logged = True
    elif tip_col not in phylo_data.columns:
        logger.error(
            f"phyloglm input is missing the '{tip_col}' column. "
            f"Columns present: {list(phylo_data.columns)[:15]}..."
        )

    r = call_r_script(
        "phyloglm_uni.R",
        tree_path=tree_path,
        data=phylo_data,
        args={
            "response": call_response,
            "predictors": call_predictors,
            "mode": mode,
            "tip_column": "tip",
            "covariates": list(covariates),
            "evolutionary_model": config.phyloglm_estimator,
            "btol": 20,
            "boot": 0,
            "min_count": config.min_count_per_category,
            # Tell R which side of the fit carries the defense system so the
            # prevalence gate follows it. Without this the reverse direction
            # gated on the PLASMID column's balance and never checked the
            # defense system at all — a system present in 2 of 15,000 species
            # was fit, separated, and emitted a Wald p-value into the
            # reverse-direction FDR family.
            "defense_side": mode,
            # Gate the response on having both levels present, so a subset
            # driven to ~97% positive is not fit silently at zero power.
            "min_count_response": config.min_count_per_category,
        },
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / f"phyloglm_uni_{covariate_mode}_{outcome_label}_{direction}",
    )

    if not r.ok:
        logger.error(
            f"phyloglm_uni [{covariate_mode}/{outcome_label}/{direction}] failed: {r.error}"
        )
        return pd.DataFrame()

    df = r.dataframe.rename(columns={"test_label": "defense_system"})
    df["outcome_label"] = outcome_label
    df["direction"] = direction
    df["covariate_mode"] = covariate_mode

    # Degenerate fits arrive with an NA p-value from the R side (coefficient
    # pinned at the btol bound, non-finite SE, convergence warning). They are
    # excluded from the FDR family automatically because apply_fdr is
    # NaN-aware, and the odds ratio below is left NaN rather than reporting
    # e^20 ~ 5e8 as an effect size.
    finite_fit = df["phyloglm_p_value"].notna()
    df["phyloglm_fdr_qvalue"] = apply_fdr(df["phyloglm_p_value"],
                                          method=config.fdr_method).values
    coef = df["phyloglm_coefficient"].where(finite_fit)
    se = df["phyloglm_std_err"].where(finite_fit)
    df["phyloglm_odds_ratio"] = np.exp(coef)
    df["phyloglm_ci_low"] = np.exp(coef - 1.96 * se)
    df["phyloglm_ci_high"] = np.exp(coef + 1.96 * se)

    n_sig = int((df["phyloglm_fdr_qvalue"] < config.alpha).sum())
    n_run = int(finite_fit.sum())
    n_skipped = int(len(df) - n_run)
    skip_note = ""
    if n_skipped and "skip_reason" in df.columns:
        top = (df.loc[~finite_fit, "skip_reason"].dropna()
               .astype(str).str.split(":").str[0].value_counts().head(3))
        if len(top):
            skip_note = "; skipped: " + ", ".join(
                f"{k}={v}" for k, v in top.items())
    logger.info(
        f"  phyloglm [{covariate_mode}/{outcome_label}/{direction}]: "
        f"{n_run} systems fit; {n_sig} at FDR < {config.alpha}"
        f"{skip_note}"
    )
    # A stage where most systems failed should not look like a stage where
    # most systems were null.
    if n_run and n_skipped > n_run:
        logger.warning(
            f"  phyloglm [{covariate_mode}/{outcome_label}/{direction}]: "
            f"more systems were skipped or degenerate ({n_skipped}) than fit "
            f"({n_run}). Interpret this slice with care.")
    return df


def run_tier2_phyloglm_univariate(phylo_data: pd.DataFrame,
                                  defense_cols: List[str],
                                  tree_path: str,
                                  config: Config,
                                  logger: logging.Logger,
                                  workdir: Path,
                                  outcome_spec: Optional[Dict[str, List[Optional[str]]]] = None
                                  ) -> pd.DataFrame:
    """Run univariate phyloglm across every outcome stratum and both
    directions (if bidirectional framing is enabled).

    Returns a long-form DataFrame with one row per (defense_system,
    outcome_label, direction) combination.
    """
    if outcome_spec is None:
        outcome_spec = {"any_plasmid": [None, None, "has_plasmid_binary"]}

    # Ensure log_n_plasmids and its spline basis exist for binary-class
    # outcomes. A single linear log(n_plasmids) term has the same problem as a
    # single linear depth term: the "species with many plasmids carries one of
    # every class" saturation is not linear in log(n) on the logit scale.
    from .io_utils import add_plasmid_count_basis
    if "n_plasmids" in phylo_data.columns:
        phylo_data = phylo_data.copy()
        if "log_n_plasmids" not in phylo_data.columns:
            phylo_data["log_n_plasmids"] = np.log1p(
                phylo_data["n_plasmids"].fillna(0).clip(lower=0))
        if not any(c.startswith(config.plasmid_count_spline_prefix)
                   for c in phylo_data.columns):
            phylo_data = add_plasmid_count_basis(phylo_data, config, logger)

    directions = ["plasmid_given_defense"]
    if config.run_bidirectional:
        directions.append("defense_given_plasmid")

    logger.info(
        f"Tier 2 phyloglm (univariate, estimator={config.phyloglm_estimator}) — "
        f"{len(defense_cols)} systems, {len(phylo_data)} species, "
        f"{len(outcome_spec)} outcomes x {len(directions)} directions x "
        f"{len(config.covariate_modes)} covariate modes"
    )

    # Enumerate (covariate_mode × outcome × direction) tasks first, then
    # dispatch them to a thread pool. Each task spawns its own Rscript
    # subprocess and writes to its own workdir, so they're embarrassingly
    # parallel. Threading backend works because the Python-side cost per
    # task is just file IO and subprocess.run; the GIL is released for the
    # subprocess wait.
    tasks = []
    for covariate_mode in config.covariate_modes:
        for outcome_label in sorted(outcome_spec.keys()):
            triple = outcome_spec[outcome_label]
            if triple is None or len(triple) != 3:
                continue
            any_col = triple[2]
            if any_col is None or any_col not in phylo_data.columns:
                logger.info(f"  skipping [{outcome_label}] — binary column '{any_col}' absent")
                continue
            include_plasmid_count = (outcome_label != "any_plasmid")
            covariates = list(config.resolve_covariates(
                config.covariate_columns_for_mode(
                    covariate_mode,
                    include_plasmid_count=include_plasmid_count),
                phylo_data))

            for direction in directions:
                tasks.append((phylo_data, defense_cols, any_col,
                              outcome_label, direction, tree_path,
                              config, logger, workdir, covariates,
                              covariate_mode))

    if tasks:
        n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
        # Cap parallelism at the number of tasks — no point spawning more
        # threads than work units.
        n_jobs = max(1, min(n_jobs, len(tasks)))
        logger.info(
            f"  dispatching {len(tasks)} phyloglm calls across {n_jobs} parallel workers"
        )
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(_run_one_direction)(*t) for t in tasks
        )
        pieces: List[pd.DataFrame] = [df for df in results if df is not None and not df.empty]
    else:
        pieces = []

    if not pieces:
        return pd.DataFrame(columns=[
            "defense_system", "outcome_label", "direction", "covariate_mode",
            "phyloglm_coefficient", "phyloglm_std_err",
            "phyloglm_z_value", "phyloglm_p_value", "phyloglm_fdr_qvalue",
        ])
    combined = pd.concat(pieces, ignore_index=True)
    return combined.sort_values(
        ["covariate_mode", "outcome_label", "direction", "phyloglm_p_value"]
    ).reset_index(drop=True)

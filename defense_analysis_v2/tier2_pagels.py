"""Tier 2: Pagel's test for correlated binary-trait evolution.

Complements phyloglm by testing a different, stronger null: that the two
binary traits (defense-system presence and plasmid-class carriage) evolve
independently on the phylogeny. Rejection means the evolution of one is
informative about the other — a claim mechanistically different from
phyloglm's "species-level conditional association".

Pagel's test is bivariate and does not accept covariates. It is run against
every binary plasmid-class outcome stratum (any_plasmid_<class>). No reverse-
direction version is needed — the test is symmetric between the two traits.

Because ``fitPagel`` at full tree scale is prohibitive, each call is given a
uniform subsample of ``config.pagels_subsample_size`` species. Rather than
relying on a single draw (which introduces run-to-run noise that a single
p-value can't reveal), we take ``config.pagels_n_subsamples`` independent
subsamples, fit Pagel's test on each, and report the median logLR p-value
plus the fraction of subsamples significant at alpha. The median p is what
downstream consensus uses — it's more stable than any single subsample and
doesn't rely on asymptotic arguments we can't check here.

Both tests appearing with consistent direction is the strongest evidence.
Pagel significant but phyloglm not significant usually indicates shared-lineage
signal without species-level conditional association, which phyloglm's
covariate-adjusted fit has controlled for.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import apply_fdr


def _run_pagels_single(phylo_data: pd.DataFrame,
                       defense_cols: List[str],
                       outcome_col: str,
                       outcome_label: str,
                       tree_path: str,
                       config: Config,
                       logger: logging.Logger,
                       workdir: Path,
                       max_species: int) -> pd.DataFrame:
    """Run Pagel's test on ``config.pagels_n_subsamples`` independent uniform
    subsamples and aggregate. Returns one row per (defense_system,
    outcome_label) with median / min / max log-LR p, the fraction of
    subsamples significant at alpha, and the raw per-subsample p-values
    stashed as a semicolon-delimited string for audit.
    """
    n_sub = max(1, int(config.pagels_n_subsamples))
    per_sub_frames: List[pd.DataFrame] = []
    for i in range(n_sub):
        r = call_r_script(
            "pagels_test.R",
            tree_path=tree_path,
            data=phylo_data,
            args={
                "response": outcome_col,
                "predictors": defense_cols,
                "tip_column": "tip",
                "max_species": max_species,
                "min_count": config.min_count_per_category,
                # Distinct seed per subsample so each draw is independent.
                "seed": int(config.random_seed) + i,
            },
            logger=logger,
            r_executable=config.r_executable,
            workdir=workdir / f"pagels_{outcome_label}" / f"sub_{i:02d}",
            timeout=60 * 60 * 6,
        )
        if not r.ok:
            logger.warning(
                f"pagels_test [{outcome_label}] subsample {i} failed: {r.error}"
            )
            continue
        sub_df = r.dataframe.copy()
        sub_df["subsample_id"] = i
        per_sub_frames.append(sub_df)

    if not per_sub_frames:
        logger.error(f"pagels_test [{outcome_label}] failed for every subsample")
        return pd.DataFrame()

    long = pd.concat(per_sub_frames, ignore_index=True)
    # Aggregate per defense system: median p, fraction significant pre-FDR,
    # and an audit trail.
    rows = []
    for system, g in long.groupby("defense_system"):
        ps = g["pagel_p_value"].dropna().values
        dll = g["pagel_delta_logL"].dropna().values
        n_ok = len(ps)
        row = {
            "defense_system": system,
            "pagel_p_value": float(np.median(ps)) if n_ok else np.nan,
            "pagel_p_min": float(np.min(ps)) if n_ok else np.nan,
            "pagel_p_max": float(np.max(ps)) if n_ok else np.nan,
            "pagel_delta_logL": float(np.median(dll)) if len(dll) else np.nan,
            "pagel_n_subsamples_fit": int(n_ok),
            "pagel_frac_subsamples_sig_raw": (
                float((ps < config.alpha).mean()) if n_ok else np.nan),
            "pagel_p_values_per_subsample": ";".join(
                f"{p:.4g}" for p in ps),
            # A skip_reason survives if any subsample skipped; use the first.
            "skip_reason": (g["skip_reason"].dropna().iloc[0]
                            if "skip_reason" in g.columns
                               and g["skip_reason"].notna().any()
                            else np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["outcome_label"] = outcome_label
    # Pagel's is bivariate and has no covariates; label "none" so downstream
    # consumers can filter consistently with the other methods.
    df["covariate_mode"] = "none"
    df["pagel_fdr_qvalue"] = apply_fdr(df["pagel_p_value"],
                                        method=config.fdr_method).values
    n_sig = int((df["pagel_fdr_qvalue"] < config.alpha).sum())
    n_run = int(df["pagel_p_value"].notna().sum())
    logger.info(
        f"  Pagel [{outcome_label}]: {n_run} fit over {n_sub} subsamples; "
        f"{n_sig} at FDR < {config.alpha} on median-p"
    )
    return df


def run_pagels_test(phylo_data: pd.DataFrame,
                    defense_cols: List[str],
                    tree_path: str,
                    config: Config,
                    logger: logging.Logger,
                    workdir: Path,
                    max_species: Optional[int] = None,
                    outcome_spec: Optional[Dict[str, List[Optional[str]]]] = None
                    ) -> pd.DataFrame:
    """Run Pagel's test across every binary plasmid-class outcome.

    ``max_species`` defaults to ``config.pagels_subsample_size``. Each call
    uses ``config.pagels_n_subsamples`` independent subsamples (see module
    docstring).
    """
    if outcome_spec is None:
        outcome_spec = {"any_plasmid": [None, None, "has_plasmid_binary"]}
    if max_species is None:
        max_species = int(config.pagels_subsample_size)

    logger.info(
        f"Tier 2 Pagel's test — {len(defense_cols)} systems, "
        f"{len(outcome_spec)} outcome strata, "
        f"{config.pagels_n_subsamples} subsamples of {max_species} species each"
    )

    pieces: List[pd.DataFrame] = []
    for outcome_label in sorted(outcome_spec.keys()):
        triple = outcome_spec[outcome_label]
        if triple is None or len(triple) != 3:
            continue
        any_col = triple[2]
        if any_col is None or any_col not in phylo_data.columns:
            continue
        df = _run_pagels_single(phylo_data, defense_cols, any_col,
                                outcome_label, tree_path, config, logger,
                                workdir, max_species)
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame(columns=[
            "defense_system", "outcome_label", "pagel_p_value",
            "pagel_delta_logL", "pagel_logL_indep", "pagel_logL_dep",
            "pagel_fdr_qvalue", "skip_reason"])
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["outcome_label", "pagel_p_value"]).reset_index(drop=True)

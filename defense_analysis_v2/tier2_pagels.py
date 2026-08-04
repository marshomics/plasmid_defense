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
relying on a single draw, we take ``config.pagels_n_subsamples`` near-disjoint
subsamples, fit Pagel's test on each, and combine the per-subsample p-values
with the CAUCHY (ACAT) combination.

Not the median. This module previously reported ``np.median(ps)`` and fed it
straight to BH, justifying it as "more stable than any single subsample" and
as not relying on "asymptotic arguments we can't check here". Both claims were
backwards: taking the median is precisely the step that introduces an unstated
and false distributional assumption. The median of k p-values is not a
p-value — under H0 with k = 5 it is Beta(3, 3), giving P(median < 0.05) ~
0.0012 against a nominal 0.05. Being super-uniform it did not inflate false
positives, so BH still controlled FDR, but it cost roughly a factor of 55 in
power and the reported q-values had no interpretation on the tested
hypothesis. It also silently mixed rows with different k (systems skipped in
some subsamples) into one BH family, where they have different null
distributions.

Cauchy combination is valid under arbitrary dependence, handles varying k on a
single scale, and was already implemented in ``stats_utils`` for the
cross-method combination.

Both tests appearing with consistent direction is the strongest evidence.
Pagel significant but phyloglm not significant usually indicates shared-lineage
signal without species-level conditional association, which phyloglm's
covariate-adjusted fit has controlled for.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import apply_fdr, combine_subsample_pvalues


def _run_one_pagels_subsample(phylo_data: pd.DataFrame,
                              defense_cols: List[str],
                              outcome_col: str,
                              outcome_label: str,
                              tree_path: str,
                              config: Config,
                              logger: logging.Logger,
                              workdir: Path,
                              max_species: int,
                              subsample_id: int,
                              fit_directional: bool = False
                              ) -> Optional[pd.DataFrame]:
    """Run a single Pagel's-test subsample. Pulled out so subsamples can
    be dispatched in parallel — each runs in its own R subprocess with
    its own seed and its own workdir.
    """
    timeout_seconds = max(60, int(config.pagels_timeout_hours) * 3600)
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
            "seed": int(config.random_seed) + subsample_id,
            # B2: fit the two restricted dependent models so evolutionary
            # ordering can be read off by AIC. Gated -- see fit_directional
            # below -- because it triples the cost of the most expensive stage
            # and direction is only MEANINGFUL where the traits are dependent.
            "fit_directional": bool(fit_directional),
            "directional_screen_alpha": float(
                config.pagels_directional_screen_alpha),
            "directional_only_if_dependent": bool(
                config.pagels_directional_only_if_dependent),
        },
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / f"pagels_{outcome_label}" / f"sub_{subsample_id:02d}",
        timeout=timeout_seconds,
    )
    if not r.ok:
        logger.warning(
            f"pagels_test [{outcome_label}] subsample {subsample_id} failed: {r.error}"
        )
        return None
    sub_df = r.dataframe.copy()
    sub_df["subsample_id"] = subsample_id
    sub_df["outcome_label"] = outcome_label
    return sub_df


_DIRECTION_MODELS = {
    "independent": "pagel_aic_independent",
    "plasmid_drives_defense": "pagel_aic_plasmid_drives_defense",
    "defense_drives_plasmid": "pagel_aic_defense_drives_plasmid",
    "mutual": "pagel_aic_mutual",
}


def _summarise_direction(g: pd.DataFrame, config: Config) -> dict:
    """B2 — collapse per-subsample directional AICs into one verdict.

    Four nested models are compared per subsample:

        independent             neither character's rates depend on the other
        plasmid_drives_defense  defense transition rates depend on plasmid state
        defense_drives_plasmid  plasmid transition rates depend on defense state
        mutual                  both

    Aggregation is by AKAIKE WEIGHTS computed WITHIN each subsample and then
    averaged across subsamples. Weights must be computed within a subsample
    because AIC is not comparable across different datasets -- each subsample
    is a different 500-tip draw with its own likelihood scale, so averaging raw
    AICs across draws would be meaningless. Within-subsample weights are on a
    common 0-1 scale and average legitimately.

    The verdict compares the two DIRECTIONAL models to each other, since that
    is the question: given that the traits are associated, which character's
    evolution is conditioned on the other's state? A direction is called only
    when the mean AIC advantage exceeds
    ``config.pagels_direction_min_delta_aic`` (2 by convention).
    """
    out: dict = {
        "pagel_direction": "not_fitted",
        "pagel_direction_delta_aic": np.nan,
        "pagel_weight_independent": np.nan,
        "pagel_weight_plasmid_drives_defense": np.nan,
        "pagel_weight_defense_drives_plasmid": np.nan,
        "pagel_weight_mutual": np.nan,
        "pagel_n_subsamples_directional": 0,
    }
    cols = list(_DIRECTION_MODELS.values())
    if not all(c in g.columns for c in cols):
        return out

    sub = g[cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if sub.empty:
        return out

    aic = sub.values.astype(float)
    # Akaike weights within each subsample (row).
    delta = aic - aic.min(axis=1, keepdims=True)
    rel = np.exp(-0.5 * delta)
    weights = rel / rel.sum(axis=1, keepdims=True)
    mean_w = weights.mean(axis=0)

    names = list(_DIRECTION_MODELS.keys())
    for name, w in zip(names, mean_w):
        out[f"pagel_weight_{name}"] = float(w)
    out["pagel_n_subsamples_directional"] = int(len(sub))

    mean_aic = dict(zip(names, aic.mean(axis=0)))
    d_def = mean_aic["defense_drives_plasmid"]
    d_pla = mean_aic["plasmid_drives_defense"]
    delta_dir = float(d_pla - d_def)   # >0 favours defense_drives_plasmid
    out["pagel_direction_delta_aic"] = delta_dir

    best = min(mean_aic, key=mean_aic.get)
    thresh = float(config.pagels_direction_min_delta_aic)
    if best == "independent":
        # No dependence at all: asking about direction is not meaningful.
        out["pagel_direction"] = "independent_no_dependence"
    elif abs(delta_dir) < thresh:
        out["pagel_direction"] = ("mutual_or_ambiguous" if best == "mutual"
                                  else "ambiguous")
    elif delta_dir > 0:
        out["pagel_direction"] = "defense_drives_plasmid"
    else:
        out["pagel_direction"] = "plasmid_drives_defense"
    return out


def _aggregate_pagels_subsamples(per_sub_frames: List[pd.DataFrame],
                                  outcome_label: str,
                                  config: Config,
                                  logger: logging.Logger,
                                  n_sub: int) -> pd.DataFrame:
    """Aggregate per-subsample Pagel's results for a single outcome into the
    combined-p summary rows the rest of the pipeline expects. Pulled out so
    aggregation works whether subsamples were dispatched per-outcome or in
    a single flattened pool.
    """
    per_sub_frames = [df for df in per_sub_frames if df is not None]

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
        n_skipped = int(len(g) - n_ok)
        row = {
            "defense_system": system,
            # Cauchy (ACAT) combination across subsamples, NOT the median.
            #
            # The median of k p-values is not a p-value: under H0 with k = 5
            # it is Beta(3, 3), so P(median < 0.05) is ~0.0012. Verified: in
            # 20,000 null draws the median fired at 0.0009 against a nominal
            # 0.05, i.e. a ~55x power loss, while Cauchy hit 0.0499. The
            # median was also incomparable across systems with different k,
            # which put rows with different null distributions into one BH
            # family. Cauchy is valid under arbitrary dependence and handles
            # varying k on one scale.
            "pagel_p_value": combine_subsample_pvalues(ps),
            "pagel_p_median": float(np.median(ps)) if n_ok else np.nan,
            "pagel_p_min": float(np.min(ps)) if n_ok else np.nan,
            "pagel_p_max": float(np.max(ps)) if n_ok else np.nan,
            "pagel_delta_logL": float(np.median(dll)) if len(dll) else np.nan,
            "pagel_n_subsamples_fit": int(n_ok),
            "pagel_n_subsamples_skipped": n_skipped,
            "pagel_frac_subsamples_sig_raw": (
                float((ps < config.alpha).mean()) if n_ok else np.nan),
            "pagel_p_values_per_subsample": ";".join(
                f"{p:.4g}" for p in ps),
            # A skip_reason survives if ANY subsample skipped. Previously this
            # coexisted with a significant q-value and no consumer filtered on
            # it. Now it is advisory only, and the decision is carried by
            # `pagel_usable` below.
            "skip_reason": (g["skip_reason"].dropna().iloc[0]
                            if "skip_reason" in g.columns
                               and g["skip_reason"].notna().any()
                            else np.nan),
        }
        # A system fit in only one or two subsamples out of ten has been
        # evaluated on ~1-2% of the tree and should not carry the same weight
        # in consensus as one fit in all of them.
        row["pagel_usable"] = bool(
            n_ok >= max(2, int(np.ceil(0.5 * len(g)))))
        row.update(_summarise_direction(g, config))
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
        f"{n_sig} at FDR < {config.alpha} on Cauchy-combined p"
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

    # Build a flat list of (outcome, subsample) tasks. Dispatching them in
    # one Parallel call (rather than 17 sequential per-outcome dispatches
    # of 5 subsamples each) keeps the worker pool full at all times — a
    # slow subsample for one outcome no longer blocks the start of the
    # next outcome's subsamples.
    n_sub = max(1, int(config.pagels_n_subsamples))
    tasks = []  # (outcome_label, outcome_col, subsample_id)
    for outcome_label in sorted(outcome_spec.keys()):
        triple = outcome_spec[outcome_label]
        if triple is None or len(triple) != 3:
            continue
        any_col = triple[2]
        if any_col is None or any_col not in phylo_data.columns:
            continue
        for sub_id in range(n_sub):
            tasks.append((outcome_label, any_col, sub_id))

    if not tasks:
        return pd.DataFrame(columns=[
            "defense_system", "outcome_label", "pagel_p_value",
            "pagel_delta_logL", "pagel_logL_indep", "pagel_logL_dep",
            "pagel_fdr_qvalue", "skip_reason"])

    n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
    n_jobs = max(1, min(n_jobs, len(tasks)))
    logger.info(
        f"  dispatching {len(tasks)} Pagel's-test subsamples "
        f"({len(tasks) // n_sub} outcomes x {n_sub} subsamples) across "
        f"{n_jobs} parallel workers; per-subsample timeout "
        f"{config.pagels_timeout_hours}h"
    )

    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(_run_one_pagels_subsample)(
            phylo_data, defense_cols, any_col, outcome_label,
            tree_path, config, logger, workdir, max_species, sub_id,
            # Directional fits triple the per-system cost of the most
            # expensive stage in the pipeline, so they are restricted to the
            # outcomes a primary claim may be made about. Within a call they
            # are further gated on the standard Pagel test rejecting
            # independence: asking "which drives which?" for an independent
            # pair is not a question.
            fit_directional=bool(
                config.pagels_fit_directional_models
                and (not config.pagels_directional_primary_outcomes_only
                     or outcome_label in config.primary_outcome_labels)))
        for outcome_label, any_col, sub_id in tasks
    )

    # Group subsample results by outcome and aggregate to per-system
    # combined-p rows.
    by_outcome: Dict[str, List[pd.DataFrame]] = {}
    for df in results:
        if df is None or df.empty:
            continue
        ol = df["outcome_label"].iloc[0]
        by_outcome.setdefault(ol, []).append(df)

    pieces: List[pd.DataFrame] = []
    for outcome_label in sorted(outcome_spec.keys()):
        per_sub = by_outcome.get(outcome_label, [])
        agg = _aggregate_pagels_subsamples(per_sub, outcome_label,
                                            config, logger, n_sub)
        if not agg.empty:
            pieces.append(agg)

    if not pieces:
        return pd.DataFrame(columns=[
            "defense_system", "outcome_label", "pagel_p_value",
            "pagel_delta_logL", "pagel_logL_indep", "pagel_logL_dep",
            "pagel_fdr_qvalue", "skip_reason"])
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["outcome_label", "pagel_p_value"]).reset_index(drop=True)

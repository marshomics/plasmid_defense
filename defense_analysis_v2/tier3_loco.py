"""Tier 3: clade robustness — influence diagnostics and heterogeneity.

Two distinct analyses live here, and the distinction matters.

**Leave-one-clade-out (LOCO): an INFLUENCE DIAGNOSTIC, no p-value.**
For each clade, drop all its species and refit. The useful output is how far
the coefficient moves — a system whose association vanishes when one clade is
removed is driven by that clade. That is a real and interpretable statement.

What LOCO cannot support is a heterogeneity TEST. Cochran's Q requires the k
effect estimates to be independent, and Q ~ chi2(k-1) only under that
assumption. LOCO estimates are fit on n - n_clade species out of the same n;
at ~40,000 species dropping one GTDB class typically removes a small fraction
of tips, so any two LOCO estimates share >90% of their data and correlate
near 1. The between-estimate variance is therefore deflated far below the
chi2 reference, Q comes out much smaller than its nominal null implies, and
``chi2.sf(Q, df)`` is grossly conservative. Bonferroni was then applied on top
of that, with the module comment justifying it as wanting to "be conservative
about calling something unstable" — conservative in the wrong direction, since
the null here is *stability*. The combined effect was a heterogeneity test
built never to fire, whose uniformly non-significant output would read as
"all systems are clade-robust".

**Within-clade fits: the valid heterogeneity test.**
Fit the model separately inside each clade. Those subsets are DISJOINT, so the
estimates are independent and Cochran's Q is valid as specified. Clades below
``config.min_species_per_within_clade_fit`` are excluded because a fit on 40
species contributes noise, not heterogeneity.

Both run at gtdb_class (primary, finer) and gtdb_phylum (fallback for rare
classes).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import cochran_q


def _run_loco_for_one_clade(phylo_data: pd.DataFrame, defense_cols: List[str],
                            tree_path: str, rank: str, clade: str,
                            covariates: List[str], outcome_col: str,
                            covariate_mode: str, config: Config,
                            logger: logging.Logger,
                            workdir: Path) -> pd.DataFrame:
    """Refit phyloglm with one specific clade dropped. Pulled out so the
    per-clade fits inside a (covariate_mode, rank) sweep can be dispatched
    in parallel — each runs in its own R subprocess and writes to its own
    workdir.
    """
    sub = phylo_data[phylo_data[rank] != clade]
    n_dropped = len(phylo_data) - len(sub)

    # The size gate belongs on the DROPPED clade, not on what remains.
    # Applied to the remainder (the previous behaviour) it can essentially
    # never fire at ~40,000 species, so the documented exclusion of small
    # clades from the heterogeneity test was never implemented, and singleton
    # clades contributed near-duplicates of the full-data estimate.
    if n_dropped < config.min_species_per_loco_clade:
        logger.debug(
            f"LOCO[{covariate_mode}/{rank}]: '{clade}' has only {n_dropped} "
            f"species (< {config.min_species_per_loco_clade}); fitting for the "
            f"influence record but excluding from heterogeneity")
    if len(sub) < 50:
        logger.debug(f"LOCO[{covariate_mode}/{rank}]: skip '{clade}' "
                     f"(only {len(sub)} species remain)")
        return pd.DataFrame()
    logger.info(
        f"LOCO[{covariate_mode}/{rank}]: excluding '{clade}' ({n_dropped} "
        f"species) -> {len(sub)} species remain"
    )

    r = call_r_script(
        "phyloglm_uni.R",
        tree_path=tree_path,
        data=sub,
        args={"response": outcome_col,
              "predictors": defense_cols,
              "mode": "predictor",
              "defense_side": "predictor",
              "covariates": covariates,
              "tip_column": "tip",
              "evolutionary_model": config.phyloglm_estimator,
              "btol": 20, "boot": 0,
              "min_count": config.min_count_per_category,
              "min_count_response": config.min_count_per_category},
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / f"loco_{covariate_mode}_{rank}" /
                f"excl_{clade.replace('/', '_')}",
    )
    if not r.ok:
        logger.warning(
            f"LOCO[{covariate_mode}/{rank}]: phyloglm failed excluding "
            f"'{clade}': {r.error}"
        )
        return pd.DataFrame()
    sub_out = r.dataframe.rename(columns={"test_label": "defense_system"})
    sub_out["excluded_clade"] = clade
    sub_out["rank"] = rank
    sub_out["covariate_mode"] = covariate_mode
    sub_out["n_species_remaining"] = len(sub)
    sub_out["n_species_dropped"] = n_dropped
    sub_out["clade_large_enough"] = n_dropped >= config.min_species_per_loco_clade
    return sub_out


def _run_within_clade_fit(phylo_data: pd.DataFrame, defense_cols: List[str],
                          tree_path: str, rank: str, clade: str,
                          covariates: List[str], outcome_col: str,
                          covariate_mode: str, config: Config,
                          logger: logging.Logger,
                          workdir: Path) -> pd.DataFrame:
    """Fit the model INSIDE one clade.

    Unlike LOCO, within-clade subsets are disjoint, so the resulting estimates
    are independent and Cochran's Q against chi2(k-1) is valid.
    """
    sub = phylo_data[phylo_data[rank] == clade]
    if len(sub) < config.min_species_per_within_clade_fit:
        return pd.DataFrame()

    r = call_r_script(
        "phyloglm_uni.R",
        tree_path=tree_path,
        data=sub,
        args={"response": outcome_col,
              "predictors": defense_cols,
              "mode": "predictor",
              "defense_side": "predictor",
              "covariates": covariates,
              "tip_column": "tip",
              "evolutionary_model": config.phyloglm_estimator,
              "btol": 20, "boot": 0,
              "min_count": config.min_count_per_category,
              "min_count_response": config.min_count_per_category},
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / f"within_{covariate_mode}_{rank}" /
                f"in_{clade.replace('/', '_')}",
    )
    if not r.ok:
        logger.warning(f"within-clade[{covariate_mode}/{rank}]: fit failed "
                       f"inside '{clade}': {r.error}")
        return pd.DataFrame()
    sub_out = r.dataframe.rename(columns={"test_label": "defense_system"})
    sub_out["clade"] = clade
    sub_out["rank"] = rank
    sub_out["covariate_mode"] = covariate_mode
    sub_out["n_species_in_clade"] = len(sub)
    return sub_out


def _run_loco_for_rank(phylo_data: pd.DataFrame, defense_cols: List[str],
                       tree_path: str, rank: str, config: Config,
                       logger: logging.Logger, workdir: Path,
                       outcome_col: str, covariate_mode: str) -> pd.DataFrame:
    """Run phyloglm with each rank-level clade removed in turn, in parallel,
    returning a long-form DataFrame: (defense_system, excluded_clade,
    covariate_mode, phyloglm_*).
    """
    clades = (phylo_data[rank].dropna().unique() if rank in phylo_data.columns else [])
    # Only fit clades large enough to move the estimate. Dropping a handful of
    # species from ~40,000 returns a near-duplicate of the full-data fit: it is
    # uninformative as an influence diagnostic and is excluded from
    # heterogeneity regardless, so fitting it is pure waste.
    if config.loco_fit_only_gated_clades and len(clades):
        sizes = phylo_data[rank].value_counts()
        before = len(clades)
        clades = [c for c in clades
                  if sizes.get(c, 0) >= config.min_species_per_loco_clade]
        logger.info(
            f"LOCO[{covariate_mode}/{rank}]: {len(clades)}/{before} clades have "
            f">= {config.min_species_per_loco_clade} species and will be fit")
    covariates = list(config.resolve_covariates(
        config.covariate_columns_for_mode(
            covariate_mode, include_plasmid_count=False), phylo_data))

    if len(clades) == 0:
        return pd.DataFrame()

    n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
    n_jobs = max(1, min(n_jobs, len(clades)))
    logger.info(
        f"LOCO[{covariate_mode}/{rank}]: dispatching {len(clades)} clade "
        f"leave-outs across {n_jobs} parallel workers"
    )
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(_run_loco_for_one_clade)(
            phylo_data, defense_cols, tree_path, rank, clade,
            covariates, outcome_col, covariate_mode, config, logger, workdir)
        for clade in clades
    )
    records = [df for df in results if df is not None and not df.empty]

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def run_loco_with_cochran_q(phylo_data: pd.DataFrame, defense_cols: List[str],
                            tree_path: str, config: Config,
                            logger: logging.Logger, workdir: Path,
                            full_dataset_phyloglm: pd.DataFrame,
                            outcome_label: str = "any_plasmid",
                            outcome_col: str = "has_plasmid_binary") -> dict:
    """Top-level LOCO driver. Runs against a single outcome (default: the
    legacy has_plasmid_binary) — the point of LOCO is to test stability of
    the primary association, not to repeat the analysis for every stratum.

    Returns a dict with two DataFrames:
        ``details``  — long-form, one row per (system, excluded_clade, rank)
        ``summary``  — one row per system, with per-rank Cochran Q + I^2 and
                       stability flag (Bonferroni-corrected p > 0.05 across
                       systems implies "not detectably heterogeneous")
    """
    # Scope. LOCO is a stability check on the PRIMARY result, so running it
    # across every covariate mode and both ranks multiplies cost with no added
    # inference. Restricting to the primary mode and primary rank cuts the fit
    # count by ~6x; the fits that remain are unchanged.
    modes = ([config.primary_covariate_mode]
             if config.loco_covariate_modes_primary_only
             else [m for m in config.covariate_modes
                   if not config.is_diagnostic_mode(m)])
    ranks = ([config.loco_ranks[0]] if config.loco_ranks_primary_only
             else list(config.loco_ranks))
    logger.info(f"LOCO scope: modes={modes}, ranks={ranks}")

    detail_frames = []
    for covariate_mode in modes:
        for rank in ranks:
            df = _run_loco_for_rank(phylo_data, defense_cols, tree_path, rank,
                                    config, logger, workdir, outcome_col,
                                    covariate_mode)
            if not df.empty:
                detail_frames.append(df)
    if not detail_frames:
        return {"details": pd.DataFrame(), "summary": pd.DataFrame()}

    details = pd.concat(detail_frames, ignore_index=True)
    details["outcome_label"] = outcome_label

    # Build summary per (covariate_mode, system). Full-dataset comparator is
    # looked up per covariate_mode.
    full = full_dataset_phyloglm if full_dataset_phyloglm is not None \
        else pd.DataFrame()
    if "outcome_label" in full.columns:
        full = full[(full["outcome_label"] == outcome_label)
                    & (full.get("direction", "plasmid_given_defense")
                       == "plasmid_given_defense")]

    summary_rows = []
    for (system, covariate_mode), group in details.groupby(
            ["defense_system", "covariate_mode"]):
        # Full-dataset comparator filtered to the matching covariate_mode
        full_sub = full
        if "covariate_mode" in full.columns:
            full_sub = full[full["covariate_mode"] == covariate_mode]
        full_by_system = full_sub.set_index("defense_system") if not full_sub.empty \
            else pd.DataFrame(columns=["phyloglm_coefficient"])
        row = {"defense_system": system, "covariate_mode": covariate_mode}
        for rank in config.loco_ranks:
            sub = group[group["rank"] == rank].dropna(
                subset=["phyloglm_coefficient", "phyloglm_std_err"])
            # Only clades large enough to matter contribute; a clade of 3
            # species returns a near-duplicate of the full-data estimate.
            if "clade_large_enough" in sub.columns:
                sub = sub[sub["clade_large_enough"].fillna(False).astype(bool)]

            full_coef = np.nan
            if system in full_by_system.index:
                v = full_by_system.loc[system, "phyloglm_coefficient"]
                full_coef = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)

            # ---- INFLUENCE diagnostics (no p-value) ----
            # LOCO estimates share >90% of their data, so they are not
            # independent and no valid heterogeneity test can be built from
            # them. What they DO support is an influence statement: how far
            # does the coefficient move, and does any single clade flip it?
            coefs = sub["phyloglm_coefficient"].values
            row[f"{rank}_n_clades"] = int(len(coefs))
            row[f"{rank}_coef_min"] = float(np.min(coefs)) if len(coefs) else np.nan
            row[f"{rank}_coef_max"] = float(np.max(coefs)) if len(coefs) else np.nan
            row[f"{rank}_coef_range"] = (float(np.ptp(coefs))
                                         if len(coefs) else np.nan)
            if np.isfinite(full_coef) and len(coefs):
                deltas = np.abs(coefs - full_coef)
                row[f"{rank}_max_abs_delta_vs_full"] = float(np.max(deltas))
                row[f"{rank}_max_influence_clade"] = str(
                    sub.iloc[int(np.argmax(deltas))]["excluded_clade"])
                row[f"{rank}_direction_preserved_frac"] = float(
                    (np.sign(coefs) == np.sign(full_coef)).mean())
                row[f"{rank}_any_sign_flip"] = bool(
                    (np.sign(coefs) != np.sign(full_coef)).any())
            else:
                row[f"{rank}_max_abs_delta_vs_full"] = np.nan
                row[f"{rank}_max_influence_clade"] = np.nan
                row[f"{rank}_direction_preserved_frac"] = np.nan
                row[f"{rank}_any_sign_flip"] = np.nan
            # A clade-driven association is one where dropping a single clade
            # flips the sign or removes most of the effect.
            row[f"{rank}_clade_driven"] = bool(
                row.get(f"{rank}_any_sign_flip") is True
                or (np.isfinite(full_coef) and abs(full_coef) > 0
                    and np.isfinite(row.get(f"{rank}_max_abs_delta_vs_full", np.nan))
                    and row[f"{rank}_max_abs_delta_vs_full"] > 0.5 * abs(full_coef))
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    # NOTE: Cochran's Q is deliberately NOT computed on LOCO estimates. See the
    # module docstring. The valid heterogeneity test lives in
    # `run_within_clade_heterogeneity`, which uses disjoint within-clade fits.

    primary_col = f"{config.loco_ranks[0]}_clade_driven"
    if primary_col in summary.columns:
        for cm, sub in summary.groupby("covariate_mode"):
            n_driven = int(sub[primary_col].fillna(False).astype(bool).sum())
            logger.info(
                f"LOCO [{cm}] ({config.loco_ranks[0]} primary): {n_driven} systems "
                f"are clade-driven (sign flip or >50% coefficient shift when a "
                f"single clade is dropped)"
            )

    return {"details": details, "summary": summary}


def run_within_clade_heterogeneity(phylo_data: pd.DataFrame,
                                   defense_cols: List[str],
                                   tree_path: str, config: Config,
                                   logger: logging.Logger, workdir: Path,
                                   outcome_label: str = "any_plasmid",
                                   outcome_col: str = "has_plasmid_binary"
                                   ) -> dict:
    """Cochran's Q on WITHIN-clade fits, which are independent.

    This is the valid heterogeneity test. Each clade is fit separately, so the
    subsets are disjoint, the estimates are independent, and Q ~ chi2(k-1)
    holds as specified. Clades smaller than
    ``config.min_species_per_within_clade_fit`` are excluded.

    BH rather than Bonferroni across systems: the previous code used
    Bonferroni on Q p-values from overlapping LOCO fits and justified it as
    wanting to be conservative about calling a system unstable. That is
    conservative in the wrong direction — the null is *stability*, so being
    reluctant to reject it means being reluctant to detect instability, which
    is not caution, it is blindness. With a valid test, standard FDR applies.
    """
    detail_frames = []
    for covariate_mode in config.covariate_modes:
        if config.is_diagnostic_mode(covariate_mode):
            continue
        for rank in config.loco_ranks:
            if rank not in phylo_data.columns:
                continue
            covariates = list(config.resolve_covariates(
                config.covariate_columns_for_mode(
                    covariate_mode, include_plasmid_count=False), phylo_data))
            clades = phylo_data[rank].dropna().unique()
            sizes = phylo_data[rank].value_counts()
            eligible = [c for c in clades
                        if sizes.get(c, 0) >= config.min_species_per_within_clade_fit]
            if len(eligible) < 3:
                logger.info(
                    f"within-clade[{covariate_mode}/{rank}]: only "
                    f"{len(eligible)} clades with >= "
                    f"{config.min_species_per_within_clade_fit} species; "
                    f"heterogeneity test needs at least 3")
                continue
            logger.info(
                f"within-clade[{covariate_mode}/{rank}]: fitting inside "
                f"{len(eligible)} clades")
            n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
            n_jobs = max(1, min(n_jobs, len(eligible)))
            results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
                delayed(_run_within_clade_fit)(
                    phylo_data, defense_cols, tree_path, rank, clade,
                    covariates, outcome_col, covariate_mode, config, logger,
                    workdir)
                for clade in eligible
            )
            frames = [d for d in results if d is not None and not d.empty]
            if frames:
                detail_frames.append(pd.concat(frames, ignore_index=True))

    if not detail_frames:
        return {"details": pd.DataFrame(), "summary": pd.DataFrame()}

    details = pd.concat(detail_frames, ignore_index=True)
    details["outcome_label"] = outcome_label

    rows = []
    for (system, covariate_mode), group in details.groupby(
            ["defense_system", "covariate_mode"]):
        row = {"defense_system": system, "covariate_mode": covariate_mode}
        for rank in config.loco_ranks:
            sub = group[group["rank"] == rank].dropna(
                subset=["phyloglm_coefficient", "phyloglm_std_err"])
            q = cochran_q(sub["phyloglm_coefficient"].values,
                          sub["phyloglm_std_err"].values)
            row[f"{rank}_Q"] = q["Q"]
            row[f"{rank}_Q_df"] = q["df"]
            row[f"{rank}_Q_p"] = q["p_value"]
            row[f"{rank}_I2"] = q["I2"]
            row[f"{rank}_n_clades"] = q["n_effective"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    for rank in config.loco_ranks:
        col = f"{rank}_Q_p"
        if col not in summary.columns:
            continue
        adj = np.full(len(summary), np.nan)
        for _, sub in summary.groupby("covariate_mode"):
            mask = sub[col].notna()
            if mask.sum() == 0:
                continue
            _, p_adj, _, _ = multipletests(sub.loc[mask, col].values,
                                           method="fdr_bh")
            adj[sub.index[mask.values]] = p_adj
        summary[f"{rank}_Q_fdr_qvalue"] = adj
        summary[f"{rank}_is_heterogeneous"] = summary[f"{rank}_Q_fdr_qvalue"] < config.alpha

    primary = f"{config.loco_ranks[0]}_is_heterogeneous"
    if primary in summary.columns:
        for cm, sub in summary.groupby("covariate_mode"):
            logger.info(
                f"within-clade heterogeneity [{cm}] ({config.loco_ranks[0]}): "
                f"{int(sub[primary].fillna(False).sum())} systems heterogeneous "
                f"at FDR q < {config.alpha}")
    return {"details": details, "summary": summary}

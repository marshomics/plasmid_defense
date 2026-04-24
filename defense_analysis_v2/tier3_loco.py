"""Tier 3: leave-one-clade-out robustness with Cochran's Q heterogeneity.

Replaces the arbitrary CV > 0.5 cutoff from new_scripts. Logic:

For each clade (at a given rank, e.g. ``gtdb_class``), drop all species in
that clade and rerun the primary phyloglm test on the remainder. Record the
phyloglm coefficient and SE for every (defense system x dropped clade) cell.

Apply Cochran's Q test for heterogeneity across the leave-one-clade-out
replicates per defense system; Q ~ chi^2(k-1) under H0 of no heterogeneity.
Systems whose Q-test p-value (Bonferroni-corrected across systems, because
here we *want* to be conservative about calling something unstable) falls
below 0.05 are flagged as clade-sensitive. I^2 is reported alongside so you
can see effect-magnitude heterogeneity independent of p-values.

The analysis runs at both gtdb_class (primary, finer) and gtdb_phylum
(fallback for rare classes). Class is reported first; phylum acts as a
sensitivity on the sensitivity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import cochran_q


def _run_loco_for_rank(phylo_data: pd.DataFrame, defense_cols: List[str],
                       tree_path: str, rank: str, config: Config,
                       logger: logging.Logger, workdir: Path,
                       outcome_col: str, covariate_mode: str) -> pd.DataFrame:
    """Run phyloglm with each rank-level clade removed in turn, returning a
    long-form DataFrame: (defense_system, excluded_clade, covariate_mode,
    phyloglm_*).
    """
    clades = (phylo_data[rank].dropna().unique() if rank in phylo_data.columns else [])
    covariates = list(config.covariate_columns_for_mode(
        covariate_mode, include_plasmid_count=False))
    covariates = [c for c in covariates if c in phylo_data.columns]

    records = []
    for clade in clades:
        sub = phylo_data[phylo_data[rank] != clade]
        if len(sub) < config.min_species_per_loco_clade:
            logger.debug(f"LOCO[{covariate_mode}/{rank}]: skip '{clade}' "
                         f"(remaining n={len(sub)} < {config.min_species_per_loco_clade})")
            continue
        logger.info(
            f"LOCO[{covariate_mode}/{rank}]: excluding '{clade}' -> "
            f"{len(sub)} species remain"
        )

        r = call_r_script(
            "phyloglm_uni.R",
            tree_path=tree_path,
            data=sub,
            args={"response": outcome_col,
                  "predictors": defense_cols,
                  "mode": "predictor",
                  "covariates": covariates,
                  "tip_column": "tip",
                  "evolutionary_model": config.phylo_evolutionary_model,
                  "btol": 20, "boot": 0,
                  "min_count": config.min_count_per_category},
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
            continue
        sub_out = r.dataframe.rename(columns={"test_label": "defense_system"})
        sub_out["excluded_clade"] = clade
        sub_out["rank"] = rank
        sub_out["covariate_mode"] = covariate_mode
        sub_out["n_species_remaining"] = len(sub)
        records.append(sub_out)

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
    detail_frames = []
    for covariate_mode in config.covariate_modes:
        for rank in config.loco_ranks:
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
            sub = group[group["rank"] == rank].dropna(subset=["phyloglm_coefficient",
                                                              "phyloglm_std_err"])
            q = cochran_q(sub["phyloglm_coefficient"].values,
                          sub["phyloglm_std_err"].values)
            row[f"{rank}_Q"] = q["Q"]
            row[f"{rank}_Q_df"] = q["df"]
            row[f"{rank}_Q_p"] = q["p_value"]
            row[f"{rank}_I2"] = q["I2"]
            row[f"{rank}_n_clades"] = q["n_effective"]
            full_coef = np.nan
            if system in full_by_system.index:
                v = full_by_system.loc[system, "phyloglm_coefficient"]
                full_coef = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
            if np.isfinite(full_coef) and not sub.empty:
                row[f"{rank}_direction_preserved_frac"] = float(
                    (np.sign(sub["phyloglm_coefficient"]) == np.sign(full_coef)).mean()
                )
            else:
                row[f"{rank}_direction_preserved_frac"] = np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Bonferroni across systems within covariate_mode for each rank's Q
    # p-value. Correcting across covariate_modes mixes dependent tests.
    for rank in config.loco_ranks:
        col = f"{rank}_Q_p"
        if col not in summary.columns:
            continue
        adj_all = np.full(len(summary), np.nan)
        for covariate_mode, sub in summary.groupby("covariate_mode"):
            mask = sub[col].notna()
            if mask.sum() == 0:
                continue
            _, p_adj, _, _ = multipletests(sub.loc[mask, col].values,
                                           method="bonferroni")
            idx = sub.index[mask.values]
            adj_all[idx] = p_adj
        summary[f"{rank}_Q_p_bonferroni"] = adj_all
        summary[f"{rank}_is_heterogeneous"] = summary[f"{rank}_Q_p_bonferroni"] < 0.05

    primary_het_col = f"{config.loco_ranks[0]}_is_heterogeneous"
    if primary_het_col in summary.columns:
        for cm, sub in summary.groupby("covariate_mode"):
            n_het = int(sub[primary_het_col].sum())
            logger.info(
                f"LOCO [{cm}] ({config.loco_ranks[0]} primary): {n_het} systems "
                f"flagged heterogeneous at Bonferroni-adjusted Q p < 0.05"
            )

    return {"details": details, "summary": summary}

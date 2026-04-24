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

import numpy as np
import pandas as pd

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
            "evolutionary_model": config.phylo_evolutionary_model,
            "btol": 20,
            "boot": 0,
            "min_count": config.min_count_per_category,
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
    df["phyloglm_fdr_qvalue"] = apply_fdr(df["phyloglm_p_value"],
                                          method=config.fdr_method).values
    df["phyloglm_odds_ratio"] = np.exp(df["phyloglm_coefficient"])
    df["phyloglm_ci_low"] = np.exp(df["phyloglm_coefficient"] - 1.96 * df["phyloglm_std_err"])
    df["phyloglm_ci_high"] = np.exp(df["phyloglm_coefficient"] + 1.96 * df["phyloglm_std_err"])

    n_sig = int((df["phyloglm_fdr_qvalue"] < config.alpha).sum())
    n_run = int(df["phyloglm_p_value"].notna().sum())
    logger.info(
        f"  phyloglm [{covariate_mode}/{outcome_label}/{direction}]: "
        f"{n_run} systems fit; {n_sig} at FDR < {config.alpha}"
    )
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

    # Ensure log_n_plasmids exists for binary-class outcomes
    if "n_plasmids" in phylo_data.columns and "log_n_plasmids" not in phylo_data.columns:
        phylo_data = phylo_data.copy()
        phylo_data["log_n_plasmids"] = np.log1p(
            phylo_data["n_plasmids"].fillna(0).clip(lower=0))

    directions = ["plasmid_given_defense"]
    if config.run_bidirectional:
        directions.append("defense_given_plasmid")

    logger.info(
        f"Tier 2 phyloglm (univariate, {config.phylo_evolutionary_model}) — "
        f"{len(defense_cols)} systems, {len(phylo_data)} species, "
        f"{len(outcome_spec)} outcomes x {len(directions)} directions"
    )

    pieces: List[pd.DataFrame] = []
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
            covariates = list(config.covariate_columns_for_mode(
                covariate_mode, include_plasmid_count=include_plasmid_count))
            covariates = [c for c in covariates if c in phylo_data.columns]

            for direction in directions:
                df = _run_one_direction(phylo_data, defense_cols, any_col,
                                        outcome_label, direction, tree_path,
                                        config, logger, workdir, covariates,
                                        covariate_mode)
                if not df.empty:
                    pieces.append(df)

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

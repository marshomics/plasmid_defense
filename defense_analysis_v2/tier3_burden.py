"""Tier 3 burden: does total defense-system count differ by plasmid status,
controlling for phylogeny?

Primary test: PGLS (phylogenetic generalised least squares) with a Pagel's
lambda covariance structure — lambda is estimated by ML rather than fixed at 0
(no correction) or 1 (full Brownian). This is what replaces the plain
Mann-Whitney / point-biserial / logistic tests from
new_scripts/defense_burden_correlation.py, which ignored phylogeny entirely.

Secondary: phylogenetic logistic regression with defense burden as the
*predictor* of plasmid presence — same direction as the primary phyloglm
results, but at the aggregate level.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .r_bridge import call_r_script


def run_burden_pgls(phylo_data: pd.DataFrame, tree_path: str,
                    config: Config, logger: logging.Logger,
                    workdir: Path) -> pd.DataFrame:
    """PGLS of defense burden count ~ plasmid status, run once per
    covariate_mode. Returns a long-form DataFrame with ``covariate_mode``
    column. Pagel's lambda is estimated by ML.
    """
    if "defense_burden_count" not in phylo_data.columns:
        raise ValueError("defense_burden_count must be present on phylo_data; "
                         "call io_utils.add_defense_burden() first")

    pieces: List[pd.DataFrame] = []
    for covariate_mode in config.covariate_modes:
        # The unadjusted arm is skipped here, unlike elsewhere.
        # `defense_burden_count` is a sum of ~435 independently-saturating
        # binary indicators, so E[burden | n_strains] climbs steeply and
        # monotonically, while P(plasmid label = 1 | n_strains) does the same.
        # An unadjusted PGLS of one on the other is close to a pure
        # measurement of Cov(n_strains, n_strains) and has no diagnostic value
        # even as a positive control.
        if config.is_diagnostic_mode(covariate_mode):
            logger.info(f"Tier 3 burden: skipping '{covariate_mode}' — burden "
                        f"is the most depth-inflated quantity in the pipeline "
                        f"and an unadjusted fit is uninterpretable")
            continue
        logger.info(
            f"Tier 3 burden [{covariate_mode}]: PGLS sqrt(burden) ~ plasmid "
            f"status (Pagel's lambda estimated by ML)"
        )
        covariates = list(config.resolve_covariates(
            config.covariate_columns_for_mode(
                covariate_mode, include_plasmid_count=False), phylo_data))
        pass_cols = ["tip", "has_plasmid_binary", "defense_burden_count"] + covariates

        r = call_r_script(
            "pgls_burden.R",
            tree_path=tree_path,
            data=phylo_data[pass_cols],
            args={
                "response": "defense_burden_count",
                "predictor": "has_plasmid_binary",
                "covariates": covariates,
                "tip_column": "tip",
                # nlme::gls is Gaussian by construction and the response is a
                # bounded, right-skewed, over-dispersed count. sqrt is the
                # standard variance-stabiliser for counts and is monotone, so
                # sign and significance are preserved.
                "transform": "sqrt",
            },
            logger=logger,
            r_executable=config.r_executable,
            workdir=workdir / f"pgls_burden_{covariate_mode}",
        )

        if not r.ok:
            logger.warning(f"pgls_burden [{covariate_mode}] failed: {r.error}")
            pieces.append(pd.DataFrame([{
                "analysis": "pgls_burden", "covariate_mode": covariate_mode,
                "pgls_p_value": np.nan, "pagel_lambda": np.nan,
                "error": r.error,
            }]))
            continue
        df = r.dataframe.copy()
        df["covariate_mode"] = covariate_mode
        pieces.append(df)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def run_burden_phyloglm(phylo_data: pd.DataFrame, tree_path: str,
                        config: Config, logger: logging.Logger,
                        workdir: Path) -> pd.DataFrame:
    """Secondary: phylogenetic logistic regression of plasmid ~ burden,
    run once per covariate_mode. Uses defense_burden_count as the single
    predictor.
    """
    pieces: List[pd.DataFrame] = []
    for covariate_mode in config.covariate_modes:
        if config.is_diagnostic_mode(covariate_mode):
            continue
        logger.info(
            f"Tier 3 burden [{covariate_mode}]: phyloglm plasmid ~ burden_count"
        )
        covariates = list(config.resolve_covariates(
            config.covariate_columns_for_mode(
                covariate_mode, include_plasmid_count=False), phylo_data))
        pass_cols = ["tip", "has_plasmid_binary", "defense_burden_count"] + covariates

        r = call_r_script(
            "phyloglm_uni.R",
            tree_path=tree_path,
            data=phylo_data[pass_cols],
            args={
                "response": "has_plasmid_binary",
                "predictors": ["defense_burden_count"],
                "mode": "predictor",
                "covariates": covariates,
                "tip_column": "tip",
                "evolutionary_model": config.phyloglm_estimator,
                "btol": 20,
                "boot": 0,
                # Burden is continuous, so the presence/absence gate does not
                # apply to the predictor, but the response gate still does.
                "min_count": 1,
                "min_count_response": config.min_count_per_category,
                "defense_side": "predictor",
            },
            logger=logger,
            r_executable=config.r_executable,
            workdir=workdir / f"phyloglm_burden_{covariate_mode}",
        )
        if not r.ok:
            logger.warning(f"phyloglm_burden [{covariate_mode}] failed: {r.error}")
            pieces.append(pd.DataFrame([{
                "defense_system": "defense_burden_count",
                "covariate_mode": covariate_mode,
                "phyloglm_p_value": np.nan, "error": r.error,
            }]))
            continue
        df = r.dataframe.rename(columns={"test_label": "defense_system"})
        df["covariate_mode"] = covariate_mode
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

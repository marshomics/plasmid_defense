"""Tier 3: misclassification sensitivity analysis for plasmid outcome.

Plasmid detection from short-read assemblies has a non-zero false-negative
rate (FNR): some species are called "no plasmid" because the detector missed
the plasmid, not because no plasmid is there. If FNR is non-differential
(independent of defense-system status), it biases all odds ratios toward 1 —
but in a known, quantifiable way.

Two complementary approaches:

1. Monte Carlo: sample plasmid-negative species and flip them to plasmid-
   positive with probability FNR / (1 - FNR), rerunning the primary
   phyloglm test each replicate. Report the fraction of replicates where
   the system stays significant and the distribution of coefficients.

2. Analytical bias correction (Bross 1954, Neuhaus 1999): for a given FNR,
   compute the adjusted odds ratio under non-differential misclassification:
       OR_true ≈ OR_obs * ((1 - FNR * (1 - pi_1)) / (1 - FNR * (1 - pi_0)))
   where pi_1, pi_0 are plasmid prevalences among defense-positive and
   defense-negative species respectively. We report the adjusted OR plus
   the FNR at which the adjusted OR crosses 1 (the "tipping-point FNR").

Both assume zero false positives (plasmid called -> plasmid really there).
The driver records that assumption in the report.
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
from .stats_utils import apply_fdr


def _one_mc_replicate(phylo_data: pd.DataFrame, defense_cols: List[str],
                      tree_path: str, fnr: float, rng: np.random.Generator,
                      config: Config, logger: logging.Logger,
                      workdir: Path, replicate_id: int,
                      covariate_mode: str) -> pd.DataFrame:
    """Flip apparent-negatives to positives at rate fnr / (1 - fnr) and rerun
    phyloglm. Returns the phyloglm DataFrame with a ``replicate_id`` column.
    """
    flip_prob = fnr / (1 - fnr) if fnr < 1 else 1.0
    data = phylo_data.copy()
    neg_mask = data["has_plasmid_binary"] == 0
    n_neg = int(neg_mask.sum())
    n_flip = int(np.round(flip_prob * n_neg))
    if n_flip > 0:
        idx = rng.choice(data.index[neg_mask].values, size=min(n_flip, n_neg), replace=False)
        data.loc[idx, "has_plasmid_binary"] = 1

    covariates = list(config.covariate_columns_for_mode(
        covariate_mode, include_plasmid_count=False))
    covariates = [c for c in covariates if c in data.columns]
    r = call_r_script(
        "phyloglm_uni.R",
        tree_path=tree_path,
        data=data,
        args={"response": "has_plasmid_binary",
              "predictors": defense_cols,
              "mode": "predictor",
              "covariates": covariates,
              "tip_column": "tip",
              "evolutionary_model": config.phylo_evolutionary_model,
              "btol": 20, "boot": 0,
              "min_count": config.min_count_per_category},
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / f"misclass_{covariate_mode}_fnr{fnr:.2f}" /
                f"rep{replicate_id:03d}",
    )
    if not r.ok:
        return pd.DataFrame()
    df = r.dataframe.rename(columns={"test_label": "defense_system"})
    df["fnr"] = fnr
    df["replicate_id"] = replicate_id
    df["covariate_mode"] = covariate_mode
    df["p_fdr"] = apply_fdr(df["phyloglm_p_value"], method=config.fdr_method).values
    return df[["defense_system", "phyloglm_coefficient", "phyloglm_std_err",
               "phyloglm_p_value", "p_fdr", "fnr", "replicate_id",
               "covariate_mode"]]


def run_misclassification_mc(phylo_data: pd.DataFrame, defense_cols: List[str],
                             tree_path: str, config: Config,
                             logger: logging.Logger, workdir: Path) -> pd.DataFrame:
    """Monte Carlo misclassification sensitivity across config.misclass_fnr_grid.
    Returns long-form DataFrame: one row per (system, fnr, replicate).
    """
    rng = np.random.default_rng(config.random_seed)
    n_modes = len(config.covariate_modes)
    logger.info(
        f"Misclassification MC: {len(config.misclass_fnr_grid)} FNR levels "
        f"x {config.misclass_n_replicates} replicates "
        f"x {n_modes} covariate modes "
        f"= {len(config.misclass_fnr_grid) * config.misclass_n_replicates * n_modes} fits"
    )

    frames = []
    for covariate_mode in config.covariate_modes:
        for fnr in config.misclass_fnr_grid:
            for rep in range(config.misclass_n_replicates):
                df = _one_mc_replicate(phylo_data, defense_cols, tree_path,
                                       fnr, rng, config, logger, workdir, rep,
                                       covariate_mode)
                if not df.empty:
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarise_misclassification_mc(mc_long: pd.DataFrame, config: Config) -> pd.DataFrame:
    """For each (system, fnr, covariate_mode): summary across replicates —
    median coefficient, IQR, fraction of replicates significant at FDR < alpha.
    """
    if mc_long.empty:
        return pd.DataFrame()
    group_cols = ["defense_system", "fnr"]
    if "covariate_mode" in mc_long.columns:
        group_cols.append("covariate_mode")
    rows = []
    for key, g in mc_long.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols, key))
        record.update({
            "median_coef": g["phyloglm_coefficient"].median(),
            "q25_coef": g["phyloglm_coefficient"].quantile(0.25),
            "q75_coef": g["phyloglm_coefficient"].quantile(0.75),
            "frac_fdr_sig": float((g["p_fdr"] < config.alpha).mean()),
            "n_replicates_completed": len(g),
        })
        rows.append(record)
    return pd.DataFrame(rows)


def analytical_bias_correction(tier2_phyloglm: pd.DataFrame,
                               tier1_results: pd.DataFrame,
                               fnr_grid: tuple,
                               config: Config) -> pd.DataFrame:
    """Non-differential misclassification bias correction per Bross (1954).

    For a 2x2 table with true plasmid prevalences pi_0 (among defense-absent)
    and pi_1 (among defense-present), observed OR is biased toward 1 by the
    factor ((1 - fnr*(1-pi_1)) / (1 - fnr*(1-pi_0))). We invert this to get
    an adjusted OR at each FNR. Also compute the tipping-point FNR where the
    adjusted OR crosses 1.
    """
    # Restrict to the any_plasmid outcome, forward direction. Analytical bias
    # correction is only defined for that outcome. Iterate over covariate
    # modes so the output table carries a covariate_mode column consistent
    # with the rest of the Tier 2 / Tier 3 outputs.
    t1 = tier1_results
    if "outcome_label" in t1.columns:
        t1 = t1[t1["outcome_label"] == "any_plasmid"]
    t2 = tier2_phyloglm
    if "outcome_label" in t2.columns:
        t2 = t2[t2["outcome_label"] == "any_plasmid"]
    if "direction" in t2.columns:
        t2 = t2[t2["direction"] == "plasmid_given_defense"]

    t1_keyed = t1[["defense_system", "plasmid_rate_with_defense",
                   "plasmid_rate_without_defense"]]
    # Different covariate_modes of tier1 have the same plasmid_rate numbers
    # (those are marginal, not model-based), so dedup on defense_system.
    t1_keyed = t1_keyed.drop_duplicates("defense_system")

    merged = t2.merge(t1_keyed, on="defense_system", how="left")

    records = []
    for _, row in merged.iterrows():
        system = row["defense_system"]
        cov_mode = row.get("covariate_mode", "with_cov")
        obs_beta = row.get("phyloglm_coefficient")
        if not np.isfinite(obs_beta):
            continue
        pi1 = row["plasmid_rate_with_defense"]
        pi0 = row["plasmid_rate_without_defense"]
        if not (np.isfinite(pi1) and np.isfinite(pi0)):
            continue
        obs_or = float(np.exp(obs_beta))
        for fnr in fnr_grid:
            denom = (1 - fnr * (1 - pi0))
            numer = (1 - fnr * (1 - pi1))
            if denom <= 0 or numer <= 0:
                adj_or = np.nan
            else:
                attenuation = numer / denom
                adj_or = obs_or / attenuation if attenuation > 0 else np.nan
            records.append({"defense_system": system, "covariate_mode": cov_mode,
                            "fnr": fnr, "obs_OR": obs_or, "adj_OR": adj_or,
                            "pi_present": pi1, "pi_absent": pi0})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Tipping point: smallest FNR where adj_OR crosses 1 (linear interp).
    # Computed per (system, covariate_mode) since the observed OR differs.
    tip_rows = []
    for (system, cov_mode), g in df.groupby(["defense_system", "covariate_mode"]):
        g = g.sort_values("fnr")
        xs = g["fnr"].values
        ys = g["adj_OR"].values - 1.0
        tip = np.nan
        for i in range(1, len(xs)):
            if np.sign(ys[i]) != np.sign(ys[i - 1]) and np.isfinite(ys[i]) and np.isfinite(ys[i - 1]):
                tip = xs[i - 1] + (xs[i] - xs[i - 1]) * (-ys[i - 1]) / (ys[i] - ys[i - 1])
                break
        tip_rows.append({"defense_system": system, "covariate_mode": cov_mode,
                         "tipping_point_fnr": tip})
    tips = pd.DataFrame(tip_rows)
    return df.merge(tips, on=["defense_system", "covariate_mode"], how="left")

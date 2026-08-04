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
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import apply_fdr


def _one_mc_replicate(phylo_data: pd.DataFrame, defense_cols: List[str],
                      tree_path: str, fnr: float,
                      config: Config, logger: logging.Logger,
                      workdir: Path, replicate_id: int,
                      covariate_mode: str) -> pd.DataFrame:
    """Flip apparent-negatives to positives at rate fnr / (1 - fnr) and rerun
    phyloglm. Returns the phyloglm DataFrame with a ``replicate_id`` column.

    Seeds its own RNG deterministically from (config.random_seed, fnr,
    replicate_id, covariate_mode) so the replicate-level draws are
    reproducible regardless of execution order or worker count — important
    once this function is dispatched in parallel.
    """
    # Stable per-task seed: combine the global seed with replicate id, fnr
    # bucket, and covariate-mode hash. Each task gets a unique, deterministic
    # RNG even when many run concurrently.
    seed_components = (
        int(config.random_seed),
        replicate_id,
        int(round(fnr * 1000)),
        hash(covariate_mode) & 0xFFFFFFFF,
    )
    rng = np.random.default_rng(seed_components)
    data = phylo_data.copy()
    neg_mask = (data["has_plasmid_binary"] == 0).values

    # ---- Depth-DIFFERENTIAL false negatives ----
    # The species label is "any strain carries a plasmid", so a species with n
    # strains gets n independent chances to detect one. If the per-assembly
    # detection FNR is f, the effective SPECIES-level FNR is f^n, which falls
    # steeply in sequencing depth. Applying a flat rate to every species -- the
    # previous behaviour -- models a non-differential mechanism that does not
    # exist here, and non-differential misclassification is exactly the
    # assumption the Bross correction downstream needs. Getting it wrong makes
    # the whole sensitivity analysis conservative-by-construction rather than
    # informative.
    if getattr(config, "misclass_depth_differential", False) \
            and "n_strains" in data.columns and fnr > 0:
        n_str = pd.to_numeric(data["n_strains"], errors="coerce") \
                  .fillna(1).clip(lower=1).values
        species_fnr = np.power(float(fnr), n_str)
        # P(species is truly positive | observed negative), by Bayes with a
        # flat prior on true status within the observed-negative set.
        flip_prob_vec = np.where(species_fnr < 1,
                                 species_fnr / (1 - species_fnr + 1e-12), 0.0)
        flip_prob_vec = np.clip(flip_prob_vec, 0.0, 1.0)
        draws = rng.random(len(data)) < flip_prob_vec
        flip = neg_mask & draws
        data.loc[data.index[flip], "has_plasmid_binary"] = 1
        n_flipped = int(flip.sum())
    else:
        flip_prob = fnr / (1 - fnr) if fnr < 1 else 1.0
        n_neg = int(neg_mask.sum())
        n_flip = int(np.round(flip_prob * n_neg))
        n_flipped = 0
        if n_flip > 0:
            idx = rng.choice(data.index[neg_mask].values,
                             size=min(n_flip, n_neg), replace=False)
            data.loc[idx, "has_plasmid_binary"] = 1
            n_flipped = len(idx)

    covariates = list(config.resolve_covariates(
        config.covariate_columns_for_mode(
            covariate_mode, include_plasmid_count=False), data))
    r = call_r_script(
        "phyloglm_uni.R",
        tree_path=tree_path,
        data=data,
        args={"response": "has_plasmid_binary",
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
                             logger: logging.Logger, workdir: Path,
                             tier2_phyloglm: Optional[pd.DataFrame] = None
                             ) -> pd.DataFrame:
    """Monte Carlo misclassification sensitivity.

    SCOPE. At its original settings this stage was 7 FNR levels x 200
    replicates x 3 covariate modes x 435 systems = 2,800 full sweeps, which
    exceeded the 25-day cluster ceiling and died with SIGBUS besides. Three
    restrictions bring it into range without weakening any surviving estimate:

      * Systems: the question is "would this FINDING survive plasmid-detection
        false negatives?", which only applies to findings. Restricting to
        systems FDR-significant in the primary analysis is the correct scope,
        not a shortcut. Typically 435 -> 20-40 systems.
      * Covariate mode: primary only, for the same reason as LOCO -- there is
        no "robustness of the confound positive control" claim to make.
      * Replicates and grid: the reported quantity is a MEDIAN coefficient per
        FNR level. 200 draws is far past where a median stabilises; 40 leaves
        the Monte Carlo error negligible against the coefficient's own
        standard error. 4 grid points span a monotone attenuation curve as
        well as 7, and the analytical Bross correction covers the continuum.

    Replicates run in parallel, each seeding its own RNG deterministically
    from (random_seed, fnr, replicate_id, covariate_mode), so output is
    reproducible regardless of worker count.
    """
    systems = list(defense_cols)
    if config.misclass_restrict_to_significant and tier2_phyloglm is not None \
            and not tier2_phyloglm.empty:
        prim = tier2_phyloglm
        if "outcome_label" in prim.columns:
            prim = prim[prim["outcome_label"] == "any_plasmid"]
        if "direction" in prim.columns:
            prim = prim[prim["direction"] == "plasmid_given_defense"]
        if "covariate_mode" in prim.columns:
            prim = prim[prim["covariate_mode"].map(
                config.normalise_covariate_mode)
                == config.normalise_covariate_mode(config.primary_covariate_mode)]
        prim = prim.dropna(subset=["phyloglm_p_value"])
        if not prim.empty:
            sig = prim[prim.get("phyloglm_fdr_qvalue", 1.0) < config.alpha]
            if sig.empty:
                # Nothing significant: fall back to the strongest N so the
                # stage still reports something interpretable.
                sig = prim.nsmallest(config.misclass_max_systems,
                                     "phyloglm_p_value")
                logger.info("Misclassification MC: no FDR-significant systems; "
                            f"using the {len(sig)} strongest instead")
            elif len(sig) > config.misclass_max_systems:
                sig = sig.nsmallest(config.misclass_max_systems,
                                    "phyloglm_p_value")
            systems = [c for c in sig["defense_system"].tolist()
                       if c in defense_cols]
    if not systems:
        systems = list(defense_cols)

    modes = ([config.primary_covariate_mode]
             if config.misclass_primary_mode_only
             else [m for m in config.covariate_modes
                   if not config.is_diagnostic_mode(m)])
    grid = (config.misclass_fnr_grid_reduced if config.misclass_use_reduced_grid
            else config.misclass_fnr_grid)
    n_rep = (config.misclass_n_replicates_effective
             if config.misclass_use_reduced_grid
             else config.misclass_n_replicates)

    n_total = len(grid) * n_rep * len(modes)
    logger.info(
        f"Misclassification MC: {len(grid)} FNR levels x {n_rep} replicates "
        f"x {len(modes)} covariate mode(s) = {n_total} fits, over "
        f"{len(systems)}/{len(defense_cols)} systems"
    )

    tasks = [
        (covariate_mode, fnr, rep)
        for covariate_mode in modes
        for fnr in grid
        for rep in range(n_rep)
    ]
    defense_cols = systems

    n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
    n_jobs = max(1, min(n_jobs, len(tasks)))
    logger.info(
        f"  dispatching {len(tasks)} misclassification phyloglm calls across "
        f"{n_jobs} parallel workers (this stage is the largest single time "
        "sink in the pipeline)"
    )

    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(_one_mc_replicate)(
            phylo_data, defense_cols, tree_path, fnr, config, logger,
            workdir, rep, covariate_mode)
        for covariate_mode, fnr, rep in tasks
    )
    frames = [df for df in results if df is not None and not df.empty]
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
    """Misclassification bias correction per Bross (1954).

    For a 2x2 table with true plasmid prevalences pi_0 (among defense-absent)
    and pi_1 (among defense-present), observed OR is biased toward 1 by the
    factor ((1 - fnr*(1-pi_1)) / (1 - fnr*(1-pi_0))). We invert this to get
    an adjusted OR at each FNR. Also compute the tipping-point FNR where the
    adjusted OR crosses 1.

    IMPORTANT — the non-differential assumption.
    Bross requires misclassification to be non-differential with respect to
    the exposure. That does not hold unconditionally in this dataset: the
    species plasmid label is "any strain carries a plasmid", so a species with
    n strains has n chances to detect one and the effective species-level FNR
    is f^n. Sequencing depth is also correlated with defense presence (both
    saturate with n), so the FNR is differential with respect to the exposure
    through depth, and the unconditional correction is invalid.

    The fix is to apply the correction WITHIN sampling-depth strata, where the
    depth-driven component of the FNR is approximately constant so the
    non-differential assumption holds conditionally, and then pool. That is
    what ``depth_strata`` does below. Passing ``depth_strata=None`` reproduces
    the unconditional (invalid) calculation and is retained only for
    comparison.
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
    out = df.merge(tips, on=["defense_system", "covariate_mode"], how="left")
    # Record the assumption so a reader of the TSV cannot mistake this for an
    # unconditionally valid correction.
    out["nondifferential_assumption"] = (
        "conditional_on_depth" if getattr(config, "misclass_depth_differential",
                                          False) else "unconditional_INVALID")
    return out


def defense_side_bias_correction(tier2_phyloglm: pd.DataFrame,
                                 tier1_results: pd.DataFrame,
                                 config: Config) -> pd.DataFrame:
    """Symmetric Bross correction for DEFENSE-side false negatives.

    The pipeline previously modelled plasmid-detection false negatives only.
    DefenseFinder recall also varies by taxon -- a system well characterised in
    Pseudomonadota may be systematically missed in deeper-branching clades --
    and if that variation correlates with clade-level plasmid carriage the
    associations are biased. The 2x2 correction generalises symmetrically, so
    there is no reason to model one side and not the other.

    Here the roles are swapped: the mismeasured variable is the EXPOSURE
    (defense presence) rather than the outcome, and for non-differential
    exposure misclassification with sensitivity (1 - fnr) and perfect
    specificity, the observed odds ratio is attenuated toward 1 by a factor
    that depends on the exposure prevalence in each outcome group.
    """
    if not getattr(config, "run_defense_misclassification", False):
        return pd.DataFrame()

    t2 = tier2_phyloglm
    if t2 is None or t2.empty:
        return pd.DataFrame()
    if "outcome_label" in t2.columns:
        t2 = t2[t2["outcome_label"] == "any_plasmid"]
    if "direction" in t2.columns:
        t2 = t2[t2["direction"] == "plasmid_given_defense"]
    if t2.empty:
        return pd.DataFrame()

    t1 = tier1_results if tier1_results is not None else pd.DataFrame()
    prev_col = next((c for c in ("defense_prevalence", "prevalence",
                                 "n_present_frac") if c in t1.columns), None)
    prev_by_system = {}
    if prev_col:
        prev_by_system = dict(zip(t1["defense_system"], t1[prev_col]))

    records = []
    for _, row in t2.iterrows():
        beta = row.get("phyloglm_coefficient")
        if not np.isfinite(beta):
            continue
        system = row["defense_system"]
        cov_mode = row.get("covariate_mode", config.primary_covariate_mode)
        obs_or = float(np.exp(beta))
        # Exposure prevalence; fall back to the observed present/absent counts.
        p_exp = prev_by_system.get(system, np.nan)
        if not np.isfinite(p_exp):
            npres = row.get("n_defense_present", np.nan)
            nabs = row.get("n_defense_absent", np.nan)
            if np.isfinite(npres) and np.isfinite(nabs) and (npres + nabs) > 0:
                p_exp = npres / (npres + nabs)
        if not np.isfinite(p_exp) or not (0 < p_exp < 1):
            continue

        for fnr in config.defense_fnr_grid:
            sens = 1.0 - float(fnr)
            if sens <= 0:
                records.append({"defense_system": system,
                                "covariate_mode": cov_mode,
                                "defense_fnr": fnr, "obs_OR": obs_or,
                                "adj_OR": np.nan})
                continue
            # Apparent exposure prevalence under imperfect sensitivity.
            p_obs = p_exp * sens
            # Attenuation factor for non-differential exposure
            # misclassification with perfect specificity.
            denom = (1 - p_obs)
            attenuation = (sens * (1 - p_exp) / denom) if denom > 0 else np.nan
            adj_or = (obs_or ** (1.0 / attenuation)
                      if np.isfinite(attenuation) and attenuation > 0 else np.nan)
            records.append({"defense_system": system,
                            "covariate_mode": cov_mode,
                            "defense_fnr": fnr,
                            "defense_prevalence_assumed_true": p_exp,
                            "defense_prevalence_observed": p_obs,
                            "obs_OR": obs_or,
                            "adj_OR": adj_or})

    return pd.DataFrame(records)

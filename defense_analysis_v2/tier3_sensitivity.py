"""Tier 3: remaining sensitivity checks.

    - Phylogenetic signal (D-statistic) for every column of interest, to
      report in methods as justification for using phylogenetic correction.
    - Clade-restricted permutation test: reshuffle plasmid labels *within*
      each clade to preserve clade-level prevalence marginals. This gives a
      phylogenetically-informed empirical null. The test statistic is a
      difference of mean plasmid rates between defense-present and
      defense-absent species — a fast proxy that preserves the sign of a
      logistic coefficient but is not numerically equal to one. Intended as
      a phylogenetically-informed null on top of phyloglm, not a replacement
      for it.
    - Prevalence-matched paired test: for each defense system, match
      plasmid-positive and plasmid-negative species by defense-system
      prevalence quantiles and test for a within-matched-pair difference.
      Flags RM-style prevalence confounding. The within-bin pairing is
      randomised (seeded) so that repeated runs average over matching noise.
    - Minimum n_strains sensitivity rerun: refit phyloglm on the subset of
      species with >= min_n_strains strains. Guards against the species-
      level max() aggregation inflating defense presence in heavily-sampled
      species, which log_n_strains as a covariate alone may not fully absorb.
    - Prevalence-feature sensitivity rerun: refit phyloglm using the per-
      species *prevalence* of each defense system (mean across strains)
      rather than the binary max. Directly addresses the same saturation
      concern with a different feature construction.
    - Phylogenetic model sensitivity: re-run phyloglm under alternative
      evolutionary models (OU, BM+lambda) so conclusions aren't quietly
      dependent on the Brownian-motion assumption.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import apply_fdr


# ----------------------------------------------------------------------
# Phylogenetic signal (D-statistic)
# ----------------------------------------------------------------------

def run_phylogenetic_signal(phylo_data: pd.DataFrame, defense_cols: List[str],
                            tree_path: str, config: Config,
                            logger: logging.Logger, workdir: Path,
                            include_response: bool = True) -> pd.DataFrame:
    """Fritz & Purvis D for plasmid outcomes plus every defense system.
    Reports p_random and p_brownian for each column; you want p_random < 0.05
    to justify phylogenetic correction at all.

    Any stratified ``any_plasmid_<class>`` columns present on ``phylo_data``
    are also included so reviewers can see signal for every outcome on which
    the pipeline will report an association.
    """
    outcome_cols = []
    if include_response:
        for c in phylo_data.columns:
            if c == "has_plasmid_binary" or c.startswith("any_plasmid_"):
                outcome_cols.append(c)
    cols = outcome_cols + list(defense_cols)
    logger.info(f"D-statistic (phylogenetic signal) for {len(cols)} columns "
                f"({len(outcome_cols)} outcomes + {len(defense_cols)} defense systems)")
    r = call_r_script(
        "phylo_d.R",
        tree_path=tree_path,
        data=phylo_data,
        args={"columns": cols, "tip_column": "tip", "n_perm": 1000},
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / "phylo_d",
    )
    if not r.ok:
        logger.error(f"phylo_d failed: {r.error}")
        return pd.DataFrame()
    return r.dataframe


# ----------------------------------------------------------------------
# Clade-restricted permutation
# ----------------------------------------------------------------------

def _clade_restricted_shuffle(values: np.ndarray, clade_labels: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    """Shuffle ``values`` independently within each clade."""
    out = values.copy()
    for clade in np.unique(clade_labels):
        idx = np.where(clade_labels == clade)[0]
        if idx.size > 1:
            perm = rng.permutation(idx)
            out[idx] = values[perm]
    return out


def _one_permutation_stat(defense: np.ndarray, plasmid: np.ndarray,
                          clade_labels: np.ndarray,
                          rng: np.random.Generator) -> float:
    shuffled = _clade_restricted_shuffle(plasmid, clade_labels, rng)
    # Test statistic: sign-of-coefficient from logistic regression would be
    # slow in tight loops; use difference in means (sign-preserving proxy).
    if defense.sum() == 0 or (defense == 0).sum() == 0:
        return 0.0
    return float(shuffled[defense == 1].mean() - shuffled[defense == 0].mean())


def _permutation_one_system(col: str, binary: np.ndarray, plasmid: np.ndarray,
                            clade_labels: np.ndarray, n_perm: int,
                            seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = float(plasmid[binary == 1].mean() - plasmid[binary == 0].mean()) \
        if binary.sum() and (binary == 0).sum() else 0.0
    null = np.array([_one_permutation_stat(binary, plasmid, clade_labels, rng)
                     for _ in range(n_perm)])
    # Two-sided p
    p = float(((np.abs(null) >= np.abs(obs)).sum() + 1) / (n_perm + 1))
    return {"defense_system": col, "perm_observed": obs,
            "perm_null_mean": float(null.mean()),
            "perm_null_std": float(null.std()),
            "perm_p_value": p}


def run_clade_permutation(binary_df: pd.DataFrame, defense_cols: List[str],
                          config: Config, logger: logging.Logger) -> pd.DataFrame:
    """Clade-restricted permutation null. ``config.permutation_clade_rank``
    controls which GTDB rank defines "clade".
    """
    rank = config.permutation_clade_rank
    if rank not in binary_df.columns:
        logger.warning(f"Clade rank '{rank}' not in data; skipping clade permutation")
        return pd.DataFrame()

    logger.info(f"Clade-restricted permutation (rank={rank}, "
                f"{config.n_permutations} permutations)")
    plasmid = binary_df["has_plasmid_binary"].values
    clade_labels = binary_df[rank].values

    n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_permutation_one_system)(
            c, binary_df[c].values, plasmid, clade_labels,
            config.n_permutations, config.random_seed + i)
        for i, c in enumerate(defense_cols)
    )
    df = pd.DataFrame(results)
    df["perm_fdr_qvalue"] = apply_fdr(df["perm_p_value"], method=config.fdr_method).values
    return df


# ----------------------------------------------------------------------
# Prevalence-matched paired test
# ----------------------------------------------------------------------

def run_prevalence_matched(binary_df: pd.DataFrame, prevalence_df: pd.DataFrame,
                           defense_cols: List[str], config: Config,
                           logger: logging.Logger) -> pd.DataFrame:
    """For each system, match plasmid+ and plasmid- species on the
    system's prevalence quantile (deciles). Paired Wilcoxon on matched
    indicator differences. A system whose Tier 1 signal survives prevalence
    matching is one where the association isn't explained by
    plasmid-carriers-also-being-well-sequenced.

    Within each decile bin, candidate plasmid+ and plasmid- species are
    sampled without replacement before pairing (seeded by
    ``config.random_seed``) so that the matching isn't a deterministic
    function of the row order of the input table.
    """
    logger.info("Prevalence-matched paired test (deciles of system prevalence)")
    rng = np.random.default_rng(config.random_seed)
    results = []
    for col in defense_cols:
        prev = prevalence_df[col].values
        plasmid = binary_df["has_plasmid_binary"].values
        # Bin species by decile of prevalence, match pairs within bins
        try:
            bins = pd.qcut(prev, q=10, duplicates="drop").astype(str)
        except ValueError:
            # Too few distinct values — skip
            continue
        records = {"pos": [], "neg": []}
        for b in np.unique(bins):
            idx = np.where(bins == b)[0]
            pos = idx[plasmid[idx] == 1]
            neg = idx[plasmid[idx] == 0]
            k = min(len(pos), len(neg))
            if k < 3:
                continue
            # Randomise within-bin pairing so repeated runs average over
            # matching noise rather than freezing in the input row order.
            pos_sampled = rng.choice(pos, size=k, replace=False)
            neg_sampled = rng.choice(neg, size=k, replace=False)
            records["pos"].extend(binary_df[col].values[pos_sampled])
            records["neg"].extend(binary_df[col].values[neg_sampled])
        pos_arr = np.asarray(records["pos"], dtype=int)
        neg_arr = np.asarray(records["neg"], dtype=int)
        if len(pos_arr) < 10:
            results.append({"defense_system": col, "n_pairs": len(pos_arr),
                            "matched_p_value": np.nan, "matched_effect": np.nan})
            continue
        diff = pos_arr - neg_arr
        try:
            stat, p = stats.wilcoxon(diff)
        except ValueError:
            stat, p = np.nan, np.nan
        results.append({"defense_system": col, "n_pairs": len(pos_arr),
                        "matched_effect": float(diff.mean()),
                        "matched_wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                        "matched_p_value": float(p) if np.isfinite(p) else np.nan})
    df = pd.DataFrame(results)
    df["matched_fdr_qvalue"] = apply_fdr(df["matched_p_value"], method=config.fdr_method).values
    return df


# ----------------------------------------------------------------------
# Minimum n_strains sensitivity rerun of the primary phyloglm
# ----------------------------------------------------------------------

def run_min_n_strains_sensitivity(phylo_data: pd.DataFrame,
                                  defense_cols: List[str],
                                  tree_path: str,
                                  config: Config,
                                  logger: logging.Logger,
                                  workdir: Path) -> pd.DataFrame:
    """Refit the primary phyloglm (any_plasmid / plasmid_given_defense) on the
    subset of species with >= ``config.min_n_strains_sensitivity`` strains.

    The binary defense feature is max()-aggregated across strains, which is
    more likely to saturate to 1 for well-sampled species than for sparsely-
    sampled ones. log_n_strains as a covariate absorbs the linear part of
    that bias; this rerun checks that the qualitative picture survives when
    the undersampled tail is removed entirely.
    """
    if "n_strains" not in phylo_data.columns:
        logger.info("min_n_strains sensitivity skipped — n_strains not available")
        return pd.DataFrame()

    threshold = int(config.min_n_strains_sensitivity)
    sub = phylo_data[phylo_data["n_strains"] >= threshold].copy()
    if len(sub) < 50:
        logger.warning(
            f"min_n_strains sensitivity: only {len(sub)} species at threshold "
            f">= {threshold}; skipping"
        )
        return pd.DataFrame()

    logger.info(
        f"min_n_strains sensitivity: refitting phyloglm on "
        f"{len(sub)}/{len(phylo_data)} species (n_strains >= {threshold})"
    )

    pieces: List[pd.DataFrame] = []
    for covariate_mode in config.covariate_modes:
        covariates = list(config.covariate_columns_for_mode(
            covariate_mode, include_plasmid_count=False))
        covariates = [c for c in covariates if c in sub.columns]
        r = call_r_script(
            "phyloglm_uni.R",
            tree_path=tree_path,
            data=sub,
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
            workdir=workdir / f"min_n_strains_{covariate_mode}",
        )
        if not r.ok:
            logger.warning(f"min_n_strains_sensitivity [{covariate_mode}] failed: {r.error}")
            continue
        df = r.dataframe.rename(columns={"test_label": "defense_system"})
        df["covariate_mode"] = covariate_mode
        df["min_n_strains_threshold"] = threshold
        df["n_species_filtered_in"] = len(sub)
        df["phyloglm_fdr_qvalue"] = apply_fdr(df["phyloglm_p_value"],
                                               method=config.fdr_method).values
        pieces.append(df)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


# ----------------------------------------------------------------------
# Prevalence-feature sensitivity rerun of the primary phyloglm
# ----------------------------------------------------------------------

def run_prevalence_feature_sensitivity(phylo_data: pd.DataFrame,
                                       prevalence_data: pd.DataFrame,
                                       defense_cols: List[str],
                                       tree_path: str,
                                       config: Config,
                                       logger: logging.Logger,
                                       workdir: Path) -> pd.DataFrame:
    """Refit phyloglm against ``has_plasmid_binary`` using the per-species
    *prevalence* of each defense system (fraction of strains carrying the
    system) as the predictor, rather than the max()-aggregated binary.

    phyloglm accepts continuous predictors; the coefficient here is the
    log-odds-ratio per one-unit increase in prevalence (i.e. from 0% to
    100% of strains carrying the system). The point of this sensitivity
    is that prevalence is immune to the sampling-depth saturation that
    binary max() suffers from, so agreement between the two feature modes
    is direct evidence that the primary result isn't sampling-artefactual.

    Runs against the legacy any_plasmid outcome only — its job is a focused
    robustness check, not a parallel replica of the stratified analysis.
    """
    if prevalence_data is None or prevalence_data.empty:
        logger.info("prevalence-feature sensitivity skipped — no prevalence table")
        return pd.DataFrame()

    # Build a phylo-style table where defense columns carry *prevalence*
    # values (mean across strains) rather than binary maxes. Everything else
    # (tip labels, covariates, has_plasmid_binary) is copied from phylo_data.
    if "tip" not in phylo_data.columns or "gtdb_species" not in prevalence_data.columns:
        logger.warning("prevalence-feature sensitivity: can't align tables; skipping")
        return pd.DataFrame()

    tip_to_species = dict(zip(phylo_data["tip"], phylo_data["gtdb_species"])) \
        if "gtdb_species" in phylo_data.columns else {}
    prev_by_species = prevalence_data.set_index("gtdb_species")[defense_cols]
    # Map prevalence values onto phylo_data rows by species name
    sub = phylo_data.copy()
    if tip_to_species:
        species_aligned = sub["tip"].map(tip_to_species)
    else:
        species_aligned = sub.get("gtdb_species")
    if species_aligned is None:
        logger.warning("prevalence-feature sensitivity: missing species column; skipping")
        return pd.DataFrame()
    valid = species_aligned.isin(prev_by_species.index)
    sub = sub.loc[valid].copy()
    sp = species_aligned.loc[valid]
    for c in defense_cols:
        sub[c] = prev_by_species.loc[sp, c].values

    if len(sub) < 50:
        logger.warning(
            f"prevalence-feature sensitivity: only {len(sub)} species aligned; skipping"
        )
        return pd.DataFrame()

    logger.info(
        f"prevalence-feature sensitivity: refitting phyloglm on {len(sub)} "
        f"species using strain-prevalence as the defense feature"
    )

    pieces: List[pd.DataFrame] = []
    for covariate_mode in config.covariate_modes:
        # include_n_strains=False here: the prevalence feature is already a
        # strain-averaged quantity, so log_n_strains as a covariate would be
        # partially redundant with the feature construction itself. Genome-
        # scale covariates still go in when covariate_mode == with_cov.
        covariates = list(config.covariate_columns_for_mode(
            covariate_mode, include_plasmid_count=False,
            include_n_strains=False))
        covariates = [c for c in covariates if c in sub.columns]
        r = call_r_script(
            "phyloglm_uni.R",
            tree_path=tree_path,
            data=sub,
            args={"response": "has_plasmid_binary",
                  "predictors": defense_cols,
                  "mode": "predictor",
                  "covariates": covariates,
                  "tip_column": "tip",
                  "evolutionary_model": config.phylo_evolutionary_model,
                  "btol": 20, "boot": 0,
                  # prevalence is continuous; min_count gate is irrelevant
                  "min_count": 0},
            logger=logger,
            r_executable=config.r_executable,
            workdir=workdir / f"prev_feature_{covariate_mode}",
        )
        if not r.ok:
            logger.warning(f"prevalence_feature_sensitivity [{covariate_mode}] failed: {r.error}")
            continue
        df = r.dataframe.rename(columns={"test_label": "defense_system"})
        df["covariate_mode"] = covariate_mode
        df["feature_mode"] = "prevalence"
        df["phyloglm_fdr_qvalue"] = apply_fdr(df["phyloglm_p_value"],
                                               method=config.fdr_method).values
        pieces.append(df)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


# ----------------------------------------------------------------------
# Phylogenetic evolutionary-model sensitivity
# ----------------------------------------------------------------------

def run_phylo_model_sensitivity(phylo_data: pd.DataFrame,
                                defense_cols: List[str],
                                tree_path: str,
                                config: Config,
                                logger: logging.Logger,
                                workdir: Path) -> pd.DataFrame:
    """Refit the primary phyloglm under each model in
    ``config.phylo_model_sensitivity_models`` (typically OU and BM+lambda).

    Horizontal transfer makes Brownian motion a simplifying assumption
    rather than a mechanistic one; reviewers routinely ask how robust the
    ranking is to that choice. The output carries an ``evolutionary_model``
    column so each model's fit can be compared against the primary BM fit
    side-by-side.
    """
    models = tuple(config.phylo_model_sensitivity_models or ())
    if not models:
        return pd.DataFrame()

    logger.info(
        f"Phylogenetic model sensitivity: refitting phyloglm under {models} "
        f"(primary = {config.phylo_evolutionary_model})"
    )

    pieces: List[pd.DataFrame] = []
    for model in models:
        for covariate_mode in config.covariate_modes:
            covariates = list(config.covariate_columns_for_mode(
                covariate_mode, include_plasmid_count=False))
            covariates = [c for c in covariates if c in phylo_data.columns]
            r = call_r_script(
                "phyloglm_uni.R",
                tree_path=tree_path,
                data=phylo_data,
                args={"response": "has_plasmid_binary",
                      "predictors": defense_cols,
                      "mode": "predictor",
                      "covariates": covariates,
                      "tip_column": "tip",
                      "evolutionary_model": model,
                      "btol": 20, "boot": 0,
                      "min_count": config.min_count_per_category},
                logger=logger,
                r_executable=config.r_executable,
                workdir=workdir / f"model_sens_{model}_{covariate_mode}",
            )
            if not r.ok:
                logger.warning(
                    f"phylo_model_sensitivity [{model}/{covariate_mode}] failed: {r.error}"
                )
                continue
            df = r.dataframe.rename(columns={"test_label": "defense_system"})
            df["covariate_mode"] = covariate_mode
            df["evolutionary_model"] = model
            df["phyloglm_fdr_qvalue"] = apply_fdr(df["phyloglm_p_value"],
                                                   method=config.fdr_method).values
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)

"""Tier 3: sensitivity checks against the sampling-depth confound.

Everything in this module exists to attack one problem: the species-level
plasmid label ("any strain carries a plasmid") and the species-level defense
call (max across strains) are both ``1 - (1-p)^n_strains``, so sequencing
depth is a common cause of predictor and outcome.

    - Phylogenetic signal (D-statistic) for every column of interest, to
      report in methods as justification for using phylogenetic correction.

    - Clade-AND-DEPTH-restricted permutation test: reshuffle plasmid labels
      within joint (clade, depth-decile) cells. Shuffling within clade alone
      -- the previous behaviour -- destroys Cov(plasmid, n_strains) while
      Cov(defense, n_strains) is preserved, because defense is never shuffled.
      The observed statistic then contains a depth-mediated component the null
      distribution lacks, making the p-values ANTICONSERVATIVE by exactly the
      magnitude of the artefact under test. Stratifying on depth as well
      preserves the confound in the null, which is the whole point.

    - Depth-matched paired test: match plasmid-positive and plasmid-negative
      species on sampling depth, then compare defense presence with McNemar's
      test. This replaces the previous "prevalence-matched" test, which was
      mathematically vacuous: it binned on deciles of
      ``prevalence_df[system]`` and then tested ``binary_df[system]``, but
      strain calls are binarised before aggregation so
      ``binary == (prevalence > 0)`` exactly. The binary indicator is
      therefore CONSTANT within any prevalence bin, every paired difference
      was structurally zero, ``scipy.stats.wilcoxon`` raised on the all-zero
      vector, and every p-value in the table was NaN. Matching on depth and
      testing defense presence is what the docstring always claimed.

    - Depth-band sensitivity reruns, in BOTH directions:
        * ``high``: species with >= min_n_strains strains. The legacy filter.
          Retained because reviewers expect it, but it selects FOR the
          artefact, not against it: P(plasmid label = 1) rises monotonically
          in n_strains, so the deep tail is enriched for positives and the
          outcome loses contrast (38.5% -> 70.6% prevalence in simulation
          while discarding 71.5% of species). Now gated on retained outcome
          variance so a subset with no contrast left cannot be reported as a
          reassuring null.
        * ``low``: species with <= max_n_strains strains, where saturation
          cannot have operated. Low power per system, but an interpretable
          null. This is the informative direction and was previously absent.

    - Prevalence-feature sensitivity rerun: refit using per-species defense
      prevalence (mean across strains) instead of the max-derived binary.
      This de-saturates the PREDICTOR only; the outcome remains the
      species-propagated plasmid label, so the depth covariates MUST stay in
      the model. They were previously dropped here on the grounds that
      prevalence is "already strain-averaged", which misidentified what they
      were doing -- they stand in for depth as a common cause of both
      variables, not as a predictor correction -- leaving this fit strictly
      LESS adjusted than the primary fit it was validating.

    - Covariance-structure sensitivity: refit under Pagel's-lambda-rescaled
      trees and under the IG10 estimator. The previous "OU" arm mapped to the
      same estimator as the primary fit and was bit-identical to it.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .config import Config
from .r_bridge import call_r_script, write_shared_frame
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

    # Native vectorised implementation by default. caper was killed at the
    # cluster's 25-day wall-clock ceiling on this exact workload; the native
    # path computes the identical statistic by flattening the tree once,
    # traversing level-wise, and evaluating all permutations as one matrix.
    if getattr(config, "phylo_signal_engine", "native") == "native":
        from .phylo_signal_fast import run_phylo_signal_fast
        return run_phylo_signal_fast(
            phylo_data, cols, tree_path,
            n_perm=int(config.n_permutations),
            random_seed=int(config.random_seed),
            logger=logger, tip_column="tip")
    r = call_r_script(
        "phylo_d.R",
        tree_path=tree_path,
        data=phylo_data,
        # `caper::phylo.d` is a permutation test; without a seed the reported
        # p_random / p_brownian differ between runs of an otherwise identical
        # pipeline, which quietly breaks reproducibility of a methods-section
        # number.
        args={"columns": cols, "tip_column": "tip",
              "n_perm": int(config.n_permutations),
              "seed": int(config.random_seed)},
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

def build_permutation_strata(binary_df: pd.DataFrame, config: Config,
                             logger: logging.Logger) -> np.ndarray:
    """Joint (clade, sampling-depth) stratum label for every species.

    The permutation null must preserve the structure that generates the
    confound. Shuffling the plasmid label within clade alone breaks
    Cov(plasmid, n_strains) -- which in the observed data is strongly positive
    by construction of the label -- while leaving Cov(defense, n_strains)
    intact, because defense is never shuffled. The observed statistic then
    contains a depth-mediated component that the null lacks, so the test is
    anticonservative precisely for the systems most contaminated by sampling
    depth. Stratifying jointly on depth fixes that.
    """
    rank = config.permutation_clade_rank
    clade = (binary_df[rank].astype(str).values if rank in binary_df.columns
             else np.full(len(binary_df), "all"))
    if rank not in binary_df.columns:
        logger.warning(
            f"Clade rank '{rank}' not in data; permutation will stratify on "
            f"sampling depth only")

    if "n_strains" in binary_df.columns:
        depth = pd.to_numeric(binary_df["n_strains"], errors="coerce")
        n_bins = max(2, int(config.permutation_depth_bins))
        try:
            depth_bin = pd.qcut(depth.rank(method="first"), q=n_bins,
                                labels=False, duplicates="drop")
        except ValueError:
            depth_bin = pd.Series(0, index=binary_df.index)
        depth_bin = depth_bin.fillna(-1).astype(int).astype(str).values
    else:
        logger.warning("n_strains absent; permutation cannot stratify on depth")
        depth_bin = np.full(len(binary_df), "0")

    return np.char.add(np.char.add(clade.astype(str), "|"), depth_bin)


def _stratified_shuffle(values: np.ndarray, strata: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Shuffle ``values`` independently within each stratum."""
    out = values.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if idx.size > 1:
            out[idx] = values[rng.permutation(idx)]
    return out


def _one_permutation_stat(defense: np.ndarray, plasmid: np.ndarray,
                          strata: np.ndarray,
                          rng: np.random.Generator) -> float:
    shuffled = _stratified_shuffle(plasmid, strata, rng)
    # Test statistic: sign-of-coefficient from logistic regression would be
    # slow in tight loops; use difference in means (sign-preserving proxy).
    if defense.sum() == 0 or (defense == 0).sum() == 0:
        return 0.0
    return float(shuffled[defense == 1].mean() - shuffled[defense == 0].mean())


def _permutation_one_system(col: str, binary: np.ndarray, plasmid: np.ndarray,
                            strata: np.ndarray, n_perm: int,
                            seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = float(plasmid[binary == 1].mean() - plasmid[binary == 0].mean()) \
        if binary.sum() and (binary == 0).sum() else 0.0
    null = np.array([_one_permutation_stat(binary, plasmid, strata, rng)
                     for _ in range(n_perm)])
    # Two-sided p
    p = float(((np.abs(null) >= np.abs(obs)).sum() + 1) / (n_perm + 1))
    return {"defense_system": col, "perm_observed": obs,
            "perm_null_mean": float(null.mean()),
            "perm_null_std": float(null.std()),
            "perm_p_value": p}


def run_clade_permutation(binary_df: pd.DataFrame, defense_cols: List[str],
                          config: Config, logger: logging.Logger) -> pd.DataFrame:
    """Permutation null stratified jointly on clade AND sampling depth.

    ``config.permutation_clade_rank`` sets the taxonomic rank (class by
    default -- phylum is too coarse, since n_strains varies by orders of
    magnitude within Pseudomonadota) and ``config.permutation_depth_bins``
    sets the number of depth strata.
    """
    strata = build_permutation_strata(binary_df, config, logger)
    n_strata = int(np.unique(strata).size)
    logger.info(
        f"Depth-stratified clade permutation (rank={config.permutation_clade_rank}, "
        f"{config.permutation_depth_bins} depth bins -> {n_strata} strata, "
        f"{config.n_permutations} permutations)")

    # A stratum of size 1 contributes no shuffling, so if most cells are
    # singletons the null collapses toward the identity and the test loses
    # all power. Warn rather than fail; the operator can coarsen the rank.
    sizes = pd.Series(strata).value_counts()
    frac_singleton = float((sizes == 1).sum() / len(sizes)) if len(sizes) else 0.0
    if frac_singleton > 0.5:
        logger.warning(
            f"{frac_singleton:.0%} of permutation strata are singletons; the "
            f"null is nearly degenerate. Coarsen permutation_clade_rank or "
            f"reduce permutation_depth_bins.")

    plasmid = binary_df["has_plasmid_binary"].values

    n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
    # Use threading backend rather than joblib's default loky (process-based).
    # The threading backend works reliably across conda envs; loky has been
    # observed to fail with ModuleNotFoundError on Python 3.10 conda builds
    # missing joblib.externals.loky.backend.synchronize. The permutation
    # work here is numpy-heavy and releases the GIL, so threading is fine.
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(_permutation_one_system)(
            c, binary_df[c].values, plasmid, strata,
            config.n_permutations, config.random_seed + i)
        for i, c in enumerate(defense_cols)
    )
    df = pd.DataFrame(results)
    df["perm_fdr_qvalue"] = apply_fdr(df["perm_p_value"], method=config.fdr_method).values
    df["perm_n_strata"] = n_strata
    df["perm_strata_definition"] = (
        f"{config.permutation_clade_rank} x n_strains decile")
    return df


# ----------------------------------------------------------------------
# Depth-matched paired test
# ----------------------------------------------------------------------

def run_depth_matched(binary_df: pd.DataFrame, prevalence_df: pd.DataFrame,
                      defense_cols: List[str], config: Config,
                      logger: logging.Logger) -> pd.DataFrame:
    """Match plasmid-positive to plasmid-negative species on SAMPLING DEPTH,
    then test whether defense presence still differs between them.

    This is the test the old ``run_prevalence_matched`` docstring described
    ("the association isn't explained by plasmid-carriers-also-being-well-
    sequenced") but did not implement. That version binned on deciles of
    ``prevalence_df[system]`` and then tested ``binary_df[system]``. Strain
    calls are binarised before aggregation (``io_utils``: ``df[c] = (df[c] >
    0)``), so ``binary == (prevalence > 0)`` EXACTLY -- the binary indicator
    is constant within any prevalence bin, every paired difference was
    identically zero, ``stats.wilcoxon`` raised ``ValueError`` on the
    all-zero vector, and the caught exception wrote NaN. Every p-value and
    q-value in that table was NaN, and ``matched_effect`` was always 0.0.
    The matching variable and the tested variable were the same variable, so
    no parameter change could have rescued it.

    Matching on depth breaks the confound path (depth -> plasmid label)
    while leaving any genuine biological association intact.

    McNemar's exact test on the discordant pairs, which is the standard test
    for matched binary data. (The old code used Wilcoxon on paired binary
    indicators, which is effectively a sign test and was flagged as
    mislabelled even in the earlier review.)
    """
    if "n_strains" not in binary_df.columns:
        logger.warning("n_strains absent; skipping depth-matched test")
        return pd.DataFrame()

    n_bins = max(2, int(config.permutation_depth_bins))
    logger.info(f"Depth-matched paired test ({n_bins} bins of n_strains, "
                f"McNemar exact)")
    rng = np.random.default_rng(config.random_seed)
    plasmid = binary_df["has_plasmid_binary"].values
    depth = pd.to_numeric(binary_df["n_strains"], errors="coerce")

    # Rank-then-qcut so heavy ties in n_strains don't collapse the bins.
    try:
        bins = pd.qcut(depth.rank(method="first"), q=n_bins,
                       labels=False, duplicates="drop").fillna(-1).astype(int).values
    except ValueError:
        logger.warning("Could not bin n_strains; skipping depth-matched test")
        return pd.DataFrame()

    # Matching is on depth only, so the pairing is identical for every defense
    # system. Build it once.
    pair_pos, pair_neg = [], []
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        pos = idx[plasmid[idx] == 1]
        neg = idx[plasmid[idx] == 0]
        k = min(len(pos), len(neg))
        if k < 3:
            continue
        pair_pos.append(rng.choice(pos, size=k, replace=False))
        pair_neg.append(rng.choice(neg, size=k, replace=False))

    if not pair_pos:
        logger.warning(
            "Depth-matched test: no depth bin contained both plasmid-positive "
            "and plasmid-negative species in sufficient numbers. This is "
            "itself diagnostic -- it means plasmid status is essentially "
            "determined by sequencing depth.")
        return pd.DataFrame()

    pos_idx = np.concatenate(pair_pos)
    neg_idx = np.concatenate(pair_neg)
    n_pairs = len(pos_idx)
    logger.info(f"  {n_pairs:,} depth-matched species pairs")

    results = []
    for col in defense_cols:
        vals = binary_df[col].values
        a = vals[pos_idx].astype(int)   # defense in plasmid+ member
        b_ = vals[neg_idx].astype(int)  # defense in plasmid- member
        # Discordant pair counts
        n01 = int(np.sum((a == 0) & (b_ == 1)))
        n10 = int(np.sum((a == 1) & (b_ == 0)))
        n_discordant = n01 + n10
        if n_discordant < 10:
            results.append({"defense_system": col, "n_pairs": n_pairs,
                            "n_discordant": n_discordant,
                            "matched_effect": np.nan,
                            "matched_p_value": np.nan,
                            "skip_reason": "too_few_discordant_pairs"})
            continue
        # Exact McNemar: n10 ~ Binomial(n_discordant, 0.5) under H0.
        p = float(stats.binomtest(n10, n_discordant, 0.5).pvalue)
        results.append({
            "defense_system": col,
            "n_pairs": n_pairs,
            "n_discordant": n_discordant,
            "n_defense_only_in_plasmid_pos": n10,
            "n_defense_only_in_plasmid_neg": n01,
            # Positive = defense more common among plasmid carriers at equal
            # sequencing depth.
            "matched_effect": float((n10 - n01) / n_pairs),
            "matched_odds_ratio": float(n10 / n01) if n01 > 0 else np.inf,
            "matched_p_value": p,
            "skip_reason": np.nan,
        })

    df = pd.DataFrame(results)
    df["matched_fdr_qvalue"] = apply_fdr(df["matched_p_value"],
                                         method=config.fdr_method).values
    return df


# Backwards-compatible alias. The old name described a test that could not
# work; callers are migrated, but keep the symbol so external notebooks that
# import it get the working implementation rather than an ImportError.
run_prevalence_matched = run_depth_matched


# ----------------------------------------------------------------------
# Minimum n_strains sensitivity rerun of the primary phyloglm
# ----------------------------------------------------------------------

def _run_depth_band(phylo_data: pd.DataFrame, defense_cols: List[str],
                    tree_path: str, config: Config, logger: logging.Logger,
                    workdir: Path, band: str) -> pd.DataFrame:
    """Refit the primary phyloglm on a sampling-depth band.

    ``band`` is "high" (n_strains >= min_n_strains_sensitivity) or "low"
    (n_strains <= max_n_strains_sensitivity).
    """
    from .io_utils import add_depth_basis

    if band == "high":
        threshold = int(config.min_n_strains_sensitivity)
        mask = phylo_data["n_strains"] >= threshold
        label = f"n_strains >= {threshold}"
    else:
        threshold = int(config.max_n_strains_sensitivity)
        mask = phylo_data["n_strains"] <= threshold
        label = f"n_strains <= {threshold}"

    sub = phylo_data[mask].copy()
    if len(sub) < 50:
        logger.warning(f"depth sensitivity [{band}]: only {len(sub)} species "
                       f"({label}); skipping")
        return pd.DataFrame()

    # ---- Outcome-contrast gate ----
    # Filtering to the deep tail drives the plasmid label toward 1 because
    # P(any strain has a plasmid) rises monotonically in n_strains. Without a
    # gate, a subset that is 97% plasmid-positive is fit at near-zero power and
    # the resulting "no significant hits" is reported as reassurance rather
    # than as an absence of contrast.
    #
    # Judged on the ABSOLUTE COUNT in the minority class, not on a proportion.
    # A proportion floor is wrong here: overall prevalence is ~5.7%, and the
    # LOW-depth band sits near 1% purely because a one-strain species has one
    # chance to carry a plasmid. That band is the informative half of this
    # analysis -- the only place saturation cannot have operated -- and a 5%
    # floor would discard it despite a few hundred positive species, which is
    # ample. What actually breaks a fit is too few species in the minority
    # class, in either direction.
    n_pos = int(sub["has_plasmid_binary"].sum())
    n_neg = int(len(sub) - n_pos)
    outcome_prev = float(sub["has_plasmid_binary"].mean())
    minority = min(n_pos, n_neg)
    prevalence_ok = (minority >= config.depth_sens_min_outcome_count
                     and config.depth_sens_min_outcome_fraction
                     <= outcome_prev
                     <= config.depth_sens_max_outcome_fraction)
    if not prevalence_ok:
        logger.warning(
            f"depth sensitivity [{band}]: only {minority:,} species in the "
            f"minority outcome class after filtering to {label} "
            f"({n_pos:,} positive / {n_neg:,} negative, prevalence "
            f"{outcome_prev:.1%}; full data "
            f"{phylo_data['has_plasmid_binary'].mean():.1%}). Below "
            f"{config.depth_sens_min_outcome_count} there is not enough "
            f"contrast to interpret this rerun; results are emitted but "
            f"flagged uninterpretable.")
    else:
        logger.info(
            f"depth sensitivity [{band}]: {n_pos:,} positive / {n_neg:,} "
            f"negative species (prevalence {outcome_prev:.1%}) — interpretable")

    # Rebuild the depth spline on the FILTERED rows. The knots are quantiles
    # of the data actually being fit; reusing the full-table basis would place
    # knots outside the retained range and leave collinear columns.
    sub = add_depth_basis(sub, config, logger)

    logger.info(
        f"depth sensitivity [{band}]: refitting phyloglm on "
        f"{len(sub):,}/{len(phylo_data):,} species ({label}), "
        f"outcome prevalence {outcome_prev:.1%}")

    pieces: List[pd.DataFrame] = []
    for covariate_mode in config.covariate_modes:
        if config.is_diagnostic_mode(covariate_mode):
            continue
        covariates = config.resolve_covariates(
            config.covariate_columns_for_mode(
                covariate_mode, include_plasmid_count=False), sub)
        r = call_r_script(
            "phyloglm_uni.R",
            tree_path=tree_path,
            data=sub,
            args={"response": "has_plasmid_binary",
                  "predictors": defense_cols,
                  "mode": "predictor",
                  "defense_side": "predictor",
                  "covariates": list(covariates),
                  "tip_column": "tip",
                  "evolutionary_model": config.phyloglm_estimator,
                  "btol": 20, "boot": 0,
                  "min_count": config.min_count_per_category,
                  "min_count_response": config.min_count_per_category},
            logger=logger,
            r_executable=config.r_executable,
            workdir=workdir / f"depth_{band}_{covariate_mode}",
        )
        if not r.ok:
            logger.warning(f"depth sensitivity [{band}/{covariate_mode}] failed: {r.error}")
            continue
        df = r.dataframe.rename(columns={"test_label": "defense_system"})
        df["covariate_mode"] = covariate_mode
        df["depth_band"] = band
        df["depth_threshold"] = threshold
        df["n_species_filtered_in"] = len(sub)
        df["outcome_prevalence"] = outcome_prev
        df["n_outcome_positive"] = n_pos
        df["n_outcome_negative"] = n_neg
        df["outcome_prevalence_ok"] = prevalence_ok
        df["interpretable"] = prevalence_ok
        df["phyloglm_fdr_qvalue"] = apply_fdr(df["phyloglm_p_value"],
                                              method=config.fdr_method).values
        pieces.append(df)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def run_min_n_strains_sensitivity(phylo_data: pd.DataFrame,
                                  defense_cols: List[str],
                                  tree_path: str,
                                  config: Config,
                                  logger: logging.Logger,
                                  workdir: Path) -> pd.DataFrame:
    """Sampling-depth sensitivity, run in BOTH directions.

    The legacy version kept only species with >= ``min_n_strains_sensitivity``
    strains and described that as showing "the associations aren't an artefact
    of poorly-sampled species". That has the artefact backwards. Both the
    plasmid label and the defense call saturate upward with n_strains, so the
    well-sampled tail is where the artefact LIVES: restricting to it enriches
    for plasmid-positive species (38.5% -> 70.6% in simulation) and discards
    ~70% of the data, losing contrast on both axes at once. Fewer surviving
    hits then reads as reassuring attenuation when the mechanism is loss of
    variance.

    So:
      * ``high`` band -- the legacy filter, retained because reviewers expect
        it, now gated on retained outcome variance.
      * ``low`` band -- species with <= ``max_n_strains_sensitivity`` strains,
        where saturation cannot have operated. Weak power per system, but this
        is the direction that actually tests the confound.

    Concordance between the two bands is the informative comparison, and it is
    computed in ``build_depth_band_concordance``.
    """
    if "n_strains" not in phylo_data.columns:
        logger.info("depth sensitivity skipped — n_strains not available")
        return pd.DataFrame()

    pieces = [
        _run_depth_band(phylo_data, defense_cols, tree_path, config, logger,
                        workdir, band)
        for band in ("high", "low")
    ]
    pieces = [p for p in pieces if p is not None and not p.empty]
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    # Legacy column name so existing consumers keep working.
    out["min_n_strains_threshold"] = out["depth_threshold"]
    return out


def build_depth_band_concordance(depth_sens: pd.DataFrame,
                                 config: Config) -> pd.DataFrame:
    """Compare the high-depth and low-depth reruns per defense system.

    A genuine biological association should appear in the LOW-depth band,
    where the saturation artefact cannot have operated. An association that
    appears only in the high-depth band is what a sampling artefact looks
    like. This comparison is the actual deliverable of the depth sensitivity;
    the per-band hit counts on their own say very little.
    """
    if depth_sens is None or depth_sens.empty:
        return pd.DataFrame()
    if "depth_band" not in depth_sens.columns:
        return pd.DataFrame()

    keep = ["defense_system", "covariate_mode", "phyloglm_coefficient",
            "phyloglm_p_value", "phyloglm_fdr_qvalue", "interpretable"]
    keep = [c for c in keep if c in depth_sens.columns]
    hi = depth_sens[depth_sens["depth_band"] == "high"][keep]
    lo = depth_sens[depth_sens["depth_band"] == "low"][keep]
    if hi.empty or lo.empty:
        return pd.DataFrame()

    m = hi.merge(lo, on=["defense_system", "covariate_mode"],
                 suffixes=("_high_depth", "_low_depth"), how="outer")
    a = config.alpha
    sig_hi = m.get("phyloglm_fdr_qvalue_high_depth", pd.Series(dtype=float)) < a
    sig_lo = m.get("phyloglm_fdr_qvalue_low_depth", pd.Series(dtype=float)) < a
    same_sign = (np.sign(m.get("phyloglm_coefficient_high_depth", np.nan))
                 == np.sign(m.get("phyloglm_coefficient_low_depth", np.nan)))

    m["depth_verdict"] = np.select(
        [sig_hi & sig_lo & same_sign,
         sig_hi & sig_lo & ~same_sign,
         sig_hi & ~sig_lo,
         ~sig_hi & sig_lo],
        ["robust_to_depth",
         "direction_reverses_with_depth",
         "high_depth_only__possible_sampling_artefact",
         "low_depth_only"],
        default="ns_both")
    return m


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
    100% of strains carrying the system).

    What this DOES test: the predictor arm of the saturation. Prevalence is a
    strain-averaged quantity and does not saturate with n_strains the way
    max() does.

    What it does NOT test, and what the previous docstring claimed it did:
    the outcome arm. The response is still ``has_plasmid_binary``, the
    species-propagated label, which saturates with depth exactly as before.
    So agreement between the two feature modes is NOT "direct evidence that
    the primary result isn't sampling-artefactual" -- both fits share the same
    saturated outcome. Read this as one arm of the confound addressed, not
    the confound resolved. The depth-band concordance
    (``build_depth_band_concordance``) and the negative control are the tests
    that speak to the whole thing.

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
        if config.is_diagnostic_mode(covariate_mode):
            continue
        # include_n_strains=True. The depth covariates STAY.
        #
        # They were previously dropped here, on the stated grounds that the
        # prevalence feature is "already a strain-averaged quantity, so
        # log_n_strains would be partially redundant with the feature
        # construction itself". That misidentifies what the covariate is for.
        # It was never a predictor correction -- it stands in for sampling
        # depth as a COMMON CAUSE of the predictor and the outcome. Switching
        # to a prevalence feature de-saturates the predictor arm only; the
        # response is still `has_plasmid_binary`, the species-propagated label
        # that saturates with depth. Removing the covariates therefore left
        # this fit strictly LESS adjusted than the primary fit it was supposed
        # to validate, and made agreement between the two nearly automatic.
        covariates = config.resolve_covariates(
            config.covariate_columns_for_mode(
                covariate_mode, include_plasmid_count=False,
                include_n_strains=True), sub)
        r = call_r_script(
            "phyloglm_uni.R",
            tree_path=tree_path,
            data=sub,
            args={"response": "has_plasmid_binary",
                  "predictors": defense_cols,
                  "mode": "predictor",
                  "defense_side": "predictor",
                  "covariates": list(covariates),
                  "tip_column": "tip",
                  "evolutionary_model": config.phyloglm_estimator,
                  "btol": 20, "boot": 0,
                  # prevalence is continuous; the presence/absence gate does
                  # not apply, but the response gate still does.
                  "min_count": 0,
                  "min_count_response": config.min_count_per_category},
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
        df["depth_adjusted"] = True
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
    """Refit the primary phyloglm under alternative model assumptions.

    Two genuinely distinct axes are swept:

    1. ESTIMATOR (``config.phyloglm_estimator_sensitivity``): MPLE vs the
       Ives-Garland penalised IG10.

    2. COVARIANCE STRUCTURE (``config.phylo_lambda_sensitivity``): the tree is
       rescaled under Pagel's lambda before fitting. lambda < 1 pulls internal
       branches toward a star phylogeny, weakening the assumed phylogenetic
       covariance. Defense systems and plasmids move horizontally, so BM is a
       simplifying assumption and this is the axis reviewers push on.

    The previous version swept ``("OUfixedRoot", "BM_penalized")`` as if they
    were evolutionary models. They are not: ``phyloglm``'s ``method`` argument
    selects the ESTIMATOR, and ``phyloglm_uni.R`` mapped both "BM" and
    "OUfixedRoot" to ``logistic_MPLE``. The OU arm was therefore bit-identical
    to the primary fit -- same data, same estimator, same btol -- so half the
    sensitivity analysis measured nothing at all.
    """
    estimators = tuple(config.phyloglm_estimator_sensitivity or ())
    lambdas = tuple(config.phylo_lambda_sensitivity or ())
    # Legacy: honour phylo_model_sensitivity_models if someone set it directly.
    legacy = tuple(m for m in (config.phylo_model_sensitivity_models or ())
                   if m not in estimators and m != config.phyloglm_estimator)
    estimators = tuple(dict.fromkeys(estimators + legacy))

    if not estimators and not lambdas:
        return pd.DataFrame()

    # (estimator, lambda, label) triples.
    arms = [(e, 1.0, f"estimator={e}") for e in estimators]
    arms += [(config.phyloglm_estimator, float(lam), f"lambda={lam:g}")
             for lam in lambdas]

    logger.info(
        f"Model sensitivity: {len(arms)} arms "
        f"(primary = estimator={config.phyloglm_estimator}, lambda=1)")

    pieces: List[pd.DataFrame] = []
    for estimator, lam, arm_label in arms:
        for covariate_mode in config.covariate_modes:
            if config.is_diagnostic_mode(covariate_mode):
                continue
            covariates = config.resolve_covariates(
                config.covariate_columns_for_mode(
                    covariate_mode, include_plasmid_count=False), phylo_data)
            safe = arm_label.replace("=", "_").replace(".", "p")
            r = call_r_script(
                "phyloglm_uni.R",
                tree_path=tree_path,
                data=phylo_data,
                args={"response": "has_plasmid_binary",
                      "predictors": defense_cols,
                      "mode": "predictor",
                      "defense_side": "predictor",
                      "covariates": list(covariates),
                      "tip_column": "tip",
                      "evolutionary_model": estimator,
                      "lambda_rescale": lam,
                      "btol": 20, "boot": 0,
                      "min_count": config.min_count_per_category,
                      "min_count_response": config.min_count_per_category},
                logger=logger,
                r_executable=config.r_executable,
                workdir=workdir / f"model_sens_{safe}_{covariate_mode}",
            )
            if not r.ok:
                logger.warning(
                    f"phylo_model_sensitivity [{arm_label}/{covariate_mode}] failed: {r.error}"
                )
                continue
            df = r.dataframe.rename(columns={"test_label": "defense_system"})
            df["covariate_mode"] = covariate_mode
            df["sensitivity_arm"] = arm_label
            df["phyloglm_estimator"] = estimator
            df["lambda_rescale"] = lam
            # Legacy column name retained for existing consumers/figures.
            df["evolutionary_model"] = arm_label
            df["phyloglm_fdr_qvalue"] = apply_fdr(df["phyloglm_p_value"],
                                                   method=config.fdr_method).values
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


# ----------------------------------------------------------------------
# Negative control — the calibration check the whole pipeline rests on
# ----------------------------------------------------------------------

def run_negative_control(phylo_data: pd.DataFrame,
                         defense_cols: List[str],
                         tree_path: str,
                         config: Config,
                         logger: logging.Logger,
                         workdir: Path) -> pd.DataFrame:
    """Permute the plasmid label within joint (clade, depth-decile) strata and
    run the PRIMARY univariate sweep on the permuted outcome.

    Under a correctly specified model the number of FDR-significant systems
    should be close to zero. If instead dozens come back significant, the
    model is picking up structure that survives label permutation -- which
    here means sequencing effort -- and no downstream result is interpretable.

    This is the single most informative stage in the pipeline and should be
    run and inspected before any result is believed. It is deliberately
    expensive (``negative_control_n_replicates`` full sweeps); the alternative
    is publishing an artefact.

    Why permute within (clade, depth) rather than freely: a free permutation
    would destroy the phylogenetic autocorrelation of the outcome as well, so
    a hit would tell you only that phylogenetic structure exists. Holding
    clade and depth fixed isolates the question of interest -- is there
    defense-plasmid association BEYOND what clade membership and sequencing
    effort already explain?
    """
    if not config.run_negative_control:
        return pd.DataFrame()
    if "has_plasmid_binary" not in phylo_data.columns:
        logger.warning("negative control skipped — no plasmid outcome column")
        return pd.DataFrame()

    n_rep = max(1, int(config.negative_control_n_replicates))
    strata = build_permutation_strata(phylo_data, config, logger)
    observed = phylo_data["has_plasmid_binary"].values

    covariate_mode = config.primary_covariate_mode
    covariates = config.resolve_covariates(
        config.covariate_columns_for_mode(covariate_mode,
                                          include_plasmid_count=False),
        phylo_data)

    logger.info(
        f"NEGATIVE CONTROL: {n_rep} replicates, plasmid label permuted within "
        f"{int(np.unique(strata).size)} (clade x depth) strata, primary "
        f"covariate mode '{covariate_mode}'")

    # The frame is written ONCE and every replicate reuses it, passing only the
    # permuted outcome column as a small override file. Previously each
    # replicate re-serialised the full ~40 MB species x feature frame; 20
    # replicates of that, concurrently, is exactly the I/O profile that
    # produced SIGBUS elsewhere in the pipeline.
    shared = write_shared_frame(phylo_data, workdir, "negative_control", logger)
    tips = phylo_data["tip"].tolist()

    def _one_replicate(rep: int) -> Optional[pd.DataFrame]:
        rng = np.random.default_rng(int(config.random_seed) + 10_000 + rep)
        override = pd.DataFrame({
            "tip": tips,
            "has_plasmid_binary": _stratified_shuffle(observed, strata, rng),
        })
        r = call_r_script(
            "phyloglm_uni.R",
            tree_path=tree_path,
            shared=shared,
            overrides=override,
            args={"response": "has_plasmid_binary",
                  "predictors": defense_cols,
                  "mode": "predictor",
                  "defense_side": "predictor",
                  "covariates": list(covariates),
                  "tip_column": "tip",
                  "evolutionary_model": config.phyloglm_estimator,
                  "btol": 20, "boot": 0,
                  "min_count": config.min_count_per_category,
                  "min_count_response": config.min_count_per_category},
            logger=logger,
            r_executable=config.r_executable,
            workdir=workdir / f"negative_control_rep{rep:02d}",
            max_retries=int(config.r_max_retries),
        )
        if not r.ok:
            logger.warning(f"negative control replicate {rep} failed: {r.error}")
            return None
        df = r.dataframe.rename(columns={"test_label": "defense_system"})
        df["phyloglm_fdr_qvalue"] = apply_fdr(df["phyloglm_p_value"],
                                              method=config.fdr_method).values
        df["replicate"] = rep
        df["covariate_mode"] = covariate_mode
        return df

    # Replicates are independent, so dispatch them in parallel rather than in
    # the previous sequential loop. This is the difference between one sweep's
    # wall-clock and twenty.
    n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
    if config.max_concurrent_r_calls > 0:
        n_jobs = min(n_jobs, config.max_concurrent_r_calls)
    n_jobs = max(1, min(n_jobs, n_rep))
    logger.info(f"NEGATIVE CONTROL: dispatching {n_rep} replicates across "
                f"{n_jobs} workers")
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(_one_replicate)(rep) for rep in range(n_rep))
    pieces = [d for d in results if d is not None and not d.empty]

    if not pieces:
        logger.error("NEGATIVE CONTROL produced no results — cannot certify "
                     "calibration. Treat all downstream results as unvalidated.")
        return pd.DataFrame()

    long = pd.concat(pieces, ignore_index=True)
    rows = []
    for rep, g in long.groupby("replicate"):
        rows.append({
            "replicate": int(rep),
            "n_tested": int(g["phyloglm_p_value"].notna().sum()),
            "n_fdr_significant": int(
                (g["phyloglm_fdr_qvalue"] < config.alpha).sum()),
            "n_nominal_significant": int(
                (g["phyloglm_p_value"] < config.alpha).sum()),
            "min_p": (float(g["phyloglm_p_value"].min())
                      if g["phyloglm_p_value"].notna().any() else np.nan),
        })
    per_rep = pd.DataFrame(rows)

    mean_hits = float(per_rep["n_fdr_significant"].mean())
    mean_tested = float(per_rep["n_tested"].mean())
    mean_nominal = float(per_rep["n_nominal_significant"].mean())
    # Under correct calibration nominal p-values are uniform, so the expected
    # count below alpha is alpha * n_tested; BH-significant should be fewer.
    expected_nominal = config.alpha * mean_tested
    threshold = (config.negative_control_max_expected_hits_multiplier
                 * max(1.0, config.alpha * mean_tested))
    calibrated = mean_hits <= threshold

    per_rep["mean_fdr_significant"] = mean_hits
    per_rep["mean_nominal_significant"] = mean_nominal
    per_rep["expected_nominal_significant"] = expected_nominal
    per_rep["calibration_threshold"] = threshold
    per_rep["calibrated"] = calibrated

    msg = (f"NEGATIVE CONTROL: mean {mean_hits:.1f} FDR-significant systems "
           f"per replicate (threshold {threshold:.1f}); mean {mean_nominal:.1f} "
           f"nominally significant vs {expected_nominal:.1f} expected under "
           f"uniformity")
    if calibrated:
        logger.info(msg + " — CALIBRATED")
    else:
        logger.error(
            msg + " — NOT CALIBRATED. The primary model is detecting structure "
            "that survives permutation of the plasmid label within clade and "
            "sequencing-depth strata. The most likely explanation is residual "
            "sampling-depth confounding. Raise config.depth_spline_df, inspect "
            "the depth-band concordance table, and do not report primary "
            "associations until this passes.")
    return per_rep

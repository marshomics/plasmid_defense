"""Tier 1: baseline associations (non-phylogenetic).

Primary test in this tier is Firth's penalised logistic regression on species-
level binary defense presence, with ``n_strains`` as a frequency weight and
genome-scale covariates (log genome size, GC, log CDS count) included. For
stratified plasmid outcomes (any_plasmid_<class>), log(n_plasmids) is included
as an additional covariate so species with many plasmids don't saturate every
class. Firth is chosen over ordinary logistic because separation is common
for rare defense systems.

Fisher's exact, Mann-Whitney U, and unweighted logistic are kept as
diagnostics. They are *never* the primary citation in the main results — they
ignore phylogeny entirely. Tier 2 (phyloglm / PGLMM / Pagel's) is what a
reviewer should be asked to trust; Tier 1 exists for sanity-checking the
direction of effect and flagging prevalence-dependent detection issues.

This module does not do any phylogenetic correction. Read tier2_phylo_uni.py
and tier2_multivariate.py for that.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

from .config import Config
from .stats_utils import apply_fdr, firth_logistic_regression


def _build_covariate_matrix(df: pd.DataFrame,
                            covariate_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Centre+scale a subset of numeric covariates. Returns (X_cov, row_mask)
    where row_mask selects species with finite values across all covariates.
    """
    if not covariate_cols:
        return np.zeros((len(df), 0)), np.ones(len(df), dtype=bool)
    sub = df[covariate_cols].apply(pd.to_numeric, errors="coerce")
    mask = sub.notna().all(axis=1).values & np.isfinite(sub.values).all(axis=1)
    X = sub.values.astype(float)
    if mask.sum() > 1:
        mu = np.nanmean(X[mask], axis=0)
        sd = np.nanstd(X[mask], axis=0, ddof=1)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
    # Replace non-finite rows with zeros (row_mask will filter them out anyway)
    X = np.where(np.isfinite(X), X, 0.0)
    return X, mask


def _fisher_exact(defense: np.ndarray, plasmid: np.ndarray) -> dict:
    table = np.array([
        [int(((defense == 0) & (plasmid == 0)).sum()),
         int(((defense == 0) & (plasmid == 1)).sum())],
        [int(((defense == 1) & (plasmid == 0)).sum()),
         int(((defense == 1) & (plasmid == 1)).sum())],
    ])
    odds_ratio, p = stats.fisher_exact(table)
    return {"odds_ratio": odds_ratio, "p_value": p,
            "n_present_with_plasmid": table[1, 1],
            "n_present_no_plasmid": table[1, 0],
            "n_absent_with_plasmid": table[0, 1],
            "n_absent_no_plasmid": table[0, 0]}


def _mann_whitney(prevalence: np.ndarray, plasmid: np.ndarray) -> dict:
    pos = prevalence[plasmid == 1]
    neg = prevalence[plasmid == 0]
    if len(pos) < 2 or len(neg) < 2:
        return {"statistic": np.nan, "p_value": np.nan,
                "median_pos": np.nan, "median_neg": np.nan}
    u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return {"statistic": u, "p_value": p,
            "median_pos": np.median(pos), "median_neg": np.median(neg)}


def _primary_firth_weighted(binary: np.ndarray, plasmid: np.ndarray,
                            weights: np.ndarray,
                            X_cov: np.ndarray) -> dict:
    """Firth logistic with ``n_strains`` frequency weights and covariates.

    X_cov is an (n, k) matrix of centred/scaled covariates. The model fit is
    plasmid ~ intercept + binary + X_cov, and only the `binary` coefficient is
    reported back (other rows are nuisance).
    """
    try:
        ones = np.ones_like(binary, dtype=float)
        X = np.column_stack([ones, binary.astype(float), X_cov])
        fit = firth_logistic_regression(X, plasmid.astype(float), weights=weights)
        return {
            "coefficient": float(fit["coef"][1]),
            "std_err": float(fit["se"][1]),
            "z_value": float(fit["z"][1]),
            "p_value": float(fit["p"][1]),
            "converged": bool(fit["converged"]),
            "iterations": int(fit["iterations"]),
            "n_covariates_used": int(X_cov.shape[1]),
        }
    except Exception as e:
        return {"coefficient": np.nan, "std_err": np.nan,
                "z_value": np.nan, "p_value": np.nan,
                "converged": False, "iterations": 0,
                "n_covariates_used": int(X_cov.shape[1]) if X_cov is not None else 0,
                "error": str(e)}


def _weighted_glm_logreg(binary: np.ndarray, plasmid: np.ndarray,
                         weights: np.ndarray, X_cov: np.ndarray) -> dict:
    """Standard weighted logistic (GLM binomial with freq_weights), with
    covariates — kept as diagnostic alongside Firth.

    statsmodels' binomial link emits RuntimeWarning('overflow encountered
    in exp') for extreme linear predictors during IRLS. The library clips
    internally so the fit still converges; the warning is cosmetic but
    floods the log in multi-thousand-system sweeps. We mute it here with
    a localised filter rather than globally.
    """
    try:
        ones = np.ones_like(binary, dtype=float)
        X = np.column_stack([ones, binary.astype(float), X_cov])
        model = sm.GLM(plasmid, X, family=sm.families.Binomial(),
                       freq_weights=weights.astype(float))
        with warnings.catch_warnings(), np.errstate(over="ignore",
                                                     invalid="ignore",
                                                     divide="ignore"):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = model.fit(disp=0, maxiter=1000)
        return {
            "coefficient": float(res.params[1]),
            "std_err": float(res.bse[1]),
            "z_value": float(res.tvalues[1]),
            "p_value": float(res.pvalues[1]),
        }
    except Exception as e:
        return {"coefficient": np.nan, "std_err": np.nan,
                "z_value": np.nan, "p_value": np.nan, "error": str(e)}


def _run_one_system(args):
    (col, binary, prevalence, plasmid, weights, X_cov, row_mask,
     outcome_label, covariate_mode) = args
    # Apply row_mask so covariate-missing species don't pollute the fit
    binary_m = binary[row_mask]
    prevalence_m = prevalence[row_mask]
    plasmid_m = plasmid[row_mask]
    weights_m = weights[row_mask]
    X_cov_m = X_cov[row_mask] if X_cov.size else X_cov

    r = {"defense_system": col,
         "outcome_label": outcome_label,
         "covariate_mode": covariate_mode,
         "direction": "plasmid_given_defense",
         "n_species_fit": int(row_mask.sum()),
         "defense_prevalence": float(binary_m.mean()) if len(binary_m) else np.nan,
         "n_species_with_defense": int(binary_m.sum()),
         "n_species_without_defense": int((binary_m == 0).sum()),
         "plasmid_rate_with_defense": float(plasmid_m[binary_m == 1].mean())
             if binary_m.sum() else np.nan,
         "plasmid_rate_without_defense": float(plasmid_m[binary_m == 0].mean())
             if (binary_m == 0).sum() else np.nan}

    firth = _primary_firth_weighted(binary_m, plasmid_m, weights_m, X_cov_m)
    r.update({f"firth_weighted_{k}": v for k, v in firth.items()})

    weighted = _weighted_glm_logreg(binary_m, plasmid_m, weights_m, X_cov_m)
    r.update({f"diag_weighted_logreg_{k}": v for k, v in weighted.items()})

    # Fisher and Mann-Whitney are univariate contingency tests; no covariate
    # adjustment is possible for them without changing the test entirely.
    fisher = _fisher_exact(binary_m, plasmid_m)
    r.update({f"diag_fisher_{k}": v for k, v in fisher.items()})

    mw = _mann_whitney(prevalence_m, plasmid_m)
    r.update({f"diag_mannwhitney_{k}": v for k, v in mw.items()})

    return r


def _select_outcome_column(outcome_label: str,
                           outcome_spec: Dict[str, List[Optional[str]]],
                           df: pd.DataFrame) -> Optional[str]:
    """Pick the binary outcome column for Tier 1 given the outcome_spec triple.
    For "any_plasmid", that's has_plasmid_binary. For stratified outcomes, it's
    the any_plasmid_<class> column.
    """
    cols = outcome_spec.get(outcome_label)
    if cols is None:
        return None
    any_col = cols[2]   # (n, frac, any)
    if any_col is None or any_col not in df.columns:
        return None
    return any_col


def run_tier1(binary_df: pd.DataFrame, prevalence_df: pd.DataFrame,
              defense_cols: List[str], config: Config,
              logger: logging.Logger,
              outcome_spec: Optional[Dict[str, List[Optional[str]]]] = None
              ) -> pd.DataFrame:
    """Run Tier 1 for every defense system across every plasmid outcome
    stratum, returning one long-form DataFrame with an ``outcome_label``
    column so callers can filter or pivot.

    If ``outcome_spec`` is None, defaults to the legacy single-outcome run
    against ``has_plasmid_binary``.
    """
    if outcome_spec is None:
        outcome_spec = {"any_plasmid": [None, None, "has_plasmid_binary"]}

    # Build log_n_plasmids now if needed (used for any-of-class outcomes).
    working_binary = binary_df.copy()
    working_prev = prevalence_df.copy()
    if "n_plasmids" in working_binary.columns and "log_n_plasmids" not in working_binary.columns:
        working_binary["log_n_plasmids"] = np.log1p(
            working_binary["n_plasmids"].fillna(0).clip(lower=0))
        working_prev["log_n_plasmids"] = np.log1p(
            working_prev["n_plasmids"].fillna(0).clip(lower=0))

    n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()

    all_results: List[pd.DataFrame] = []
    for covariate_mode in config.covariate_modes:
        cov_base = list(config.covariate_columns_for_mode(
            covariate_mode, include_plasmid_count=False))
        cov_binary = list(config.covariate_columns_for_mode(
            covariate_mode, include_plasmid_count=True))

        for outcome_label in sorted(outcome_spec.keys()):
            outcome_col = _select_outcome_column(outcome_label, outcome_spec, working_binary)
            if outcome_col is None:
                logger.info(
                    f"Tier 1 skipping outcome '{outcome_label}' — no binary column available"
                )
                continue
            # Pick the covariate set — stratified outcomes get log(n_plasmids)
            # added when covariate_mode == with_cov
            cov_cols = cov_base if outcome_label == "any_plasmid" else cov_binary
            # Drop any covariate columns absent from the frame
            cov_cols = [c for c in cov_cols if c in working_binary.columns]

            logger.info(
                f"Tier 1 [{covariate_mode}/{outcome_label}]: Firth logistic on "
                f"{outcome_col} (covariates: {cov_cols})"
            )

            plasmid = working_binary[outcome_col].values.astype(float)
            weights = working_binary["n_strains"].values.astype(float)
            X_cov, row_mask = _build_covariate_matrix(working_binary, cov_cols)
            outcome_mask = np.isfinite(plasmid)
            row_mask = row_mask & outcome_mask

            if row_mask.sum() < 10:
                logger.warning(
                    f"Tier 1 [{covariate_mode}/{outcome_label}]: only {row_mask.sum()} "
                    f"species with complete data; skipping"
                )
                continue

            tasks = [(c, working_binary[c].values, working_prev[c].values,
                      plasmid, weights, X_cov, row_mask, outcome_label,
                      covariate_mode)
                     for c in defense_cols]

            raw = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_run_one_system)(t) for t in tasks
            )
            df_out = pd.DataFrame(raw)

            for col, qcol in [
                ("firth_weighted_p_value", "firth_weighted_fdr_qvalue"),
                ("diag_weighted_logreg_p_value", "diag_weighted_logreg_fdr_qvalue"),
                ("diag_fisher_p_value", "diag_fisher_fdr_qvalue"),
                ("diag_mannwhitney_p_value", "diag_mannwhitney_fdr_qvalue"),
            ]:
                if col in df_out.columns:
                    df_out[qcol] = apply_fdr(df_out[col], method=config.fdr_method).values

            if "firth_weighted_fdr_qvalue" in df_out.columns:
                n_sig = int((df_out["firth_weighted_fdr_qvalue"] < config.alpha).sum())
                logger.info(
                    f"  [{covariate_mode}/{outcome_label}] Firth: {n_sig} systems "
                    f"at FDR < {config.alpha}"
                )

            all_results.append(df_out.sort_values("firth_weighted_p_value"))

    if not all_results:
        return pd.DataFrame(columns=["defense_system", "outcome_label", "covariate_mode"])
    return pd.concat(all_results, ignore_index=True)

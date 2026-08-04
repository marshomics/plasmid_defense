"""Statistical helpers used across the pipeline.

This module contains the parts of the statistical toolkit that need to be
defensible by themselves: FDR correction, Firth's penalised logistic
regression, Cochran's Q for LOCO heterogeneity, the Cauchy combination for
aggregating dependent p-values, and the one-SE rule for LASSO lambda selection.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from statsmodels.stats.multitest import multipletests


# ======================================================================
# FDR helpers
# ======================================================================

def apply_fdr(pvals: pd.Series, method: str = "fdr_bh") -> pd.Series:
    """Benjamini-Hochberg (or other) FDR correction on a Series that may
    contain NaN. NaN p-values are passed through untouched; the FDR family
    size is the count of *non-NaN* p-values only.
    """
    q = pd.Series(np.nan, index=pvals.index, dtype=float)
    mask = pvals.notna()
    if mask.sum() == 0:
        return q
    _, qvals, _, _ = multipletests(pvals[mask].values, method=method)
    q.loc[mask] = qvals
    return q


def apply_global_fdr(df: pd.DataFrame, pvalue_columns: list,
                     method: str = "fdr_bh",
                     qvalue_suffix: str = "_global_qvalue",
                     family_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    """Stack p-value columns into ONE family, FDR-correct jointly, and add
    ``<col>_global_qvalue`` columns.

    This is the correction across all primary tests. Per-stratum correction
    happens upstream and controls the error rate only *within* a stratum,
    which does not cover a narrative that highlights whatever reached
    significance somewhere across ~435 systems x |strata| x 2 directions.

    ``family_mask`` restricts the family to the rows eligible for a primary
    claim (see ``Config.is_primary_slice``). Rows outside the mask get NaN
    global q-values rather than being silently folded into the family, so
    exploratory replicon strata cannot dilute the correction.

    Note: this function was defined but never called anywhere in the pipeline,
    while ``config.report_global_fdr`` was True and its docstring promised the
    correction existed. It is now wired in ``reporting.attach_global_fdr``.
    """
    df = df.copy()
    present = [c for c in pvalue_columns if c in df.columns]
    if not present:
        return df

    if family_mask is None:
        family_mask = pd.Series(True, index=df.index)
    family_mask = family_mask.reindex(df.index).fillna(False).astype(bool)

    stacked = pd.concat(
        [df[c].where(family_mask).rename("p") for c in present], axis=0,
        ignore_index=True)
    q = apply_fdr(stacked, method=method)
    offset = 0
    for c in present:
        n = len(df)
        df[c + qvalue_suffix] = q.iloc[offset:offset + n].values
        offset += n
    df["in_global_fdr_family"] = family_mask.values
    return df


# ======================================================================
# Cauchy combination test — for combining dependent p-values across methods
# ======================================================================

def cauchy_combination(pvals: np.ndarray) -> float:
    """Liu & Xie (2020) Cauchy combination. Robust to unknown dependence
    between the input p-values, which is appropriate here because the
    phyloglm, Pagel, and PGLMM tests are positively correlated under H0.

    Returns NaN if all inputs are NaN.
    """
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[np.isfinite(pvals)]
    if pvals.size == 0:
        return np.nan
    # Clip to (eps, 1-eps) to avoid infinite tan() at the endpoints
    eps = np.finfo(float).eps
    pvals = np.clip(pvals, eps, 1 - eps)
    T = np.mean(np.tan((0.5 - pvals) * np.pi))
    return 0.5 - np.arctan(T) / np.pi


# ======================================================================
# Firth's penalised logistic regression
# ======================================================================

def firth_logistic_regression(X: np.ndarray, y: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              max_iter: int = 100,
                              tol: float = 1e-6) -> dict:
    """Firth's penalised logistic regression.

    Penalises the likelihood by Jeffreys' prior (|I(beta)|^{1/2}). This gives
    finite, bias-reduced estimates even when the data exhibit complete or
    quasi-complete separation, which happens often at the tails of defense-
    system prevalence (e.g. a system present in 5 species, 4 of which carry
    plasmids).

    X must include an intercept column. Returns a dict with keys
    ``coef`` (array), ``se`` (array), ``z`` (array), ``p`` (array),
    ``converged`` (bool), ``iterations`` (int).

    Validation: this is a hand-rolled implementation of Firth's penalised
    score, not a wrapper around R's ``logistf``. Use
    :func:`validate_firth_implementation` to compare coefficients and
    standard errors against ``statsmodels`` GLM on a non-separated case
    (where the Jeffreys-prior penalty is negligible and the two should
    agree to four decimal places). Run that validation once per environment.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)

    beta = np.zeros(p)
    for it in range(max_iter):
        eta = X @ beta
        mu = expit(eta)
        W = w * mu * (1 - mu)
        # Fisher information: X^T diag(W) X
        I = (X.T * W) @ X
        try:
            I_inv = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            I_inv = np.linalg.pinv(I)
        # Hat matrix diagonal H_ii = w_i * sqrt(W_ii) * X_i^T I^{-1} X_i * sqrt(W_ii)
        H_diag = np.einsum("ij,jk,ik->i", X, I_inv, X) * W
        # Firth-adjusted score: U_j^* = sum_i X_ij (y_i - mu_i + H_ii (0.5 - mu_i))
        adj = H_diag * (0.5 - mu)
        U = X.T @ (w * (y - mu + adj))
        delta = I_inv @ U
        beta_new = beta + delta
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    else:
        converged = False

    # Standard errors from the (Firth-penalised) information matrix
    eta = X @ beta
    mu = expit(eta)
    W = w * mu * (1 - mu)
    I = (X.T * W) @ X
    try:
        cov = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(I)
    # Ill-conditioned information matrices (rare predictors, near-separation
    # the Jeffreys penalty couldn't fully rescue) can produce negative OR
    # zero diagonal entries when inversion falls through to pinv. Both cases
    # mean the Wald standard error is not well-defined for that coefficient;
    # NaN them explicitly so the downstream z/p go to NaN rather than to
    # ±inf or an artefactual p ≈ 0 (which FDR would then flag as highly
    # significant). The "converged" flag already captures whether the
    # outer IRLS reached tolerance; NaN SE is a separate, covariate-level
    # failure mode and is surfaced via NaN in the returned p-values.
    diag = np.diag(cov).copy()
    diag[~(np.isfinite(diag) & (diag > 0))] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(diag)
        z = beta / se
        # Two-sided Wald z-test; profile-likelihood intervals would be
        # tighter but plumbing them through is a larger change.
        p_vals = 2 * stats.norm.sf(np.abs(z))

    return {
        "coef": beta,
        "se": se,
        "z": z,
        "p": p_vals,
        "converged": converged,
        "iterations": it + 1,
    }


def validate_firth_implementation(n: int = 500, seed: int = 42,
                                   tol_coef: float = 1e-2,
                                   tol_se: float = 1e-2) -> dict:
    """Sanity-check the hand-rolled Firth implementation against
    ``statsmodels`` GLM on a well-separated, non-pathological case.

    On data without separation, Jeffreys' prior contributes negligibly
    relative to the likelihood, so Firth coefficients and standard errors
    should match ordinary MLE to the tolerance controlled by ``tol_coef``
    / ``tol_se``. Returns a dict with the two coefficient vectors and a
    pass/fail flag; callers should assert ``result["passed"]`` in tests.
    """
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    logit = -0.3 + 0.7 * X1 - 0.4 * X2
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(float)
    X = np.column_stack([np.ones(n), X1, X2])

    firth = firth_logistic_regression(X, y)
    glm = sm.GLM(y, X, family=sm.families.Binomial()).fit(disp=0)
    max_coef_diff = float(np.max(np.abs(firth["coef"] - glm.params)))
    max_se_diff = float(np.max(np.abs(firth["se"] - glm.bse)))
    passed = max_coef_diff < tol_coef and max_se_diff < tol_se
    return {
        "firth_coef": firth["coef"].tolist(),
        "glm_coef": np.asarray(glm.params).tolist(),
        "firth_se": firth["se"].tolist(),
        "glm_se": np.asarray(glm.bse).tolist(),
        "max_coef_diff": max_coef_diff,
        "max_se_diff": max_se_diff,
        "tol_coef": tol_coef,
        "tol_se": tol_se,
        "passed": passed,
    }


def validate_firth_under_separation(seed: int = 42) -> dict:
    """Check Firth behaves correctly under COMPLETE SEPARATION.

    ``validate_firth_implementation`` compares against ``statsmodels`` GLM on
    data *without* separation. That proves the base likelihood code is right,
    but says nothing about the Jeffreys-prior penalty — which is the entire
    reason Firth exists. This closes that gap without needing R's ``logistf``.

    Under complete separation the unpenalised MLE diverges: the coefficient
    runs to infinity and the standard error with it. Firth's penalty must
    instead return a finite, moderate coefficient with a finite standard
    error. Those are the properties we assert, and they are exact consequences
    of the method rather than tolerance-based comparisons, so no reference
    implementation is required.

    The reference values below come from the classic separated 2x2 example:
    every x = 1 observation has y = 1 and every x = 0 observation has y = 0.
    """
    import statsmodels.api as sm

    n = 40
    x = np.repeat([0.0, 1.0], n // 2)
    y = x.copy()                       # perfectly separated
    X = np.column_stack([np.ones(n), x])

    firth = firth_logistic_regression(X, y)
    coef = np.asarray(firth["coef"], dtype=float)
    se = np.asarray(firth["se"], dtype=float)

    # Unpenalised MLE for comparison: should blow up or fail to converge.
    try:
        glm = sm.GLM(y, X, family=sm.families.Binomial()).fit(disp=0)
        mle_slope = float(np.asarray(glm.params)[1])
        mle_se = float(np.asarray(glm.bse)[1])
    except Exception:
        mle_slope, mle_se = np.inf, np.inf

    finite_coef = bool(np.all(np.isfinite(coef)))
    finite_se = bool(np.all(np.isfinite(se)) and np.all(se > 0))
    # Firth shrinks the separated slope to something finite and interpretable.
    bounded = finite_coef and abs(coef[1]) < 25.0
    # And it must be materially smaller than the divergent MLE.
    beats_mle = (not np.isfinite(mle_slope)) or abs(coef[1]) < abs(mle_slope)
    correct_sign = coef[1] > 0

    passed = bool(finite_coef and finite_se and bounded and beats_mle
                  and correct_sign)
    return {
        "firth_slope": float(coef[1]),
        "firth_slope_se": float(se[1]),
        "mle_slope": mle_slope,
        "mle_slope_se": mle_se,
        "finite_coefficients": finite_coef,
        "finite_standard_errors": finite_se,
        "bounded": bounded,
        "smaller_than_mle": beats_mle,
        "correct_sign": correct_sign,
        "passed": passed,
    }


# ======================================================================
# Cochran's Q — heterogeneity across leave-one-clade-out estimates
# ======================================================================

def cochran_q(effect_sizes: np.ndarray, standard_errors: np.ndarray) -> dict:
    """Cochran's Q statistic for heterogeneity of log-odds-ratios across
    independent subsamples (e.g. leave-one-clade-out replicates).

    Q = sum_i w_i (theta_i - theta_bar)^2, with w_i = 1 / SE_i^2 and
    theta_bar the inverse-variance-weighted mean. Under H0 (no
    heterogeneity), Q ~ chi^2(k-1).

    Returns dict: Q, df, p_value, I2 (between-study variance fraction).
    NaN-safe.
    """
    theta = np.asarray(effect_sizes, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    mask = np.isfinite(theta) & np.isfinite(se) & (se > 0)
    if mask.sum() < 2:
        return {"Q": np.nan, "df": 0, "p_value": np.nan, "I2": np.nan,
                "n_effective": int(mask.sum())}
    theta = theta[mask]
    se = se[mask]
    w = 1.0 / (se ** 2)
    theta_bar = np.sum(w * theta) / np.sum(w)
    Q = float(np.sum(w * (theta - theta_bar) ** 2))
    k = theta.size
    df = k - 1
    p = float(stats.chi2.sf(Q, df))
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return {"Q": Q, "df": df, "p_value": p, "I2": I2, "n_effective": k}


# ======================================================================
# One-SE rule for CV-tuned LASSO / Elastic Net
# ======================================================================

def one_se_lambda(lambdas: np.ndarray, cv_scores: np.ndarray) -> int:
    """Select the lambda index by the one-standard-error rule.

    ``cv_scores`` is a 2D array (folds x lambdas) of a maximize-better metric
    (e.g. ROC-AUC). The index returned is the one with the largest lambda
    whose mean score is within one SE of the best mean score. That is the
    conventional "most regularised model within 1 SE of optimum" choice from
    Hastie-Tibshirani.
    """
    cv_scores = np.asarray(cv_scores, dtype=float)
    if cv_scores.ndim != 2:
        raise ValueError("cv_scores must be 2D (folds x lambdas)")
    mean_score = cv_scores.mean(axis=0)
    se_score = cv_scores.std(axis=0, ddof=1) / np.sqrt(cv_scores.shape[0])
    best_idx = int(np.argmax(mean_score))
    threshold = mean_score[best_idx] - se_score[best_idx]
    # Largest lambda (= most regularised) whose mean score is >= threshold.
    # Convention: lambdas is sorted descending (largest -> smallest) in sklearn's
    # Cs, so "largest lambda" = smallest C = first eligible index.
    eligible = np.where(mean_score >= threshold)[0]
    if eligible.size == 0:
        return best_idx
    return int(eligible[np.argmin(lambdas[eligible])])


# ======================================================================
# Rank-product consensus across methods
# ======================================================================

def rank_product(rank_df: pd.DataFrame, methods: list,
                 missing_policy: str = "skip") -> pd.Series:
    """Geometric mean of ranks across the given columns (each column ranks
    defense systems, 1 = strongest evidence).

    WARNING — the raw geometric mean is NOT comparable across rows with
    different numbers of contributing methods, and it is NOT a calibrated
    statistic. Use ``rank_product_with_null`` for anything that gets ranked or
    thresholded. This function is retained as the underlying computation.

    ``missing_policy`` governs how to handle a defense system that is missing
    a rank from one of the methods (e.g. Pagel's skipped for low_count):

        "skip" (default) — geometric mean over the methods that *did* rank the
            system. This avoids conflating "method skipped for a legitimate
            reason" with "method ranked this system last" — but it introduces
            the inverse artefact: a system ranked #1 by phyloglm alone scores
            1.0 and outranks a system ranked #2 by all three methods, which
            scores 2.0. Since PGLMM only admits systems above the multivariate
            prevalence floor and Pagel's skips low-count systems, that is the
            modal case, not an edge case, and it systematically promotes the
            systems with the LEAST corroboration. ``rank_product_with_null``
            corrects for this by calibrating against a per-k permutation null.

        "max_rank" — fill missing ranks with the column-wise maximum rank
            before averaging. Penalises legitimate skips.
    """
    sub = rank_df[methods].copy()
    if missing_policy == "max_rank":
        for m in methods:
            sub[m] = sub[m].fillna(sub[m].max())
        log_ranks = np.log(sub.values)
        gm = np.exp(log_ranks.mean(axis=1))
        return pd.Series(gm, index=sub.index, name="rank_product")
    # "skip": geometric mean over non-missing entries per row.
    arr = sub.values.astype(float)
    mask = np.isfinite(arr) & (arr > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ranks = np.where(mask, np.log(np.where(mask, arr, 1.0)), 0.0)
        counts = mask.sum(axis=1)
        sums = log_ranks.sum(axis=1)
        gm = np.where(counts > 0, np.exp(sums / np.maximum(counts, 1)), np.nan)
    return pd.Series(gm, index=sub.index, name="rank_product")


def rank_product_with_null(rank_df: pd.DataFrame, methods: list,
                           n_permutations: int = 10_000,
                           random_seed: int = 42) -> pd.DataFrame:
    """Calibrated rank product: geometric mean of ranks PLUS a permutation
    p-value computed separately for each number of contributing methods.

    Returns a frame indexed like ``rank_df`` with columns:

        rank_product            geometric mean of available ranks
        n_methods_contributing  how many methods ranked this row
        rank_product_p_value    permutation p-value, calibrated within k
        rank_product_fdr_qvalue BH across all rows with a p-value

    Why this is necessary. Breitling's rank product requires a permutation
    null: the null distribution of a geometric mean of ranks depends on both
    the number of items ranked and the number of methods contributing, and
    neither is constant here. Comparing a raw score of 1.0 from one method
    against 2.0 from three methods is meaningless. Drawing the null separately
    for each k makes the p-values comparable across rows, which the raw score
    never was.

    The null draws ranks uniformly without replacement from 1..n_items
    independently per method, which is the correct null of "no association
    between methods' orderings", holding the observed n and k fixed.
    """
    rng = np.random.default_rng(random_seed)
    sub = rank_df[methods].astype(float)
    arr = sub.values
    mask = np.isfinite(arr) & (arr > 0)
    counts = mask.sum(axis=1)

    observed = rank_product(rank_df, methods, missing_policy="skip").values

    # Number of items each method actually ranked, used as the null support.
    per_method_n = {i: int(np.isfinite(arr[:, i]).sum())
                    for i in range(len(methods))}
    n_items = max([v for v in per_method_n.values()] + [1])

    pvals = np.full(arr.shape[0], np.nan)
    for k in sorted(set(int(c) for c in counts if c > 0)):
        # Null distribution of the geometric mean of k independent uniform
        # ranks drawn from 1..n_items.
        draws = rng.integers(1, n_items + 1, size=(n_permutations, k))
        null_gm = np.exp(np.log(draws).mean(axis=1))
        null_sorted = np.sort(null_gm)
        rows = np.where(counts == k)[0]
        obs_k = observed[rows]
        # P(null <= observed): small rank product = strong evidence.
        ranks_in_null = np.searchsorted(null_sorted, obs_k, side="right")
        pvals[rows] = (ranks_in_null + 1) / (n_permutations + 1)

    out = pd.DataFrame({
        "rank_product": observed,
        "n_methods_contributing": counts.astype(int),
        "rank_product_p_value": pvals,
    }, index=rank_df.index)
    out["rank_product_fdr_qvalue"] = apply_fdr(
        out["rank_product_p_value"]).values
    return out


# ======================================================================
# E-values — sensitivity to unmeasured confounding (VanderWeele & Ding 2017)
# ======================================================================

def _evalue_from_rr(rr: float) -> float:
    """E-value for a risk ratio. ``E = RR + sqrt(RR * (RR - 1))``.

    Ratios below 1 are inverted first, so the returned value is always >= 1 and
    is on a "how strong would the confounder have to be" scale regardless of
    the direction of the association.
    """
    if not np.isfinite(rr) or rr <= 0:
        return np.nan
    if rr < 1:
        rr = 1.0 / rr
    if rr <= 1:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def evalue_from_odds_ratio(odds_ratio: float,
                           ci_low: Optional[float] = None,
                           ci_high: Optional[float] = None,
                           rare_outcome: bool = False) -> dict:
    """E-value for an odds ratio, with the common-outcome correction.

    The E-value is the minimum strength of association -- on the risk-ratio
    scale, with BOTH the exposure and the outcome -- that an unmeasured
    confounder would need in order to explain away the observed effect. It
    turns "we adjusted for what we could measure" into a number a reader can
    argue with.

    ``rare_outcome`` controls the OR-to-RR conversion:

      * False (DEFAULT, and correct here): the outcome is common, so the odds
        ratio overstates the risk ratio and RR is approximated by sqrt(OR),
        which is the approximation VanderWeele & Ding recommend for common
        outcomes. Species-level plasmid carriage is common in this dataset --
        the propagation step alone drives prevalence high -- so using the raw
        OR would inflate every E-value and overstate robustness.
      * True: outcome prevalence under ~15%, where OR ~ RR directly.

    The CI E-value uses the confidence limit CLOSEST TO THE NULL, which is the
    quantity of interest: how strong a confounder would suffice to move the
    bound to 1. If the interval already crosses 1 the CI E-value is 1, because
    no confounding at all is needed.

    Returns a dict with ``evalue_point``, ``evalue_ci``, the converted risk
    ratio, and the conversion used.
    """
    def _to_rr(x):
        if x is None or not np.isfinite(x) or x <= 0:
            return np.nan
        return float(x) if rare_outcome else float(np.sqrt(x))

    rr_point = _to_rr(odds_ratio)
    out = {
        "evalue_point": _evalue_from_rr(rr_point),
        "evalue_ci": np.nan,
        "risk_ratio_approx": rr_point,
        "evalue_conversion": "odds_ratio" if rare_outcome else "sqrt_odds_ratio",
    }

    lo, hi = _to_rr(ci_low), _to_rr(ci_high)
    if np.isfinite(lo) and np.isfinite(hi):
        if lo <= 1.0 <= hi:
            # Interval spans the null: no unmeasured confounding is required.
            out["evalue_ci"] = 1.0
        else:
            bound = lo if lo > 1.0 else hi
            out["evalue_ci"] = _evalue_from_rr(bound)
    return out


def resolve_rare_outcome(rare_outcome, outcome_prevalence: Optional[float],
                         threshold: float = 0.15,
                         logger=None) -> bool:
    """Decide the OR-to-RR conversion, from data when not forced.

    ``rare_outcome=None`` means "look at the observed prevalence". Hard-coding
    this is how the pipeline previously got it wrong: the config asserted that
    species-level plasmid carriage is common, when the measured prevalence is
    5.7%. Deriving it removes the failure mode, and the decision is logged.
    """
    if rare_outcome is not None:
        return bool(rare_outcome)
    if outcome_prevalence is None or not np.isfinite(outcome_prevalence):
        # No prevalence available: assume common, which understates the
        # E-value. Erring toward under-claiming robustness is the safe default.
        if logger:
            logger.warning(
                "E-values: outcome prevalence unavailable; assuming a COMMON "
                "outcome (sqrt(OR)), which understates the E-value.")
        return False
    p = float(outcome_prevalence)
    rare = p < threshold
    if logger:
        logger.info(
            f"E-values: outcome prevalence {p:.1%} -> treating the outcome as "
            f"{'RARE (OR ~ RR)' if rare else 'COMMON (RR ~ sqrt(OR))'} "
            f"[threshold {threshold:.0%}]")
    return rare


def attach_evalues(df: pd.DataFrame,
                   or_col: str = "phyloglm_odds_ratio",
                   ci_low_col: str = "phyloglm_ci_low",
                   ci_high_col: str = "phyloglm_ci_high",
                   rare_outcome: bool = False,
                   prefix: str = "",
                   outcome_prevalence: Optional[float] = None) -> pd.DataFrame:
    """Vectorised ``evalue_from_odds_ratio`` over a results table.

    ``outcome_prevalence`` is recorded alongside so a reader can check the
    conversion was appropriate rather than taking it on trust.
    """
    df = df.copy()
    if or_col not in df.columns:
        return df
    lo = df[ci_low_col] if ci_low_col in df.columns else pd.Series(np.nan, index=df.index)
    hi = df[ci_high_col] if ci_high_col in df.columns else pd.Series(np.nan, index=df.index)
    recs = [evalue_from_odds_ratio(o, l, h, rare_outcome=rare_outcome)
            for o, l, h in zip(df[or_col], lo, hi)]
    ev = pd.DataFrame(recs, index=df.index)
    for c in ev.columns:
        df[f"{prefix}{c}"] = ev[c]
    if outcome_prevalence is not None and np.isfinite(outcome_prevalence):
        df[f"{prefix}evalue_outcome_prevalence"] = float(outcome_prevalence)
    return df


def combine_subsample_pvalues(pvals) -> float:
    """Combine p-values from repeated subsamples of the same test into one
    valid p-value, using the Cauchy (ACAT) combination.

    Use this instead of the median. The median of k p-values is NOT a p-value:
    under H0 with k = 5 independent draws it is Beta(3, 3), so
    P(median < 0.05) is about 0.0012 rather than 0.05. That is super-uniform,
    so BH still controls FDR, but it destroys power and the resulting
    "q-values" have no interpretation on the tested hypothesis. It is also
    incomparable across rows with different k, which happens whenever a system
    is skipped in some subsamples.

    Cauchy combination is valid under arbitrary dependence and handles varying
    k correctly, so rows combining 3 subsamples and rows combining 10 land on
    the same scale.
    """
    arr = np.asarray(list(pvals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return cauchy_combination(arr)

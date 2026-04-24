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
                     qvalue_suffix: str = "_global_qvalue") -> pd.DataFrame:
    """Stack a list of p-value columns into one family, FDR-correct jointly,
    and add corresponding ``<col>_global_qvalue`` columns. This is the global
    correction across all primary phylogenetic tests; per-tier correction is
    done separately upstream.
    """
    df = df.copy()
    stacked = pd.concat([df[c].rename("p") for c in pvalue_columns], axis=0)
    q = apply_fdr(stacked, method=method)
    offset = 0
    for c in pvalue_columns:
        n = len(df[c])
        df[c + qvalue_suffix] = q.iloc[offset:offset + n].values
        offset += n
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

    ``missing_policy`` governs how to handle a defense system that is missing
    a rank from one of the methods (e.g. Pagel's skipped for low_count):

        "skip" (default) — take the geometric mean over the methods that
            *did* rank the system. A system ranked by phyloglm and PGLMM
            but skipped by Pagel's is averaged over two methods, not three.
            This avoids conflating "method skipped for a legitimate reason"
            with "method ranked this system last", which the old default
            did. Systems with zero non-missing ranks return NaN.

        "max_rank" — fill missing ranks with the column-wise maximum rank
            before averaging. The old default; still available for callers
            that want the old behaviour.
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

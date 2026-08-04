"""Tier 2 multivariate: phylogenetic GLMM (primary) plus regularised
selection on phylogenetically-decorrelated residuals (secondary).

The question this module answers: which defense systems independently
predict plasmid carriage, controlling simultaneously for (a) the other
defense systems, (b) shared ancestry, (c) genome-scale covariates, and
(optionally) (d) pairwise defense interactions for top-ranked candidates?

Outcome modes:
  - binary  : ``any_plasmid_<class>`` (or ``has_plasmid_binary``) with
              log(n_plasmids) as an additional covariate for stratified classes
  - binomial: cbind(k, n-k) where k = plasmids-of-class-X per species and
              n = total plasmids for that species. Natural weighting by
              plasmid count, so a species with 2 plasmids and a species with
              2000 contribute their own precision rather than one overwhelming
              the other.

Both binary and binomial modes are run for each stratified outcome when
``plasmid_stratified_primary_mode == "fraction"``; the binomial fit is flagged
as primary in that case and binary is carried alongside as the backward-
compatible complement.

Regularised selection (LASSO / Elastic Net) is retained as a secondary method
on phylogenetically-decorrelated residuals — unchanged from the prior
release. It is run once on the primary ``has_plasmid_binary`` outcome only,
since the interpretation of "features selected beyond phylogenetic noise"
becomes ambiguous across many fraction outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import apply_fdr, one_se_lambda


@dataclass
class MultivariateResult:
    pglmm: pd.DataFrame
    lasso: pd.DataFrame
    elastic_net: pd.DataFrame
    stability: pd.DataFrame


def _filter_common_systems(binary_df: pd.DataFrame,
                           defense_cols: List[str],
                           min_prevalence: float) -> List[str]:
    keep = [c for c in defense_cols
            if binary_df[c].mean() >= min_prevalence and binary_df[c].mean() <= (1 - min_prevalence)]
    return keep


def _pick_interaction_pairs(ranked_systems: List[str],
                            k: int) -> List[Tuple[str, str]]:
    """All pairwise interactions among the top-k systems by rank."""
    top = ranked_systems[:k]
    pairs: List[Tuple[str, str]] = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            pairs.append((top[i], top[j]))
    return pairs


def _maybe_subsample_for_pglmm(phylo_data: pd.DataFrame, config: Config,
                                logger: logging.Logger) -> pd.DataFrame:
    """If ``config.pglmm_max_species`` is set, return a uniform random
    subsample of phylo_data of that size, stratified by phylum if
    possible. Otherwise return the input unchanged.

    Phylogenetic GLMM cost scales as N^2-N^3 in tip count; for a 40k-tip
    tree, dropping to 10-15k makes the fit complete in hours rather than
    days. The subsample is seeded so two PGLMM fits in the same run see
    the same species set (otherwise coefficients across outcomes wouldn't
    be on a comparable footing).
    """
    cap = getattr(config, "pglmm_max_species", None)
    if not cap or cap <= 0 or len(phylo_data) <= cap:
        return phylo_data
    rng = np.random.default_rng(config.random_seed)
    rank = config.permutation_clade_rank
    if rank in phylo_data.columns:
        # Proportional sample within each phylum, rounded up so small
        # phyla don't disappear entirely.
        groups = phylo_data.groupby(rank, dropna=False)
        per_group = groups.size()
        target_per_group = (per_group * (cap / len(phylo_data))).round().astype(int)
        target_per_group = target_per_group.clip(lower=1, upper=per_group)
        idx_keep = []
        for grp_name, n_keep in target_per_group.items():
            idx_pool = phylo_data.index[phylo_data[rank] == grp_name].to_numpy()
            if len(idx_pool) <= n_keep:
                idx_keep.extend(idx_pool)
            else:
                idx_keep.extend(rng.choice(idx_pool, size=int(n_keep),
                                           replace=False))
        sub = phylo_data.loc[idx_keep].reset_index(drop=True)
    else:
        keep = rng.choice(phylo_data.index.to_numpy(),
                          size=int(cap), replace=False)
        sub = phylo_data.loc[keep].reset_index(drop=True)
    logger.info(
        f"  PGLMM subsampling: {len(phylo_data):,} -> {len(sub):,} species "
        f"(cap={cap}, stratified-by={rank if rank in phylo_data.columns else 'uniform'})"
    )
    return sub


def _run_pglmm_single(phylo_data: pd.DataFrame,
                      predictors: List[str],
                      outcome_col_any: str,
                      outcome_cols_binomial: Optional[Tuple[str, str]],
                      covariates: List[str],
                      interaction_pairs: List[Tuple[str, str]],
                      outcome_mode: str,
                      tree_path: str,
                      config: Config,
                      logger: logging.Logger,
                      workdir: Path,
                      subdir: str) -> pd.DataFrame:
    """Single PGLMM fit. ``outcome_mode`` is 'binary' or 'binomial'.
    outcome_col_any is used for binary; outcome_cols_binomial = (k, n) for
    binomial.
    """
    phylo_data = _maybe_subsample_for_pglmm(phylo_data, config, logger)
    args = {
        "predictors": predictors,
        "tip_column": "tip",
        "bayes": False,
        # REML is now passed explicitly. The R script previously derived it as
        # `REML = !use_bayes`, tying a variance-estimation choice to an
        # inference paradigm, so every PQL binomial fit silently got
        # REML = TRUE with no deliberation.
        "reml": True,
        "covariates": list(covariates),
        "interaction_pairs": [list(p) for p in interaction_pairs],
        "outcome_mode": outcome_mode,
    }
    if outcome_mode == "binary":
        args["response"] = outcome_col_any
    else:
        args["response_k_column"] = outcome_cols_binomial[0]
        args["response_n_column"] = outcome_cols_binomial[1]

    timeout_seconds = max(60, int(config.pglmm_timeout_hours) * 3600)
    r = call_r_script(
        "pglmm_mv.R",
        tree_path=tree_path,
        data=phylo_data,
        args=args,
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / subdir,
        timeout=timeout_seconds,
    )
    if not r.ok:
        logger.error(f"pglmm_mv [{subdir}] failed: {r.error}")
        return pd.DataFrame()

    df = r.dataframe.rename(columns={"term": "defense_system"})

    # FDR is applied to main effects and interaction terms as SEPARATE
    # families. Previously only the intercept was excluded, so ~18 defense
    # main effects, 5 genome covariates, and 28 interaction terms were
    # corrected together as one family of ~51, and reporting then presented
    # the interaction q-values as if they had been their own family.
    term = df["defense_system"].astype(str)
    is_intercept = term == "(Intercept)"
    is_interaction = term.str.contains(":", regex=False)
    is_covariate = term.isin(list(config.genome_covariate_columns)
                             + list(config.depth_spline_columns())
                             + list(config.plasmid_count_spline_columns()))
    is_main = ~is_intercept & ~is_interaction & ~is_covariate

    df["term_type"] = np.select(
        [is_intercept, is_interaction, is_covariate],
        ["intercept", "interaction", "covariate"], default="main_effect")
    df["pglmm_fdr_qvalue"] = np.nan
    for family_mask in (is_main, is_interaction):
        if family_mask.sum() == 0:
            continue
        q = apply_fdr(df.loc[family_mask, "pglmm_p_value"],
                      method=config.fdr_method)
        df.loc[family_mask, "pglmm_fdr_qvalue"] = q.values

    # Interaction terms are chosen by univariate phyloglm rank on the SAME
    # data, so their p-values are post-selection. Flag them; do not present
    # them as confirmatory.
    df.loc[is_interaction, "post_selection_inference"] = True

    # Surface convergence status in the log and in the output so degenerate
    # fits (singular phylogenetic variance, non-finite SEs, non-zero convcode)
    # are not mistaken for real results. The R script already computes
    # ``pglmm_converged`` and ``pglmm_fit_degenerate`` if available.
    # Degenerate fits FORFEIT their p-values.
    #
    # This block previously computed the flags, logged "Treating coefficients
    # as provisional", and returned the frame unchanged -- and no consumer
    # (consensus, reporting, plotting) ever read the flags. "Provisional"
    # existed only in a log line while the q-values above went straight into
    # the results. A fit with singular phylogenetic variance or non-finite
    # standard errors is not weak evidence, it is not evidence.
    def _as_bool(series, default=False):
        if series is None:
            return default
        v = series.iloc[0] if len(series) else default
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "t", "yes")
        try:
            return bool(v)
        except Exception:
            return default

    if "pglmm_fit_degenerate" in df.columns or "pglmm_converged" in df.columns:
        degenerate = _as_bool(df.get("pglmm_fit_degenerate"), False)
        converged = _as_bool(df.get("pglmm_converged"), True)
        if degenerate or not converged:
            logger.warning(
                f"pglmm_mv [{subdir}]: fit is degenerate "
                f"(converged={converged}, degenerate={degenerate}). "
                f"P-values and q-values are set to NaN so the fit cannot enter "
                f"FDR, consensus, or a figure; coefficients are retained for "
                f"forensics only."
            )
            df["pglmm_p_value"] = np.nan
            df["pglmm_fdr_qvalue"] = np.nan
            df["pglmm_fit_degenerate"] = True

    return df.sort_values("pglmm_p_value").reset_index(drop=True)


def run_pglmm_multivariate(phylo_data: pd.DataFrame,
                           defense_cols: List[str],
                           tree_path: str,
                           config: Config,
                           logger: logging.Logger,
                           workdir: Path,
                           outcome_spec: Optional[Dict[str, List[Optional[str]]]] = None,
                           ranked_systems: Optional[List[str]] = None
                           ) -> pd.DataFrame:
    """Primary multivariate test across every outcome stratum.

    Returns a long-form DataFrame indexed by (defense_system, outcome_label,
    outcome_mode). Includes intercept rows and interaction-term rows — callers
    that want only main effects should filter on ``defense_system`` being in
    ``defense_cols``.
    """
    if outcome_spec is None:
        outcome_spec = {"any_plasmid": [None, None, "has_plasmid_binary"]}

    common = _filter_common_systems(phylo_data, defense_cols,
                                    config.min_prevalence_multivariate)
    logger.info(f"Tier 2 multivariate PGLMM — {len(common)}/{len(defense_cols)} "
                f"systems pass ≥{config.min_prevalence_multivariate:.0%} prevalence gate")

    if not common:
        return pd.DataFrame(columns=[
            "defense_system", "outcome_label", "outcome_mode",
            "pglmm_coefficient", "pglmm_std_err",
            "pglmm_z_value", "pglmm_p_value", "pglmm_fdr_qvalue"])

    interaction_pairs: List[Tuple[str, str]] = []
    if config.add_multivariate_interactions and ranked_systems:
        # Only consider systems that passed the prevalence gate for interactions
        ranked_common = [s for s in ranked_systems if s in common]
        interaction_pairs = _pick_interaction_pairs(
            ranked_common, config.n_interaction_systems)
        logger.info(f"  adding {len(interaction_pairs)} interaction pairs")

    # Ensure log_n_plasmids is present for stratified outcomes
    if "n_plasmids" in phylo_data.columns and "log_n_plasmids" not in phylo_data.columns:
        phylo_data = phylo_data.copy()
        phylo_data["log_n_plasmids"] = np.log1p(
            phylo_data["n_plasmids"].fillna(0).clip(lower=0))

    pieces: List[pd.DataFrame] = []

    for covariate_mode in config.covariate_modes:
        for outcome_label in sorted(outcome_spec.keys()):
            triple = outcome_spec[outcome_label]
            if triple is None or len(triple) != 3:
                continue
            n_col, frac_col, any_col = triple

            if any_col is None or any_col not in phylo_data.columns:
                logger.info(f"  skipping [{outcome_label}] — binary outcome absent")
                continue

            include_plasmid_count = (outcome_label != "any_plasmid")
            covariates = list(config.resolve_covariates(
                config.covariate_columns_for_mode(
                    covariate_mode,
                    include_plasmid_count=include_plasmid_count),
                phylo_data))

            # Binary outcome fit
            df_bin = _run_pglmm_single(
                phylo_data, common, any_col, None, covariates, interaction_pairs,
                "binary", tree_path, config, logger, workdir,
                subdir=f"pglmm_mv_{covariate_mode}_{outcome_label}_binary",
            )
            if not df_bin.empty:
                df_bin["outcome_label"] = outcome_label
                df_bin["outcome_mode"] = "binary"
                df_bin["covariate_mode"] = covariate_mode
                # `is_primary` marks the fit that consensus and the headline
                # tables actually consume. Binary is primary because it is the
                # only mode phyloglm and Pagel's can also fit, so consensus
                # combines comparable estimands. (This column was previously
                # written twice and read nowhere, while config declared the
                # binomial fit primary and every consumer filtered it out.)
                df_bin["is_primary"] = (
                    outcome_label == "any_plasmid"
                    or config.plasmid_stratified_primary_mode == "binary"
                )
                pieces.append(df_bin)

            # Binomial (fraction) outcome fit
            if (outcome_label != "any_plasmid"
                    and n_col is not None
                    and n_col in phylo_data.columns
                    and "n_plasmids" in phylo_data.columns):
                df_bnm = _run_pglmm_single(
                    phylo_data, common, None, (n_col, "n_plasmids"), covariates,
                    interaction_pairs, "binomial", tree_path, config, logger,
                    workdir,
                    subdir=f"pglmm_mv_{covariate_mode}_{outcome_label}_binomial",
                )
                if not df_bnm.empty:
                    df_bnm["outcome_label"] = outcome_label
                    df_bnm["outcome_mode"] = "binomial"
                    df_bnm["covariate_mode"] = covariate_mode
                    # The binomial fit is a concordance check on the binary
                    # primary, not a silent alternative primary.
                    df_bnm["is_primary"] = (
                        config.plasmid_stratified_primary_mode == "fraction"
                    )
                    df_bnm["role"] = "concordance_check"
                    pieces.append(df_bnm)

            n_sig = 0
            if pieces:
                last = pieces[-1]
                n_sig = int((last["pglmm_fdr_qvalue"] < config.alpha).sum())
            logger.info(
                f"  PGLMM [{covariate_mode}/{outcome_label}]: last fit -> "
                f"{n_sig} at FDR < {config.alpha}"
            )

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _phylo_residualise(phylo_data: pd.DataFrame,
                       defense_cols: List[str],
                       tree_path: str,
                       config: Config,
                       logger: logging.Logger,
                       workdir: Path) -> Optional[pd.DataFrame]:
    r = call_r_script(
        "phylo_residuals.R",
        tree_path=tree_path,
        data=phylo_data,
        args={"response": "has_plasmid_binary",
              "predictors": defense_cols,
              "tip_column": "tip"},
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / "phylo_residuals",
    )
    if not r.ok:
        logger.warning(
            f"phylo_residuals failed: {r.error}. Falling back to RAW "
            f"standardised features — downstream output will be tagged "
            f"phylo_residualised=False so it is not mislabelled as "
            f"phylogenetically decorrelated.")
        return None

    resid = r.dataframe
    # The R script writes a companion status table recording, per predictor,
    # whether decorrelation actually succeeded or silently fell back to
    # scale(x). Without this the LASSO output claimed phylo-residualised
    # features regardless of what it received.
    status_path = Path(str(r.output_path) + ".status.tsv") \
        if getattr(r, "output_path", None) else None
    n_failed, resp_ok = 0, True
    if status_path is not None and status_path.exists():
        try:
            st = pd.read_csv(status_path, sep="\t")
            n_failed = int((~st["decorrelated"].astype(bool)).sum())
            if "response_decorrelated" in st.columns and len(st):
                resp_ok = bool(st["response_decorrelated"].astype(bool).iloc[0])
        except Exception as exc:
            logger.debug(f"could not read phylo_residuals status table: {exc}")
    if n_failed or not resp_ok:
        logger.warning(
            f"phylo_residuals: {n_failed} predictor(s) fell back to raw "
            f"standardised values; response decorrelated = {resp_ok}. "
            f"Downstream coefficients are only partially decorrelated.")
    resid.attrs["n_predictors_not_decorrelated"] = n_failed
    resid.attrs["response_decorrelated"] = resp_ok
    return resid


def _cv_lasso_path(X: np.ndarray, y: np.ndarray, n_cv: int,
                   l1_ratio: float, Cs: np.ndarray, seed: int) -> tuple:
    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
    fold_scores = np.zeros((n_cv, len(Cs)))
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        for ci, C in enumerate(Cs):
            if l1_ratio >= 1.0:
                clf = LogisticRegression(penalty="l1", C=C, solver="saga",
                                         max_iter=5000, random_state=seed)
            else:
                clf = LogisticRegression(penalty="elasticnet", C=C,
                                         l1_ratio=l1_ratio, solver="saga",
                                         max_iter=5000, random_state=seed)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            try:
                fold_scores[fold_idx, ci] = roc_auc_score(y[te], p)
            except ValueError:
                fold_scores[fold_idx, ci] = np.nan
    refit_coefs = np.zeros((len(Cs), X.shape[1]))
    for ci, C in enumerate(Cs):
        if l1_ratio >= 1.0:
            clf = LogisticRegression(penalty="l1", C=C, solver="saga",
                                     max_iter=5000, random_state=seed)
        else:
            clf = LogisticRegression(penalty="elasticnet", C=C,
                                     l1_ratio=l1_ratio, solver="saga",
                                     max_iter=5000, random_state=seed)
        clf.fit(X, y)
        refit_coefs[ci] = clf.coef_.ravel()
    return fold_scores, refit_coefs


def _run_regularised(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                     l1_ratio: float, config: Config,
                     logger: logging.Logger) -> pd.DataFrame:
    Cs = np.logspace(-3, 2, 40)
    lambdas = 1.0 / (X.shape[0] * Cs)

    fold_scores, refit = _cv_lasso_path(
        X, y, config.cv_folds, l1_ratio, Cs, seed=config.random_seed
    )
    pick_idx = one_se_lambda(lambdas, fold_scores) if config.lasso_one_se_rule \
        else int(np.argmax(np.nanmean(fold_scores, axis=0)))
    chosen_C = float(Cs[pick_idx])
    coefs = refit[pick_idx]

    rng = np.random.default_rng(config.random_seed)
    sel_freq = np.zeros(X.shape[1])
    n = X.shape[0]
    m = int(np.ceil(config.lasso_stability_subsample_frac * n))
    for _ in range(config.lasso_stability_n_subsamples):
        idx = rng.choice(n, size=m, replace=False)
        try:
            if l1_ratio >= 1.0:
                clf = LogisticRegression(penalty="l1", C=chosen_C, solver="saga",
                                         max_iter=5000, random_state=config.random_seed)
            else:
                clf = LogisticRegression(penalty="elasticnet", C=chosen_C,
                                         l1_ratio=l1_ratio, solver="saga",
                                         max_iter=5000, random_state=config.random_seed)
            clf.fit(X[idx], y[idx])
            sel_freq += (np.abs(clf.coef_.ravel()) > 1e-8).astype(int)
        except Exception:
            continue
    sel_freq /= config.lasso_stability_n_subsamples

    return pd.DataFrame({
        "defense_system": feature_names,
        "coefficient": coefs,
        "selected_one_se": np.abs(coefs) > 1e-8,
        "stability_selection_freq": sel_freq,
        "stable_selection": sel_freq >= config.lasso_stability_threshold,
        "chosen_C": chosen_C,
        "chosen_lambda": 1.0 / (X.shape[0] * chosen_C),
        "one_se_rule": config.lasso_one_se_rule,
        "l1_ratio": l1_ratio,
    })


def run_regularised_on_residuals(phylo_data: pd.DataFrame,
                                 defense_cols: List[str],
                                 tree_path: str,
                                 config: Config,
                                 logger: logging.Logger,
                                 workdir: Path) -> tuple:
    """Run LASSO + Elastic Net on phylogenetically-decorrelated residuals
    against the legacy ``has_plasmid_binary`` outcome. Stratified outcomes are
    handled by the PGLMM path; running this for every stratum multiplies
    compute without adding interpretive clarity.
    """
    common = _filter_common_systems(phylo_data, defense_cols,
                                    config.min_prevalence_multivariate)
    logger.info(f"Regularised selection on phylo-decorrelated residuals "
                f"(LASSO + Elastic Net) — {len(common)} systems")
    if not common:
        empty = pd.DataFrame(columns=["defense_system", "coefficient",
                                      "selected_one_se", "stability_selection_freq"])
        return empty, empty.copy()

    resid = _phylo_residualise(phylo_data, common, tree_path, config,
                               logger, workdir)
    if resid is None:
        X = phylo_data[common].values.astype(float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        y = phylo_data["has_plasmid_binary"].values.astype(int)
    else:
        # phylo_residuals.R applies normalise_tips() to the tree+data tip
        # labels — strip outer single quotes, trim whitespace, replace
        # interior spaces with underscores. The Python-side phylo_data
        # still carries the ORIGINAL tip column (with spaces). Normalise
        # phylo_data's `tip` column the same way before joining so the
        # .loc lookup actually finds the rows.
        def _normalise_tip(s):
            s = str(s).strip().strip("'").strip()
            return s.replace(" ", "_")
        phylo_data_norm = phylo_data.copy()
        phylo_data_norm["tip"] = phylo_data_norm["tip"].apply(_normalise_tip)
        resid = resid.set_index("tip")
        # Pre-flight asserts so a join mismatch surfaces immediately rather
        # than as an opaque pandas KeyError after multi-day residualisation.
        # Lost ~6 days of compute the one time this assertion was missing.
        if not resid.index.is_unique:
            n_dup = int(resid.index.duplicated().sum())
            raise RuntimeError(
                f"phylo_residuals.R returned {n_dup} duplicate tip labels; "
                "expected unique. Check that the R script's normalise_tips() "
                "isn't collapsing distinct species (e.g. via bracket stripping)."
            )
        norm_tips = set(phylo_data_norm["tip"])
        overlap = sum(t in norm_tips for t in resid.index)
        if overlap == 0:
            sample_resid = list(resid.index[:3])
            sample_phylo = list(phylo_data_norm["tip"].head(3))
            raise RuntimeError(
                f"LASSO residual join failed: zero overlap between residuals "
                f"({len(resid.index)} rows) and phylo_data ({len(phylo_data_norm)} "
                f"rows) after tip normalisation. Most likely a tip-label form "
                f"mismatch between R's normalise_tips() and the Python-side "
                f"_normalise_tip(). Sample resid keys: {sample_resid}; sample "
                f"phylo_data keys: {sample_phylo}."
            )
        if overlap < len(resid.index):
            logger.warning(
                f"LASSO residual join: {overlap}/{len(resid.index)} residual "
                "tips match phylo_data — partial mismatch. Continuing but "
                "investigate if downstream LASSO coefficients look wrong."
            )
        aligned = phylo_data_norm.set_index("tip").loc[resid.index]
        X = resid[[f"predictor_{c}" for c in common]].values.astype(float)
        # The R script computes `response_residual` and previously nothing used
        # it -- only X was decorrelated, so "the phylogeny has already been
        # partialled out" was half true at best. LogisticRegression needs a
        # binary target, so the residualised response is dichotomised at its
        # median, which preserves the decorrelation while keeping the model
        # well-posed. Falls back to the raw label if residualisation failed.
        if ("response_residual" in resid.columns
                and resid.attrs.get("response_decorrelated", False)):
            rr = pd.to_numeric(resid["response_residual"], errors="coerce")
            y = (rr > rr.median()).astype(int).values
        else:
            y = aligned["has_plasmid_binary"].values.astype(int)

    lasso = _run_regularised(X, y, common, l1_ratio=1.0,
                             config=config, logger=logger)
    elastic_net = _run_regularised(X, y, common, l1_ratio=0.5,
                                   config=config, logger=logger)
    return lasso, elastic_net


def run_tier2_multivariate(phylo_data: pd.DataFrame,
                           defense_cols: List[str],
                           tree_path: str,
                           config: Config,
                           logger: logging.Logger,
                           workdir: Path,
                           outcome_spec: Optional[Dict[str, List[Optional[str]]]] = None,
                           ranked_systems: Optional[List[str]] = None
                           ) -> MultivariateResult:
    """Run PGLMM across every outcome stratum (primary) + LASSO/Elastic Net on
    phylo-residualised features (secondary, against has_plasmid_binary only).
    """
    pglmm = run_pglmm_multivariate(phylo_data, defense_cols, tree_path,
                                   config, logger, workdir,
                                   outcome_spec=outcome_spec,
                                   ranked_systems=ranked_systems)
    lasso, enet = run_regularised_on_residuals(phylo_data, defense_cols,
                                               tree_path, config, logger,
                                               workdir)

    # Merge PGLMM (primary any_plasmid / binary outcome only, excluding
    # intercept and interactions) with LASSO + Elastic Net for a compact
    # stability overview.
    merge_key = "defense_system"
    if pglmm.empty:
        stability = lasso.rename(columns={"coefficient": "lasso_coef",
                                          "selected_one_se": "lasso_selected",
                                          "stability_selection_freq": "lasso_stab_freq"})
        stability = stability.merge(
            enet.rename(columns={"coefficient": "enet_coef",
                                 "selected_one_se": "enet_selected",
                                 "stability_selection_freq": "enet_stab_freq"}),
            on=merge_key, suffixes=("", "_enet"), how="outer")
    else:
        # Restrict the stability overview to the with_cov any_plasmid binary
        # fit so that merging by defense_system is unambiguous.
        mask = (
            (pglmm["defense_system"] != "(Intercept)")
            & (~pglmm["defense_system"].str.contains(":", regex=False))
            & (pglmm.get("outcome_label", "any_plasmid") == "any_plasmid")
            & (pglmm.get("outcome_mode", "binary") == "binary")
            & (pglmm.get("covariate_mode", "with_cov") == "with_cov")
        )
        pglmm_small = pglmm[mask][[
            "defense_system", "pglmm_coefficient", "pglmm_p_value",
            "pglmm_fdr_qvalue"]].copy()
        stability = pglmm_small.merge(
            lasso[["defense_system", "coefficient", "selected_one_se",
                   "stability_selection_freq", "stable_selection"]]
                .rename(columns={"coefficient": "lasso_coef",
                                 "selected_one_se": "lasso_selected_one_se",
                                 "stability_selection_freq": "lasso_stab_freq",
                                 "stable_selection": "lasso_stable"}),
            on=merge_key, how="outer",
        ).merge(
            enet[["defense_system", "coefficient", "selected_one_se",
                  "stability_selection_freq", "stable_selection"]]
                .rename(columns={"coefficient": "enet_coef",
                                 "selected_one_se": "enet_selected_one_se",
                                 "stability_selection_freq": "enet_stab_freq",
                                 "stable_selection": "enet_stable"}),
            on=merge_key, how="outer",
        )

    return MultivariateResult(pglmm=pglmm, lasso=lasso, elastic_net=enet,
                              stability=stability)

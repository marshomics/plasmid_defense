"""Clade-blocked Random Forest for defense-system predictor ranking.

The old pipeline's RF used stratified k-fold CV on species-level data, which
over-estimates generalisation because species from the same clade appear in
both train and test splits (the i.i.d. assumption underlying classical CV
is violated in a tree-structured dataset). This version uses LeaveOneGroupOut
with GTDB class (or phylum, configurable) as the blocking group, which is
the standard correction for phylogenetic non-independence in ML evaluation.

Two flavours:
    - binary     : max-across-strains defense presence/absence
    - prevalence : proportion of strains carrying each system

Both report clade-blocked CV ROC-AUC, per-class fold-level AUCs (so variance
across clades is visible), Gini feature importance, and phylogenetically-
blocked permutation importance (permutation done across species within each
test fold, preserving clade structure).

Blocked CV is pessimistic for small clades; test folds with < 10 held-out
species are skipped. Report counts of usable folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from .config import Config


@dataclass
class RFResult:
    binary: pd.DataFrame
    prevalence: pd.DataFrame
    fold_aucs: pd.DataFrame


def _run_rf_one_matrix(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                       groups: np.ndarray, config: Config,
                       logger: logging.Logger,
                       feature_label: str) -> tuple:
    """Clade-blocked RF: one fold per unique ``groups`` value. Returns
    (feature_importance_df, fold_auc_df).
    """
    logo = LeaveOneGroupOut()
    unique_groups = np.unique(groups)
    logger.info(f"  RF ({feature_label}): {len(unique_groups)} CV folds (clades)")

    aucs = []
    fold_perm_import = []
    for g in unique_groups:
        test_mask = (groups == g)
        if test_mask.sum() < 10 or (~test_mask).sum() < 20:
            # Too few species held out / too few to train on
            continue
        if y[test_mask].sum() in (0, int(test_mask.sum())):
            # No variance in outcome on held-out clade
            continue

        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            n_jobs=config.n_jobs if config.n_jobs > 0 else -1,
            random_state=config.random_seed,
            oob_score=False,
        )
        model.fit(X[~test_mask], y[~test_mask])
        probs = model.predict_proba(X[test_mask])[:, 1]
        try:
            auc = roc_auc_score(y[test_mask], probs)
        except ValueError:
            continue
        aucs.append({"held_out_clade": g,
                     "n_held_out": int(test_mask.sum()),
                     "n_plasmid_pos_held_out": int(y[test_mask].sum()),
                     "cv_auc": float(auc)})
        # Per-fold permutation importance on the held-out clade.
        try:
            pi = permutation_importance(model, X[test_mask], y[test_mask],
                                        n_repeats=10, random_state=config.random_seed,
                                        scoring="roc_auc", n_jobs=1)
            fold_perm_import.append(pi.importances_mean)
        except Exception:
            pass

    fold_auc_df = pd.DataFrame(aucs)
    fold_auc_df["feature_set"] = feature_label

    # Aggregate model on full data for Gini importance (cheapest summary)
    full_model = RandomForestClassifier(
        n_estimators=1000, min_samples_leaf=3,
        n_jobs=config.n_jobs if config.n_jobs > 0 else -1,
        random_state=config.random_seed,
    )
    full_model.fit(X, y)
    gini = full_model.feature_importances_

    # Average permutation importance across folds where it was computable
    if fold_perm_import:
        mean_perm = np.mean(np.vstack(fold_perm_import), axis=0)
        std_perm = np.std(np.vstack(fold_perm_import), axis=0)
    else:
        mean_perm = np.full(X.shape[1], np.nan)
        std_perm = np.full(X.shape[1], np.nan)

    import_df = pd.DataFrame({
        "defense_system": feature_names,
        "rf_gini_importance": gini,
        "rf_perm_importance_mean": mean_perm,
        "rf_perm_importance_std": std_perm,
        "feature_set": feature_label,
    }).sort_values("rf_perm_importance_mean", ascending=False)

    mean_auc = fold_auc_df["cv_auc"].mean() if not fold_auc_df.empty else np.nan
    logger.info(f"  RF ({feature_label}): usable folds={len(fold_auc_df)}, "
                f"mean clade-blocked AUC={mean_auc:.3f}")

    return import_df, fold_auc_df


def run_clade_blocked_rf(binary_df: pd.DataFrame,
                         prevalence_df: pd.DataFrame,
                         defense_cols: List[str],
                         config: Config,
                         logger: logging.Logger,
                         clade_rank: str = "gtdb_class") -> RFResult:
    """Run RF on both feature sets (binary, prevalence) with clade-blocked CV.

    A species whose clade is NaN is dropped. If the number of eligible clades
    falls below 5 at the primary rank, we fall back to ``gtdb_phylum``.
    """
    if clade_rank not in binary_df.columns:
        raise ValueError(f"{clade_rank} not in binary_df columns")

    eligible = binary_df[clade_rank].notna()
    # Fallback to phylum if too few classes
    if binary_df.loc[eligible, clade_rank].nunique() < 5 and "gtdb_phylum" in binary_df.columns:
        logger.info(f"  RF: only {binary_df.loc[eligible, clade_rank].nunique()} "
                    f"classes; falling back to gtdb_phylum")
        clade_rank = "gtdb_phylum"
        eligible = binary_df[clade_rank].notna()

    y = binary_df.loc[eligible, "has_plasmid_binary"].values.astype(int)
    groups = binary_df.loc[eligible, clade_rank].values

    X_bin = binary_df.loc[eligible, defense_cols].values.astype(float)
    X_prev = prevalence_df.loc[eligible, defense_cols].values.astype(float)

    logger.info(f"Clade-blocked Random Forest (LeaveOneGroupOut, "
                f"blocking on {clade_rank}, {len(np.unique(groups))} groups)")

    bin_imp, bin_aucs = _run_rf_one_matrix(X_bin, y, defense_cols, groups,
                                           config, logger, "binary")
    prev_imp, prev_aucs = _run_rf_one_matrix(X_prev, y, defense_cols, groups,
                                             config, logger, "prevalence")

    fold_aucs = pd.concat([bin_aucs, prev_aucs], ignore_index=True)
    return RFResult(binary=bin_imp, prevalence=prev_imp, fold_aucs=fold_aucs)

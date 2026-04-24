"""Consensus scoring across phylogenetic methods, per (outcome_label,
covariate_mode) slice.

Rank-product across the defensible methods (phyloglm univariate, PGLMM
multivariate, Pagel's). Non-phylogenetic tests are intentionally excluded —
the point of consensus is to find defense systems with a robust phylogenetic
signal across multiple model assumptions.

Also a Cauchy combination p-value across the same three methods. Cauchy is
robust to positive dependence, which is expected here.

Long-form inputs: each of the three phylogenetic method tables carries an
``outcome_label`` column, and phyloglm / PGLMM also carry ``covariate_mode``
and (for phyloglm) ``direction``. Pagel's has no covariate_mode by
construction (bivariate test); it gets repeated across covariate modes so
rank-product is well-defined per slice.

Consensus is built for the *primary direction* (plasmid_given_defense) of
each (outcome_label, covariate_mode) slice. Reverse-direction results travel
alongside as a separate output table.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .stats_utils import cauchy_combination, rank_product


def _filter_primary(phyloglm: pd.DataFrame, outcome_label: str,
                    covariate_mode: str) -> pd.DataFrame:
    df = phyloglm.copy() if phyloglm is not None else pd.DataFrame()
    if df.empty:
        return df
    if "outcome_label" in df.columns:
        df = df[df["outcome_label"] == outcome_label]
    if "direction" in df.columns:
        df = df[df["direction"] == "plasmid_given_defense"]
    if "covariate_mode" in df.columns:
        df = df[df["covariate_mode"] == covariate_mode]
    return df


def _filter_pglmm(pglmm_mv: pd.DataFrame, outcome_label: str,
                  covariate_mode: str) -> pd.DataFrame:
    df = pglmm_mv.copy() if pglmm_mv is not None else pd.DataFrame()
    if df.empty:
        return df
    if "outcome_label" in df.columns:
        df = df[df["outcome_label"] == outcome_label]
    if "outcome_mode" in df.columns:
        # Prefer binary mode for consensus since phyloglm and Pagel's are binary
        df = df[df["outcome_mode"] == "binary"]
    if "covariate_mode" in df.columns:
        df = df[df["covariate_mode"] == covariate_mode]
    df = df[df["defense_system"] != "(Intercept)"]
    df = df[~df["defense_system"].astype(str).str.contains(":", regex=False)]
    return df


def _filter_pagels(pagels: pd.DataFrame, outcome_label: str) -> pd.DataFrame:
    df = pagels.copy() if pagels is not None else pd.DataFrame()
    if df.empty:
        return df
    if "outcome_label" in df.columns:
        df = df[df["outcome_label"] == outcome_label]
    return df


def build_consensus_table(tier2_phyloglm: pd.DataFrame,
                          tier2_pagels: pd.DataFrame,
                          pglmm_mv: pd.DataFrame,
                          outcome_label: str = "any_plasmid",
                          covariate_mode: str = "with_cov") -> pd.DataFrame:
    """Merge the phylogenetic-only method results for one
    (outcome_label, covariate_mode) slice into a consensus table.
    """
    uni = _filter_primary(tier2_phyloglm, outcome_label, covariate_mode)
    mv = _filter_pglmm(pglmm_mv, outcome_label, covariate_mode)
    pag = _filter_pagels(tier2_pagels, outcome_label)

    needed_uni = ["defense_system", "phyloglm_coefficient",
                  "phyloglm_p_value", "phyloglm_fdr_qvalue"]
    if uni.empty or not all(c in uni.columns for c in needed_uni):
        return pd.DataFrame(columns=needed_uni + ["outcome_label",
                                                   "covariate_mode",
                                                   "rank_product"])

    uni = uni[needed_uni].copy()

    needed_pag = ["defense_system", "pagel_p_value", "pagel_fdr_qvalue"]
    if pag.empty or not all(c in pag.columns for c in needed_pag):
        pag = pd.DataFrame(columns=needed_pag)
    else:
        pag = pag[needed_pag].copy()

    needed_mv = ["defense_system", "pglmm_coefficient", "pglmm_p_value",
                 "pglmm_fdr_qvalue"]
    if mv.empty or not all(c in mv.columns for c in needed_mv):
        mv = pd.DataFrame(columns=needed_mv)
    else:
        mv = mv[needed_mv].copy()

    out = uni.merge(mv, on="defense_system", how="outer") \
             .merge(pag, on="defense_system", how="outer")
    out["outcome_label"] = outcome_label
    out["covariate_mode"] = covariate_mode

    for src in ["phyloglm_p_value", "pglmm_p_value", "pagel_p_value"]:
        if src not in out.columns:
            out[src] = np.nan
        # na_option="keep" lets rank_product distinguish "method skipped"
        # from "method ranked last", so Pagel's low-count skips don't
        # artefactually demote a system that phyloglm + PGLMM both rank
        # highly. (Low-n Pagel's is a known property of this dataset —
        # see the tier2_pagels module docstring.)
        out[src + "_rank"] = out[src].rank(method="average", na_option="keep")

    rank_cols = ["phyloglm_p_value_rank", "pglmm_p_value_rank", "pagel_p_value_rank"]
    out["rank_product"] = rank_product(out, rank_cols,
                                       missing_policy="skip").values
    out["n_methods_contributing"] = out[rank_cols].notna().sum(axis=1).astype(int)
    out["rank_product_rank"] = out["rank_product"].rank(method="average",
                                                        na_option="bottom")

    def _combine(row):
        return cauchy_combination(np.asarray([
            row.get("phyloglm_p_value"),
            row.get("pglmm_p_value"),
            row.get("pagel_p_value"),
        ], dtype=float))
    out["cauchy_combined_p"] = out.apply(_combine, axis=1)

    def _agree(row):
        a = row.get("phyloglm_coefficient")
        b = row.get("pglmm_coefficient")
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.nan
        return int(np.sign(a) == np.sign(b))
    out["phylo_direction_agreement"] = out.apply(_agree, axis=1)

    return out.sort_values("rank_product")


def build_consensus_by_outcome(tier2_phyloglm: pd.DataFrame,
                               tier2_pagels: pd.DataFrame,
                               pglmm_mv: pd.DataFrame,
                               outcome_spec: Optional[Dict[str, List[Optional[str]]]] = None,
                               covariate_modes: Optional[List[str]] = None
                               ) -> pd.DataFrame:
    """Build one consensus table per (outcome_label, covariate_mode) slice
    and concatenate. Covariate modes default to whichever ones appear in the
    phyloglm input (or ``('with_cov', 'without_cov')`` if absent).
    """
    labels = list(outcome_spec.keys()) if outcome_spec else []
    for src in (tier2_phyloglm, tier2_pagels, pglmm_mv):
        if src is not None and "outcome_label" in src.columns:
            for lab in src["outcome_label"].dropna().unique().tolist():
                if lab not in labels:
                    labels.append(lab)
    if not labels:
        labels = ["any_plasmid"]

    if covariate_modes is None:
        seen = set()
        for src in (tier2_phyloglm, pglmm_mv):
            if src is not None and "covariate_mode" in src.columns:
                seen.update(src["covariate_mode"].dropna().unique().tolist())
        covariate_modes = sorted(seen) if seen else ["with_cov", "without_cov"]

    pieces: List[pd.DataFrame] = []
    for lab in labels:
        for cm in covariate_modes:
            df = build_consensus_table(tier2_phyloglm, tier2_pagels, pglmm_mv,
                                        outcome_label=lab, covariate_mode=cm)
            if not df.empty:
                pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def build_covariate_impact(tier2_phyloglm: pd.DataFrame) -> pd.DataFrame:
    """Compare phyloglm with vs. without covariates per
    (defense_system, outcome_label). Reports the coefficient delta, the
    q-value delta, and verdict tags:
        "stable"              — significant in both, same direction
        "emerges_under_cov"   — significant only with covariates (Simpson-style)
        "attenuated_by_cov"   — significant without, not with (genome-capacity confounding)
        "direction_reversed"  — opposite signs between the two
        "ns_both"             — not significant in either

    A positive delta |coef_nocov| - |coef_cov| means covariate adjustment
    shrank the effect (genome-capacity confounding suspected). Output is
    long-form with one row per (defense_system, outcome_label) where both
    modes were fit.
    """
    if tier2_phyloglm is None or tier2_phyloglm.empty:
        return pd.DataFrame()
    df = tier2_phyloglm
    if "direction" in df.columns:
        df = df[df["direction"] == "plasmid_given_defense"]
    if "covariate_mode" not in df.columns:
        return pd.DataFrame()

    with_cov = df[df["covariate_mode"] == "with_cov"][[
        "defense_system", "outcome_label", "phyloglm_coefficient",
        "phyloglm_p_value", "phyloglm_fdr_qvalue"]].rename(columns={
            "phyloglm_coefficient": "coef_with_cov",
            "phyloglm_p_value": "p_with_cov",
            "phyloglm_fdr_qvalue": "q_with_cov"})
    wo_cov = df[df["covariate_mode"] == "without_cov"][[
        "defense_system", "outcome_label", "phyloglm_coefficient",
        "phyloglm_p_value", "phyloglm_fdr_qvalue"]].rename(columns={
            "phyloglm_coefficient": "coef_without_cov",
            "phyloglm_p_value": "p_without_cov",
            "phyloglm_fdr_qvalue": "q_without_cov"})
    merged = with_cov.merge(wo_cov, on=["defense_system", "outcome_label"],
                             how="outer")
    if merged.empty:
        return merged

    merged["coef_delta"] = merged["coef_with_cov"] - merged["coef_without_cov"]
    merged["abs_coef_shrinkage"] = (merged["coef_without_cov"].abs()
                                    - merged["coef_with_cov"].abs())
    merged["neglog10p_delta"] = (-np.log10(merged["p_with_cov"])
                                  - (-np.log10(merged["p_without_cov"])))

    def _verdict(row, alpha: float = 0.05):
        qw, qn = row.get("q_with_cov"), row.get("q_without_cov")
        cw, cn = row.get("coef_with_cov"), row.get("coef_without_cov")
        sig_w = np.isfinite(qw) and qw < alpha
        sig_n = np.isfinite(qn) and qn < alpha
        if sig_w and sig_n:
            if np.isfinite(cw) and np.isfinite(cn) and np.sign(cw) != np.sign(cn):
                return "direction_reversed"
            return "stable"
        if sig_w and not sig_n:
            return "emerges_under_cov"
        if sig_n and not sig_w:
            return "attenuated_by_cov"
        return "ns_both"
    merged["verdict"] = merged.apply(_verdict, axis=1)

    return merged.sort_values(["outcome_label", "verdict", "p_with_cov"])

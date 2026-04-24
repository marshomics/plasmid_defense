"""Combined results table and human-readable summary report.

The combined table merges every per-system output into a single TSV keyed on
defense_system. The summary text file names the top findings by rank product
and flags systems where phyloglm significance disappears at reasonable
plasmid-misclassification rates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .taxonomy import classify_defense_system


def _filter_primary_any_plasmid(df: pd.DataFrame,
                                covariate_mode: str = "with_cov") -> pd.DataFrame:
    """Restrict a long-form result table to the legacy any_plasmid outcome,
    primary direction (plasmid_given_defense), binary outcome mode,
    with_cov covariate mode. Tables without the long-form columns are
    passed through unchanged.
    """
    if df is None or df.empty:
        return df
    if "outcome_label" in df.columns:
        df = df[df["outcome_label"] == "any_plasmid"]
    if "direction" in df.columns:
        df = df[df["direction"] == "plasmid_given_defense"]
    if "outcome_mode" in df.columns:
        df = df[df["outcome_mode"] == "binary"]
    if "covariate_mode" in df.columns:
        # Pagel's tags as "none"; let either the explicit mode OR "none"
        # through so we don't drop it.
        df = df[df["covariate_mode"].isin([covariate_mode, "none"])]
    return df


def build_combined_results(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge every tier's per-system result table into a single DataFrame
    keyed on defense_system, restricted to the legacy any_plasmid outcome
    (primary direction, binary mode). Stratified-outcome results travel in
    separate tier outputs; ``build_per_outcome_summary`` gives the cross-
    stratum view.
    """
    order = [
        "tier1", "tier2_phyloglm", "tier2_pagels", "tier2_pglmm_mv",
        "tier3_loco_summary", "tier3_perm", "tier3_prevalence_matched",
        "misclass_summary", "misclass_analytical_summary",
        "lasso", "elastic_net", "rf_binary", "rf_prevalence",
        "phylo_vs_nonphylo", "consensus",
    ]
    merged = None
    for name in order:
        df = outputs.get(name)
        if df is None or df.empty:
            continue
        df = _filter_primary_any_plasmid(df)
        if df.empty:
            continue
        if name == "tier2_pglmm_mv":
            df = df[df["defense_system"] != "(Intercept)"].copy()
            df = df[~df["defense_system"].astype(str).str.contains(":", regex=False)]
        if "defense_system" not in df.columns:
            continue
        if name == "misclass_summary":
            df = _collapse_misclass_summary(df)
        if name == "misclass_analytical_summary":
            df = _collapse_misclass_analytical(df)
        # Drop duplicate per-system rows to keep the merge key unique.
        df = df.drop_duplicates(subset=["defense_system"], keep="first")
        if merged is None:
            merged = df.copy()
        else:
            suffix = "__" + name
            merged = merged.merge(df, on="defense_system", how="outer",
                                  suffixes=("", suffix))
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("rank_product" if "rank_product" in merged.columns
                              else merged.columns[1])


def build_per_outcome_summary(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cross-stratum compact summary: for each
    (defense_system, outcome_label, covariate_mode) from the primary-direction
    phyloglm output, report coefficient, q-value, and concordance with the
    matching PGLMM fit. Pagel's q-value is merged on outcome_label only
    (Pagel's has no covariate_mode).
    """
    phyloglm = outputs.get("tier2_phyloglm")
    if phyloglm is None or phyloglm.empty:
        return pd.DataFrame()
    uni = phyloglm
    if "direction" in uni.columns:
        uni = uni[uni["direction"] == "plasmid_given_defense"]
    keep_uni = ["defense_system", "outcome_label", "phyloglm_coefficient",
                "phyloglm_p_value", "phyloglm_fdr_qvalue"]
    if "covariate_mode" in uni.columns:
        keep_uni.append("covariate_mode")
    uni = uni[keep_uni]

    mv = outputs.get("tier2_pglmm_mv", pd.DataFrame())
    if not mv.empty:
        if "outcome_mode" in mv.columns:
            mv = mv[mv["outcome_mode"] == "binary"]
        mv = mv[mv["defense_system"] != "(Intercept)"]
        mv = mv[~mv["defense_system"].astype(str).str.contains(":", regex=False)]
        keep = ["defense_system", "outcome_label", "pglmm_coefficient",
                "pglmm_p_value", "pglmm_fdr_qvalue"]
        merge_on = ["defense_system", "outcome_label"]
        if "covariate_mode" in mv.columns:
            keep.append("covariate_mode")
            merge_on.append("covariate_mode")
        if all(c in mv.columns for c in keep):
            mv = mv[keep]
            uni = uni.merge(mv, on=merge_on, how="left")

    pag = outputs.get("tier2_pagels", pd.DataFrame())
    if not pag.empty and "outcome_label" in pag.columns:
        pag = pag[["defense_system", "outcome_label", "pagel_p_value",
                   "pagel_fdr_qvalue"]]
        uni = uni.merge(pag, on=["defense_system", "outcome_label"], how="left")

    sort_cols = ["outcome_label"]
    if "covariate_mode" in uni.columns:
        sort_cols.append("covariate_mode")
    sort_cols.append("phyloglm_p_value")
    return uni.sort_values(sort_cols).reset_index(drop=True)


def build_phylo_vs_nonphylo_comparison(tier1: pd.DataFrame,
                                       tier2_phyloglm: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison of the primary Tier 1 (non-phylogenetic) and
    Tier 2 (phyloglm) tests, flagging:
        - sign disagreement between Tier 1 coefficient and phyloglm coefficient
        - attenuation: |Tier 1 coef| >> |phyloglm coef| (phylogeny was
          absorbing most of the association)
        - inflation: opposite (rare but possible for Simpson-like reversals)
    Reviewers routinely ask "how much did phylogenetic correction change the
    picture"; this answers that directly.
    """
    if tier1 is None or tier1.empty or tier2_phyloglm is None or tier2_phyloglm.empty:
        return pd.DataFrame()

    # Both tables are now long-form with outcome_label (and tier2 has direction
    # too). Restrict to the legacy any_plasmid outcome, primary direction, for
    # the phylo-vs-nonphylo comparison — running it per-stratum explodes the
    # output without adding interpretive value.
    t1 = tier1
    if "outcome_label" in t1.columns:
        t1 = t1[t1["outcome_label"] == "any_plasmid"]
    t2 = tier2_phyloglm
    if "outcome_label" in t2.columns:
        t2 = t2[t2["outcome_label"] == "any_plasmid"]
    if "direction" in t2.columns:
        t2 = t2[t2["direction"] == "plasmid_given_defense"]
    if t1.empty or t2.empty:
        return pd.DataFrame()

    t1 = t1[["defense_system", "firth_weighted_coefficient",
             "firth_weighted_p_value", "firth_weighted_fdr_qvalue",
             "diag_fisher_odds_ratio"]].drop_duplicates("defense_system").copy()
    t2 = t2[["defense_system", "phyloglm_coefficient",
             "phyloglm_p_value", "phyloglm_fdr_qvalue"]] \
             .drop_duplicates("defense_system").copy()
    merged = t1.merge(t2, on="defense_system", how="outer")

    merged["category"] = merged["defense_system"].map(classify_defense_system)

    # Sign and magnitude comparisons
    def _sign_agree(row):
        a, b = row["firth_weighted_coefficient"], row["phyloglm_coefficient"]
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.nan
        return int(np.sign(a) == np.sign(b))
    merged["sign_agreement"] = merged.apply(_sign_agree, axis=1)

    def _attenuation_ratio(row):
        a, b = abs(row["firth_weighted_coefficient"]), abs(row["phyloglm_coefficient"])
        if not (np.isfinite(a) and np.isfinite(b)) or a == 0:
            return np.nan
        return b / a     # < 1 -> phyloglm shrank the effect
    merged["phylo_attenuation_ratio"] = merged.apply(_attenuation_ratio, axis=1)

    def _log_p_drop(row):
        a, b = row["firth_weighted_p_value"], row["phyloglm_p_value"]
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
            return np.nan
        return -np.log10(a) - (-np.log10(b))   # positive = phyloglm is less sig
    merged["neglog10p_drop"] = merged.apply(_log_p_drop, axis=1)

    def _classify(row):
        q1 = row.get("firth_weighted_fdr_qvalue", np.nan)
        q2 = row.get("phyloglm_fdr_qvalue", np.nan)
        sig_t1 = np.isfinite(q1) and q1 < 0.05
        sig_t2 = np.isfinite(q2) and q2 < 0.05
        if sig_t1 and sig_t2:
            if row.get("sign_agreement") == 1:
                return "robust"                # same direction, both significant
            return "direction_reversed"
        if sig_t1 and not sig_t2:
            return "phylo_explained"           # absorbed by phylogeny
        if not sig_t1 and sig_t2:
            return "emerged_under_phylo"       # rarer: real once phylogeny accounted
        return "not_significant_either"
    merged["verdict"] = merged.apply(_classify, axis=1)

    return merged.sort_values("phyloglm_p_value").reset_index(drop=True)


def _collapse_misclass_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long (system, fnr) MC summary into a compact per-system row:
    the fnr at which frac_fdr_sig drops below 50% and at 0.
    """
    records = []
    for system, g in df.groupby("defense_system"):
        g = g.sort_values("fnr")
        below_half = g[g["frac_fdr_sig"] < 0.5]
        fnr_half = below_half["fnr"].iloc[0] if len(below_half) else None
        none_sig = g[g["frac_fdr_sig"] == 0]
        fnr_zero = none_sig["fnr"].iloc[0] if len(none_sig) else None
        records.append({"defense_system": system,
                        "misclass_fnr_below_50pct_sig": fnr_half,
                        "misclass_fnr_to_zero_sig": fnr_zero})
    return pd.DataFrame(records)


def _collapse_misclass_analytical(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[["defense_system", "tipping_point_fnr"]].drop_duplicates()
    return keep.rename(columns={"tipping_point_fnr": "misclass_analytical_tipping_point_fnr"})


def write_summary_report(outputs: Dict[str, pd.DataFrame], output_dir: Path,
                         alpha: float = 0.05) -> Path:
    """Human-readable summary citing only phylogenetic results as primary.
    Reports per-outcome top findings.
    """
    lines = []
    add = lines.append

    add("Defense-Plasmid Association Analysis — summary")
    add("=" * 60)
    add("")
    add(f"Per-tier FDR threshold: q < {alpha}. Non-phylogenetic Tier 1 results")
    add("are diagnostic only and are not cited below. Covariates: genome size,")
    add("GC content, CDS count, log(n_strains) — the species-level sampling")
    add("depth covariate that partials out max()-saturation of binary defense")
    add("features — and log(n_plasmids) for stratified outcomes.")
    add("")
    add("Primary outcome per stratum:")
    add("  * any_plasmid — binary has_plasmid_binary.")
    add("  * Stratified classes (mobility, size, reptype) — binomial (k, n-k)")
    add("    fit via PGLMM, where k = plasmids-of-class-X per species and")
    add("    n = total plasmids for that species. The binary any_plasmid_<X>")
    add("    fit travels alongside as a backward-compatible secondary.")
    add("")

    phyloglm = outputs.get("tier2_phyloglm")
    if phyloglm is not None and not phyloglm.empty:
        primary = phyloglm
        if "direction" in primary.columns:
            primary = primary[primary["direction"] == "plasmid_given_defense"]
        outcome_labels = (primary["outcome_label"].dropna().unique().tolist()
                          if "outcome_label" in primary.columns
                          else ["any_plasmid"])
        cov_modes = (primary["covariate_mode"].dropna().unique().tolist()
                     if "covariate_mode" in primary.columns else ["with_cov"])
        add("Phylogenetic logistic regression (primary univariate):")
        for cov_mode in sorted(cov_modes):
            add(f"  --- covariate_mode = {cov_mode} ---")
            for lab in sorted(outcome_labels):
                sub = primary
                if "outcome_label" in sub.columns:
                    sub = sub[sub["outcome_label"] == lab]
                if "covariate_mode" in sub.columns:
                    sub = sub[sub["covariate_mode"] == cov_mode]
                sig = sub[sub["phyloglm_fdr_qvalue"] < alpha]
                add(f"    [{lab}] — {len(sig)} defense systems at FDR q < {alpha}")
                for _, r in sig.head(10).iterrows():
                    add(f"      {r['defense_system']:40s}  "
                        f"coef={r['phyloglm_coefficient']:+.3f}  "
                        f"OR={r.get('phyloglm_odds_ratio', float('nan')):.2f}  "
                        f"q={r['phyloglm_fdr_qvalue']:.3g}")
        add("")

        # Covariate impact rollup (if both modes present)
        impact = outputs.get("covariate_impact")
        if impact is not None and not impact.empty:
            counts = impact["verdict"].value_counts().to_dict()
            add("Covariate impact (phyloglm with_cov vs without_cov, primary direction):")
            for verdict in ["stable", "emerges_under_cov", "attenuated_by_cov",
                            "direction_reversed", "ns_both"]:
                add(f"  {verdict:24s} {counts.get(verdict, 0)}")
            add("")

        reverse = phyloglm
        if "direction" in reverse.columns:
            reverse = reverse[reverse["direction"] == "defense_given_plasmid"]
            if not reverse.empty:
                add("Reverse direction (defense_i ~ plasmid-class):")
                for lab in sorted(outcome_labels):
                    sub = reverse[reverse["outcome_label"] == lab] \
                        if "outcome_label" in reverse.columns else reverse
                    sig = sub[sub["phyloglm_fdr_qvalue"] < alpha]
                    add(f"  [{lab}] — {len(sig)} defense systems at FDR q < {alpha}")
                    for _, r in sig.head(10).iterrows():
                        add(f"    {r['defense_system']:40s}  "
                            f"coef={r['phyloglm_coefficient']:+.3f}  "
                            f"q={r['phyloglm_fdr_qvalue']:.3g}")
                add("")

    pglmm = outputs.get("tier2_pglmm_mv")
    if pglmm is not None and not pglmm.empty:
        mv = pglmm[pglmm["defense_system"] != "(Intercept)"]
        mv = mv[~mv["defense_system"].astype(str).str.contains(":", regex=False)]
        add("Multivariate PGLMM (defense systems + phylogeny, per covariate_mode):")
        if "outcome_label" in mv.columns:
            cov_modes_mv = (mv["covariate_mode"].dropna().unique().tolist()
                            if "covariate_mode" in mv.columns else ["with_cov"])
            for cov_mode in sorted(cov_modes_mv):
                add(f"  --- covariate_mode = {cov_mode} ---")
                sub_cm = mv
                if "covariate_mode" in mv.columns:
                    sub_cm = mv[mv["covariate_mode"] == cov_mode]
                for lab in sorted(sub_cm["outcome_label"].dropna().unique().tolist()):
                    for mode in sorted(sub_cm.loc[sub_cm["outcome_label"] == lab,
                                                  "outcome_mode"]
                                       .dropna().unique().tolist()
                                       if "outcome_mode" in sub_cm.columns
                                       else ["binary"]):
                        sub = sub_cm[(sub_cm["outcome_label"] == lab)
                                     & (sub_cm.get("outcome_mode", "binary") == mode)]
                        sig = sub[sub["pglmm_fdr_qvalue"] < alpha]
                        add(f"    [{lab} / {mode}] — {len(sig)} defense systems at "
                            f"FDR q < {alpha}")
                        for _, r in sig.head(10).iterrows():
                            add(f"      {r['defense_system']:40s}  "
                                f"coef={r['pglmm_coefficient']:+.3f}  "
                                f"q={r['pglmm_fdr_qvalue']:.3g}")
        else:
            sig = mv[mv["pglmm_fdr_qvalue"] < alpha]
            add(f"  {len(sig)} defense systems at FDR q < {alpha}")
        add("")

        # Interaction terms, if present
        inters = pglmm[pglmm["defense_system"].astype(str).str.contains(":", regex=False)]
        if not inters.empty:
            sig_i = inters[inters["pglmm_fdr_qvalue"] < alpha] if \
                "pglmm_fdr_qvalue" in inters.columns else pd.DataFrame()
            add(f"Pairwise defense x defense interactions (PGLMM):")
            add(f"  {len(sig_i)} interaction terms at FDR q < {alpha} (total tested: {len(inters)})")
            for _, r in sig_i.head(10).iterrows():
                add(f"    {r['defense_system']:60s}  coef={r['pglmm_coefficient']:+.3f}  "
                    f"q={r['pglmm_fdr_qvalue']:.3g}  "
                    f"[{r.get('outcome_label', 'any_plasmid')}]")
            add("")

    loco = outputs.get("tier3_loco_summary")
    if loco is not None and not loco.empty:
        col = "gtdb_class_is_heterogeneous" if "gtdb_class_is_heterogeneous" in loco.columns \
            else "gtdb_phylum_is_heterogeneous"
        if col in loco.columns:
            n_het = int(loco[col].sum())
            add(f"Leave-one-clade-out heterogeneity (Cochran Q, Bonferroni-adjusted):")
            add(f"  {n_het} systems flagged clade-sensitive in primary rank")
            add("")

    misclass = outputs.get("misclass_summary")
    if misclass is not None and not misclass.empty:
        collapsed = _collapse_misclass_summary(misclass)
        stable = collapsed[collapsed["misclass_fnr_to_zero_sig"].isna()]
        add(f"Plasmid misclassification sensitivity (Monte Carlo):")
        add(f"  {len(stable)} systems remain significant across the full "
            f"FNR grid ({list(outputs['misclass_summary']['fnr'].unique())})")
        add("")

    # Sampling-depth / feature-mode / phylo-model sensitivity reruns
    mnss = outputs.get("tier3_min_n_strains_sens")
    if mnss is not None and not mnss.empty:
        add("Minimum-n_strains sensitivity (primary phyloglm, species with >= "
            f"{int(mnss['min_n_strains_threshold'].iloc[0])} strains only):")
        for cm, sub in mnss.groupby("covariate_mode"):
            sig = sub[sub["phyloglm_fdr_qvalue"] < alpha]
            add(f"  [{cm}] — {len(sig)} defense systems at FDR q < {alpha} "
                f"after filtering to {int(sub['n_species_filtered_in'].iloc[0])} species")
        add("")

    prev_sens = outputs.get("tier3_prev_feature_sens")
    if prev_sens is not None and not prev_sens.empty:
        add("Prevalence-feature sensitivity (defense feature = mean-across-strains):")
        for cm, sub in prev_sens.groupby("covariate_mode"):
            sig = sub[sub["phyloglm_fdr_qvalue"] < alpha]
            add(f"  [{cm}] — {len(sig)} defense systems at FDR q < {alpha}")
        add("")

    model_sens = outputs.get("tier3_phylo_model_sens")
    if model_sens is not None and not model_sens.empty:
        add("Phylogenetic-model sensitivity (primary phyloglm refit under "
            "alternative evolutionary models):")
        for model, sub in model_sens.groupby("evolutionary_model"):
            for cm, sub2 in sub.groupby("covariate_mode"):
                sig = sub2[sub2["phyloglm_fdr_qvalue"] < alpha]
                add(f"  [model={model} / {cm}] — {len(sig)} defense systems "
                    f"at FDR q < {alpha}")
        add("")

    consensus = outputs.get("consensus")
    if consensus is not None and not consensus.empty:
        add("Consensus (rank product across phyloglm + PGLMM + Pagel's):")
        if "outcome_label" in consensus.columns:
            cov_modes_con = (consensus["covariate_mode"].dropna().unique().tolist()
                              if "covariate_mode" in consensus.columns
                              else ["with_cov"])
            for cov_mode in sorted(cov_modes_con):
                add(f"  --- covariate_mode = {cov_mode} ---")
                sub_cm = consensus
                if "covariate_mode" in consensus.columns:
                    sub_cm = consensus[consensus["covariate_mode"] == cov_mode]
                for lab in sorted(sub_cm["outcome_label"].dropna().unique().tolist()):
                    sub = sub_cm[sub_cm["outcome_label"] == lab]
                    add(f"    [{lab}] top 10:")
                    for _, r in sub.head(10).iterrows():
                        add(f"      {r['defense_system']:40s}  "
                            f"rank_product={r['rank_product']:.1f}  "
                            f"cauchy_p={r['cauchy_combined_p']:.3g}")
        else:
            for _, r in consensus.head(15).iterrows():
                add(f"    {r['defense_system']:40s}  "
                    f"rank_product={r['rank_product']:.1f}  "
                    f"cauchy_p={r['cauchy_combined_p']:.3g}")
        add("")

    out_path = output_dir / "summary_report.txt"
    out_path.write_text("\n".join(lines))
    return out_path


def save_all(outputs: Dict[str, pd.DataFrame], output_dir: Path,
             prefix: str = "") -> Dict[str, Path]:
    """Persist every DataFrame in ``outputs`` as a TSV. Returns map name -> path."""
    paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        if df is None or df.empty:
            continue
        path = output_dir / f"{prefix}{name}.tsv"
        df.to_csv(path, sep="\t", index=False)
        paths[name] = path
    return paths

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
                                covariate_mode: str = "full") -> pd.DataFrame:
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

    # ---- Negative control first. Nothing below it means anything if this
    # ---- section says the model is not calibrated.
    nc = outputs.get("negative_control")
    add("NEGATIVE CONTROL (read this first)")
    add("-" * 60)
    if nc is None or nc.empty:
        add("  NOT RUN. The plasmid label was never permuted, so the pipeline's")
        add("  calibration against the sampling-depth confound is unverified.")
        add("  Run with --stages negative_control before believing any result.")
    else:
        calibrated = bool(nc["calibrated"].iloc[0]) if "calibrated" in nc else False
        mean_hits = float(nc["mean_fdr_significant"].iloc[0]) \
            if "mean_fdr_significant" in nc else float("nan")
        thresh = float(nc["calibration_threshold"].iloc[0]) \
            if "calibration_threshold" in nc else float("nan")
        add(f"  Replicates: {len(nc)}")
        add(f"  Mean FDR-significant systems on permuted labels: {mean_hits:.1f}")
        add(f"  Calibration threshold: {thresh:.1f}")
        if calibrated:
            add("  VERDICT: CALIBRATED — permuting the plasmid label within")
            add("  (clade x sequencing-depth) strata destroys the signal, as it")
            add("  should. Associations below are not explained by clade")
            add("  membership or sequencing effort alone.")
        else:
            add("  VERDICT: *** NOT CALIBRATED ***")
            add("  The model finds associations even when the plasmid label has")
            add("  been permuted within clade and depth strata. The most likely")
            add("  cause is residual sampling-depth confounding. DO NOT REPORT")
            add("  the associations below as biological findings until this")
            add("  passes. Raise config.depth_spline_df and re-run.")
    add("")
    add(f"FDR threshold: q < {alpha}, applied per stratum AND globally across")
    add("all primary tests (*_global_qvalue). Non-phylogenetic Tier 1 results")
    add("are diagnostic only and are not cited below.")
    add("")
    add("Covariates: log(genome size), GC content, log(CDS count), a natural-")
    add("spline basis on log(n_strains) — the sampling-depth adjustment — and a")
    add("spline on log(n_plasmids) for stratified outcomes.")
    add("")
    add("Sampling-depth note. The species plasmid label is propagated ('any")
    add("strain carries a plasmid') and the species defense call is max across")
    add("strains, so both saturate as 1-(1-p)^n_strains and sequencing depth is")
    add("a common cause of predictor and outcome. Unadjusted, this yields OR")
    add("~2.5 with a 100% false-positive rate under a strict null. The")
    add("'unadjusted' covariate mode is retained ONLY as a positive control for")
    add("that confound and is excluded from consensus and from every primary")
    add("claim. Read the negative-control section before anything else.")
    add("")
    add("Primary outcome per stratum:")
    add("  * any_plasmid — binary has_plasmid_binary.")
    add("  * Stratified classes (mobility, size, reptype) — binary")
    add("    any_plasmid_<X> with a log(n_plasmids) spline. Binary is primary")
    add("    because phyloglm and Pagel's have no binomial mode, so consensus")
    add("    must combine comparable estimands. The binomial (k, n-k) PGLMM fit")
    add("    runs alongside as a concordance check.")
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
                     if "covariate_mode" in primary.columns else ["full"])
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
            add("Covariate impact (phyloglm full vs unadjusted, primary direction).")
            add("NOTE: 'attenuated' here largely reflects removal of the")
            add("sampling-depth confound, not genome-capacity confounding.")
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
                            if "covariate_mode" in mv.columns else ["full"])
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
                              else ["full"])
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

    # ------------------------------------------------------------------
    # A4 — pre-registered entry-mode prediction
    # ------------------------------------------------------------------
    conf = outputs.get("entry_mode_confirmatory")
    add("PRE-REGISTERED ENTRY-MODE (ssDNA) PREDICTION")
    add("-" * 60)
    add("  Conjugative plasmids enter as single-stranded DNA; restriction-like")
    add("  systems cleave double-stranded DNA. Prediction, fixed in advance:")
    add("  dsDNA-restricting systems deplete NON-conjugative plasmids more than")
    add("  abortive-infection / signalling systems do. One confirmatory test,")
    add("  one degree of freedom.")
    if conf is None or conf.empty:
        add("  NOT RUN.")
    else:
        r = conf.iloc[0]
        add(f"  dsDNA-restricting systems (n={int(r['n_predicted_systems'])}): "
            f"weighted mean coefficient {r['weighted_mean_coef_predicted']:+.3f}")
        add(f"  abortive/signalling     (n={int(r['n_not_predicted_systems'])}): "
            f"weighted mean coefficient {r['weighted_mean_coef_not_predicted']:+.3f}")
        add(f"  difference = {r['observed_difference']:+.3f}, "
            f"one-sided permutation p = {r['p_one_sided_preregistered']:.4g} "
            f"({int(r['n_permutations']):,} permutations of group labels)")
        if bool(r["prediction_supported"]):
            add("  VERDICT: PREDICTION SUPPORTED. The contrast behaves as the")
            add("  ssDNA-evasion mechanism requires, which is evidence that these")
            add("  systems act at plasmid entry rather than merely co-occurring.")
        else:
            add("  VERDICT: not supported at alpha. An informative negative — it")
            add("  argues against entry mode being the axis on which these")
            add("  systems discriminate between plasmids.")
    add("")

    # ------------------------------------------------------------------
    # B1 — matched sister pairs
    # ------------------------------------------------------------------
    sp = outputs.get("sister_pair_summary")
    add("PHYLOGENETICALLY MATCHED SISTER PAIRS")
    add("-" * 60)
    add("  Within-pair contrast among sister species matched on sequencing")
    add("  depth. Controls phylogeny and depth BY CONSTRUCTION rather than by")
    add("  model, so a system significant here is the most defensible result")
    add("  the species-level data can produce.")
    if sp is None or sp.empty:
        add("  NOT RUN.")
    else:
        testable = sp[sp["sister_p_value"].notna()]
        sig = testable[testable["sister_fdr_qvalue"] < alpha]
        add(f"  {len(testable)}/{len(sp)} systems had enough depth-matched "
            f"discordant pairs; {len(sig)} significant at FDR q < {alpha}")
        for _, r in sig.head(15).iterrows():
            add(f"    {str(r['defense_system'])[:40]:40s}  "
                f"OR={r.get('sister_odds_ratio', float('nan')):.2f}  "
                f"pairs={int(r.get('n_discordant_pairs', 0))}  "
                f"q={r['sister_fdr_qvalue']:.3g}")
        cmp_ = outputs.get("sister_vs_primary")
        if cmp_ is not None and not cmp_.empty and "sister_verdict" in cmp_:
            add("  Agreement with the primary regression:")
            for k, v in cmp_["sister_verdict"].value_counts().items():
                add(f"    {str(k):48s} {v}")
    add("")

    # ------------------------------------------------------------------
    # B2 — directionality
    # ------------------------------------------------------------------
    pag = outputs.get("tier2_pagels")
    if pag is not None and not pag.empty and "pagel_direction" in pag.columns:
        add("EVOLUTIONARY DIRECTIONALITY (Pagel dependent-transition models)")
        add("-" * 60)
        add("  Which character's transition rates are conditioned on the")
        add("  other's state, by AIC over nested Mk models. This is the only")
        add("  analysis that speaks to ordering rather than association.")
        counts = pag["pagel_direction"].value_counts()
        for k, v in counts.items():
            add(f"    {str(k):32s} {v}")
        drives = pag[pag["pagel_direction"] == "defense_drives_plasmid"]
        if not drives.empty:
            top = drives.reindex(
                drives["pagel_direction_delta_aic"].sort_values(
                    ascending=False).index).head(10)
            add("  Strongest 'defense state drives plasmid gain/loss':")
            for _, r in top.iterrows():
                add(f"    {str(r['defense_system'])[:40]:40s}  "
                    f"dAIC={r['pagel_direction_delta_aic']:+.1f}")
        add("")

    # ------------------------------------------------------------------
    # B3 — matched-feature negative control
    # ------------------------------------------------------------------
    fc = outputs.get("feature_control_comparison")
    fcr = outputs.get("feature_control_results")
    add("MATCHED-FEATURE NEGATIVE CONTROL")
    add("-" * 60)
    add("  Would an arbitrary trait with the same prevalence and phylogenetic")
    add("  clustering show the same association? Calibrates the effect scale.")
    if fcr is None or fcr.empty:
        add("  NOT RUN.")
    else:
        n_fit = int(fcr["phyloglm_p_value"].notna().sum())
        n_sig = int((fcr["phyloglm_fdr_qvalue"] < alpha).sum())
        add(f"  {n_sig}/{n_fit} biologically arbitrary control features reach "
            f"FDR q < {alpha} "
            f"({100 * n_sig / max(n_fit, 1):.1f}%)")
        if n_fit and n_sig / max(n_fit, 1) > 0.10:
            add("  WARNING: a large fraction of arbitrary traits reach")
            add("  significance. Effect sizes should be read against this null,")
            add("  not against zero.")
        if fc is not None and not fc.empty:
            n_ex = int(fc["exceeds_matched_null"].sum())
            add(f"  {n_ex}/{len(fc)} defense systems exceed their matched null")
            top = fc[fc["exceeds_matched_null"]].nlargest(
                10, "control_percentile")
            for _, r in top.iterrows():
                add(f"    {str(r['defense_system'])[:40]:40s}  "
                    f"percentile={r['control_percentile']:.1f}  "
                    f"q={r['control_empirical_fdr_qvalue']:.3g}")
    add("")

    # ------------------------------------------------------------------
    # B4 — E-values
    # ------------------------------------------------------------------
    ph = outputs.get("tier2_phyloglm")
    if ph is not None and not ph.empty and "evalue_point" in ph.columns:
        add("SENSITIVITY TO UNMEASURED CONFOUNDING (E-values)")
        add("-" * 60)
        add("  The E-value is the minimum association strength, on the risk-")
        add("  ratio scale, that an unmeasured confounder would need with BOTH")
        add("  the defense system and plasmid carriage to explain the effect")
        add("  away. The OR-to-RR conversion is chosen from the OBSERVED")
        add("  outcome prevalence and recorded in evalue_conversion.")
        sub = ph
        if "direction" in sub.columns:
            sub = sub[sub["direction"] == "plasmid_given_defense"]
        if "outcome_label" in sub.columns:
            sub = sub[sub["outcome_label"] == "any_plasmid"]
        sig = sub[sub.get("phyloglm_fdr_qvalue", pd.Series(dtype=float)) < alpha]
        sig = sig.dropna(subset=["evalue_point"])
        if sig.empty:
            add("  No FDR-significant primary associations to evaluate.")
        else:
            add(f"  Median E-value across {len(sig)} significant associations: "
                f"{sig['evalue_point'].median():.2f} "
                f"(CI-limit E-value {sig['evalue_ci'].median():.2f})")
            add("  Most robust associations:")
            for _, r in sig.nlargest(10, "evalue_point").iterrows():
                add(f"    {str(r['defense_system'])[:40]:40s}  "
                    f"OR={r.get('phyloglm_odds_ratio', float('nan')):.2f}  "
                    f"E={r['evalue_point']:.2f}  E(CI)={r['evalue_ci']:.2f}")
            weak = sig[sig["evalue_ci"] < 1.25]
            if not weak.empty:
                add(f"  {len(weak)} associations have a CI E-value below 1.25 —")
                add("  a very weak unmeasured confounder would suffice to")
                add("  explain those away.")
        add("")

    out_path = output_dir / "summary_report.txt"
    out_path.write_text("\n".join(lines))
    return out_path


def attach_global_fdr(tier2_phyloglm: pd.DataFrame, config) -> pd.DataFrame:
    """Add ``phyloglm_p_value_global_qvalue`` across all PRIMARY tests.

    ``config.report_global_fdr`` has always been True and
    ``stats_utils.apply_global_fdr`` has always existed, but nothing called it,
    so the documented "single global FDR across all primary tests" did not
    exist and correction was per-stratum only. Per-stratum BH controls the
    error rate within a stratum, which does not cover a narrative that
    highlights whatever reached significance somewhere across ~435 systems x
    |strata| x 2 directions x |covariate modes|.

    The family is restricted to ``config.primary_outcome_labels`` at
    ``config.primary_covariate_mode``, so exploratory replicon strata and the
    unadjusted positive control cannot dilute it.
    """
    from .stats_utils import apply_global_fdr

    if tier2_phyloglm is None or tier2_phyloglm.empty:
        return tier2_phyloglm
    if not getattr(config, "report_global_fdr", False):
        return tier2_phyloglm

    df = tier2_phyloglm.copy()
    labels = df.get("outcome_label", pd.Series("any_plasmid", index=df.index))
    modes = df.get("covariate_mode",
                   pd.Series(config.primary_covariate_mode, index=df.index))
    mask = pd.Series(
        [config.is_primary_slice(str(l), str(m)) for l, m in zip(labels, modes)],
        index=df.index)
    # Both directions belong in one family: a claim in either direction is a
    # claim, and reporting them with identical "q < alpha" language while
    # correcting them separately understates the multiplicity.
    return apply_global_fdr(df, ["phyloglm_p_value"],
                            method=config.fdr_method, family_mask=mask)


def build_binomial_concordance(pglmm_mv: pd.DataFrame, config) -> pd.DataFrame:
    """Compare the binary and binomial PGLMM fits per stratified outcome.

    config declares binary primary because it is the only mode all three
    consensus methods can fit. The binomial cbind(k, n-k) fit is the more
    natural framing for "what fraction of this species' plasmids are of class
    X", so it runs alongside — but previously it was written and then filtered
    out by every consumer (consensus, combined results, figures), while the
    config and the summary report both described it as primary.

    Making it a declared concordance check means a stratified claim requires
    the two framings to agree in sign.
    """
    if pglmm_mv is None or pglmm_mv.empty:
        return pd.DataFrame()
    if "outcome_mode" not in pglmm_mv.columns:
        return pd.DataFrame()

    df = pglmm_mv[pglmm_mv["defense_system"] != "(Intercept)"]
    df = df[~df["defense_system"].astype(str).str.contains(":", regex=False)]
    keys = [c for c in ("defense_system", "outcome_label", "covariate_mode")
            if c in df.columns]
    cols = keys + ["pglmm_coefficient", "pglmm_p_value", "pglmm_fdr_qvalue"]
    cols = [c for c in cols if c in df.columns]

    binary = df[df["outcome_mode"] == "binary"][cols]
    binom = df[df["outcome_mode"] == "binomial"][cols]
    if binary.empty or binom.empty:
        return pd.DataFrame()

    m = binary.merge(binom, on=keys, suffixes=("_binary", "_binomial"),
                     how="inner")
    if m.empty:
        return m
    a = config.alpha
    sig_b = m["pglmm_fdr_qvalue_binary"] < a
    sig_n = m["pglmm_fdr_qvalue_binomial"] < a
    same = np.sign(m["pglmm_coefficient_binary"]) == \
        np.sign(m["pglmm_coefficient_binomial"])
    m["concordance"] = np.select(
        [sig_b & sig_n & same, sig_b & sig_n & ~same, sig_b ^ sig_n],
        ["both_significant_same_sign", "both_significant_OPPOSITE_SIGN",
         "one_mode_only"],
        default="ns_both")
    m["reportable_as_primary"] = (
        sig_b & (same | ~sig_n))
    return m


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

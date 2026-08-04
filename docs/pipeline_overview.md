# defense_analysis_v2 — pipeline overview

## Scientific goal

Identify bacterial anti-phage defense systems (type or subtype) whose presence in a microbe makes plasmid carriage more or less likely, and the reverse direction. Each defense system gets a quantified association with plasmid presence, and the strongest claims are the systems showing consistent direction across multiple methods.

Headline question, plain language:
> Does having a particular defense system mean a bacterium is more (or less) likely to harbor a plasmid, or vice versa?

## Data structure

- **Defense systems:** strain-level classification from DefenseFinder on 342,000 microbial assemblies. ~435 distinct defense systems at subtype level; ~80-100 at type level. Strain-level binary calls are aggregated to species level via `max()` across strains (presence = at least one strain in the species carries the system).
- **Plasmids:** 142,000 individual plasmids with metadata (mobility class, size, replicon type). Plasmid carriage was propagated species-wide upstream — if any one strain in a species carries a plasmid, all strains of that species are labeled `has_plasmid=yes`. The species-level invariant is enforced in `io_utils.aggregate_to_species_level`.
- **Tree:** Project-specific GTDB cut at `human_animal_free_90percent_species_level/`. ~39,681 unique tip labels. Some labels carry `[species NNNNN]` annotations that are meaningful identifiers and must not be stripped during normalisation.
- **Covariates:** log(genome size), GC content, log(CDS count), plus a **natural-spline basis** on `log(n_strains)` (sampling-depth adjustment) and on `log(n_plasmids)` for stratified outcomes. A single linear `log(n_strains)` term is insufficient — see the sampling-depth section of the README.

## Two-axis stratification

**Plasmid outcomes:**
- `any_plasmid` (legacy binary, primary direction)
- mobility classes: `conjugative`, `mobilizable`, `non-mobilizable`
- size bins: `size_small`, `size_medium`, `size_large`
- top-N replicon types (`reptype_IncF`, `reptype_IncN`, etc.)

For each stratum both binary (any-of-class) and binomial (k-of-n) framings are run. **Binary is primary** — phyloglm and Pagel's have no binomial mode, so consensus must combine comparable estimands. The binomial PGLMM fit is a declared concordance check (`binomial_concordance.tsv`).

**Directions:**
- `plasmid_given_defense`: plasmid presence as outcome, defense as predictor — "does this defense system predict plasmid carriage?" (the defense-prevalence gate follows the defense system in both directions)
- `defense_given_plasmid`: defense presence as outcome, plasmid as predictor — "does plasmid carriage predict this defense system?"

Both directions are run per stratum and per covariate-adjustment mode (`with_cov` vs `without_cov`).

## Headline structure of evidence

For a defense system to be called a confident hit, the design wants:
1. **Phyloglm univariate** — direction agrees across `plasmid_given_defense` and `defense_given_plasmid`, FDR-significant
2. **Pagel's correlated-evolution** — rejects independent evolution of the two binary traits
3. **PGLMM multivariate** — coefficient holds up after conditioning on the other defense systems and phylogenetic random effect

The consensus stage (`consensus.py`) ranks systems by rank-product across these three methods. Cauchy-combined p-value travels alongside. Phyloglm + Pagel's alone give `consensus_tier = partial`. A `corroborated` claim needs all three.

## Bidirectional and stratified outputs

Even with PGLMM dropped, the pipeline produces:
- Per-outcome FDR-corrected p-values per defense system, both directions
- Effect sizes (odds ratios) with 95% CIs
- Stratified results (by mobility / size / replicon)
- Sensitivity analyses (LOCO, misclassification MC, n_strains threshold, prevalence vs binary feature)

## Output files

Per granularity (`subtype_level` and `type_level`):
- `tier1.tsv` — Firth logistic + diagnostic tests
- `tier2_phyloglm.tsv` — primary univariate phylogenetic results
- `tier2_pagels.tsv` — Pagel's correlated evolution
- `tier2_pglmm_mv.tsv` — multivariate PGLMM (only if it ran)
- `lasso.tsv`, `elastic_net.tsv`, `mv_stability.tsv` — regularized selection
- `rf_binary.tsv`, `rf_prevalence.tsv`, `rf_fold_aucs.tsv` — clade-blocked Random Forest
- `burden_pgls.tsv`, `burden_phyloglm.tsv` — total defense count vs plasmid
- `tier3_loco_detail.tsv`, `tier3_loco_summary.tsv` — leave-one-clade-out **influence diagnostic** (no p-value; LOCO estimates are not independent)
- `tier3_phylo_signal.tsv` — D-statistic phylogenetic signal
- `tier3_perm.tsv` — clade-restricted permutation null
- `tier3_depth_matched.tsv` — depth-matched McNemar test (replaces the vacuous prevalence-matched test)
- `negative_control.tsv` — **calibration check; read first**
- `tier3_depth_sens.tsv`, `tier3_depth_band_concordance.tsv` — high- and low-depth reruns and their concordance
- `tier3_within_clade_summary.tsv` — valid Cochran's Q heterogeneity
- `tier3_prev_feature_sens.tsv`, `tier3_phylo_model_sens.tsv` — robustness reruns
- `misclass_mc_long.tsv`, `misclass_summary.tsv` — plasmid FNR Monte Carlo
- `misclass_analytical.tsv` — Bross 1954 analytical bias correction
- `consensus.tsv`, `covariate_impact.tsv` — rank-product across methods
- `phylo_vs_nonphylo.tsv` — Tier 1 vs Tier 2 comparison
- Aggregates only at end: `combined_all_results.tsv`, `per_outcome_summary.tsv`, `summary_report.txt`, `figures/`

## Codebase layout

- `defense_analysis_v2/config.py` — all knobs as Config dataclass
- `defense_analysis_v2/defense_plasmid_analysis.py` — CLI driver with stage dispatch + checkpointing
- `defense_analysis_v2/io_utils.py` — strain→species aggregation, plasmid stratification
- `defense_analysis_v2/tree_utils.py` — tree loading, deduplication, pruning
- `defense_analysis_v2/r_bridge.py` — subprocess wrapper for R scripts
- `defense_analysis_v2/r_scripts/*.R` — phyloglm_uni.R, pglmm_mv.R, pagels_test.R, etc.
- `defense_analysis_v2/tier1.py` — Firth logistic baseline
- `defense_analysis_v2/tier2_*.py` — phyloglm, pagels, PGLMM/LASSO/EN, random forest
- `defense_analysis_v2/tier3_*.py` — burden, LOCO, misclassification, other sensitivity
- `defense_analysis_v2/consensus.py` — rank-product across methods
- `defense_analysis_v2/reporting.py` — combined tables, summary text
- `defense_analysis_v2/plotting.py` — publication figures

## Reading the consensus output

`consensus.tsv` has one row per (defense_system, outcome_label, covariate_mode). Key columns:
- `rank_product_p_value` / `rank_product_fdr_qvalue` — **calibrated** against a permutation null drawn per number of contributing methods. Use these, not the raw `rank_product` score, which is not comparable across rows with different method counts.
- `cauchy_combined_p` — Cauchy-combined p-value across methods
- `n_methods_contributing` — 2 or 3 depending on whether PGLMM ran for that slice
- `phylo_direction_agreement` — 1 if phyloglm and PGLMM coefficients agree in sign, 0 if disagree

A confident hit: `consensus_tier = corroborated` (>=3 methods, calibrated rank-product q < 0.05, directions agree), significant under the **global** FDR (`phyloglm_p_value_global_qvalue`), `depth_verdict = robust_to_depth` in the depth-band concordance, and a passing negative control.

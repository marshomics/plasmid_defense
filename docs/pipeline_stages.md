# Pipeline stages — what each does and why

Each stage produces one or more TSVs under `<output_dir>/<granularity>/`. Stages run in the order below, checkpointed individually so a job killed mid-pipeline resumes from the last finished stage.

## Tier 1 — diagnostic baseline (NOT primary evidence)

**Stage name:** `tier1` → `tier1.tsv`

Firth-penalised logistic regression with `n_strains` as frequency weight and genome-scale covariates. Runs across every plasmid-outcome stratum, both directions. Plus diagnostic Fisher's exact, Mann-Whitney U, and ordinary weighted logistic.

**Contribution to the question:** Cheap non-phylogenetic sanity check. Identifies the direction of effect before any phylogenetic correction. Reviewers will ask for it but it should NOT be the citation in the headline — it ignores phylogeny entirely and is confounded by clade structure.

## Tier 2 phyloglm — primary univariate phylogenetic test

**Stage name:** `phyloglm` → `tier2_phyloglm.tsv`

`phylolm::phyloglm` fits logistic regression with a Brownian-motion phylogenetic covariance on the full ~40k-tip tree. One fit per (defense_system, outcome_label, direction, covariate_mode). With ~435 systems × 17 outcomes × 2 directions × 2 covariate modes = up to ~30,000 fits. Parallelised at the (outcome × direction × covariate) level via joblib; each task internally iterates 435 systems serially within one R subprocess.

**Contribution:** The strongest single-method evidence the pipeline produces. Each defense system gets an FDR-corrected p-value and coefficient (log odds-ratio) under the phylogenetic null. Both directions are reported.

**Typical wall-clock:** 3-5 days at `--n-jobs 20`, BLAS single-threaded.

## Tier 2 Pagel's — correlated binary-trait evolution

**Stage name:** `pagels` → `tier2_pagels.tsv`

`phytools::fitPagel` fits independent vs correlated continuous-time Markov models for each (defense_system, plasmid_outcome) binary pair. Tests a different null than phyloglm: that the two traits evolve independently on the tree. Subsamples to `pagels_subsample_size = 500` tips × `pagels_n_subsamples = 5` independent draws; aggregates with median p-value per system.

**Contribution:** Orthogonal evidence to phyloglm. A system significant under Pagel's but not phyloglm typically reflects shared-lineage signal that phyloglm's covariate-adjusted fit has controlled for. A system significant under both is the strongest evidence of an evolutionarily-correlated trait pair.

**Why subsampled:** Full-tree fitPagel on 40k tips × 435 systems × 5 subsamples × 17 outcomes = ~75,000 wall-clock hours. Computationally impossible. The 5×500-tip design with median-p aggregation is the standard accommodation.

## Tier 2 PGLMM — multivariate phylogenetic GLMM

**Stage name:** `pglmm_mv` → `tier2_pglmm_mv.tsv` (plus `lasso.tsv`, `elastic_net.tsv`, `mv_stability.tsv`)

`phyr::pglmm` fits a binomial GLMM with all 18 (post-prevalence-gate) defense systems as fixed effects, plus pairwise interactions for the top-8 by phyloglm rank, plus a phylogenetic random effect on the tree. One fit per (covariate_mode × outcome × outcome_mode).

**Contribution:** Tells you which defense systems independently predict plasmid carriage after conditioning on the other defense systems and shared ancestry. If phyloglm says system X is associated but PGLMM says no (or much weaker), then X's signal was confounded with another system in the multivariate sense.

**Memory and threading constraints:** This is the hardest stage to run. See `pglmm_step_recommendations.md`.

## Tier 2 RF — clade-blocked Random Forest

**Stage name:** `rf` → `rf_binary.tsv`, `rf_prevalence.tsv`, `rf_fold_aucs.tsv`

Random forest classifier with LeaveOneGroupOut CV blocked by GTDB class. Per-fold permutation importance. Two feature flavours: binary (max-across-strains) and prevalence (mean-across-strains).

**Contribution:** Non-parametric check on the linear models. If RF feature importance ranks systems similarly to phyloglm, the linearity assumption is fine. The clade-blocked CV is the right fix for the i.i.d. violation that standard k-fold imposes on phylogenetic data.

## Tier 2 LASSO / Elastic Net — regularized selection on phylo residuals

**Stage name:** `lasso` (or runs inside `pglmm_mv` if that stage is enabled)

Two-step: (1) phylogenetically decorrelate predictors via `nlme::gls(predictor ~ 1, correlation = corBrownian)` on the full tree, taking residuals as the decorrelated feature; (2) run sklearn LASSO/Elastic Net with one-SE lambda selection and stability selection across subsamples.

**Contribution:** Secondary regularized-selection list. Not part of the consensus rank-product. Useful for cross-checking that the same top systems emerge under L1/L2 regularization as under per-system testing.

**Important computational note:** The phylo-residualisation step runs ONE R subprocess that does 18 sequential `nlme::gls` fits, each requiring Cholesky of a 40k×40k dense covariance. Single-threaded BLAS makes each fit take days. Multi-threaded BLAS makes the stage tractable (~4-5 days total). See `blas_threading_lessons.md`.

## Tier 3 burden — total defense count vs plasmid

**Stage name:** `burden` → `burden_pgls.tsv`, `burden_phyloglm.tsv`

PGLS (Pagel's lambda estimated by ML) of defense burden (sum of defense systems present) ~ plasmid carriage. Plus secondary phyloglm with burden as the predictor.

**Contribution:** Tests the aggregate "do plasmid-carrying species have more or fewer defense systems overall" question, after phylogenetic correction. Independent of which specific systems.

## Tier 3 LOCO — leave-one-clade-out stability

**Stage name:** `loco` → `tier3_loco_detail.tsv`, `tier3_loco_summary.tsv`

For each clade at GTDB class level, drop all species in that clade and refit the primary phyloglm. Cochran's Q test for heterogeneity of coefficients across leave-out replicates. Bonferroni-corrected within covariate_mode.

**Contribution:** Identifies defense systems whose primary-test result is driven by a single clade. Systems flagged as heterogeneous shouldn't be cited as broad bacterial-kingdom findings.

## Tier 3 phylogenetic signal — D-statistic

**Stage name:** `phylo_signal` → `tier3_phylo_signal.tsv`

`caper::phylo.d` reports Fritz & Purvis D for every binary feature and outcome.

**Contribution:** Methods-section justification for using phylogenetic correction at all. Reviewers expect to see D-statistics showing that traits are clustered on the tree (otherwise why do phyloglm).

## Tier 3 clade-restricted permutation

**Stage name:** `clade_perm` → `tier3_perm.tsv`

Reshuffle plasmid labels within each phylum (preserving clade-level prevalence). Empirical p-value over 1000 permutations.

**Contribution:** Empirical null distribution that respects clade-level prevalence structure. A complement to the analytic phyloglm test.

## Tier 3 prevalence matching

**Stage name:** `prev_match` → `tier3_prevalence_matched.tsv`

For each defense system, match plasmid+ and plasmid- species on the system's own prevalence quantile (deciles). Paired test on the matched indicators.

**Contribution:** Flags whether the association survives prevalence matching — i.e. whether plasmid-carriers happening to be well-sequenced (and therefore having higher max-aggregated defense rates) is driving the apparent effect.

## Tier 3 n_strains sensitivity

**Stage name:** `min_n_strains_sens` → `tier3_min_n_strains_sens.tsv`

Refit primary phyloglm on the subset of species with ≥5 strains.

**Contribution:** Guards against the max-aggregation saturation bias. Heavily-sampled species saturate to 1 for almost every defense system. If the primary result survives when poorly-sampled species are excluded, the signal isn't a sampling-depth artefact.

## Tier 3 prevalence-feature sensitivity

**Stage name:** `prev_feature_sens` → `tier3_prev_feature_sens.tsv`

Refit primary phyloglm using strain-mean prevalence (instead of max) as the defense feature. Continuous predictor.

**Contribution:** Another guard against max-aggregation bias, from a different angle. Concordance with the primary binary-feature result strengthens confidence.

## Tier 3 phylogenetic model sensitivity

**Stage name:** `phylo_model_sens` → `tier3_phylo_model_sens.tsv`

Refit primary phyloglm under OU and BM+lambda models (instead of BM).

**Contribution:** Tests whether the BM assumption is load-bearing. Defense systems and plasmids move horizontally so BM is a simplification; reviewers ask about it. If rankings agree across models, BM is fine for primary.

## Tier 3 misclassification MC

**Stage name:** `misclass_mc` → `misclass_mc_long.tsv`, `misclass_summary.tsv`

Monte Carlo over plasmid false-negative rate (FNR) grid 0.00, 0.05, 0.10, ..., 0.30. For each FNR, flip a fraction of plasmid-negative species to positive (modeling missed plasmid detections) and refit phyloglm. 200 replicates per FNR. ~2,800 total phyloglm fits.

**Contribution:** Reports the FNR at which each defense system loses significance. A system surviving up to FNR=0.30 is robust to reasonable plasmid-detection error. Quantifies the assumption.

## Tier 3 misclassification analytical

**Stage name:** `misclass_analytical` → `misclass_analytical.tsv`

Bross 1954 closed-form adjustment for non-differential misclassification of the outcome. Computes adjusted OR at each FNR and a "tipping point" FNR where the adjusted OR crosses 1.

**Contribution:** Analytical complement to the Monte Carlo. Fast; provides per-system FNR ceiling above which the observed association vanishes.

## Consensus

**Stage name:** `consensus` → `consensus.tsv`, `covariate_impact.tsv`

Rank-product across phyloglm + Pagel's + PGLMM. Cauchy-combined p-value travels alongside. One row per (defense_system, outcome_label, covariate_mode). Direction agreement between phyloglm and PGLMM is reported.

**Contribution:** The headline output for "which defense systems are robustly associated with plasmid carriage." Use `rank_product` as the primary ranking column; cite individual method p-values as supporting evidence.

## Phylo vs non-phylo comparison

**Stage name:** `phylo_vs_nonphylo` → `phylo_vs_nonphylo.tsv`

Side-by-side Tier 1 (non-phylogenetic) vs Tier 2 (phyloglm) coefficients, with verdict tags: `robust`, `phylo_explained`, `emerged_under_phylo`, `direction_reversed`, `not_significant_either`.

**Contribution:** Directly addresses the reviewer question "how much did phylogenetic correction change the picture." A system labeled `robust` survives both; `phylo_explained` is the most common signal-killer (phylogeny absorbed the association).

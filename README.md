# defense_analysis_v2

Phylogenetically-corrected comparative analysis of bacterial anti-phage defense systems against plasmid carriage. Given a strain-level defense-system presence/absence table, species-level plasmid annotations, and a GTDB tree, the pipeline fits a tiered set of phylogenetic models (phyloglm, Pagel's correlated-evolution, PGLMM, PGLS for burden) and reports which defense systems are associated with plasmid presence, in both directions, across mobility / size / replicon-type strata, with a full suite of sensitivity analyses.

A full walk-through of what each tier does and why lives in [`defense_analysis_v2/__init__.py`](defense_analysis_v2/__init__.py).

## Install

One command:

```bash
bash install.sh
```

Or, if you've already `chmod +x install.sh`:

```bash
./install.sh
```

Options:

```
./install.sh            # install into the active Python + system R
./install.sh --venv     # create ./.venv and install into it
./install.sh --conda    # create a conda env from environment.yml
./install.sh --no-r     # skip R (Python-only install for plotting / Tier 1)
```

The installer:

1. verifies Python ≥ 3.9 and (unless `--no-r`) R ≥ 4.0,
2. installs the Python package in editable mode (`pip install -e .`),
3. installs the R packages via `Rscript install_r_packages.R`,
4. imports `defense_analysis_v2` as a smoke test.

### Manual install (if you don't want the script)

```bash
# Python side
pip install -e .                    # or: pip install -r requirements.txt

# R side
Rscript install_r_packages.R
```

### Requirements

**Python ≥ 3.9** with: numpy, pandas, scipy, statsmodels, scikit-learn, joblib, dendropy, matplotlib. All pulled in automatically by `pip install -e .`

**R ≥ 4.0** with: ape, phylolm, phytools, caper, phyr, nlme, jsonlite. All pulled in by `install_r_packages.R`.

On macOS, R source builds of `phyr` / `phytools` may need Xcode command-line tools (`xcode-select --install`). On Debian/Ubuntu, `apt-get install libxml2-dev libcurl4-openssl-dev libssl-dev gfortran` covers the usual missing system libraries.

## Input data

The pipeline expects three TSV files and one Newick tree. Paths are configurable (see `defense_analysis_v2/config.py`); the CLI accepts them via flags.

**Strain-level defense table** (`--input`). One row per strain genome. Required columns:

```
genome   source   has_plasmid   gtdb_domain ... gtdb_species   <defense_system_1>   <defense_system_2>   ...
GCF_0001 refseq   yes           Bacteria    ... s__Escherichia coli   1   0   ...
```

- `has_plasmid` is `yes` / `no` and must be constant within each species (the pipeline enforces this invariant — it's the species-level propagation you're working with).
- Defense columns are non-negative integer counts; any positive value is treated as "present". Column names = defense-system identifiers (subtype or type).

Run once at subtype granularity and once at type granularity (the driver does both by default via `--granularity both`).

**Plasmid metadata table** (optional, enables the stratified outcomes). One row per plasmid, joined to species via `gtdb_species`. Expected columns: `gtdb_species`, `predicted_mobility_updated`, `rep_type(s)`, `size`. See `config.Config` for the exact column names and how to override them.

**Genome covariates table** (optional). One row per strain, keyed on `genome`. Supplies `corrected_genome_size`, `gc_avg`, `cds_number` for genome-capacity adjustment.

**Phylogenetic tree** (`--tree`). Newick file with species names as tip labels (with or without underscores — the matcher tries several normalisations).

## Quick start

```bash
defense-plasmid-analyze \
  --input      data/strain_defense_subtype.tsv \
  --input-type data/strain_defense_type.tsv \
  --tree       data/species_tree.nwk \
  --output-dir results/
```

This runs all stages on both granularities. To run only a subset of stages:

```bash
defense-plasmid-analyze \
  --input data/strain_defense_subtype.tsv \
  --tree  data/species_tree.nwk \
  --output-dir results/ \
  --granularity subtype_level \
  --stages tier1 phyloglm pglmm_mv consensus figures
```

Available stages: `negative_control`, `tier1`, `phyloglm`, `pagels`, `pglmm_mv`, `lasso`, `rf`, `burden`, `loco`, `within_clade_het`, `phylo_signal`, `clade_perm`, `depth_match`, `misclass_mc`, `misclass_analytical`, `defense_misclass`, `depth_sens`, `prev_feature_sens`, `phylo_model_sens`, `entry_mode`, `sister_pairs`, `feature_control`, `consensus`, `phylo_vs_nonphylo`, `figures`.

`prev_match` and `min_n_strains_sens` are accepted as aliases for `depth_match` and `depth_sens`.

Run the negative control before believing anything else:

```bash
defense-plasmid-analyze --input ... --tree ... --output-dir results/ \
  --stages negative_control
```

Estimate cost before submitting — every stage that failed on the cluster
failed predictably:

```bash
defense-plasmid-analyze --input ... --tree ... --output-dir ... --estimate-cost
```

This prints per-stage R calls, model fits, projected wall-clock, peak memory and
a fits/doesn't-fit verdict against the cluster envelope, then exits. See
[`docs/cluster_optimization_v3.md`](docs/cluster_optimization_v3.md).

`defense-plasmid-analyze --help` lists every flag. Programmatic access: `from defense_analysis_v2 import config, defense_plasmid_analysis`.

## Outputs

For each granularity the driver writes to `<output_dir>/<subtype_level|type_level>/`:

- `<stage>.tsv` — one long-form table per stage (e.g. `tier2_phyloglm.tsv`)
- `combined_all_results.tsv` — the legacy any-plasmid view, one row per defense system, columns merged across tiers
- `per_outcome_summary.tsv` — cross-stratum compact summary: coefficient, q-value, and consensus across methods per (defense_system, outcome_label, covariate_mode)
- `negative_control.tsv` — per-replicate FDR-significant counts on permuted labels, plus the `calibrated` verdict
- `tier3_depth_band_concordance.tsv` — per-system verdict across the high- and low-depth reruns
- `binomial_concordance.tsv` — binary vs binomial PGLMM agreement per stratified outcome
- `entry_mode_confirmatory.tsv` — the single pre-registered ssDNA-prediction test
- `entry_mode_composition.tsv` — per-system within-species plasmid-composition effects
- `sister_pair_summary.tsv`, `sister_vs_primary.tsv` — matched-pair results and their agreement with the regression
- `feature_control_comparison.tsv` — each system's effect against a prevalence- and clustering-matched null
- `summary_report.txt` — human-readable top-findings report
- `figures/` — PNG (300 dpi) + SVG pairs

Primary direction claims live in the `plasmid_given_defense` rows; the reverse direction (`defense_given_plasmid`) is in the same tables filtered on `direction`.

The `covariate_mode` column distinguishes `full` (log genome size / GC / log CDS + depth spline — **the only mode a primary claim may cite**), `depth_only` (depth spline alone, isolating the genome-capacity contribution), and `unadjusted` (**a positive control for the sampling-depth confound, not an alternative model**). Legacy labels `with_cov` / `without_cov` resolve to `full` / `unadjusted`.

## Repository layout

```
.
├── defense_analysis_v2/      # installable Python package
│   ├── __init__.py
│   ├── config.py             # dataclass with all pipeline knobs
│   ├── defense_plasmid_analysis.py   # CLI driver
│   ├── io_utils.py           # strain -> species aggregation, plasmid stratification
│   ├── tree_utils.py         # GTDB tree loading + tip matching
│   ├── stats_utils.py        # Firth, Cochran's Q, Cauchy combination, calibrated rank product
│   ├── r_bridge.py           # subprocess wrapper around Rscript
│   ├── r_scripts/            # phyloglm_uni.R, pglmm_mv.R, pagels_test.R, …
│   ├── tier1.py              # non-phylogenetic diagnostic baseline (Firth logistic)
│   ├── tier2_phylo_uni.py    # univariate phyloglm
│   ├── tier2_pagels.py       # Pagel's correlated-evolution
│   ├── tier2_multivariate.py # PGLMM + LASSO/Elastic Net on phylo residuals
│   ├── tier2_random_forest.py# clade-blocked RF
│   ├── tier3_burden.py       # PGLS on defense burden count
│   ├── tier3_loco.py         # LOCO influence diagnostic + within-clade Cochran's Q
│   ├── tier3_misclassification.py    # plasmid + defense FNR sensitivity (MC + analytical)
│   ├── tier3_sensitivity.py  # negative control, depth-matched, depth-stratified permutation, depth bands
│   ├── tier3_entry_mode.py   # A4 pre-registered ssDNA entry-mode prediction
│   ├── tier3_sister_pairs.py # B1 depth-matched phylogenetic sister pairs
│   ├── tier3_feature_control.py # B3 prevalence/clustering-matched null features
│   ├── phylo_signal_fast.py  # native vectorised Fritz & Purvis D (replaces caper)
│   ├── cost_model.py         # per-stage wall-clock / memory projection
│   ├── consensus.py          # calibrated rank product + Cauchy combination across methods
│   ├── reporting.py          # combined tables + summary_report.txt
│   └── plotting.py           # publication figures
├── pyproject.toml            # package metadata + dependencies
├── requirements.txt          # pip install -r alternative
├── environment.yml           # conda env alternative
├── install_r_packages.R      # idempotent CRAN install
├── install.sh                # one-shot Python + R installer
├── tests/
│   ├── test_pipeline_fixes.py       # regression tests for the confound + statistics fixes
│   └── test_analysis_extensions.py  # regression tests for A4 / B1-B4
├── confound_sim.py           # reproduces the null-simulation numbers quoted above
├── confound_sim2.py          # sweep showing where a linear log(n) term fails
├── LICENSE
├── CITATION.cff
└── README.md
```

## The sampling-depth problem — read this first

The species plasmid label is propagated upstream ("any strain in the species carries a plasmid") and the species defense call is `max()` across strains. Both are therefore *"at least one positive among n_strains"*, i.e. `1 − (1−p)^n` in expectation, which makes **sequencing depth a common cause of the predictor and the outcome**. Left uncorrected this manufactures an odds ratio of ~2.5 with a **100% false-positive rate under a strict null** (reproduce with `confound_sim.py`).

A single linear `log(n_strains)` covariate is *not* sufficient. The logit of `1 − (1−p)^n` is not linear in `log n`, and its curvature depends on `p`, so a rare defense system and common plasmid carriage sit on differently-shaped saturation curves. In simulation one linear term left 28% false positives when defense was common and plasmid carriage rare, and **40% when sampling depth was clade-structured** — which is what GTDB actually looks like.

The pipeline therefore:

1. adjusts with a **natural cubic spline basis** on `log(n_strains)` (`config.depth_spline_df`, default 5);
2. runs a **negative control** first (`--stages negative_control`) that permutes the plasmid label within joint (clade × depth-decile) strata and re-runs the primary sweep. Under a correct model the FDR-significant count should be near zero. **If it isn't, nothing downstream is interpretable** — the summary report leads with this verdict;
3. treats the unadjusted fit as a **positive control for the confound**, not as an alternative model. `covariate_modes` is `("full", "depth_only", "unadjusted")`, `unadjusted` is excluded from consensus and from every primary claim, and `primary_covariate_mode` names the one mode a headline number may cite.

## Method summary

Tier 1 is non-phylogenetic (Firth-weighted logistic with covariates, plus Fisher / Mann-Whitney / ordinary logistic diagnostics). It's explicitly labelled diagnostic and is never cited as primary evidence.

Tier 2 is where the primary claims live:

- **phyloglm** (`phylolm::phyloglm`) — one univariate fit per defense system, both directions, across every plasmid-outcome stratum. Fits that hit the `btol` bound, separate, return non-finite standard errors, or emit a convergence warning **forfeit their p-value** so they cannot enter FDR, consensus, or a figure. The prevalence gate follows the defense system in *both* directions.
- **Pagel's test** (`phytools::fitPagel`) — correlated binary-trait evolution, on 10 subsamples with the per-subsample p-values combined by the **Cauchy (ACAT) combination**. Not the median: the median of k p-values is not a p-value (Beta(3,3) at k=5, so P(median < 0.05) ≈ 0.0012).
- **PGLMM** (`phyr::pglmm`) — multivariate fit controlling for the other defense systems, with pairwise interactions on the top-8 univariate hits. Main effects and interactions are **separate FDR families**; interaction terms are flagged post-selection. The phylogenetic variance component is selected **by name**, not position.
- **Clade-blocked Random Forest** — LeaveOneGroupOut CV on GTDB class with per-fold permutation importance.

Tier 3 is robustness:

- **Negative control** — the calibration check described above.
- **Depth sensitivity, both bands** — high-depth (`n_strains ≥ 5`, gated on retained outcome variance) *and* low-depth (`n_strains ≤ 2`, where saturation cannot have operated). The **concordance between bands** is the deliverable; a system significant only in the deep tail is flagged `high_depth_only__possible_sampling_artefact`.
- **Depth-matched McNemar test** — match plasmid+ to plasmid− species on sampling depth, then test defense presence on the discordant pairs.
- **Depth-stratified clade permutation** — reshuffle plasmid labels within (GTDB class × depth decile).
- **Within-clade Cochran's Q** — the valid heterogeneity test, on disjoint within-clade fits. **LOCO** is retained as an *influence diagnostic* (coefficient shift, sign flips) with no p-value, because leave-one-clade-out estimates share >90% of their data.
- **PGLS burden** (`nlme::gls` with `corPagel`, λ estimated by ML) on `sqrt(burden)` — a variance-stabilising transform for a count outcome under a Gaussian likelihood. Never run unadjusted.
- **D-statistic** phylogenetic signal for every column (seeded).
- **Prevalence-feature sensitivity** — refit with strain-mean prevalence as the predictor, *keeping* the depth covariates. This de-saturates the predictor arm only; the outcome remains the propagated label, so it is one arm of the confound addressed, not the confound resolved.
- **Model sensitivity** — estimator (MPLE vs IG10) and **covariance structure** via Pagel's-λ-rescaled trees. `phyloglm`'s `method` selects the estimator, not an evolutionary process.
- **Misclassification sensitivity, both sides** — plasmid-detection FNR (Monte Carlo with a **depth-differential** rate `f^n`, plus Bross applied within depth strata) and a symmetric analytical correction for **DefenseFinder false negatives**.

Consensus is a **calibrated** rank product across phyloglm + PGLMM + Pagel's, with a permutation null drawn per number of contributing methods, plus an FDR-corrected Cauchy combination and a `consensus_tier` label that requires corroboration. FDR is applied per stratum **and globally** across the pre-declared primary outcomes (`config.primary_outcome_labels`).

## Mechanism and causal-inference analyses

Four analyses go beyond the association screen. Three of them are, by
construction, immune to the sampling-depth confound that governs the main
sweep, which makes them orthogonal evidence rather than further robustness
checks on the same estimand.

**Pre-registered entry-mode prediction (`entry_mode`).** Conjugative plasmids
enter as single-stranded DNA; restriction-like systems cleave double-stranded
DNA. So dsDNA-restricting systems (RM, Type IV restriction, BREX, DISARM,
Wadjet, Dnd) should deplete *non-conjugative* plasmids more than
abortive-infection and signalling systems do. The partition is declared in
`config.entry_mode_predicted_categories` and **must be fixed before any
entry-mode result is inspected** — the inferential value comes entirely from
having done so. The primary model is a *within-species composition* contrast
(what fraction of a species' plasmids are non-conjugative), so every
species-level property including sequencing depth differences out. The
confirmatory test is one permutation test with one degree of freedom, with
group labels permuted across systems so the dependence among per-system
estimates is preserved.

**Phylogenetically matched sister pairs (`sister_pairs`).** Within-pair
contrasts among sister species matched on sequencing depth. Controls phylogeny
*and* depth by construction rather than by model. Pairs are built from the
direct leaf children of each internal node rather than strict cherries, because
polytomies are resolved arbitrarily upstream and cherry extraction would keep
an arbitrary subset of each polytomy. Exact McNemar, with conditional logistic
adjustment for residual within-pair depth imbalance.

**Evolutionary directionality (`pagels`).** `fitPagel` is now run under
`dep.var = "x"`, `"y"` and `"xy"`, giving nested models comparable by AIC.
This distinguishes *defense state drives plasmid gain/loss* from *plasmid
carriage drives defense gain/loss* — the only analysis in the design that
addresses ordering rather than association, and the direct answer to the "or
vice versa" half of the question. Aggregated by Akaike weights computed within
each subsample, since AIC is not comparable across different subsamples.

**Matched-feature negative control (`feature_control`).** Would an arbitrary
trait with the same prevalence and the same phylogenetic clustering show the
same association? Synthetic traits are simulated as
`sqrt(lambda)*z_BM + sqrt(1-lambda)*z_iid` — exactly Pagel's lambda covariance
— then thresholded to the target prevalence. Each real system's effect is
reported as a percentile against its prevalence-matched null. A system at the
60th percentile of arbitrary traits is not doing anything special, whatever its
q-value says.

**E-values** are attached to every primary association: the minimum strength an
unmeasured confounder would need, on the risk-ratio scale, with both the
defense system and plasmid carriage. Computed from `sqrt(OR)` because plasmid
carriage is common here — using the raw OR would inflate every E-value.

## Assumptions this codebase does not verify

- **Defense calls are chromosomal.** Nothing in the pipeline restricts DefenseFinder output to chromosomal contigs, and the input schema carries no replicon location. A plasmid-encoded defense system would correlate with plasmid presence by definition. Enforce this upstream.
- **The tree.** The GTDB cut is project-specific (`human_animal_free_90percent_species_level`); polytomies are resolved arbitrarily with ε-length branches. The phylogenetic correction inherits any systematic error in it.

See `defense_analysis_v2/__init__.py` for the full rationale of every primary choice, and `defense_analysis_v2_review.md` (if present in the repo root) for the scientific-defensibility review of the pipeline.

## Running on a subset / for development

```bash
# Install dev extras (pytest, ruff, mypy)
pip install -e '.[dev]'

# Single stage, single granularity, on a laptop-sized subset
defense-plasmid-analyze \
  --input test_data/strain_defense_small.tsv \
  --tree  test_data/species_tree_small.nwk \
  --output-dir /tmp/defense_smoke \
  --granularity subtype_level \
  --stages tier1 phyloglm consensus \
  --n-jobs 4 --n-permutations 100
```

## Reproducibility

`config.random_seed = 42` seeds every downstream RNG (joblib parallelism, Pagel's subsampling, LASSO stability selection, misclassification MC, depth-matching, the D-statistic permutation, and the negative control).

Stage checkpointing is keyed on `config.fingerprint()`, a hash of every field that can change a numeric result. Changing a threshold and re-running no longer silently reuses a stale TSV — the cache is invalidated and a warning is logged. Paths and compute knobs are excluded from the hash. Set via the `Config` dataclass if you need a different seed. Tree preprocessing (polytomy resolution, ε-branch fix) happens once on the Python side so every R call gets an identically-conditioned tree.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this pipeline, please cite via [CITATION.cff](CITATION.cff).

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

Available stages: `tier1`, `phyloglm`, `pagels`, `pglmm_mv`, `lasso`, `rf`, `burden`, `loco`, `phylo_signal`, `clade_perm`, `prev_match`, `misclass_mc`, `misclass_analytical`, `min_n_strains_sens`, `prev_feature_sens`, `phylo_model_sens`, `consensus`, `phylo_vs_nonphylo`, `figures`.

`defense-plasmid-analyze --help` lists every flag. Programmatic access: `from defense_analysis_v2 import config, defense_plasmid_analysis`.

## Outputs

For each granularity the driver writes to `<output_dir>/<subtype_level|type_level>/`:

- `<stage>.tsv` — one long-form table per stage (e.g. `tier2_phyloglm.tsv`)
- `combined_all_results.tsv` — the legacy any-plasmid view, one row per defense system, columns merged across tiers
- `per_outcome_summary.tsv` — cross-stratum compact summary: coefficient, q-value, and consensus across methods per (defense_system, outcome_label, covariate_mode)
- `summary_report.txt` — human-readable top-findings report
- `figures/` — PNG (300 dpi) + SVG pairs

Primary direction claims live in the `plasmid_given_defense` rows; the reverse direction (`defense_given_plasmid`) is in the same tables filtered on `direction`. The `covariate_mode` column distinguishes `with_cov` (genome-size / GC / CDS / log(n_strains) adjusted) from `without_cov` fits.

## Repository layout

```
.
├── defense_analysis_v2/      # installable Python package
│   ├── __init__.py
│   ├── config.py             # dataclass with all pipeline knobs
│   ├── defense_plasmid_analysis.py   # CLI driver
│   ├── io_utils.py           # strain -> species aggregation, plasmid stratification
│   ├── tree_utils.py         # GTDB tree loading + tip matching
│   ├── stats_utils.py        # Firth, Cochran's Q, Cauchy combination, rank product
│   ├── r_bridge.py           # subprocess wrapper around Rscript
│   ├── r_scripts/            # phyloglm_uni.R, pglmm_mv.R, pagels_test.R, …
│   ├── tier1.py              # non-phylogenetic diagnostic baseline (Firth logistic)
│   ├── tier2_phylo_uni.py    # univariate phyloglm
│   ├── tier2_pagels.py       # Pagel's correlated-evolution
│   ├── tier2_multivariate.py # PGLMM + LASSO/Elastic Net on phylo residuals
│   ├── tier2_random_forest.py# clade-blocked RF
│   ├── tier3_burden.py       # PGLS on defense burden count
│   ├── tier3_loco.py         # leave-one-clade-out + Cochran's Q
│   ├── tier3_misclassification.py    # plasmid FNR sensitivity (MC + analytical)
│   ├── tier3_sensitivity.py  # prevalence-matched, clade-permutation, n_strains sens, phylo-model sens
│   ├── consensus.py          # rank product + Cauchy combination across methods
│   ├── reporting.py          # combined tables + summary_report.txt
│   └── plotting.py           # publication figures
├── pyproject.toml            # package metadata + dependencies
├── requirements.txt          # pip install -r alternative
├── environment.yml           # conda env alternative
├── install_r_packages.R      # idempotent CRAN install
├── install.sh                # one-shot Python + R installer
├── LICENSE
├── CITATION.cff
└── README.md
```

## Method summary

Tier 1 is non-phylogenetic (Firth-weighted logistic with covariates, plus Fisher / Mann-Whitney / ordinary logistic diagnostics). It's explicitly labelled diagnostic and is never cited as primary evidence.

Tier 2 is where the primary claims live:
- **phyloglm** (`phylolm::phyloglm`, BM default) — one univariate fit per defense system, both directions, across every plasmid-outcome stratum.
- **Pagel's test** (`phytools::fitPagel`) — correlated binary-trait evolution. Run on 5 × 1,500-species uniform subsamples with the median log-LR p reported (the fraction of subsamples significant is also surfaced).
- **PGLMM** (`phyr::pglmm`) — multivariate fit that controls for the other defense systems simultaneously, with optional pairwise interactions on the top-8 univariate hits. Binary + binomial outcome modes.
- **Clade-blocked Random Forest** — LeaveOneGroupOut CV on GTDB class with per-fold permutation importance.

Tier 3 is robustness:
- **PGLS burden** (`nlme::gls` with `corPagel`, λ estimated by ML) — does total defense count differ by plasmid status after accounting for ancestry?
- **LOCO** with **Cochran's Q** heterogeneity test at GTDB class (primary) and phylum (fallback).
- **D-statistic** phylogenetic signal for every column, as methods-section justification.
- **Clade-restricted permutation** — reshuffle plasmid labels within each phylum.
- **Prevalence matching** on deciles of the defense system's own prevalence.
- **Min-n_strains** and **prevalence-feature** sensitivity reruns — immunise against species-level `max()` saturation bias.
- **Phylogenetic-model sensitivity** — refit phyloglm under OU and BM+lambda.
- **Plasmid misclassification sensitivity** — Monte Carlo across an FNR grid plus analytical Bross (1954) bias correction with tipping-point FNR.

Consensus is rank-product across phyloglm + PGLMM + Pagel's, computed per (outcome_label, covariate_mode) slice, with Cauchy-combined p-value alongside.

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

`config.random_seed = 42` seeds every downstream RNG (joblib parallelism, Pagel's subsampling, LASSO stability selection, misclassification MC, prevalence-matching). Set via the `Config` dataclass if you need a different seed. Tree preprocessing (polytomy resolution, ε-branch fix) happens once on the Python side so every R call gets an identically-conditioned tree.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this pipeline, please cite via [CITATION.cff](CITATION.cff).

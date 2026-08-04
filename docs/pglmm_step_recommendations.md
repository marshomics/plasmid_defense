# PGLMM and LASSO stages — failures encountered and what works

The multivariate phylogenetic GLMM stage (`pglmm_mv`) and the LASSO/Elastic Net stage (`lasso`) were the two hardest parts of the pipeline to get running. They share infrastructure (`tier2_multivariate.py`) and have overlapping pain points around BLAS threading, memory ceilings, and now tip-label normalisation drift. This document records every failure mode encountered and the configurations that ultimately worked.

## Failure modes encountered

### 1. Integer overflow in RcppArmadillo at full tree size

**Error:**
```
Error: SpMat::init(): requested size is too large; suggest to enable ARMA_64BIT_WORD
```

**Cause:** Default Armadillo (the C++ matrix library RcppArmadillo wraps) uses 32-bit signed integer indexing for sparse matrices. The index limit is 2³¹ ≈ 2.15 billion elements. With 39,681 tips, the phylogenetic covariance matrix is ~1.57 billion elements — under the limit for the dense matrix itself, but sparse-matrix internals (separate row/column index arrays plus values) push past 2³¹.

**Fix:** Recompile RcppArmadillo with `ARMA_64BIT_WORD=1`, then rebuild phyr from source against the new RcppArmadillo. Procedure:

```bash
mamba activate mmseqs
mkdir -p ~/.R
cat >> ~/.R/Makevars <<'EOF'
PKG_CPPFLAGS += -DARMA_64BIT_WORD=1
CXXFLAGS = -O3 -DARMA_64BIT_WORD=1
CXX17FLAGS = -O3 -DARMA_64BIT_WORD=1
EOF

R -e 'install.packages("RcppArmadillo", type="source", repos="https://cloud.r-project.org", Ncpus=4)'
R -e 'install.packages("phyr", type="source", repos="https://cloud.r-project.org", Ncpus=4)'
```

**Critical detail:** Use `+=` in PKG_CPPFLAGS, not `=`. Plain `=` overrides the package's own PKG_CPPFLAGS, which includes `-I../inst/include`. Without that include path, the latest RcppArmadillo can't find `<RcppArmadillo/Lighter>` and the compile fails. Took several rounds of debugging to realize this.

After recompile, verify with a 50,000-tip pglmm test (see test snippet in main troubleshooting log).

### 2. Per-process memory ceiling on SGE

**Error:** `rc=-9` (SIGKILL) after R consumes 175+ GB resident.

**Cause:** Tübingen MPI's SGE enforces `h_vmem` per-process (despite `consumable=YES`). With `-l h_vmem=30G`, every R process is capped at 30 GB regardless of slot count. PGLMM at full tree needs ~175-250 GB for one fit.

**Fix:** Either request higher per-process `h_vmem` (limited by node total — cluster nodes max 128 GB), or subsample with `pglmm_max_species`. Pure full-tree PGLMM not feasible on this cluster without a node with >256 GB RAM.

### 3. Timeout at 2 hours (original hardcoded)

**Error:** `Timed out after 7200s` for PGLMM fits.

**Cause:** Original `pglmm_mv.R` had `timeout=60 * 60 * 2` hardcoded in `call_r_script`. Wholly insufficient at any meaningful tree size.

**Fix:** Made configurable via `config.pglmm_timeout_hours`. Current default: 48. Set higher if needed.

### 4. Per-outcome convergence failures

**Error (per outcome):** `Estimation of B failed. Check for lack of variation in Y.`

**Cause:** Specific outcomes (like `conjugative_binomial`) have very low variance after conditioning on species with non-zero plasmid count. PQL hits a boundary of the random-effect parameter space and aborts.

**Behavior:** This is a per-outcome failure, not a pipeline failure. The relevant outcome's PGLMM row is empty in the output. Consensus stage falls back to phyloglm + Pagel's for that outcome.

**Workaround:** None at the package level. Document affected outcomes in methods.

### 5. Single-threaded BLAS even with env vars set

**Error:** PGLMM runs but at 100% CPU (single thread). Per-fit takes 24-48h instead of 30-90 min.

**Cause(s) found in sequence:**
- `OMP_NUM_THREADS` missing from environment (only OPENBLAS_NUM_THREADS was set). MKL's GNU thread variant reads `OMP_NUM_THREADS`, defaults to 1 when absent.
- `MKL_THREADING_LAYER=INTEL` was ignored by conda's MKL build.
- SGE allocated `-pe parallel 1` so cgroup affinity gave only one CPU.

**Fix:** Set all three thread env vars (OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS) to N, and request `-pe parallel N` so SGE actually allocates N cores.

```bash
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
```

with `-pe parallel 20` in the SGE header.

### 6. LASSO crash after multi-day phylo-residualisation (tip-label mismatch)

**Error (after `_phylo_residualise` returns successfully):**
```
KeyError: "None of [Index(['s__JACPOT01_sp016183435', 's__JACDAF01_[species_13657]', ...])] are in the [index]"
```

**When:** Inside `run_regularised_on_residuals`, at the line
```python
aligned = phylo_data.set_index("tip").loc[resid.index]
```

**Cost when first hit:** ~6 days. Phylo-residualisation completed in R, residuals were written, Python tried to align them with `phylo_data` and crashed immediately. All the residual computation was lost.

**Cause:** Mid-project, all the R scripts (`phyloglm_uni.R`, `pglmm_mv.R`, `pagels_test.R`, `phylo_residuals.R`, etc.) were given a `normalise_tips` function that strips outer single quotes, trims whitespace, and replaces interior spaces with underscores. This was added to fix tree-vs-data intersect failures caused by ape's standard underscore-to-space conversion on unquoted Newick labels.

The R script normalises both `tree$tip.label` AND `data[[tip_column]]`, writes its output keyed on the normalised tip labels. But `phylo_data` on the Python side still carries the ORIGINAL tip column values (with spaces, possibly with quotes). The `.loc[resid.index]` join then fails because the index forms don't match.

This bug affected ONLY the LASSO/EN path. Other stages don't read residual files back into Python — they consume R output entirely inside R, so the normalisation drift was invisible.

**Fix:** Apply the same normalisation on the Python side before the join. In `tier2_multivariate.py:run_regularised_on_residuals`:

```python
def _normalise_tip(s):
    s = str(s).strip().strip("'").strip()
    return s.replace(" ", "_")

phylo_data_norm = phylo_data.copy()
phylo_data_norm["tip"] = phylo_data_norm["tip"].apply(_normalise_tip)
resid = resid.set_index("tip")
# Pre-flight assert so future drift surfaces immediately, not after 6 days:
norm_tips = set(phylo_data_norm["tip"])
overlap = sum(t in norm_tips for t in resid.index)
if overlap == 0:
    raise RuntimeError(
        "LASSO residual join failed: zero overlap after tip normalisation. "
        "Most likely a tip-label form mismatch between R's normalise_tips() "
        "and the Python-side _normalise_tip()."
    )
aligned = phylo_data_norm.set_index("tip").loc[resid.index]
```

This is now in the codebase as of the bug discovery. If you ever add a new R-side normalisation step (e.g. case-folding, accent stripping, etc.), update `_normalise_tip` in `tier2_multivariate.py` to mirror it, or the join will silently mismatch again.

**Lesson:** Any R script that writes tip-keyed output for Python to consume should be paired with a Python-side normaliser that mirrors the R-side one. The pre-flight overlap assert costs microseconds and catches the next 6-day waste.

### 7. Deprecation warnings (harmless)

**Warnings:**
```
'as(<matrix>, "dgTMatrix")' is deprecated.
the 'findbars' function has moved to the reformulas package.
```

**Cause:** Newer versions of `Matrix` and `lme4` deprecate calls that phyr still uses internally. Cosmetic; doesn't affect results.

**Fix:** Ignore. Could be silenced by reinstalling Matrix + lme4 from source aligned with phyr's expected versions, but not worth it.

## Recommended PGLMM configuration for future runs

### Config file

```python
# defense_analysis_v2/config.py
pglmm_timeout_hours: int = 48
pglmm_max_species: Optional[int] = 15000   # phylum-stratified subsample
min_prevalence_multivariate: float = 0.10
```

### SGE script

```bash
#$ -pe parallel 20
#$ -l h_vmem=12G          # 20 × 12 = 240G total; per-process cap 12G fits 15k-tip PGLMM
#$ -l h_rt=600:0:0
#$ -j y

. ~/.bashrc
mamba activate mmseqs

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

defense-plasmid-analyze \
  --granularity both \
  --n-jobs 1 \
  --stages pglmm_mv \
  --output-dir /ebio/abt3_projects2/Gut_genetics2/data/defensefinder/plasmid_vs_defense_v2
```

Per-fit wall-clock at 15k tips, 20-thread BLAS: ~30-90 minutes. Total PGLMM stage with ~68 fits: 1-3 days.

If you want the full ~40k-tip PGLMM despite the cluster's 128 GB ceiling, the only option is `pglmm_max_species = 30000` (≈98 GB per fit) on a 128 GB node with all 20 cores: requires `-pe parallel 4 -l h_vmem=30G` (120 GB total, per-process 30G) plus the BLAS threading config. Per-fit ~4-8 hours, total stage ~12-24 days.

## What to verify before submitting a PGLMM job

1. **RcppArmadillo recompiled with ARMA_64BIT_WORD.** Test with a 50,000-tip pglmm fit:

   ```r
   library(ape); library(phyr)
   set.seed(42); n <- 50000
   tree <- rtree(n)
   data <- data.frame(species = tree$tip.label, y = rbinom(n, 1, 0.3), x = rnorm(n))
   phyr::pglmm(y ~ x + (1 | species__), data = data, family = "binomial",
               cov_ranef = list(species = tree))
   ```

   Should complete without `SpMat::init(): requested size is too large`. If it errors, the recompile didn't take.

2. **`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` are all in your SGE script.** Setting only OPENBLAS won't help when MKL is the loaded BLAS. Setting only MKL won't help when CHOLMOD (if used) is the bottleneck.

3. **`pglmm_max_species` is set in `config.py`.** Without it, defaults to None and tries full tree → either RcppArmadillo overflow or OOM.

4. **SGE script's slot count and `OPENBLAS_NUM_THREADS` match.** Don't set OPENBLAS=20 with `-pe parallel 1`; you'll only get 1 core regardless.

## Skipping PGLMM if it's still painful

The consensus rank-product is built from phyloglm + Pagel's + PGLMM but degrades gracefully to 2 methods if PGLMM is absent. To skip:

```bash
defense-plasmid-analyze \
  --granularity both \
  --n-jobs 20 \
  --stages tier1 phyloglm pagels rf burden loco phylo_signal clade_perm \
           prev_match misclass_mc misclass_analytical min_n_strains_sens \
           prev_feature_sens phylo_model_sens consensus phylo_vs_nonphylo figures \
  --output-dir ...
```

Note `pglmm_mv` and `lasso` are not in the stage list. The consensus stage's `n_methods_contributing` column will be 2 for every system. Methods section should say: "Multivariate PGLMM was attempted but not used in the final consensus because of computational constraints; the consensus rank-product combines univariate phyloglm and Pagel's correlated-evolution test."

That's a defensible analysis at this dataset's scale. Reviewers will accept it given the tree-size and per-process memory ceiling on the cluster.

## Recipe for adding PGLMM later as a separate job

If you've already run the rest of the pipeline and want to add PGLMM:

```bash
# Run only PGLMM
defense-plasmid-analyze --stages pglmm_mv --granularity both --n-jobs 1 ...

# Then re-run consensus to incorporate it
defense-plasmid-analyze --stages consensus phylo_vs_nonphylo figures \
  --force-rerun consensus phylo_vs_nonphylo figures \
  --granularity both --n-jobs 4 ...
```

The `--force-rerun consensus` is necessary because the existing `consensus.tsv` will have been built from 2 methods; we need to refresh it with PGLMM included.

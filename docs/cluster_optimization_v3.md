# Cluster optimisation — defense_analysis_v2 v3.2

Supersedes `cluster_optimization_log.md`, which documents the v2 pipeline. The
environment constraints are unchanged (128 GB node ceiling, per-process
`h_vmem`, 600 h `h_rt`); what changed is that the four stages that could not
complete now fit, and the pipeline can tell you the cost before you submit.

## The four blocked stages

| Stage | v2 blocking issue | v3.2 status |
|---|---|---|
| **LOCO** | SIGBUS under concurrent R subprocess + pandas I/O | Runs. 1,200 → **50** R calls, ~13 h |
| **misclass_mc** | Same SIGBUS; 2,800 sweeps > 25-day ceiling | Runs. 4,200 → **160** R calls, ~4 h |
| **phylo_signal** | caper killed at the 25-day `h_rt` | Runs. Native D statistic, **~1 h** |
| **PGLMM full tree** | ~175 GB per fit > 128 GB node | Runs at 15k tips, **25 GB** |

Projected totals at 20 workers, 39,681 species × 435 systems × 17 outcomes:

```
                        before      after
parallel wall-clock    2,818 h      258 h      (117 days -> 11 days)
R invocations           6,648        692
peak memory             175 GB       25 GB
```

Check it yourself before submitting anything:

```bash
defense-plasmid-analyze --input ... --tree ... --output-dir ... --estimate-cost
```

That prints per-stage R calls, model fits, wall-clock, memory, and a
fits/doesn't-fit verdict against the cluster envelope, then exits without
running. Every one of the four v2 failures was predictable from the data
dimensions and the config; none of them needed to be discovered 25 days in.

---

## 1. The SIGBUS was an I/O problem, and it is fixed at the source

**Diagnosis.** Every `call_r_script` invocation serialised the entire
species × feature frame — 39,681 × ~460 at ~40 MB — and a full v2 run made
~4,800 such calls. That is **~190 GB of temporary writes**, with up to 20
workers streaming concurrently onto shared scratch. Bus errors on a write path
are the classic signature of filesystem pressure, and the v2 log already
records intermittent bus errors traced to `/tmp` fill events.

Worth being precise: this was **not** the wall-clock bottleneck. Measured, the
serialisation costs about 3 h across the whole pipeline. It was a *reliability*
problem — the thing that killed LOCO and misclass_mc — not a speed problem.

**Fix.** `r_bridge.SharedFrame`: stages that call R many times over the same
data write it **once** and pass tiny side files instead.

- `row_filter_file` — one column of tips to keep (LOCO, within-clade fits)
- `override_file` — a keyed table of columns to replace (negative control,
  misclassification MC, feature control)

R-side support is in `r_scripts/_shared_data.R`, sourced by every script.
Per-call I/O drops from ~40 MB to a few hundred KB — roughly three orders of
magnitude — and total temp writes fall from ~190 GB to under 1 GB.

**Belt and braces.** `call_r_script` now retries calls killed by a *signal*
(negative return code: SIGBUS, SIGKILL, SIGSEGV) up to `config.r_max_retries`,
with backoff. R-level errors return positive codes, are deterministic, and are
never retried. `config.max_concurrent_r_calls` caps concurrent R subprocesses
independently of `n_jobs` if you want a harder limit.

---

## 2. phylo_signal: replace caper, don't shrink the analysis

caper was killed at 600 h on 452 columns × 1000 permutations × 40k tips. The
tempting fix is to cut `n_perm` to 100, which buys a 10× reduction and costs
you a tenth of the permutation resolution.

That is not necessary. The D statistic is a single O(n) tree traversal per
evaluation; the cost was R-level implementation overhead, not arithmetic.
`phylo_signal_fast.py` restructures it:

- the tree is parsed into flat arrays **once**, not per column;
- the post-order traversal runs **level-wise**, so the Python loop is over tree
  depth (a few hundred steps) rather than over ~80,000 nodes;
- all permutations are evaluated **simultaneously** as a (nodes × permutations)
  matrix.

Same statistic, full 1000 permutations, **~1 h instead of 600+**.

Validated at both reference points on a balanced test tree: a randomly assigned
trait returns D = 1.02 (theory: 1.0), a Brownian-thresholded trait returns
D = −0.14 (theory: 0.0), and a single contiguous clade returns D = −0.81. The
nulls use rank-based thresholding so every null replicate has *exactly* the
observed prevalence — a null at a different prevalence is not comparable.

`config.phylo_signal_engine = "caper"` restores the old path for cross-checking
on a small subset.

---

## 3. Scope reductions that follow from the science

The remaining wall-clock was the **number of model fits**, not I/O. Each
reduction below removes comparisons that were never load-bearing; none of them
weakens a fit that survives.

### LOCO: 1,200 → 50 R calls

- **Primary covariate mode only.** LOCO is a stability check on the *primary
  result*. There is no "stability of the confound positive control" claim to
  make, and `depth_only` is a decomposition rather than a result. 3× saving.
- **Primary rank only** (GTDB class; phylum available via
  `loco_ranks_primary_only=False`). 2× saving.
- **Only size-gated clades are fit.** Dropping 3 species from 40,000 returns a
  near-duplicate of the full-data estimate: uninformative as an influence
  diagnostic, and already excluded from the heterogeneity test. Fitting it was
  pure waste. ~200 → ~50 clades.

### misclassification MC: 4,200 → 160 R calls

- **Restricted to FDR-significant systems.** The question is "would this
  *finding* survive plasmid-detection false negatives?", which only applies to
  findings. This is the correct scope, not a shortcut. 435 → ~20–40 systems.
  Falls back to the strongest `misclass_max_systems` if nothing reaches FDR.
- **Primary covariate mode only**, same reasoning as LOCO.
- **40 replicates, not 200.** The reported quantity is a *median* coefficient
  per FNR level. Verified numerically: the Monte Carlo standard error of a
  median over 40 draws is under a quarter of the coefficient's own standard
  error, so the extra 160 draws buy nothing measurable.
- **4 FNR grid points, not 7.** The attenuation curve is monotone, and the
  analytical Bross correction covers the continuum anyway.

### Pagel directionality: ~3× cost → ~1.2×

The B2 directional fits triple the per-system cost of the most expensive stage.
They are now gated twice, both times on semantics rather than budget:

- **Primary outcomes only** — directionality is a headline claim, not something
  to compute for 17 exploratory strata.
- **Only where the standard Pagel test rejects independence** (lenient screen at
  α = 0.10). Asking "which drives which?" for an independent pair is not a
  question; the aggregator already returns `independent_no_dependence` for
  those rows. Typically removes 80–90% of systems.

### entry_mode: 435 R invocations → 20

The A4 composition model needs a univariate fit per system. Calling
`pglmm_mv.R` once per system paid interpreter start-up, package loading and
tree parsing 435 times, and those fixed costs dominated — the model itself runs
only on species with enough plasmids to *have* a composition (a few thousand
tips, not 40,000).

`r_scripts/pglmm_uni_binomial.R` iterates systems internally, and the sweep is
**chunked across workers** so it both amortises start-up *and* parallelises. One
process would have been a 124 h serial block; 20 chunks make it ~6 h.

Automatic fallback to the empirical-logit PGLS if the binomial PGLMM cannot
complete, recorded in the output's `engine` column so the substitution is never
invisible.

### negative_control: sequential → parallel

Replicates are independent and were being run in a `for` loop. Dispatched in
parallel they cost one sweep's wall-clock instead of twenty.

---

## 4. SGE submission

The two-config split from v2 still applies, and the reasoning is unchanged:
20 parallel R subprocesses each spawning 20 BLAS threads is 400 threads on 20
cores. Set all three thread variables — different libraries in the stack read
different ones, and a missing `OMP_NUM_THREADS` silently pins threading to 1.

**Config A — parallel R subprocesses** (`tier1`, `phyloglm`, `pagels`, `loco`,
`within_clade_het`, `misclass_mc`, `negative_control`, `feature_control`,
`entry_mode`, `depth_match`, `clade_perm`, `rf`, `burden`, `sister_pairs`):

```bash
#$ -pe parallel 20
#$ -l h_vmem=10G
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=/ebio/abt3_scratch/jmarsh/tmp_$JOB_ID
defense-plasmid-analyze --n-jobs 20 --stages <list> ...
```

**Config B — single R process, heavy internal BLAS** (`lasso`,
`prev_feature_sens`, `phylo_model_sens`, `depth_sens`):

```bash
#$ -pe parallel 20
#$ -l h_vmem=12G
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20
export TMPDIR=/ebio/abt3_scratch/jmarsh/tmp_$JOB_ID
defense-plasmid-analyze --n-jobs 1 --stages <list> ...
```

**Config C — PGLMM fat memory** (`pglmm_mv`):

```bash
#$ -pe parallel 4
#$ -l h_vmem=30G
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20
defense-plasmid-analyze --n-jobs 1 --stages pglmm_mv ...
```

Still requires `pglmm_max_species = 15000` (now the default, not `None`) and
RcppArmadillo built with `ARMA_64BIT_WORD=1` via `~/.R/Makevars` — use `+=` for
`PKG_CPPFLAGS`, not `=`, or the package's own include path is clobbered.

**Config D — phylo_signal is now pure Python.** No R, no BLAS threading, one
core, ~1 h:

```bash
#$ -pe parallel 1
#$ -l h_vmem=16G
defense-plasmid-analyze --n-jobs 1 --stages phylo_signal ...
```

---

## 5. Suggested submission order

```bash
# 0. Plan. Costs nothing, prevents the 25-day surprise.
defense-plasmid-analyze ... --estimate-cost

# 1. Calibration. If this fails, stop — nothing downstream is interpretable.
#    Config A. ~5 h.
defense-plasmid-analyze ... --stages negative_control --n-jobs 20

# 2. Primary. Config A. ~28 h.
defense-plasmid-analyze ... --stages tier1 phyloglm --n-jobs 20

# 3. Everything that scopes off the primary result. Config A. ~25 h.
defense-plasmid-analyze ... --stages pagels loco within_clade_het \
    misclass_mc misclass_analytical defense_misclass depth_match \
    clade_perm sister_pairs feature_control entry_mode rf burden --n-jobs 20

# 4. BLAS-heavy. Config B. ~110 h (lasso dominates).
defense-plasmid-analyze ... --stages lasso depth_sens prev_feature_sens \
    phylo_model_sens --n-jobs 1

# 5. PGLMM. Config C. ~20 h.
defense-plasmid-analyze ... --stages pglmm_mv --n-jobs 1

# 6. Cheap tail. Config D / A.
defense-plasmid-analyze ... --stages phylo_signal consensus \
    phylo_vs_nonphylo figures --n-jobs 4
```

Checkpointing is unchanged except that the cache is now keyed on
`config.fingerprint()`. Changing a threshold and re-running no longer silently
reuses a stale TSV — the cache invalidates and warns. Paths and compute knobs
are excluded from the hash, so switching `--n-jobs` does not force a rerun.

---

## 6. What is still expensive, and why that is correct

**`lasso` ~100 h.** Dominated by sequential `nlme::gls(predictor ~ 1,
correlation = corBrownian)` fits, each a dense Cholesky on the full covariance.
That is inherent to phylogenetic residualisation; BLAS threading already takes
it from ~90 days to ~4 days. Reducible only by subsampling the tree, which
would change the estimand.

**`pagels` ~55 h.** Already subsampled to 500 tips × 10 draws. The Cauchy
combination makes those draws worth more than the old median did, so this is
buying real power.

**`pglmm_mv` ~20 h at 25 GB.** Memory scales as O(N²), so the 15,000-tip cap is
load-bearing. Document it in the methods: the multivariate fit uses a
phylum- and depth-stratified subsample.

Nothing in this list is a stage that gets skipped.

---

## 7. Methods-section consequences

Three of the reductions change what gets reported and should be stated:

- LOCO influence diagnostics are computed at GTDB class for clades with ≥ 50
  species, under the primary covariate model.
- The misclassification Monte Carlo covers FDR-significant systems at four FNR
  levels with 40 replicates; the analytical Bross correction covers all systems
  and the full FNR continuum.
- Pagel directional models are fitted for primary outcomes where the standard
  test rejects trait independence at α = 0.10.

None of these is "we ran out of compute". Each is the scope the question
actually has.

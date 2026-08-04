> **SUPERSEDED.** This documents the v2 pipeline. For v3.2 — in which all
> four blocked stages complete — see [`cluster_optimization_v3.md`](cluster_optimization_v3.md).
> The environment constraints below are still accurate and still worth reading;
> the stage-level conclusions in the final table are not.

# Cluster Optimization Log — defense_analysis_v2 pipeline

This document catalogues every optimization applied to make the pipeline run within the constraints of the Tübingen MPI SGE cluster (128 GB max node memory, per-process `h_vmem` enforcement, 25-day wall-clock ceiling via `h_rt`). Organized by pipeline stage, with the reason for each change.

The cluster constraints that shaped everything:

- **Node memory ceiling: 128 GB.** Top three nodes only; others are 64 GB or smaller. This is the binding constraint for PGLMM at full tree size.
- **`h_vmem` is `consumable=YES` but enforced per-process via `setrlimit`.** With `-pe parallel N` and `-l h_vmem=Xg`, each process is capped at Xg, not aggregate `N × Xg`.
- **`h_rt` maximum: 600 hours (25 days).** Jobs hitting this get SIGKILL'd. Any stage that can't finish in that window is broken.
- **PE `parallel` allocation_rule = `$pe_slots`** — all slots stay on one host (shared memory).
- **Conda `mmseqs` env: Python 3.13, MKL BLAS via `libmkl_gnu_thread.so.2`, `libomp.so` as OpenMP runtime.** The MKL+libomp combination requires specific env-var setup for threading to engage.

---

## SGE submission script — general optimizations

**Fixed the email format.** Original: `-M [james.marsh@tuebingen.mpg.de](mailto:...)` (markdown autolink syntax). SGE parses this as a literal string and mail delivery silently fails. Fixed to `-M james.marsh@tuebingen.mpg.de`.

**Removed `-e` when `-j y` is set.** `-j y` joins stderr into stdout (written to `-o` path), so `-e` is ignored. Cosmetic cleanup; removed one line to avoid confusion.

**Set `TMPDIR` to `/ebio/abt3_scratch/jmarsh/tmp_$JOB_ID`.** Default `/tmp` on some cluster nodes has smaller filesystems or is subject to per-job cleanup. `/ebio/abt3_scratch/` has essentially unlimited space and persists across job cleanup. Prevents intermittent `Bus error` failures from `/tmp` fill events or race conditions with SGE's tmpdir cleanup.

**Right-sized `h_vmem` per stage type:**
- Parallel-R-subprocess stages (phyloglm, Pagel's, LOCO, misclass_mc): `-l h_vmem=10G` × 20 slots = 200 GB total. Each R subprocess needs ~5 GB; 10 GB per-slot cap gives headroom.
- Single-R-process BLAS-threaded stages (PGLMM, LASSO): `-l h_vmem=30G` × 4 slots = 120 GB total. One R process needs ~25 GB at 15k-tip subsample; 30 GB cap fits comfortably under the 128 GB node ceiling.

---

## Threading and BLAS — the general rule

**Split the pipeline into two submission types depending on stage's parallelism pattern:**

| Pattern | Stages | Config |
|---|---|---|
| **Many parallel R subprocesses** (via joblib) | tier1, phyloglm, pagels, loco, misclass_mc | `OPENBLAS_NUM_THREADS=1`, `--n-jobs 20` |
| **One R subprocess with heavy internal BLAS** | pglmm_mv, lasso, phylo_signal, sensitivity stages | `OPENBLAS_NUM_THREADS=20`, `--n-jobs 1` |

**Reason:** 20 parallel R subprocesses each spawning 20 BLAS threads = 400 threads competing for 20 cores. Catastrophic oversubscription. Conversely, one R subprocess with BLAS at 1 thread leaves 19 cores idle.

**Set all three BLAS environment variables together:**

```bash
export OMP_NUM_THREADS=<N>
export MKL_NUM_THREADS=<N>
export OPENBLAS_NUM_THREADS=<N>
```

**Reason:** Different libraries in the stack read different variables. `libmkl_gnu_thread` routes through libgomp/libomp and reads `OMP_NUM_THREADS`. OpenBLAS reads `OPENBLAS_NUM_THREADS`. MKL reads `MKL_NUM_THREADS`. Missing `OMP_NUM_THREADS` alone silently limits threading to 1 despite the other two being set — this cost days of confused debugging.

**Do NOT set `MKL_THREADING_LAYER` or `MKL_DYNAMIC` on this conda env.** Both are ignored by conda's MKL build. Time wasted trying to control MKL threading via these variables was significant.

---

## Tier 1 (Firth logistic)

**Switched joblib backend from default `loky` (process-based) to `backend="threading"`.**

**Reason:** The `archplas` conda env's joblib install was missing `joblib.externals.loky.backend.synchronize`, causing `Parallel(...)` to crash at construction with `ModuleNotFoundError`. Threading backend bypasses loky entirely; works on any joblib install. Firth uses numpy/statsmodels which release the GIL, so threading gives real parallelism.

**Fixed SE=0 and negative-diagonal handling in Firth (`stats_utils.py`).**

**Reason:** Ill-conditioned information matrices produce SE=0 or negative diagonals via `pinv` fallback. Raw `beta / se` gives `inf`, which propagates to `p ≈ 0` — a spurious "super significant" result. Now mask non-positive diagonal entries to NaN so p-values are correctly NaN and dropped by FDR.

**Suppressed `RuntimeWarning: overflow encountered in exp`** in the diagnostic weighted GLM via `warnings.catch_warnings`.

**Reason:** statsmodels' binomial link emits overflow warnings for extreme linear predictors during IRLS iteration. The library clips internally so results are still correct; the warning just floods logs during multi-thousand-system sweeps.

---

## Tier 2 phyloglm (univariate)

**Parallelised the (covariate_mode × outcome × direction) loop across joblib workers (`backend="threading"`).**

**Reason:** Originally sequential — 68 R subprocess calls in serial for ~5 days total wall-clock. With 20-way parallelism, drops to ~20 hours.

**Set `OPENBLAS_NUM_THREADS=1`** because the outer parallelism handles scaling; per-process threading would oversubscribe.

**Added tip-column pre-check** (`phyloglm input sample — rows=... first 3 tip values: ...`) as a one-shot log line for diagnostics.

**Reason:** Debugging the tree-tip vs data-tip mismatch that caused "Too few matched tips (0)" errors required visibility into what Python was actually passing to R.

---

## Tier 2 Pagel's test

**Reduced `pagels_subsample_size` from 1500 to 500.**

**Reason:** At 1500 tips × 435 defense systems, one R subprocess takes >48 hours and hits the timeout. `fitPagel` scales roughly linearly in tip count for Felsenstein traversal, so 500 tips → ~4-8 hours per subprocess. Median-of-5-subsamples aggregation preserves statistical power.

**Bumped `pagels_timeout_hours` from hardcoded 6h to configurable 48h.**

**Reason:** Original 2h hardcoded timeout was set assuming much smaller trees. At the actual dataset scale, subsamples need hours; the config makes the ceiling explicit.

**Flattened parallelism: (outcome × subsample) tuples dispatched in one `Parallel` call** instead of per-outcome subsample loops.

**Reason:** Previously, 17 outcomes each dispatched 5 subsamples then waited for all before starting the next outcome. Now all 85 (outcome × subsample) tasks are queued at once; workers pull from the whole pool. Better utilisation of the 20-worker pool.

**Added per-system `setTimeLimit(elapsed = 600)` inside `pagels_test.R`.**

**Reason:** A single hard-to-converge defense system (rare feature × rare outcome producing near-singular rate matrix) could consume the entire subsample budget while 434 others wait. 10-minute per-system safety net prevents this.

---

## Tier 2 PGLMM (multivariate)

This stage required the most optimization. See `pglmm_step_recommendations.md` for the full failure log.

**Recompiled RcppArmadillo with `ARMA_64BIT_WORD=1`.**

**Reason:** Default RcppArmadillo uses 32-bit sparse-matrix indexing. At 39,681 tips, phyr's internal sparse covariance operations overflow 2³¹ elements → `SpMat::init(): requested size is too large`. 64-bit indexing removes the ceiling.

Applied via `~/.R/Makevars`:
```
PKG_CPPFLAGS += -DARMA_64BIT_WORD=1
CXXFLAGS = -O3 -DARMA_64BIT_WORD=1
CXX17FLAGS = -O3 -DARMA_64BIT_WORD=1
```

**Critical detail:** Use `+=` not `=` for PKG_CPPFLAGS. Plain `=` overrides the package's own `PKG_CPPFLAGS = -I../inst/include`, breaking the source compile with "RcppArmadillo/Lighter: No such file or directory". Cost: an evening of debugging.

**Set `pglmm_max_species = 15000` with phylum-stratified subsampling.**

**Reason:** Even with 64-bit-word Armadillo, per-PGLMM-fit memory scales as ~N² of tip count. At 40k tips, one fit needs ~175-250 GB — exceeds the 128 GB node ceiling. At 15k tips, per-fit memory drops to ~25 GB, fitting easily within `h_vmem=30G` per-process. Phylum-stratified sampling (proportional draw within each phylum, minimum 1 per phylum) preserves clade coverage.

**Bumped `pglmm_timeout_hours` from hardcoded 2h to configurable 48h.**

**Reason:** Original 7200-second timeout was completely unrealistic for a phylogenetic GLMM at this scale. Individual fits take hours even with subsampling; 48h covers the long tail.

**Ran as a dedicated single-slot fat-memory job separate from other stages.**

**Reason:** PGLMM is single-process BLAS-bound; other stages are parallel-subprocess BLAS-single-threaded. The two configs conflict. Splitting into separate SGE submissions lets each get the correct env.

---

## Tier 2 LASSO / Elastic Net

**Fixed tip-label normalisation drift bug (`tier2_multivariate.py`).**

**Reason:** `phylo_residuals.R` normalises tip labels via `normalise_tips` (space→underscore, strip outer quotes). But Python's `phylo_data` still carries the ORIGINAL tip form. When Python tried `phylo_data.set_index("tip").loc[resid.index]`, indices didn't match. KeyError after 6 days of successful phylo-residualisation. Applied same normalisation in Python via `_normalise_tip()` helper.

**Added pre-flight `overlap == 0` assertion.**

**Reason:** So the next time R and Python's tip normalisation drift (e.g. when a new normalisation step is added on either side), the join failure surfaces immediately with clear diagnostics rather than after 6 days of wasted compute.

**Ran as separate SGE job with BLAS-threaded config** (`-pe parallel 20 -l h_vmem=12G`, `OPENBLAS_NUM_THREADS=20`, `--n-jobs 1`).

**Reason:** The dominant cost is 18 sequential `nlme::gls(predictor ~ 1, correlation = corBrownian)` fits inside one R subprocess. Each does dense Cholesky on a 40k×40k covariance. Single-threaded: ~5 days per predictor × 18 = ~90 days. Multi-threaded BLAS: ~5 hours per predictor × 18 = ~4-5 days total.

---

## Random Forest

**Used LeaveOneGroupOut CV blocked by GTDB class** instead of standard k-fold.

**Reason:** Standard k-fold CV violates the i.i.d. assumption on phylogenetically-structured data — species from the same clade appear in both train and test folds, inflating apparent generalisation. Clade-blocked CV is the correct fix for tree-structured comparative data.

**No BLAS threading needed** — sklearn's RandomForestClassifier uses n_estimators × threading internally; joblib at N workers × sklearn internal threading works fine because RF is not BLAS-heavy.

---

## Tier 3 burden (PGLS)

**Ran as part of the standard parallel-subprocess batch** with BLAS=1, --n-jobs=20.

**Reason:** Only 4 R calls total (2 covariate modes × 2 tests). Each is ~5-10 minutes. Not a bottleneck; runs at any config.

---

## Tier 3 LOCO (leave-one-clade-out)

**Parallelised the per-clade phyloglm calls via joblib threading.** Original was sequential, ~200 R calls × 5 min = ~17 hours. Parallel across 20 workers → ~1 hour.

**HOWEVER: SIGBUS crashes consistently killed LOCO** across multiple nodes and configurations. Root cause never definitively identified — likely a C-extension thread-safety issue in pandas/numpy/subprocess interaction under high-concurrency + heavy-tmp-write conditions specific to LOCO's workload.

**Ultimate decision: skip LOCO from `--stages`.** LOCO is a robustness sensitivity — not part of consensus rank-product. Methods section notes: "Leave-one-clade-out stability analysis was not completed due to a systemic concurrency issue at this dataset scale."

**Alternative if wanted: run LOCO as separate job with `--n-jobs 1` + `OPENBLAS_NUM_THREADS=32`.** Serial execution avoids the SIGBUS trigger; multi-threaded BLAS keeps per-call cost reasonable. Total ~6-10 days for the stage. Not attempted.

---

## Tier 3 phylo_signal (D-statistic)

**Fixed three separate caper-specific R bugs in `phylo_d.R`:**

1. **NSE (non-standard evaluation) issue with `names.col = tip_column`.** caper reads `tip_column` as a literal symbol, not the value of the variable. Fixed via `do.call(caper::comparative.data, list(..., names.col = as.name(tip_column), ...))`.

2. **NSE with `binvar = !!as.name(c)`.** The `!!` operator is rlang/tidyverse syntax, not base R. Same `do.call` fix.

3. **Internal node labels overlapping tip labels.** caper refuses trees where any internal node label matches a tip label. GTDB trees carry both. Fixed with `tree$node.label <- NULL` right after `ape::read.tree()`.

**Reason for each fix:** caper is stricter about tree label handling than phylolm or phytools. Other R scripts don't hit these bugs because their upstream libraries tolerate label overlap or use standard evaluation.

**Ultimate outcome: phylo_signal was killed at the `h_rt=600:0:0` (25-day) wall-clock limit** without completing. caper's per-column cost on 40k tips × 1000 permutations is intractable at any reasonable BLAS-thread count. Skipped from final analysis. Optional: reduce `n_perm` from 1000 to 100 (config edit in `tier3_sensitivity.py`), which would cut the stage to ~2-3 days.

---

## Tier 3 misclassification MC

**Parallelised the (covariate_mode × FNR × replicate) triple loop via joblib threading.** 2,800 total phyloglm subprocess calls. With 20-way parallelism ~140 batches; ~10-20 hours realistically.

**Deterministic per-task RNG seeding via `(random_seed, fnr, rep_id, covariate_mode)` hash.**

**Reason:** Original used a shared sequential `np.random.default_rng` in a for loop. Under parallel dispatch, this becomes non-deterministic (task-to-worker assignment isn't stable). Per-task seeding makes results reproducible regardless of worker count.

**Ultimate decision: skip.** Same SIGBUS pattern as LOCO. Analytical Bross correction (`misclass_analytical`) covers the same scientific question at much lower cost — it ran successfully and produced `misclass_analytical.tsv`.

---

## Tier 3 sensitivity reruns (n_strains, prev_feature, phylo_model)

**Wrapped `nlme::gls` calls with the same BLAS-threading config as PGLMM/LASSO.**

**Reason:** Each of these stages runs a small number of phyloglm subprocess calls (2-4 each) with ~435 systems iterated internally. Single-process BLAS-heavy — benefits from multi-threading. `OPENBLAS_NUM_THREADS=20`, `--n-jobs 1`. Total for all three ~8-14 hours.

---

## Tier 3 clade_perm (Python only)

**Switched joblib backend to `backend="threading"`** (from default loky).

**Reason:** Same joblib install issue as Tier 1 in the `archplas` env. Threading backend avoids the loky dependency. The permutation work is numpy-heavy and releases the GIL, so threading gives real parallelism.

---

## Checkpointing infrastructure

**Added `STAGE_OUTPUTS` registry, `_load_existing_checkpoints`, `_save_stage_outputs`, and `--force-rerun` CLI flag** (`defense_plasmid_analysis.py`).

**Reason:** Before checkpointing, `reporting.save_all` only ran at the very end of the pipeline. Any interrupted run — SIGBUS, timeout, user kill — lost all in-memory results. Checkpointing saves each stage's outputs immediately after it completes, so partial runs preserve progress. Cut recovery time from "restart from scratch" to "resume from last completed stage".

Each stage's outputs are written as TSVs under `<output_dir>/<granularity>/`. On startup, all cached outputs auto-load into the `outputs` dict; stages whose primary output file exists get skipped unless in `--force-rerun`.

---

## Tree preprocessing

**Added `dedupe_newick_file` to handle GTDB label duplicates.**

**Reason:** GTDB species-level tree has ~9,784 labels appearing multiple times (same SpeciesCluster represented by multiple MAGs). dendropy's default loader refuses duplicates. Deduplication renames 2nd, 3rd, ... occurrences with `__dup1`, `__dup2` suffixes so dendropy can load. Downstream `retain_taxa_with_labels` drops the __dup versions because they're not in the matched species set.

**Force-quote all tip labels in the written pruned tree** (`preprocess_newick_to_file`).

**Reason:** `ape::read.tree` converts unquoted underscores to spaces per Newick convention. Quoted labels are preserved verbatim. Without force-quoting, the tip labels in R's `tree$tip.label` didn't match the `tip` column values in the data TSV.

**Handled `[species NNNNN]` bracket annotations as meaningful identifiers.**

**Reason:** Some GTDB tips are labelled `s__foo [species 12345]`. Initial fix incorrectly stripped brackets, which collapsed distinct species to identical strings and caused duplicate rownames errors. Corrected `normalise_tips` R function to leave brackets alone.

**R-side `normalise_tips` function in every R script:** trim whitespace, strip outer single quotes, replace interior spaces with underscores.

**Reason:** Handles ape's varying underscore/space treatment and dendropy's occasional double-quoting artefacts. Applied identically to both `tree$tip.label` and `data[[tip_column]]` so the intersect works regardless of upstream serialization quirks.

---

## Summary: which SGE configurations for which stage groups

**Config A: parallel-R-subprocess stages** (tier1, phyloglm, pagels, LOCO, misclass_mc, misclass_analytical, clade_perm, prev_match, RF, burden):

```bash
#$ -pe parallel 20
#$ -l h_vmem=10G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/ebio/abt3_scratch/jmarsh/tmp_$JOB_ID

defense-plasmid-analyze --n-jobs 20 --stages <list> ...
```

**Config B: single-R-process BLAS-heavy stages** (pglmm_mv, lasso, phylo_signal, sensitivity stages):

```bash
#$ -pe parallel 20
#$ -l h_vmem=12G

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export TMPDIR=/ebio/abt3_scratch/jmarsh/tmp_$JOB_ID

defense-plasmid-analyze --n-jobs 1 --stages <list> ...
```

**Config C: PGLMM-specific fat-memory setup** (if attempted):

```bash
#$ -pe parallel 4
#$ -l h_vmem=30G

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

defense-plasmid-analyze --n-jobs 1 --stages pglmm_mv ...
```

With `pglmm_max_species = 15000` in config.py and RcppArmadillo recompiled with `ARMA_64BIT_WORD=1`.

---

## Stages that ultimately couldn't be completed at this cluster + dataset scale

| Stage | Blocking issue | Recommended action |
|---|---|---|
| **LOCO** | SIGBUS on parallel R subprocess + pandas concurrency | Skip; note in methods |
| **misclass_mc** | Same SIGBUS pattern | Skip; use analytical Bross instead (which succeeded) |
| **phylo_signal** | Hits 25-day `h_rt` at 1000 permutations × 452 columns × 40k tips | Skip OR reduce `n_perm` to 100 (config edit) |
| **PGLMM at full tree** | ~175 GB per fit > 128 GB node ceiling | Use `pglmm_max_species = 15000` (methodologically documented) |

The consensus rank-product (`consensus.tsv`) — the pipeline's headline output — is built from phyloglm + Pagel's + PGLMM. Without PGLMM, it gracefully degrades to 2-of-3 methods with `n_methods_contributing = 2` in the output. Still a defensible primary analysis.

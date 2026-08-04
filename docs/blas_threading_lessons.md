# BLAS / OpenMP threading — per-stage configuration

## The rule

BLAS threading and joblib parallelism cannot both be high simultaneously. Each stage needs one or the other:

| Stage type | Parallelism source | Required env settings |
|---|---|---|
| **Parallel-R-subprocess stages** (phyloglm, pagels, misclass_mc, loco, etc.) | joblib spawns N R subprocesses | `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `--n-jobs N` |
| **Single-R-process stages** (pglmm_mv, lasso) | one R process does big BLAS calls | `OPENBLAS_NUM_THREADS=N`, `OMP_NUM_THREADS=N`, `MKL_NUM_THREADS=N`, `--n-jobs 1` |

Why: 20 parallel R subprocesses each spawning 20 BLAS threads = 400 threads on 20 cores. Oversubscription catastrophe. Conversely, a single R process with BLAS at 1 thread leaves 19 cores idle. The right config depends on which stage is running.

**Implication: split the pipeline into two SGE jobs** if running both kinds of stage. Or accept that one type runs sub-optimally if you keep them in one job.

## What worked at Tübingen MPI

For single-R-process stages (PGLMM, LASSO phylo-residualization):

```bash
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
# Don't set MKL_THREADING_LAYER — conda's MKL ignores it
# Don't set MKL_DYNAMIC=FALSE — also ignored in this env
```

The pipeline's R is linked against Intel MKL (loaded via `libmkl_rt.so.2`), which on this conda env routes through `libmkl_gnu_thread.so.2` regardless of `MKL_THREADING_LAYER`. The actual OpenMP runtime that powers the threading is `libomp.so` (LLVM/Intel's). Despite the symbol-mismatch concern (MKL's gnu_thread variant nominally expects libgomp, but libomp from recent conda-forge builds includes GNU ABI compatibility), threading engages correctly without preloading libgomp.

For parallel-R-subprocess stages:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

`--n-jobs 20` (or whatever the slot count is) controls parallelism via joblib spawning R subprocesses.

## What didn't work

- **`MKL_THREADING_LAYER=INTEL`** — conda's MKL build ignores it. Stays on GNU thread layer.
- **`MKL_DYNAMIC=FALSE` alone** — doesn't engage threading on its own; necessary but not sufficient.
- **`LD_PRELOAD=/path/to/libgomp.so.1`** — wasn't needed. libomp's GNU ABI compatibility layer handled it.
- **`KMP_DUPLICATE_LIB_OK=True`** — already set in the conda env; not a fix on its own.
- **Single high-memory slot (`-pe parallel 1 -l h_vmem=256G`)** — SGE granted 64 CPUs of affinity but only one slot of accounting. The single R process saw `Cpus_allowed_list: 0-63` but BLAS still went single-threaded because of missing OMP env var. Even after fixing the env, you only get one process which uses CPU well but won't parallelise across cores for tasks where R-level glue dominates (PGLMM PQL iterations).

## Verifying threading is engaged

Three independent checks. Use all three when you doubt.

**1. Mathematical proof from cumulative CPU time:**

```bash
ps aux | grep '[R]' | awk '$5 > 1000000 {print}'
```

The `%CPU` column shows cumulative CPU usage as percentage. Above 100% means the process accumulated more CPU-seconds than wall-seconds — physically impossible without multi-threading. 1700% = ~17× average parallelism.

**2. Per-thread snapshot:**

```bash
top -H -p <PID> -bn1 | head -30
```

Shows individual OS threads as rows. A single-threaded process has 1 row in state `R`; a 20-threaded BLAS process shows ~13-18 rows in `R` and the rest in `S` (parked at OpenMP barriers waiting for the next parallel region).

**3. Thread count:**

```bash
ls /proc/<PID>/task/ | wc -l
```

Returns total OS threads. For a multi-threaded R process at `OPENBLAS_NUM_THREADS=20`, expect ~20-25 (the BLAS workers plus libR housekeeping). Single-threaded: ~3-5.

**Most precise: `pidstat -t`:**

```bash
pidstat -t -p <PID> 2 3
```

Three 2-second intervals showing per-thread CPU usage. Sustained non-zero `%CPU` on many threads = active parallel work. Threads with 0% are either parked between parallel regions or genuinely idle.

## Stage-specific configuration reference

| Stage | Threading | Slots | h_vmem per slot | n-jobs |
|---|---|---|---|---|
| tier1 | parallel subprocesses (joblib) | 20 | 10G | 20 |
| phyloglm | parallel R subprocesses | 20 | 10G | 20 |
| pagels | parallel R subprocesses | 20 | 10G | 20 |
| pglmm_mv | single R, multi-threaded BLAS | 20 | 12G (total 240G needed) | 1 |
| lasso | single R, multi-threaded BLAS | 20 | 12G (total 240G needed) | 1 |
| rf | parallel via sklearn (joblib) | 20 | 8G | 20 |
| burden | parallel | 20 | 8G | 20 |
| loco | parallel | 20 | 10G | 20 |
| misclass_mc | parallel (heaviest stage by task count) | 20 | 10G | 20 |
| consensus, figures, etc | trivially fast | 4 | 8G | 4 |

## Recommended SGE submission pattern

Two separate jobs, because the threading config can't be set differently mid-job.

**Job A: parallel-subprocess stages (BLAS=1, joblib=20).**

```bash
#$ -pe parallel 20
#$ -l h_vmem=10G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

defense-plasmid-analyze \
  --granularity both --n-jobs 20 \
  --stages tier1 phyloglm pagels rf burden loco phylo_signal clade_perm \
           prev_match misclass_mc misclass_analytical min_n_strains_sens \
           prev_feature_sens phylo_model_sens consensus phylo_vs_nonphylo figures \
  --output-dir /ebio/abt3_projects2/Gut_genetics2/data/defensefinder/plasmid_vs_defense_v2
```

**Job B: single-process BLAS-threaded stages.**

```bash
#$ -pe parallel 20
#$ -l h_vmem=12G

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

defense-plasmid-analyze \
  --granularity both --n-jobs 1 \
  --stages pglmm_mv lasso \
  --output-dir /ebio/abt3_projects2/Gut_genetics2/data/defensefinder/plasmid_vs_defense_v2
```

Then a final merge:

```bash
defense-plasmid-analyze \
  --stages consensus phylo_vs_nonphylo figures \
  --force-rerun consensus phylo_vs_nonphylo figures \
  --granularity both --n-jobs 4
```

Cached outputs from both Job A and Job B feed into the consensus stage.

## Default flag in pyproject.toml: `--n-jobs`

The CLI flag is `--n-jobs`, not `--threads` or `--n-cores`. It controls joblib's `n_jobs` parameter for parallel-subprocess stages. It does NOT control BLAS threads inside a single R subprocess — that's controlled by environment variables read at R startup.

For LASSO specifically, `--n-jobs` has no functional effect because the stage doesn't use joblib internally. Set it to 1 for clarity.

## Diagnostic libraries to confirm

```bash
# Which BLAS library is loaded?
cat /proc/<PID>/maps | grep -iE 'mkl|libopenblas|libblas' | awk '{print $NF}' | sort -u

# Which threading variant of MKL?
cat /proc/<PID>/maps | grep -iE 'mkl_(gnu_thread|intel_thread|sequential)' | awk '{print $NF}' | sort -u

# Which OpenMP runtime?
cat /proc/<PID>/maps | grep -iE 'libgomp|libomp|libiomp' | awk '{print $NF}' | sort -u

# What env vars does the running process actually have?
cat /proc/<PID>/environ | tr '\0' '\n' | grep -E 'OMP|MKL|BLAS|KMP'
```

In a working multi-threaded PGLMM/LASSO run, expect: `libmkl_rt.so.2` + `libmkl_gnu_thread.so.2` + `libomp.so` + all the OMP/MKL env vars set to N.

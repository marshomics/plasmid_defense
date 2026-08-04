# Memory and reference docs for defense_analysis_v2

These were written as memory entries for future Claude sessions but are stored here because the session's memory directory wasn't writable from the sandbox. Treat them as long-form reference docs; copy into `~/Library/Application Support/Claude/.../memory/` manually if you want them auto-loaded in future conversations.

## Documents

- **[pipeline_overview.md](pipeline_overview.md)** — Scientific goal, data structure, codebase layout, headline interpretation of the consensus output. Read this first if returning to the project after a gap.

- **[pipeline_stages.md](pipeline_stages.md)** — Every stage explained: what it does, what it produces, how it contributes to the headline question. Reference when planning partial reruns or interpreting a specific TSV.

- **[blas_threading_lessons.md](blas_threading_lessons.md)** — Threading configuration per stage. What worked at Tübingen MPI, what didn't, and the per-stage env-var matrix. Read before submitting any new SGE job for this pipeline.

- **[pglmm_step_recommendations.md](pglmm_step_recommendations.md)** — Every PGLMM failure encountered (RcppArmadillo overflow, SIGKILL, timeouts, convergence failures, single-threaded BLAS) and the configuration that ultimately worked. Read before re-running the PGLMM stage from scratch.

## Headline conclusions from the work so far

1. The pipeline is scientifically defensible. Full review in `../defense_analysis_v2_review.md`.

2. The pipeline produces results without PGLMM. Phyloglm + Pagel's consensus is enough; PGLMM is a third opinion that strengthens but isn't required for a publication.

3. Full-tree PGLMM is not feasible on this cluster. RcppArmadillo's 32-bit sparse index limit (~46k tips) AND per-process memory ceiling (~128 GB) both block it. Use `pglmm_max_species = 15000` with phylum-stratified subsampling; methodologically defensible if documented.

4. BLAS threading and joblib parallelism are mutually exclusive. Single-R-process stages (PGLMM, LASSO) want BLAS=N, n-jobs=1. Many-R-process stages (phyloglm, misclass MC) want BLAS=1, n-jobs=N. Split into separate SGE jobs.

5. Tübingen MPI's MKL+libomp stack threads correctly with `OMP_NUM_THREADS=N` + `OPENBLAS_NUM_THREADS=N` + `MKL_NUM_THREADS=N`. Don't bother with `MKL_THREADING_LAYER` or `MKL_DYNAMIC` or `LD_PRELOAD` — they're ignored on this conda env.

6. Stage-level checkpointing is implemented. After each stage finishes, its output is written to a TSV in the granularity-level output directory. Subsequent runs reload them via `_load_existing_checkpoints` and skip the stage. Use `--force-rerun <stage>` to override.

## Cluster facts (Tübingen MPI)

- Login host: `chimi` (also `morty` for some sessions)
- Compute node max memory: 128 GB
- Top three nodes: `node526`, `node525`, `node524` (each 128 GB)
- Other nodes: 64 GB or less
- SGE `h_vmem` is `consumable=YES` but enforced per-process via `setrlimit`
- PE `parallel` has `allocation_rule = $pe_slots` — slots stay on one host (shared memory)
- No `smp` or `openmp` PE available; `parallel` is the only single-host parallel option
- R 4.5.2 in the `mmseqs` conda env
- MKL 2.x with `libmkl_gnu_thread.so.2` as the loaded threading variant
- libomp.so (LLVM OpenMP) is the OMP runtime; libgomp also available

## Common diagnostic commands

```bash
# Find your R process on a compute node
qstat -u $USER                                              # find node + jobid
ssh <node>
ps aux | grep '[R]' | awk '$5 > 1000000 {print}'           # find R process

# Confirm threading is engaged
PID=<your_R_pid>
ls /proc/$PID/task/ | wc -l                                # thread count
top -H -p $PID -bn1 | head -30                             # per-thread breakdown
pidstat -t -p $PID 2 3                                     # per-thread over 6s
top -bn1 | grep ' R '                                      # one-shot summary

# Confirm BLAS configuration
cat /proc/$PID/environ | tr '\0' '\n' | grep -E 'OMP|MKL|BLAS|KMP'
cat /proc/$PID/maps | grep -iE 'mkl|libgomp|libomp' | awk '{print $NF}' | sort -u

# Watch pipeline progress
tail -f /ebio/abt3_projects2/Gut_genetics2/data/defensefinder/plasmid_vs_defense_v2/subtype_level/log.txt
grep -E "PGLMM.*last fit|phyloglm.*systems fit|checkpointed" \
  /ebio/abt3_projects2/Gut_genetics2/data/defensefinder/plasmid_vs_defense_v2/subtype_level/log.txt | tail -20
```

- [cluster_optimization_v3.md](cluster_optimization_v3.md) — current cluster guidance; how the four previously-blocked stages were brought inside the envelope, and the `--estimate-cost` planner.

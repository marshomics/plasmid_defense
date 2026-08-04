#!/bin/bash
# Submit the complete defense_analysis_v2 v3.2 pipeline to the SGE cluster.
#
#   ./run_pipeline_sge.sh                 # both granularities, full pipeline
#   ./run_pipeline_sge.sh subtype_level   # one granularity only
#   ./run_pipeline_sge.sh both --dry-run  # print the qsub calls, submit nothing
#
# WHY THIS IS NOT ONE COMMAND
# ---------------------------
# `defense-plasmid-analyze` with no --stages does run everything, and that is
# the right call on a workstation. On the cluster it is wrong, because the
# stages need CONFLICTING threading configurations:
#
#   * Stages that spawn many parallel R subprocesses need BLAS pinned to 1
#     thread. Otherwise 20 workers x 20 BLAS threads = 400 threads on 20 cores.
#   * Stages that run ONE R process with heavy internal BLAS need the opposite:
#     BLAS at 20 threads and joblib at 1, or 19 cores sit idle.
#
# A single submission has to pick one, and either choice cripples half the
# pipeline. So the work is split into four jobs with the correct environment
# each, chained by -hold_jid.
#
# Critical-path wall-clock is ~150 h (~6.5 days) per granularity; the two
# granularities run concurrently. Run --estimate-cost first.

set -euo pipefail

GRANULARITY="${1:-both}"
DRY_RUN="${2:-}"

# ---------------------------------------------------------------- paths ----
# These match config.py defaults; override here if your layout differs.
INPUT_SUBTYPE="/ebio/abt3_projects2/Gut_genetics2/data/defensefinder/all_combined/defense_finder_human_animal_free_combined_reshaped_nodefenseincluded_hasplasmid.txt"
INPUT_TYPE="/ebio/abt3_projects2/Gut_genetics2/data/defensefinder/all_combined/defense_finder_human_animal_free_combined_reshaped_type_nodefenseincluded_hasplasmid.txt"
TREE="/ebio/abt3_scratch/jmarsh/tract_score3/gtdb_custom_trees/human_animal_free_90percent_species_level/output/gtdbtk.rooted.speciesnames.tree"
OUTDIR="/ebio/abt3_projects2/Gut_genetics2/data/defensefinder/plasmid_vs_defense_v3"
LOGDIR="${OUTDIR}/sge_logs"
EMAIL="james.marsh@tuebingen.mpg.de"

# Conda environment to activate inside each job.
#
# Defaults to whatever environment you are IN when you run this script, which
# is almost always the right answer and avoids a whole class of silent
# failure: hard-coding a name that does not match your actual environment
# means every job activates the wrong interpreter (or fails to activate at
# all) and either dies immediately or runs a different install of the package.
# Override by exporting CONDA_ENV before calling.
CONDA_ENV="${CONDA_ENV:-${CONDA_PREFIX:-}}"
if [[ -z "${CONDA_ENV}" ]]; then
  echo "ERROR: no conda environment detected. Activate the environment that" >&2
  echo "has defense-plasmid-analyze installed, or export CONDA_ENV=<name|path>." >&2
  exit 1
fi

[[ "${DRY_RUN}" != "--dry-run" ]] && mkdir -p "${LOGDIR}"
true

# ------------------------------------------------------------ stage sets ----
# Order within a job follows DEFAULT_STAGES, so dependencies inside a job are
# already satisfied (e.g. misclass_mc scopes itself off tier2_phyloglm, which
# runs earlier in the same job).

# Job 1 — calibration + primary. Everything else is scoped off these.
STAGES_CORE="negative_control tier1 phyloglm"

# Job 2 — parallel-R-subprocess stages that depend on the primary result.
STAGES_PARALLEL="pagels loco within_clade_het clade_perm depth_match \
misclass_mc misclass_analytical defense_misclass sister_pairs \
feature_control entry_mode rf burden"

# Job 3 — one R process, heavy internal BLAS.
STAGES_BLAS="lasso depth_sens prev_feature_sens phylo_model_sens"

# Job 4 — PGLMM, fat memory, O(N^2).
STAGES_PGLMM="pglmm_mv"

# Job 5 — pure Python + aggregation. phylo_signal is now native (no R).
STAGES_FINAL="phylo_signal consensus phylo_vs_nonphylo figures"

# --------------------------------------------------------------- helper ----
submit () {
  local name="$1" gran="$2" stages="$3" slots="$4" vmem="$5" threads="$6" \
        njobs="$7" hold="$8"
  local holdopt=""
  [[ -n "${hold}" ]] && holdopt="#\$ -hold_jid ${hold}"

  local script
  script=$(cat <<EOF
#!/bin/bash
#\$ -N ${name}
#\$ -o ${LOGDIR}/${name}.log
#\$ -j y
#\$ -cwd
#\$ -V
#\$ -pe parallel ${slots}
#\$ -l h_vmem=${vmem}
#\$ -l h_rt=590:0:0
#\$ -M ${EMAIL}
#\$ -m ea
${holdopt}

set -euo pipefail

# All three must be set together. Different libraries in the stack read
# different variables — libmkl_gnu_thread routes through libgomp and reads
# OMP_NUM_THREADS, OpenBLAS reads OPENBLAS_NUM_THREADS, MKL reads
# MKL_NUM_THREADS. A missing OMP_NUM_THREADS silently pins threading to 1
# regardless of the other two.
export OMP_NUM_THREADS=${threads}
export MKL_NUM_THREADS=${threads}
export OPENBLAS_NUM_THREADS=${threads}

# Not /tmp. Per-job cleanup races and fill events on /tmp produced
# intermittent bus errors; scratch is effectively unlimited and persists.
export TMPDIR=/ebio/abt3_scratch/jmarsh/tmp_\${JOB_ID}
mkdir -p "\${TMPDIR}"
trap 'rm -rf "\${TMPDIR}"' EXIT

# CONDA_ENV may be a full prefix path (from CONDA_PREFIX) or a bare name.
if [[ -d "${CONDA_ENV}" ]]; then
  source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"
else
  source activate ${CONDA_ENV} 2>/dev/null || conda activate ${CONDA_ENV}
fi

# The console script must be the v3.2 build. A stale install runs the old
# pipeline and the output looks entirely normal, so fail loudly here rather
# than 6 days later.
if ! defense-plasmid-analyze --help 2>&1 | grep -q negative_control; then
  echo "FATAL: defense-plasmid-analyze in this environment is STALE." >&2
  echo "Run 'pip install -e <repo root>' and resubmit." >&2
  exit 1
fi

defense-plasmid-analyze \\
  --input       "${INPUT_SUBTYPE}" \\
  --input-type  "${INPUT_TYPE}" \\
  --tree        "${TREE}" \\
  --output-dir  "${OUTDIR}" \\
  --granularity ${gran} \\
  --stages ${stages} \\
  --n-jobs ${njobs}
EOF
)

  if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "=== ${name}  (slots=${slots} vmem=${vmem} threads=${threads} n_jobs=${njobs} hold=${hold:-none}) ==="
    echo "    stages: ${stages}"
  else
    echo "${script}" | qsub
  fi
}

# ----------------------------------------------------------------- main ----
GRANS=()
case "${GRANULARITY}" in
  both)          GRANS=(subtype_level type_level) ;;
  subtype_level) GRANS=(subtype_level) ;;
  type_level)    GRANS=(type_level) ;;
  *) echo "usage: $0 [both|subtype_level|type_level] [--dry-run]" >&2; exit 1 ;;
esac

# The jobs below invoke the CONSOLE SCRIPT, not this checkout. If the
# installed copy is stale, every job silently runs the old pipeline and the
# results look entirely normal. Verify before submitting.
if [[ "${DRY_RUN}" != "--dry-run" ]]; then
  if ! command -v defense-plasmid-analyze >/dev/null; then
    echo "ERROR: defense-plasmid-analyze is not on PATH. Run 'pip install -e .'" >&2
    exit 1
  fi
  if ! defense-plasmid-analyze --help 2>&1 | grep -q negative_control; then
    echo "ERROR: the installed defense-plasmid-analyze is STALE — it does not" >&2
    echo "know the 'negative_control' stage. Run 'pip install -e .' from the" >&2
    echo "repository root, then re-run this script." >&2
    exit 1
  fi
fi

echo "Output directory: ${OUTDIR}"
echo "Conda environment: ${CONDA_ENV}"
echo "Granularities:    ${GRANS[*]}"
echo

for G in "${GRANS[@]}"; do
  TAG="${G%%_level}"

  # 1. Calibration + primary. ~33 h.
  #    If the negative control fails, STOP — nothing downstream is
  #    interpretable. Check ${OUTDIR}/${G}/negative_control.tsv before
  #    trusting anything the later jobs produce.
  submit "dp_${TAG}_core" "${G}" "${STAGES_CORE}" 20 10G 1 20 ""

  # 2. Parallel-R-subprocess stages. ~86 h. Holds on core because several
  #    stages scope themselves off the primary phyloglm result.
  submit "dp_${TAG}_par" "${G}" "${STAGES_PARALLEL}" 20 10G 1 20 \
         "dp_${TAG}_core"

  # 3. BLAS-heavy single-process stages. ~116 h — this is the critical path,
  #    dominated by lasso's sequential nlme::gls per predictor.
  submit "dp_${TAG}_blas" "${G}" "${STAGES_BLAS}" 20 12G 20 1 \
         "dp_${TAG}_core"

  # 4. PGLMM. ~20 h at 25 GB. Needs pglmm_max_species=15000 (the default) and
  #    RcppArmadillo built with ARMA_64BIT_WORD=1.
  submit "dp_${TAG}_pglmm" "${G}" "${STAGES_PGLMM}" 4 30G 20 1 \
         "dp_${TAG}_core"

  # 5. Aggregation. ~3 h. phylo_signal is native Python now, so no R and no
  #    BLAS threading needed.
  submit "dp_${TAG}_final" "${G}" "${STAGES_FINAL}" 4 16G 4 4 \
         "dp_${TAG}_par,dp_${TAG}_blas,dp_${TAG}_pglmm"
done

echo
if [[ "${DRY_RUN}" == "--dry-run" ]]; then
  echo "DRY RUN — nothing was submitted. Re-run without --dry-run to submit."
  exit 0
fi

echo "Submitted. Monitor with: qstat -u \$USER"
echo "Logs:                    ${LOGDIR}/"
echo
echo "READ FIRST when the core job finishes:"
echo "  ${OUTDIR}/<granularity>/negative_control.tsv   (column: calibrated)"
echo "If calibrated == False, stop and raise config.depth_spline_df before"
echo "interpreting anything else."

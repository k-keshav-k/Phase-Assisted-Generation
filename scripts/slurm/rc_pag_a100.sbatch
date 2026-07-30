#!/bin/bash
#SBATCH --job-name=rc_pag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/rc_pag/%j_%x.out
#SBATCH --error=logs/rc_pag/%j_%x.err

set -uo pipefail

: "${PROJECT_DIR:?PROJECT_DIR must be exported by submit_rc_pag.sh}"
: "${OVERLAY_PATH:?OVERLAY_PATH must be exported by submit_rc_pag.sh}"
: "${SIF_PATH:?SIF_PATH must be exported by submit_rc_pag.sh}"
: "${RC_PAG_STAGE:?RC_PAG_STAGE must be set}"
: "${RC_PAG_CONFIG:?RC_PAG_CONFIG must be set}"
: "${RC_PAG_OUTPUT_ROOT:?RC_PAG_OUTPUT_ROOT must be set}"

RC_PAG_SCRATCH_ROOT="${RC_PAG_SCRATCH_ROOT:-/scratch/${USER}/rc_pag}"
mkdir -p "${PROJECT_DIR}/logs/rc_pag" "${RC_PAG_SCRATCH_ROOT}"

export HF_HOME="${HF_HOME:-${RC_PAG_SCRATCH_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RC_PAG_SCRATCH_ROOT}/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RC_PAG_SCRATCH_ROOT}/uv-cache}"
export TMPDIR="${TMPDIR:-${RC_PAG_SCRATCH_ROOT}/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${UV_CACHE_DIR}" "${TMPDIR}"

child_pid=""
requeue_requested=0

checkpoint_and_requeue() {
    requeue_requested=1
    if [[ -n "${child_pid}" ]]; then
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
}

trap checkpoint_and_requeue USR1 TERM

singularity exec --nv \
    --overlay "${OVERLAY_PATH}:rw" \
    "${SIF_PATH}" \
    /bin/bash -lc '
set -euo pipefail
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
conda activate "${RC_PAG_CONDA_ENV:-llmr}"
cd "$PROJECT_DIR"
uv sync --group phase_cpd_dream --python "${RC_PAG_PYTHON:-3.11}"

args=(
    "$RC_PAG_STAGE"
    --config "$RC_PAG_CONFIG"
    --output-root "$RC_PAG_OUTPUT_ROOT"
    --device cuda
    --resume
)
if [[ "${RC_PAG_ALLOW_CONFIRMATORY:-0}" == "1" ]]; then
    args+=(--allow-confirmatory)
fi
if [[ -n "${RC_PAG_LIMIT:-}" ]]; then
    args+=(--limit "$RC_PAG_LIMIT")
fi
exec uv run --group phase_cpd_dream python scripts/run_rc_pag.py "${args[@]}"
' &
child_pid=$!
set +e
wait "${child_pid}"
status=$?
set -e

if [[ "${requeue_requested}" == "1" ]]; then
    echo "Preemption signal received; completed records are atomic. Requeuing ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

exit "${status}"

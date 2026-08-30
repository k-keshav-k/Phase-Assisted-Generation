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

RC_PAG_MAX_CUDA_REQUEUES="${RC_PAG_MAX_CUDA_REQUEUES:-2}"
if [[ ! "${RC_PAG_MAX_CUDA_REQUEUES}" =~ ^[0-9]+$ ]]; then
    echo "RC_PAG_MAX_CUDA_REQUEUES must be a non-negative integer" >&2
    exit 2
fi

RC_PAG_SCRATCH_ROOT="${RC_PAG_SCRATCH_ROOT:-/scratch/${USER}/rc_pag}"
mkdir -p "${PROJECT_DIR}/logs/rc_pag" "${RC_PAG_SCRATCH_ROOT}"

export HF_HOME="${HF_HOME:-${RC_PAG_SCRATCH_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RC_PAG_SCRATCH_ROOT}/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RC_PAG_SCRATCH_ROOT}/uv-cache}"
export TMPDIR="${TMPDIR:-${RC_PAG_SCRATCH_ROOT}/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${UV_CACHE_DIR}" "${TMPDIR}"

# Prefer cached artifacts, fetch only missing pinned revisions, then run offline.
# Keep RC_PAG_HF_OFFLINE as a backwards-compatible override when the new mode
# was not set explicitly.
if [[ -z "${RC_PAG_HF_MODE:-}" && -n "${RC_PAG_HF_OFFLINE:-}" ]]; then
    case "${RC_PAG_HF_OFFLINE,,}" in
        1|true|yes) RC_PAG_HF_MODE="offline" ;;
        0|false|no) RC_PAG_HF_MODE="online" ;;
        *)
            echo "RC_PAG_HF_OFFLINE must be a boolean value" >&2
            exit 2
            ;;
    esac
else
    RC_PAG_HF_MODE="${RC_PAG_HF_MODE:-auto}"
    RC_PAG_HF_MODE="${RC_PAG_HF_MODE,,}"
fi
case "${RC_PAG_HF_MODE}" in
    auto|offline|online) ;;
    *)
        echo "RC_PAG_HF_MODE must be auto, offline, or online" >&2
        exit 2
        ;;
esac
export RC_PAG_HF_MODE

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

uv run --group phase_cpd_dream python scripts/check_rc_pag_cuda.py

uv run --group phase_cpd_dream python scripts/bootstrap_rc_pag_hf.py \
    --config "$RC_PAG_CONFIG" \
    --mode "$RC_PAG_HF_MODE"

if [[ "$RC_PAG_HF_MODE" == "online" ]]; then
    export HF_HUB_OFFLINE=0
    export HF_DATASETS_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
else
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

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
if [[ -n "${RC_PAG_REUSE_FROM:-}" ]]; then
    args+=(--reuse-development-from "$RC_PAG_REUSE_FROM")
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

if [[ "${status}" -eq 75 ]]; then
    restart_count="${SLURM_RESTART_COUNT:-0}"
    if (( restart_count < RC_PAG_MAX_CUDA_REQUEUES )); then
        echo "CUDA allocation smoke check failed on ${SLURMD_NODENAME:-unknown node}." >&2
        echo "Requeuing resumable job ${SLURM_JOB_ID} (${restart_count}/${RC_PAG_MAX_CUDA_REQUEUES})." >&2
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    fi
    echo "CUDA allocation smoke check failed after the automatic CUDA requeue limit " \
        "(${RC_PAG_MAX_CUDA_REQUEUES})." >&2
    echo "Job=${SLURM_JOB_ID:-unknown} node=${SLURMD_NODENAME:-unknown} " \
        "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    echo "Resubmit once; if the failure repeats, send this node/job information to cluster support." >&2
    exit 75
fi

exit "${status}"

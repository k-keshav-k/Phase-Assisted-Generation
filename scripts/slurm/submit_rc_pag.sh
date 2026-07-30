#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
RC_PAG_STAGE="${1:-pilot}"
RC_PAG_CONFIG="${RC_PAG_CONFIG:-${PROJECT_DIR}/configs/experiments/rc_pag_neurips.yaml}"
RC_PAG_OUTPUT_ROOT="${RC_PAG_OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/rc_pag}"
RC_PAG_ACCOUNT="${RC_PAG_ACCOUNT:-csci_ga_3033_131-2026sp}"
RC_PAG_PARTITION="${RC_PAG_PARTITION:-c12m85-a100-1}"
RC_PAG_TIME="${RC_PAG_TIME:-24:00:00}"
OVERLAY_PATH="${OVERLAY_PATH:-/scratch/${USER}/overlay-25GB-500K.ext3}"
SIF_PATH="${SIF_PATH:-/scratch/${USER}/ubuntu-20.04.3.sif}"
WORKER="${SCRIPT_DIR}/rc_pag_a100.sbatch"

case "${RC_PAG_STAGE}" in
    preflight|pilot|collect|screen|calibrate|confirm) ;;
    fit|report|paper)
        echo "${RC_PAG_STAGE} is CPU-only; run it with scripts/run_rc_pag.py instead of reserving an A100." >&2
        exit 2
        ;;
    all)
        echo "The A100 wrapper rejects 'all'; submit gated GPU stages one at a time." >&2
        exit 2
        ;;
    *) echo "Unknown RC-PAG stage: ${RC_PAG_STAGE}" >&2; exit 2 ;;
esac

command -v sbatch >/dev/null 2>&1 || { echo "sbatch is not available" >&2; exit 1; }
[[ -d "${PROJECT_DIR}" ]] || { echo "PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2; exit 1; }
[[ -f "${RC_PAG_CONFIG}" ]] || { echo "Config does not exist: ${RC_PAG_CONFIG}" >&2; exit 1; }
[[ -f "${OVERLAY_PATH}" ]] || { echo "Overlay does not exist: ${OVERLAY_PATH}" >&2; exit 1; }
[[ -f "${SIF_PATH}" ]] || { echo "Singularity image does not exist: ${SIF_PATH}" >&2; exit 1; }

if [[ "${RC_PAG_STAGE}" == "confirm" || "${RC_PAG_STAGE}" == "all" ]]; then
    : "${RC_PAG_ALLOW_CONFIRMATORY:?Set RC_PAG_ALLOW_CONFIRMATORY=1 for confirmatory generation}"
    [[ "${RC_PAG_ALLOW_CONFIRMATORY}" == "1" ]] || {
        echo "RC_PAG_ALLOW_CONFIRMATORY must equal 1" >&2
        exit 2
    }
fi

mkdir -p "${PROJECT_DIR}/logs/rc_pag" "${RC_PAG_OUTPUT_ROOT}"

export PROJECT_DIR OVERLAY_PATH SIF_PATH RC_PAG_STAGE RC_PAG_CONFIG RC_PAG_OUTPUT_ROOT
export RC_PAG_ALLOW_CONFIRMATORY="${RC_PAG_ALLOW_CONFIRMATORY:-0}"
export RC_PAG_LIMIT="${RC_PAG_LIMIT:-}"

echo "Submitting RC-PAG stage=${RC_PAG_STAGE} partition=${RC_PAG_PARTITION} time=${RC_PAG_TIME}"
sbatch \
    --account="${RC_PAG_ACCOUNT}" \
    --partition="${RC_PAG_PARTITION}" \
    --time="${RC_PAG_TIME}" \
    --chdir="${PROJECT_DIR}" \
    --export=ALL \
    "${WORKER}"

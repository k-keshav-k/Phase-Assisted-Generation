#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_WORKSHOP_CONFIG="${PROJECT_ROOT}/configs/experiments/rc_pag_neurips_workshop_v7.yaml"
KNOWN_V5_RUN="${PROJECT_ROOT}/artifacts/rc_pag/rc-pag-7688c5235bd4"

if [[ -n "${RC_PAG_LIMIT:-}" ]]; then
    echo "RC_PAG_LIMIT is not allowed for the complete confirmatory run." >&2
    exit 2
fi

export RC_PAG_ALLOW_CONFIRMATORY=1
export RC_PAG_TIME="${RC_PAG_ALL_TIME:-48:00:00}"
export RC_PAG_CONFIG="${RC_PAG_CONFIG:-${DEFAULT_WORKSHOP_CONFIG}}"
if [[ -z "${RC_PAG_REUSE_FROM:-}" && -d "${KNOWN_V5_RUN}" ]]; then
    export RC_PAG_REUSE_FROM="${KNOWN_V5_RUN}"
fi

echo "Submitting the complete RC-PAG pipeline in one resumable A100 job."
echo "Confirmation profile: ${RC_PAG_CONFIG}"
echo "Fresh workshop matrix: 5,184 confirmation generations on the preregistered complement."
echo "Fresh v7 workload: 10,184 prompt-method generations (3,028 plain; 7,156 instrumented)."
echo "With compatible v4/v5 trace reuse: 8,984 generations (3,028 plain; 5,956 instrumented)."
echo "V7 requires 8% tuning headroom, certifies <=2% harm, and gates the headline on"
echo "a >5% paired-bootstrap NFE-reduction lower bound for each model."
if [[ -n "${RC_PAG_REUSE_FROM:-}" ]]; then
    echo "Reusing only compatible native traces from: ${RC_PAG_REUSE_FROM}."
fi
echo "Stages: preflight -> pilot -> collect -> fit -> screen -> calibrate -> confirm -> report -> paper"
exec "${SCRIPT_DIR}/submit_rc_pag.sh" all

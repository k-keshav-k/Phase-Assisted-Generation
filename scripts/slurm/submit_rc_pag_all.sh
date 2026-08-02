#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_WORKSHOP_CONFIG="${PROJECT_ROOT}/configs/experiments/rc_pag_neurips_workshop_v5.yaml"

if [[ -n "${RC_PAG_LIMIT:-}" ]]; then
    echo "RC_PAG_LIMIT is not allowed for the complete confirmatory run." >&2
    exit 2
fi

export RC_PAG_ALLOW_CONFIRMATORY=1
export RC_PAG_TIME="${RC_PAG_ALL_TIME:-48:00:00}"
export RC_PAG_CONFIG="${RC_PAG_CONFIG:-${DEFAULT_WORKSHOP_CONFIG}}"

echo "Submitting the complete RC-PAG pipeline in one resumable A100 job."
echo "Confirmation profile: ${RC_PAG_CONFIG}"
echo "Fresh workshop matrix: 5,184 confirmation generations on the preregistered complement."
echo "Fresh v5 workload: 10,784 prompt-method generations (3,328 plain; 7,456 instrumented)."
echo "With compatible v4 reuse: 8,984 generations (3,028 plain; 5,956 instrumented)."
echo "The v5 gate requires 8% tuning headroom, then certifies <=2% harm and >5% NFE savings."
if [[ -n "${RC_PAG_REUSE_FROM:-}" ]]; then
    echo "Requested compatible v4 trace and paired-q500 reuse: ${RC_PAG_REUSE_FROM}."
fi
echo "Stages: preflight -> pilot -> collect -> fit -> rollout -> refit -> screen"
echo "        calibrate -> confirm -> report -> paper"
exec "${SCRIPT_DIR}/submit_rc_pag.sh" all

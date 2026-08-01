#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_WORKSHOP_CONFIG="${PROJECT_ROOT}/configs/experiments/rc_pag_neurips_workshop_v4.yaml"

if [[ -n "${RC_PAG_LIMIT:-}" ]]; then
    echo "RC_PAG_LIMIT is not allowed for the complete confirmatory run." >&2
    exit 2
fi

export RC_PAG_ALLOW_CONFIRMATORY=1
export RC_PAG_TIME="${RC_PAG_ALL_TIME:-48:00:00}"
export RC_PAG_CONFIG="${RC_PAG_CONFIG:-${DEFAULT_WORKSHOP_CONFIG}}"

echo "Submitting the complete RC-PAG pipeline in one resumable A100 job."
echo "Confirmation profile: ${RC_PAG_CONFIG}"
echo "Fresh workshop matrix: 5,184 confirmation generations on the v1 complement."
echo "Post-pilot workload: 9,384 prompt-method generations (2,628 plain; 6,756 instrumented)."
echo "The v4 gate jointly certifies <=2% harm and >=5% paired NFE savings per model."
if [[ -n "${RC_PAG_REUSE_FROM:-}" ]]; then
    echo "Requested compatible v3/v4 raw-trace reuse: ${RC_PAG_REUSE_FROM}."
fi
echo "Stages: preflight -> pilot -> collect -> fit -> screen -> calibrate -> confirm"
echo "        report -> paper"
exec "${SCRIPT_DIR}/submit_rc_pag.sh" all

#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSHOP_CONFIG="${SCRIPT_DIR}/../../configs/experiments/rc_pag_neurips_workshop_v2.yaml"

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
if [[ -n "${RC_PAG_REUSE_FROM:-}" ]]; then
    echo "Validated reuse source: ${RC_PAG_REUSE_FROM} (LLaDA local estimator only)."
fi
echo "Stages: preflight -> pilot -> collect -> fit -> screen -> calibrate -> confirm"
echo "        report -> paper"
exec "${SCRIPT_DIR}/submit_rc_pag.sh" all

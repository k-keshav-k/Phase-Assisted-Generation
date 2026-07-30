#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSHOP_CONFIG="${SCRIPT_DIR}/../../configs/experiments/rc_pag_neurips_workshop.yaml"

if [[ -n "${RC_PAG_LIMIT:-}" ]]; then
    echo "RC_PAG_LIMIT is not allowed for the complete confirmatory run." >&2
    exit 2
fi

export RC_PAG_ALLOW_CONFIRMATORY=1
export RC_PAG_TIME="${RC_PAG_ALL_TIME:-48:00:00}"
export RC_PAG_CONFIG="${RC_PAG_CONFIG:-${DEFAULT_WORKSHOP_CONFIG}}"

echo "Submitting the complete RC-PAG pipeline in one resumable A100 job."
echo "Confirmation profile: ${RC_PAG_CONFIG}"
echo "Workshop matrix: 6,000 confirmation generations; 16,200 projected GPU runs total."
echo "Stages: preflight -> pilot -> collect -> fit -> screen -> calibrate -> confirm"
echo "        report -> paper"
exec "${SCRIPT_DIR}/submit_rc_pag.sh" all

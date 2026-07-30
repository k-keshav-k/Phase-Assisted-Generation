#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${RC_PAG_LIMIT:-}" ]]; then
    echo "RC_PAG_LIMIT is not allowed for the complete confirmatory run." >&2
    exit 2
fi

export RC_PAG_ALLOW_CONFIRMATORY=1
export RC_PAG_TIME="${RC_PAG_ALL_TIME:-48:00:00}"

echo "Submitting the complete RC-PAG pipeline in one resumable A100 job."
echo "Stages: preflight -> pilot -> collect -> fit -> screen -> calibrate -> confirm"
echo "        report -> paper"
exec "${SCRIPT_DIR}/submit_rc_pag.sh" all

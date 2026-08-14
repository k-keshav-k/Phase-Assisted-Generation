#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_WORKSHOP_CONFIG="${PROJECT_ROOT}/configs/experiments/rc_pag_neurips_workshop_v9.yaml"
KNOWN_V8_RUN="${PROJECT_ROOT}/artifacts/rc_pag/rc-pag-d36b982c2388"

if [[ -n "${RC_PAG_LIMIT:-}" ]]; then
    echo "RC_PAG_LIMIT is not allowed for the complete confirmatory run." >&2
    exit 2
fi

export RC_PAG_ALLOW_CONFIRMATORY=1
export RC_PAG_TIME="${RC_PAG_ALL_TIME:-48:00:00}"
export RC_PAG_CONFIG="${RC_PAG_CONFIG:-${DEFAULT_WORKSHOP_CONFIG}}"
if [[ -z "${RC_PAG_REUSE_FROM:-}" && -d "${KNOWN_V8_RUN}" ]]; then
    export RC_PAG_REUSE_FROM="${KNOWN_V8_RUN}"
fi

echo "Submitting the complete RC-PAG pipeline in one resumable A100 job."
echo "Confirmation profile: ${RC_PAG_CONFIG}"
echo "Fresh workshop matrix: 5,184 confirmation generations on the preregistered complement."
echo "V9 starts with a 32-prompt/model numerical audit and a fresh 64-prompt/model gate."
echo "With v8 AdaBlock-reference reuse: 8,168 total GPU generations."
echo "Later stages require exact token/state parity, complete guard evidence, no row-work increase,"
echo "and a >5% paired-bootstrap latency-reduction lower bound for both models."
if [[ -n "${RC_PAG_REUSE_FROM:-}" ]]; then
    echo "Reusing only compatible AdaBlock audit references from: ${RC_PAG_REUSE_FROM}."
fi
echo "Stages: preflight -> audit/pilot -> no-op collect/fit -> screen -> calibrate -> confirm -> report -> paper"
exec "${SCRIPT_DIR}/submit_rc_pag.sh" all

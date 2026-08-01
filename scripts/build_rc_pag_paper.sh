#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRITEUP_DIR="${PROJECT_DIR}/writeup"
GENERATED_DIR="${WRITEUP_DIR}/generated"
BUILD_DIR="${WRITEUP_DIR}/build"
RUN_DIR="${1:-}"

command -v latexmk >/dev/null 2>&1 || {
    echo "latexmk is required to build the workshop paper" >&2
    exit 1
}

if [[ ! -f "${WRITEUP_DIR}/neurips_2026.sty" && "${RC_PAG_ALLOW_DRAFT_STYLE:-0}" != "1" ]]; then
    echo "Missing writeup/neurips_2026.sty. Download the official NeurIPS 2026 style," >&2
    echo "or set RC_PAG_ALLOW_DRAFT_STYLE=1 only for a non-submission draft." >&2
    exit 2
fi

uv run python - "${RUN_DIR}" "${GENERATED_DIR}" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

run_arg, generated_arg = sys.argv[1:]
generated = Path(generated_arg).resolve()
if generated.name != "generated" or generated.parent.name != "writeup":
    raise SystemExit(f"refusing to replace unexpected generated directory: {generated}")
if generated.exists():
    shutil.rmtree(generated)
generated.mkdir(parents=True)

if not run_arg:
    print("No run directory supplied; building an explicitly results-pending draft.")
    raise SystemExit(0)

run_dir = Path(run_arg).expanduser().resolve()
required = (
    run_dir / "risk_certificate.json",
    run_dir / "report" / "inputs.json",
    run_dir / "report" / "claim_audit.json",
    run_dir / "paper_manifest.json",
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("incomplete RC-PAG run; missing: " + ", ".join(missing))

certificate = json.loads(required[0].read_text(encoding="utf-8"))
inputs = json.loads(required[1].read_text(encoding="utf-8"))
audit = json.loads(required[2].read_text(encoding="utf-8"))
if bool(certificate.get("mock")) or bool(inputs.get("mock")):
    raise SystemExit("refusing to build numerical paper results from mock evidence")
if any(int(count) < 1 for count in inputs.get("coverage", {}).values()):
    raise SystemExit("confirmatory coverage contains an empty method/dataset/model cell")

names = {str(row.get("name", "")) for row in certificate.get("candidates", ())}
models = {name.split("/", 1)[0] for name in names if "/" in name}
if (
    certificate.get("loss") != "adablock_correct_candidate_wrong"
    or models != {"llada", "dream"}
    or len(names) != 2
):
    raise SystemExit("v2 certificate must cover exactly two frozen model-policy pairs")

tables = run_dir / "report" / "tables"
figures = run_dir / "report" / "figures"
for name in (
    "main_results.tex",
    "calibration.tex",
    "ablations.tex",
    "estimator_ablation.tex",
    "headline.tex",
):
    source = tables / name
    if not source.is_file():
        raise SystemExit(f"missing generated table: {source}")
    shutil.copy2(source, generated / name)
for name in ("nfe_accuracy.pdf", "risk_compute.pdf", "reliability.pdf"):
    source = figures / name
    if not source.is_file():
        raise SystemExit(f"missing generated figure: {source}")
    shutil.copy2(source, generated / name)

(generated / "evidence_status.json").write_text(
    json.dumps(
        {
            "run_dir": str(run_dir),
            "headline_eligible": bool(audit.get("headline_eligible")),
            "failed_gates": audit.get("failed_gates", []),
            "mock": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Validated and staged non-mock results from {run_dir}")
PY

mkdir -p "${BUILD_DIR}"
(
    cd "${WRITEUP_DIR}"
    latexmk \
        -pdf \
        -interaction=nonstopmode \
        -halt-on-error \
        -outdir="${BUILD_DIR}" \
        rc_pag_workshop.tex
)

LOG_PATH="${BUILD_DIR}/rc_pag_workshop.log"
PDF_PATH="${BUILD_DIR}/rc_pag_workshop.pdf"
MAIN_PAGES="$(grep -o 'RC-PAG-MAIN-PAGES:[0-9]*' "${LOG_PATH}" | tail -n 1 | cut -d: -f2)"
if [[ -z "${MAIN_PAGES}" ]]; then
    echo "Could not determine main-text page count from ${LOG_PATH}" >&2
    exit 1
fi
if (( MAIN_PAGES > 8 )); then
    echo "Main text is ${MAIN_PAGES} pages; the DiffuLM workshop limit is 8." >&2
    exit 3
fi

TOTAL_PAGES="unknown"
if command -v pdfinfo >/dev/null 2>&1; then
    TOTAL_PAGES="$(pdfinfo "${PDF_PATH}" | awk '/^Pages:/ {print $2}')"
fi
echo "Built ${PDF_PATH} (main text: ${MAIN_PAGES} pages; total: ${TOTAL_PAGES} pages)."

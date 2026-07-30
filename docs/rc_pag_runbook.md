# RC-PAG one-A100 runbook

This runbook executes the frozen Risk-Calibrated PAG (RC-PAG) protocol on one NYU Langone
A100. It deliberately separates development, untouched calibration, and confirmation. Do
not change the configuration after calibration begins; a changed config creates a different
protocol hash and requires new calibration.

The default config is `configs/experiments/rc_pag_neurips.yaml`. Its current config hash is
`c855954c1a30f8d8c84a70edd115f0b784ae4bf0ba09337c5df268252506eab7`, so the default run is
`artifacts/rc_pag/rc-pag-c855954c1a30`. The certificate now treats each of six policies on
each of two models as a separate hypothesis and controls all 12 simultaneously.

For an unattended run that executes every GPU and CPU stage inside one A100 allocation, use
the concise [one-command guide](rc_pag_one_command.md). The gated steps below remain useful
when each stage should be inspected manually. The one-command launcher defaults to the
workshop-sized confirmation profile; the stage-by-stage commands below retain the larger
four-method protocol.

## 1. Local verification

From the repository root:

```bash
uv sync --group phase_cpd_dream
uv run pytest tests/experiments tests/llada/test_generate_pag.py \
  tests/dream/test_generation_utils_pag.py -q

mock_root="$(mktemp -d)"
uv run python scripts/run_rc_pag.py all --mock --output-root "${mock_root}" --limit 2
```

Mock output verifies orchestration only. It is watermarked in every certificate and the paper
builder refuses to import it.

## 2. Cluster variables

Clone or synchronize the repository to a shared NYU filesystem, then set paths for your
account. The defaults match the course allocation used by this project; override them if
NYU changes the allocation.

```bash
export PROJECT_DIR=/path/to/research
export RC_PAG_OUTPUT_ROOT=/scratch/${USER}/rc_pag/artifacts
export RC_PAG_ACCOUNT=csci_ga_3033_131-2026sp
export RC_PAG_PARTITION=c12m85-a100-1
export OVERLAY_PATH=/scratch/${USER}/overlay-25GB-500K.ext3
export SIF_PATH=/scratch/${USER}/ubuntu-20.04.3.sif
export RC_PAG_SCRATCH_ROOT=/scratch/${USER}/rc_pag/cache
```

Authenticate with Hugging Face interactively before submission if the pinned model revisions
require it. Never put a token in this repository or in a Slurm command. The worker stores
model, dataset, uv, and temporary caches under `RC_PAG_SCRATCH_ROOT` and loads only one model
at a time.

## 3. Run the gated funnel

Run one stage at a time. Inspect the log and manifest before submitting the next stage. Every
stage is atomic and resumes completed prompt/method records after preemption.

### A. GPU preflight and pilot

```bash
RC_PAG_TIME=01:00:00 scripts/slurm/submit_rc_pag.sh preflight
RC_PAG_TIME=04:00:00 scripts/slurm/submit_rc_pag.sh pilot
```

The pilot uses 32 prompts/model and runs both a full-budget decode and a forced early-stop
shadow smoke decode. This catches model-specific cache cloning and shadow failures before trace
collection. After it completes:

```bash
run_dir="${RC_PAG_OUTPUT_ROOT}/rc-pag-c855954c1a30"
python -m json.tool "${run_dir}/compute_projection.json"
python -m json.tool "${run_dir}/manifests/pilot.json"
```

Stop if either backend fails, records contain non-finite values, memory is close to the A100
limit, or the projected time/storage is unacceptable. Adjusting the protocol means starting a
new run, not editing an in-progress directory.

### B. GPU trace collection

```bash
RC_PAG_TIME=24:00:00 scripts/slurm/submit_rc_pag.sh collect
```

This is one full trajectory for each of 600 prompts per model. Training counterfactuals are
derived from that trajectory, avoiding a separate model run per candidate.

### C. CPU estimator fit

The fit does not need an A100. Run it on a login/CPU allocation with access to the same
`RC_PAG_OUTPUT_ROOT`:

```bash
uv run --group phase_cpd_dream python scripts/run_rc_pag.py fit \
  --output-root "${RC_PAG_OUTPUT_ROOT}" --device cpu --resume
```

Inspect `estimators/manifest.json`. Histogram gradient boosting is primary; logistic
regression remains a preregistered capacity check and must not be chosen after calibration.

### D. GPU screening

```bash
RC_PAG_TIME=24:00:00 scripts/slurm/submit_rc_pag.sh screen
```

Screening tries fixed, AdaBlock, Fast-dLLM-style, SchED-style, entropy, confidence,
stability, constant-budget, size-lookup, original PAG, residual PAG, local/history RC-PAG,
and an offline oracle. `screening_summary.json` chooses the lowest-NFE nonlearned method that
does not lose more than the frozen tuning accuracy allowance. Fast-dLLM and SchED entries are
explicitly recorded as style reproductions, not official implementations.

### E. GPU calibration

```bash
RC_PAG_TIME=24:00:00 scripts/slurm/submit_rc_pag.sh calibrate
python -m json.tool "${run_dir}/risk_certificate.json"
```

Do not inspect calibration examples and then modify thresholds. The six policies were frozen
before this stage. The exact tests use a Bonferroni cutoff of `0.05 / 12`. Confirmation is
blocked unless both local and history variants certify separately for LLaDA and Dream. It is
also blocked if the selected history policies do not beat the screened nonlearned method on
calibration NFE.

### F. Full confirmatory generation

This is the expensive stage and requires an explicit opt-in:

```bash
export RC_PAG_ALLOW_CONFIRMATORY=1
RC_PAG_TIME=48:00:00 scripts/slurm/submit_rc_pag.sh confirm
```

It runs AdaBlock, the frozen best nonlearned rule, local RC-PAG, and history RC-PAG on paired
GSM8K test, MATH-500, sanitized MBPP, and HumanEval prompts for both models. Do not use
`RC_PAG_LIMIT` for confirmation; the CLI rejects it.

### G. CPU report and paper manifest

```bash
uv run --group phase_cpd_dream python scripts/run_rc_pag.py report \
  --output-root "${RC_PAG_OUTPUT_ROOT}" --device cpu --resume
uv run --group phase_cpd_dream python scripts/run_rc_pag.py paper \
  --output-root "${RC_PAG_OUTPUT_ROOT}" --device cpu --resume
python -m json.tool "${run_dir}/report/claim_audit.json"
```

The report always writes the complete evidence. It emits a positive headline only if every
predeclared claim gate passes. A valid negative or mixed result is still publishable but must
use the automatically generated qualified headline.

## 4. Build the workshop PDF

Download the official NeurIPS 2026 style from the official author-kit/Overleaf template and
place `neurips_2026.sty` in `writeup/`. Then build from the non-mock run:

```bash
scripts/build_rc_pag_paper.sh "${run_dir}"
```

The builder validates non-mock evidence, complete coverage, and all 12 model-policy
certificate rows before copying tables and vector figures. It fails if the main text exceeds
the workshop's eight-page limit. For a results-pending local layout check only:

```bash
RC_PAG_ALLOW_DRAFT_STYLE=1 scripts/build_rc_pag_paper.sh
```

The PDF is `writeup/build/rc_pag_workshop.pdf`.

## 5. Compute-minimization rules

- Prefer the stage-by-stage commands when manual pilot inspection or minimum A100 reservation
  time matters. Use `submit_rc_pag_all.sh` when one unattended, resumable allocation is more
  important; it also runs CPU-only stages on the allocated node.
- Cache pinned weights/datasets on scratch, keep one model resident, and resume the same run.
- Stop after pilot on backend or budget failure; stop after calibration on certificate or
  NFE futility failure.
- Reuse one collected full trajectory for estimator labels; use same-state shadows only for
  calibration, where they are statistically necessary.
- Profile elapsed time as well as NFE. NFE is the primary architecture-independent compute
  measure, but a method with CPU synchronization overhead may not improve latency.
- Do not run extra thresholds after seeing calibration. If a new idea is essential, declare a
  new protocol and collect a new calibration set.

## 6. Final publication audit

Before submission, verify that:

1. all model and dataset revisions match the YAML and the paper;
2. `claim_audit.json` governs the abstract, results, and conclusion wording;
3. every result cell has the registered paired sample count;
4. risk is described as strict shadow disagreement, never task-correctness risk;
5. style reproductions are not presented as official baseline code;
6. NFE counting includes the initial proposal forward pass;
7. the PDF is anonymous, searchable, eight main-text pages or fewer, and uses the official
   NeurIPS 2026 style; and
8. the submission metadata and author list are added only after the double-blind manuscript
   is finalized.

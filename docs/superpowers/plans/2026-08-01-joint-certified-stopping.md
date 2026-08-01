# Joint-Certified Stable Stopping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-estimator early-stop decoder whose frozen model policies must jointly certify at most 2% harmful regression and at least 5% mean paired NFE reduction.

**Architecture:** Extend the online local feature schema with temporal-JS summaries, simplify `RiskStoppingPolicy` to a score/patience rule, and add a bounded paired-compute test beside the exact harm test. Introduce a frozen v4 config while retaining old protocols for reproducibility.

**Tech Stack:** Python 3.11, NumPy, SciPy, scikit-learn, PyTorch, pytest, YAML, Slurm, LaTeX.

---

### Task 1: Temporal stability features and single-score policy

**Files:**
- Modify: `src/pag/experiments/rc_pag_features.py`
- Modify: `src/pag/experiments/rc_pag_policy.py`
- Modify: `tests/experiments/test_rc_pag_features.py`
- Modify: `tests/experiments/test_rc_pag_policy.py`

- [ ] Add failing tests asserting `local.temporal_js_mean`, `local.temporal_js_max`, and quantiles are extracted only over masked tokens.
- [ ] Run `uv run pytest tests/experiments/test_rc_pag_features.py tests/experiments/test_rc_pag_policy.py -q` and confirm the new assertions fail.
- [ ] Add the JS summaries to `_LOCAL_FEATURE_NAMES` and `extract_features`.
- [ ] Remove benefit prediction and independent tail/JS eligibility from the stopping decision; retain score threshold, minimum step, and patience.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Joint harm and compute certificate

**Files:**
- Modify: `src/pag/experiments/risk_control.py`
- Modify: `tests/experiments/test_risk_control.py`

- [ ] Add failing tests in which harm passes but 5% compute reduction fails, both pass, negative savings are rejected, and multiplicity counts two hypotheses per candidate.
- [ ] Run `uv run pytest tests/experiments/test_risk_control.py -q` and confirm failure.
- [ ] Implement the Hoeffding--Bentkus p-value and inverted one-sided lower confidence bound for bounded paired savings.
- [ ] Extend certificate rows with harm and compute p-values/bounds; define `certified` as their conjunction.
- [ ] Preserve the legacy risk-only mode for v1-v3 reproducibility and run the tests.

### Task 3: Frozen v4 protocol and orchestration

**Files:**
- Create: `configs/experiments/rc_pag_neurips_workshop_v4.yaml`
- Modify: `src/pag/experiments/rc_pag_config.py`
- Modify: `src/pag/experiments/rc_pag_orchestrator.py`
- Modify: `src/pag/experiments/rc_pag_runtime.py`
- Modify: `tests/experiments/test_rc_pag_config.py`
- Modify: `tests/experiments/test_rc_pag_orchestrator.py`
- Modify: `tests/experiments/test_rc_pag_runtime.py`

- [ ] Add failing config tests for one estimator, three thresholds, no benefit/tail/JS gates, and a 5% calibration compute target.
- [ ] Add failing orchestration tests that require paired NFE ratios and block confirmation when either half of the joint certificate fails.
- [ ] Teach config validation/loading about v4 and its `risk.minimum_nfe_reduction` field.
- [ ] Remove v4 benefit-estimator fitting/loading and pass paired normalized savings into calibration.
- [ ] Keep old protocol branches unchanged and run focused experiment tests.

### Task 4: Reporting, launcher, and manuscript

**Files:**
- Modify: `src/pag/experiments/rc_pag_report.py`
- Modify: `scripts/slurm/submit_rc_pag_all.sh`
- Modify: `docs/rc_pag_one_command.md`
- Modify: `docs/rc_pag_runbook.md`
- Modify: `writeup/rc_pag_workshop.tex`
- Modify: `writeup/rc_pag_references.bib`
- Modify: corresponding tests under `tests/experiments/`

- [ ] Add failing assertions for v4 as the launcher default and for both certificate columns in generated reports.
- [ ] Update report JSON/CSV/LaTeX generation and claim gates to require joint certification.
- [ ] Point the one-command launcher to v4 and document the one command, fresh artifacts, resume behavior, and expected fallback.
- [ ] Rewrite the paper method and theorem around the joint certificate; cite Prophet, SWD, STDec, LATCH, and RC-Jot without claiming novelty for temporal stability alone.
- [ ] Run focused report, launcher, and paper tests.

### Task 5: Full verification

**Files:**
- Verify all changed files.

- [ ] Run `uv run pytest -q` and record the exact pass count.
- [ ] Run `uv run ruff check src tests scripts phase_cpd phase_predict`.
- [ ] Run `bash -n scripts/slurm/submit_rc_pag_all.sh scripts/slurm/submit_rc_pag.sh scripts/slurm/rc_pag_a100.sbatch`.
- [ ] Run `scripts/build_rc_pag_paper.sh` in draft mode when real v4 artifacts are not local.
- [ ] Inspect `git diff --check` and the final diff for unrelated or destructive changes.

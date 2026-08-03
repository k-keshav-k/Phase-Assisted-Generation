# Risk-Budgeted Verified PAG v6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen v6 decoder that allocates calibrated local disagreement risk across a prompt, prioritizes high-NFE stops, certifies end-to-end harm, and evaluates raw paired compute without assuming nonnegative per-prompt savings.

**Architecture:** Preserve every v1--v5 path. V6 reuses only native full-budget traces, fits a held-out calibrated local-risk scorer plus a remaining-block-NFE regressor, and instantiates a stateful exact-agreement controller with a prompt risk ledger and stop cap. Screening uses fresh rows; calibration certifies harm only; confirmation applies preregistered paired-bootstrap compute gates.

**Tech Stack:** Python 3.11, PyTorch, scikit-learn histogram gradient boosting and isotonic regression, NumPy/SciPy, pytest, Ruff, YAML, Slurm/bash, LaTeX.

---

## File map

- Create `configs/experiments/rc_pag_neurips_workshop_v6.yaml`: immutable v6 protocol.
- Modify `src/pag/experiments/rc_pag_config.py`: v6 fields, validation, and loading.
- Modify `src/pag/experiments/rc_pag_policy.py`: calibrated scorer persistence and ledger state.
- Modify `src/pag/experiments/rc_pag_orchestrator.py`: v6 fitting, reuse, harm-only calibration, raw compute diagnostics.
- Modify `src/pag/experiments/rc_pag_runtime.py`: v6 artifact loading and policy construction.
- Modify `src/pag/experiments/rc_pag_report.py`: aggregate bootstrap compute gate.
- Modify `scripts/run_rc_pag.py`, `scripts/slurm/submit_rc_pag*.sh`: v6 routing/default.
- Modify `docs/rc_pag_one_command.md`, `docs/rc_pag_runbook.md`, `writeup/rc_pag_workshop.tex`: execution and claims.
- Modify focused files under `tests/experiments/`: red-green coverage for every boundary.

### Task 1: Frozen v6 configuration

**Files:**
- Create: `configs/experiments/rc_pag_neurips_workshop_v6.yaml`
- Modify: `src/pag/experiments/rc_pag_config.py`
- Test: `tests/experiments/test_rc_pag_config.py`

- [ ] **Step 1: Write failing config tests**

Add assertions equivalent to:

```python
def test_v6_freezes_ledger_family_and_fresh_splits(v6_config):
    assert v6_config.protocol_version == "v6"
    assert v6_config.risk.minimum_nfe_reduction is None
    assert v6_config.readiness.minimum_tuning_nfe_reduction_per_model == 0.08
    assert [(c.total_risk_budget, c.max_prompt_stops, c.min_predicted_nfe_savings)
            for c in v6_config.candidates] == [(0.02, 1, 4.0), (0.05, 2, 3.0), (0.10, 3, 2.0)]
    assert sum(hi - lo + 1 for lo, hi in v6_config.splits["tuning"].values()) == 150
    assert sum(hi - lo + 1 for lo, hi in v6_config.splits["calibration"].values()) == 500
```

Also assert exact ranges and disjointness from the v5 rollout/tuning/calibration ranges.

- [ ] **Step 2: Run the focused test and confirm red**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py -q`

Expected: failure because v6 and ledger fields are unknown.

- [ ] **Step 3: Extend immutable configuration types and validation**

Add backward-compatible defaults:

```python
@dataclass(frozen=True, slots=True)
class PolicyCandidateSpec:
    # existing fields...
    total_risk_budget: float | None = None
    max_prompt_stops: int | None = None

@dataclass(frozen=True, slots=True)
class ClaimGateSpec:
    # existing fields...
    minimum_model_nfe_reduction_lower_ci: float | None = None
```

Register v6, require variant `rc_pag_budgeted`, exact agreement, budgets/stops
`[(.02,1,4),(.05,2,3),(.10,3,2)]`, HGB only, calibration size 500, harm-only risk,
8% readiness, fresh profile, and a 5% aggregate lower-CI gate. Leave v1--v5 defaults unchanged.

- [ ] **Step 4: Add the exact frozen YAML**

Copy pinned model/dataset/decoding/confirmation identities from v5, remove `rollout`, set tuning to
GSM8K `400--449`, MATH `300--349`, MBPP `200--249`, and calibration to GSM8K `854--1241`, MATH
`433--507`, MBPP `295--331`. Set `minimum_nfe_reduction: null` and the three ledger candidates.

- [ ] **Step 5: Run config tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py -q`

Expected: all config tests pass.

```bash
git add configs/experiments/rc_pag_neurips_workshop_v6.yaml src/pag/experiments/rc_pag_config.py tests/experiments/test_rc_pag_config.py
git commit -m "feat: freeze risk-budgeted PAG v6 protocol"
```

### Task 2: Calibrated local-risk scorer

**Files:**
- Modify: `src/pag/experiments/rc_pag_policy.py`
- Test: `tests/experiments/test_rc_pag_policy.py`

- [ ] **Step 1: Write red tests for held-out calibration and persistence**

Cover monotone bounded predictions, prompt-disjoint inputs, single-class calibration fallback,
metadata hash verification, and save/load equality:

```python
calibrated = CalibratedRiskEstimator.fit(
    training_examples=train,
    calibration_examples=calibration,
    kind="hist_gradient_boosting",
    include_history=False,
    history_window=4,
    seed=7,
)
assert 0.0 <= calibrated.predict_risk(features) <= 1.0
assert set(calibrated.training_prompt_ids).isdisjoint(calibrated.calibration_prompt_ids)
```

- [ ] **Step 2: Confirm the tests fail**

Run: `uv run pytest tests/experiments/test_rc_pag_policy.py -q`

Expected: import/attribute failures for `CalibratedRiskEstimator`.

- [ ] **Step 3: Implement the calibrated scorer**

Use `RiskEstimator.fit` for the base scorer and `IsotonicRegression(y_min=0, y_max=1,
out_of_bounds="clip")` on held-out raw scores. If held-out labels contain one class, store an
explicit constant calibrator. Persist base model state, calibrator, prompt-ID hashes, feature names,
and target name `local_full_trajectory_disagreement` in one joblib plus JSON metadata.

- [ ] **Step 4: Run policy tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_policy.py -q`

Expected: all policy tests pass.

```bash
git add src/pag/experiments/rc_pag_policy.py tests/experiments/test_rc_pag_policy.py
git commit -m "feat: calibrate local RC-PAG risk scores"
```

### Task 3: Prompt risk ledger and verified charging

**Files:**
- Modify: `src/pag/experiments/rc_pag_policy.py`
- Test: `tests/experiments/test_rc_pag_policy.py`

- [ ] **Step 1: Write red ledger transition tests**

Exercise this sequence:

```python
policy = RiskStoppingPolicy(
    scorer,
    threshold=1.0,
    min_steps=2,
    patience=2,
    benefit_scorer=benefit,
    min_predicted_nfe_savings=3.0,
    require_exact_agreement=True,
    total_risk_budget=0.05,
    max_prompt_stops=2,
)
assert not policy.observe(first).should_stop
assert policy.risk_spent == 0.0
assert policy.observe(matching_second).should_stop
assert policy.risk_spent == pytest.approx(0.02)
assert policy.prompt_stops == 1
```

Also test proposal change does not charge, benefit gate, insufficient remaining budget, block reset
preserving ledger, prompt reset clearing ledger, and maximum stop count.

- [ ] **Step 2: Confirm red**

Run the named ledger tests with `uv run pytest tests/experiments/test_rc_pag_policy.py -q`.

- [ ] **Step 3: Implement ledger state without changing v1--v5**

Add optional `total_risk_budget` and `max_prompt_stops` constructor fields. Initialize ledger fields
in `reset_prompt`, not `start_block`. Include ledger eligibility in the existing gate. Charge the
current verified state's score only when exact agreement produces `should_stop=True`. Extend
`StopDecision` with defaulted `risk_spent` and `prompt_stops` audit fields.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_policy.py -q`

Expected: all old and new policy tests pass.

```bash
git add src/pag/experiments/rc_pag_policy.py tests/experiments/test_rc_pag_policy.py
git commit -m "feat: add verified prompt risk ledger"
```

### Task 4: V6 trace fitting and raw-trace reuse

**Files:**
- Modify: `src/pag/experiments/rc_pag_orchestrator.py`
- Test: `tests/experiments/test_rc_pag_orchestrator.py`

- [ ] **Step 1: Write failing fitting/reuse tests**

Assert v6 uses active stages without rollout/refit, partitions 600 prompt groups 4:1, writes
`MODEL_rc_pag_budgeted_risk.joblib` and `MODEL_remaining_nfe.joblib`, and accepts v5 raw traces while
copying no v5 advantage estimator or screen artifact. Reject incomplete temporal-JS traces.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py -q`

- [ ] **Step 3: Implement v6 fitting**

Build local and benefit examples with `_v4_training_payloads`. Partition groups by deterministic
prompt order (`index % 5 == 0` calibration, remaining training). Fit/save the calibrated risk head
and remaining-NFE head, compute held-out AUROC/Brier/MAE, and record exact prompt counts/hashes in
`estimators/manifest.json`.

- [ ] **Step 4: Implement v6 reuse**

Add `_prepare_v6_reuse` that validates source protocol v4 or v5, model/dataset identities, trace
count, native schedules, and temporal JS, then copies only `collect/MODEL/full_budget_shadow` rows.
Write `reuse_scope: raw_exact_loop_traces_only_for_v6_refit`.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py -q`

```bash
git add src/pag/experiments/rc_pag_orchestrator.py tests/experiments/test_rc_pag_orchestrator.py
git commit -m "feat: fit v6 ledger estimators from reusable traces"
```

### Task 5: Runtime artifact loading

**Files:**
- Modify: `src/pag/experiments/rc_pag_runtime.py`
- Test: `tests/experiments/test_rc_pag_runtime.py`

- [ ] **Step 1: Write a failing runtime construction test**

Build a v6 candidate and assert runtime loads exactly the calibrated-risk and remaining-NFE files,
validates identical local feature schemas, enables exact agreement, and forwards budget/stop fields.
Also assert missing or uncalibrated artifacts fail before model generation.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/experiments/test_rc_pag_runtime.py -q`

- [ ] **Step 3: Implement v6 dispatch**

Branch on `candidate.variant == "rc_pag_budgeted"`, load
`MODEL_rc_pag_budgeted_risk.joblib` through `CalibratedRiskEstimator` and
`MODEL_remaining_nfe.joblib` through `RemainingNFEEstimator`, validate local HGB/temporal-JS schema,
and pass the frozen ledger fields into `RiskStoppingPolicy`. Preserve v1--v5 branches exactly.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_runtime.py -q`

```bash
git add src/pag/experiments/rc_pag_runtime.py tests/experiments/test_rc_pag_runtime.py
git commit -m "feat: run risk-budgeted v6 policies"
```

### Task 6: Harm-only calibration with raw compute diagnostics

**Files:**
- Modify: `src/pag/experiments/rc_pag_orchestrator.py`
- Modify: `src/pag/experiments/rc_pag_report.py`
- Test: `tests/experiments/test_rc_pag_orchestrator.py`
- Test: `tests/experiments/test_rc_pag_report.py`

- [ ] **Step 1: Write red calibration/report tests**

Construct paired rows containing a negative normalized saving. Assert v6 calibration does not clip
or reject it, calls `certify_candidates(..., minimum_nfe_reduction=None)`, uses two harm hypotheses,
stores raw mean savings and a paired bootstrap interval as diagnostics, and marks
`certificate_mode: harm_only_with_paired_compute_evidence`.

For confirmation, assert each model's aggregate raw NFE-reduction lower bound must exceed 0.05 for
headline eligibility.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_report.py -q`.

- [ ] **Step 3: Implement harm-only v6 calibration**

Keep v4/v5 joint logic unchanged. In v6, collect raw per-prompt savings including negative values,
certify binary harm only, and store a bootstrap interval computed from paired candidate/baseline NFE.
Confirmation proceeds only when both frozen model policies are harm-certified.

- [ ] **Step 4: Implement aggregate confirmation compute gate**

Pool paired in-domain rows within each model, bootstrap candidate-minus-AdaBlock NFE, convert both
interval endpoints to reduction using the paired baseline mean, and require the reduction lower
bound to exceed the configured 0.05. Preserve per-cell accuracy gates and all raw tables.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_report.py -q`.

```bash
git add src/pag/experiments/rc_pag_orchestrator.py src/pag/experiments/rc_pag_report.py tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_report.py
git commit -m "feat: separate v6 harm certification from compute evidence"
```

### Task 7: Launcher, concise guide, and manuscript

**Files:**
- Modify: `scripts/run_rc_pag.py`
- Modify: `scripts/slurm/submit_rc_pag.sh`
- Modify: `scripts/slurm/submit_rc_pag_all.sh`
- Modify: `docs/rc_pag_one_command.md`
- Modify: `docs/rc_pag_runbook.md`
- Modify: `writeup/rc_pag_workshop.tex`
- Test: `tests/experiments/test_submit_rc_pag_all.py`
- Test: `tests/experiments/test_rc_pag_paper.py`

- [ ] **Step 1: Write failing launcher/paper assertions**

Assert the one-command default is v6, v5 artifact reuse is forwarded unchanged, the stage list has
no rollout/refit for v6, and the paper distinguishes exact harm certification from bootstrap compute
evidence while documenting the v4 accumulation and v5 AUROC failures.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/experiments/test_submit_rc_pag_all.py tests/experiments/test_rc_pag_paper.py -q`.

- [ ] **Step 3: Update execution surfaces**

Make `rc_pag_neurips_workshop_v6.yaml` the default, print v6 stages/workload/reuse scope, and keep the
single command `scripts/slurm/submit_rc_pag_all.sh`. Document the recommended v5 reuse path,
resumability, controlled-stop semantics, and artifact locations.

- [ ] **Step 4: Update the manuscript honestly**

Replace v5 advantage-gating claims with the prompt ledger equations, frozen candidates, fresh splits,
harm-only theorem, and raw compute bootstrap criterion. State v4/v5 as development failures and do
not insert numerical v6 claims before validated artifacts exist.

- [ ] **Step 5: Run tests, build draft, and commit**

Run:

```bash
uv run pytest tests/experiments/test_submit_rc_pag_all.py tests/experiments/test_rc_pag_paper.py -q
RC_PAG_ALLOW_DRAFT_STYLE=1 scripts/build_rc_pag_paper.sh
```

Expected: tests pass; main manuscript is at most eight pages.

```bash
git add scripts/run_rc_pag.py scripts/slurm/submit_rc_pag.sh scripts/slurm/submit_rc_pag_all.sh docs/rc_pag_one_command.md docs/rc_pag_runbook.md writeup/rc_pag_workshop.tex tests/experiments/test_submit_rc_pag_all.py tests/experiments/test_rc_pag_paper.py
git commit -m "docs: make risk-budgeted v6 the workshop default"
```

### Task 8: End-to-end verification

**Files:**
- Modify only files required by failures in the scoped v6 implementation.

- [ ] **Step 1: Run focused RC-PAG tests**

```bash
uv run pytest tests/experiments/test_rc_pag_config.py tests/experiments/test_rc_pag_policy.py tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_runtime.py tests/experiments/test_rc_pag_report.py tests/experiments/test_submit_rc_pag_all.py tests/experiments/test_rc_pag_paper.py -q
```

Expected: all pass.

- [ ] **Step 2: Run scoped lint and shell validation**

Run Ruff on every changed Python file, `ruff format --check` on the same set, `git diff --check`, and
`bash -n scripts/slurm/submit_rc_pag.sh scripts/slurm/submit_rc_pag_all.sh`.

- [ ] **Step 3: Run the complete suite**

Run: `make test`

Expected: zero failures; existing PyTorch warnings may remain.

- [ ] **Step 4: Run a clean mock v6 pipeline**

```bash
mock_root="$(mktemp -d)"
uv run python scripts/run_rc_pag.py all \
  --config configs/experiments/rc_pag_neurips_workshop_v6.yaml \
  --mock --allow-confirmatory --output-root "${mock_root}"
```

Expected: stages through `paper` complete, estimator artifacts and harm-only certificate exist, and
the resume command targets the same v6 run ID.

- [ ] **Step 5: Inspect the final diff and commit verification-only fixes**

Confirm the worktree contains no unrelated files and no generated mock artifacts. Commit only if
verification required a code change:

```bash
git add src/pag/experiments/rc_pag_config.py src/pag/experiments/rc_pag_policy.py \
  src/pag/experiments/rc_pag_orchestrator.py src/pag/experiments/rc_pag_runtime.py \
  src/pag/experiments/rc_pag_report.py tests/experiments
git commit -m "test: verify risk-budgeted PAG v6"
```

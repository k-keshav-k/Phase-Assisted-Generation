# Risk-Adaptive Verified Speculation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a v8 RC-PAG protocol that uses the fitted local risk model to size exact AdaBlock verification graphs, preserving AdaBlock's output while reducing model-forward evaluations.

**Architecture:** A model-independent module constructs speculative token-state graphs and accepts only edges exactly reproduced by the existing AdaBlock transition. Dream and LLaDA provide batched active-window forward callbacks; the orchestration layer treats verified speculation as a new candidate variant and records exactness/acceptance diagnostics.

**Tech Stack:** Python 3.11, PyTorch, scikit-learn/joblib risk estimators, pytest, YAML, Slurm.

---

### Task 1: Pure verified-speculation core

**Files:**
- Create: `src/pag/experiments/rc_pag_speculation.py`
- Test: `tests/experiments/test_rc_pag_speculation.py`

- [ ] **Step 1: Write failing policy and verifier tests**

Add tests that require `RiskAdaptiveSpeculationPolicy` to map low/medium/high risk to frozen node
budgets, require `build_linear_draft` to reveal only masked positions, and require
`verify_draft` to return an exact sequence of reference transitions after both acceptance and
rejection.

- [ ] **Step 2: Run the focused test and observe the import failure**

Run: `uv run pytest tests/experiments/test_rc_pag_speculation.py -q`

Expected: collection fails because `pag.experiments.rc_pag_speculation` does not exist.

- [ ] **Step 3: Implement the immutable graph and verifier types**

Implement `SpeculationPlan`, `SpeculationResult`, `RiskAdaptiveSpeculationPolicy`,
`build_linear_draft`, `verify_draft`, `repeat_tensor_tree`, and diagnostic serialization.  Validate
all probabilities, depths, shapes, parent order, and monotone unmasking.  `verify_draft` must always
apply at least one verified transition and raise if a proposed child changes an already unmasked
token.

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest tests/experiments/test_rc_pag_speculation.py -q`

Expected: all tests pass.

### Task 2: Configuration and runtime routing

**Files:**
- Modify: `src/pag/experiments/rc_pag_config.py`
- Modify: `src/pag/experiments/rc_pag_runtime.py`
- Modify: `tests/experiments/test_rc_pag_config.py`
- Modify: `tests/experiments/test_rc_pag_runtime.py`

- [ ] **Step 1: Add failing v8 parsing and routing tests**

Add a v8 candidate with `variant: rc_pag_verified`, `max_speculation_depth`,
`medium_speculation_depth`, `deep_risk_threshold`, `medium_risk_threshold`, and
`draft_width_multiplier`.  Assert invalid depth/threshold ordering is rejected and that the runtime
loads only the compatible local risk estimator.

- [ ] **Step 2: Run the focused tests**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py tests/experiments/test_rc_pag_runtime.py -q`

Expected: failures identify the missing fields and runtime branch.

- [ ] **Step 3: Implement v8 config and runtime construction**

Extend `PolicyCandidateSpec` with validated speculation defaults.  Add protocol v8 to modern exact
AdaBlock routing.  Construct `RiskAdaptiveSpeculationPolicy` from the local histogram-boosting head,
pass it separately from `RiskStoppingPolicy`, and expose provenance `rc_pag_verified`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py tests/experiments/test_rc_pag_runtime.py -q`

Expected: all tests pass.

### Task 3: LLaDA and Dream exact batched adapters

**Files:**
- Modify: `AdaBlock-dLLM/llada/generate_adablock.py`
- Modify: `AdaBlock-dLLM/dream/model/generation_utils_adablock.py`
- Modify: `tests/experiments/test_rc_pag_runtime.py`
- Modify: `tests/dream/test_dream_adablock.py`
- Modify: `tests/llada/test_llada_adablock.py`

- [ ] **Step 1: Add failing fake-model integration tests**

Use deterministic fake models and caches to compare sequential AdaBlock with verified speculation.
Cover full draft acceptance, first-edge rejection, a partially accepted draft, cache batch expansion,
and schedule diagnostics.  Assert identical tokens and fewer counted NFEs when at least two edges are
accepted.

- [ ] **Step 2: Run focused adapter tests**

Run: `uv run pytest tests/experiments/test_rc_pag_runtime.py tests/dream tests/llada -q`

Expected: failures identify absent `speculation_policy` hooks.

- [ ] **Step 3: Implement batched active-window verification**

At each noninitial refinement, use the previous verified logits to build a bounded draft, repeat the
frozen block cache over its node batch, evaluate nodes in one model call, and apply the exact existing
transfer function to every verified parent.  Store `speculation_steps` alongside `risk_steps`.
Sequential AdaBlock and all v1--v7 policy paths must remain byte-for-byte behaviorally unchanged.

- [ ] **Step 4: Run adapter and regression tests**

Run: `uv run pytest tests/experiments/test_rc_pag_runtime.py tests/dream tests/llada -q`

Expected: all tests pass.

### Task 4: v8 experiment funnel and exactness gates

**Files:**
- Create: `configs/experiments/rc_pag_neurips_workshop_v8.yaml`
- Modify: `src/pag/experiments/rc_pag_orchestrator.py`
- Modify: `src/pag/experiments/rc_pag_report.py`
- Modify: `tests/experiments/test_rc_pag_orchestrator.py`
- Modify: `tests/experiments/test_rc_pag_report.py`

- [ ] **Step 1: Add failing screen-selection tests**

Require v8 selection to reject any candidate with a generated-ID disagreement, rank remaining
candidates by paired latency then NFE, and stop before calibration if either model has less than 5%
NFE reduction or nonpositive latency reduction.

- [ ] **Step 2: Run orchestration tests**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_report.py -q`

Expected: tests fail on missing v8 exactness fields and selection logic.

- [ ] **Step 3: Implement v8 summaries and gates**

Aggregate sequence disagreements, evaluated nodes, accepted transitions, rejection depth, equivalent
sequential steps, NFE reduction, and latency reduction.  Make sequence equality a hard eligibility
condition.  Keep the existing paired accuracy certificate as a redundant confirmatory check.

- [ ] **Step 4: Run orchestration tests**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_report.py -q`

Expected: all tests pass.

### Task 5: One-command launcher, reuse, and run documentation

**Files:**
- Modify: `scripts/run_rc_pag.py`
- Modify: `scripts/slurm/submit_rc_pag_all.sh`
- Modify: `docs/rc_pag_one_command.md`
- Modify: `tests/experiments/test_run_rc_pag.py`
- Modify: `tests/experiments/test_submit_rc_pag_all.py`

- [ ] **Step 1: Add failing default-launcher tests**

Assert v8 is the default configuration, compatible v4--v7 full-budget traces and local risk heads can
be reused, the launcher describes lossless verification, and the one-command document contains the
exact `bash scripts/slurm/submit_rc_pag_all.sh` invocation.

- [ ] **Step 2: Run launcher tests**

Run: `uv run pytest tests/experiments/test_run_rc_pag.py tests/experiments/test_submit_rc_pag_all.py -q`

Expected: tests fail because v7 remains the default.

- [ ] **Step 3: Update the resumable v8 launcher and concise runbook**

Point defaults to v8, prefer the newest compatible artifact source, preserve offline dataset behavior,
and print the exactness and 5% compute gates.  Document estimated pilot/screen hours separately from
confirmation and explain resumption.

- [ ] **Step 4: Run launcher tests**

Run: `uv run pytest tests/experiments/test_run_rc_pag.py tests/experiments/test_submit_rc_pag_all.py -q`

Expected: all tests pass.

### Task 6: Full verification

**Files:**
- Modify only files needed for fixes found by verification.

- [ ] **Step 1: Run the complete focused experiment suite**

Run: `uv run pytest tests/experiments tests/dream tests/llada -q`

Expected: all tests pass.

- [ ] **Step 2: Run lint before formatting**

Run: `uv run ruff check src tests scripts phase_cpd phase_predict`

Expected: no lint errors.

- [ ] **Step 3: Format and confirm a clean diff**

Run: `uv run ruff format src tests scripts phase_cpd phase_predict`

Expected: formatting completes; `git diff --check` produces no output.

- [ ] **Step 4: Run the v8 mock funnel**

Run: `uv run python scripts/run_rc_pag.py all --config configs/experiments/rc_pag_neurips_workshop_v8.yaml --mock --output-root /tmp/rc-pag-v8-mock`

Expected: the pipeline completes or reaches its preregistered compute gate without an exception, and
all produced candidate records contain verification diagnostics.

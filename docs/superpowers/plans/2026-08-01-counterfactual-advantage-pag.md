# Counterfactual Advantage-Gated RC-PAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen v5 protocol that learns prompt-level harmful-regression and normalized-compute heads from paired on-policy rollouts, then verifies proposed stops before committing.

**Architecture:** Preserve v1-v4 and the native AdaBlock loops. Add v5-only rollout/refit stages, two compact estimator artifacts per model, and exact consecutive-proposal verification inside the policy; retain the existing joint calibration certificate and confirmation funnel.

**Tech Stack:** Python 3.11, NumPy, scikit-learn, PyTorch, pytest, YAML, Bash/Slurm.

---

### Task 1: Frozen v5 configuration

**Files:**
- Create: `configs/experiments/rc_pag_neurips_workshop_v5.yaml`
- Modify: `src/pag/experiments/rc_pag_config.py`
- Modify: `tests/experiments/test_rc_pag_config.py`

- [ ] **Step 1: Add failing v5 config tests**

Assert `protocol_version == "v5"`, rollout size 150, calibration size 500, three exact-agreement candidates with `(0.02, 0.05)`, `(0.05, 0.08)`, `(0.10, 0.10)`, and readiness 0.08.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py -q`

- [ ] **Step 3: Extend immutable config types and validation**

Add `rollout_per_model`, `require_exact_agreement`, protocol-specific split counts, v5 method/profile registration, and v5-only validation while leaving v1-v4 accepted byte-for-byte.

- [ ] **Step 4: Add the frozen YAML**

Use old v4 tuning rows as `rollout`; assign disjoint 150-prompt v5 tuning rows and 500 calibration rows. Keep the fresh 864-prompt confirmation complement.

- [ ] **Step 5: Run focused config tests**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py -q`

### Task 2: Bounded advantage estimator and verified policy

**Files:**
- Modify: `src/pag/experiments/rc_pag_policy.py`
- Modify: `tests/experiments/test_rc_pag_policy.py`

- [ ] **Step 1: Add failing estimator and policy tests**

Cover bounded `[0,1]` predictions, persistence, first eligible decision becoming pending, exact masked-token agreement stopping on the following step, disagreement restarting verification, and block/prompt reset clearing pending state.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `uv run pytest tests/experiments/test_rc_pag_policy.py -q`

- [ ] **Step 3: Implement the bounded estimator**

Add a `NormalizedNFEReductionEstimator` backed by `HistGradientBoostingRegressor`; reject out-of-range training targets and clip deployment predictions to `[0,1]`.

- [ ] **Step 4: Implement pending verification**

Extend `RiskStoppingPolicy` with `require_exact_agreement`. Store the first eligible proposal, require the next eligible proposal to match all currently masked positions, and expose `pending_verification` / `agreement_verified` reasons without changing v4 behavior.

- [ ] **Step 5: Run focused policy tests**

Run: `uv run pytest tests/experiments/test_rc_pag_policy.py -q`

### Task 3: Counterfactual rollout labels and refit stages

**Files:**
- Modify: `src/pag/experiments/rc_pag_orchestrator.py`
- Modify: `tests/experiments/test_rc_pag_orchestrator.py`

- [ ] **Step 1: Add failing grouped-label tests**

Construct paired AdaBlock/seed records and assert stopped decisions receive prompt-level harm and normalized-saving labels; reject missing pairs, negative savings, and rows with no executed stops.

- [ ] **Step 2: Add failing v5 stage-flow test**

Assert active stages include `rollout` and `refit` only for v5, mock resume is idempotent, and screen loads refitted advantage artifacts.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py -q`

- [ ] **Step 4: Implement active stage sequencing and rollout**

Add `rollout` and `refit` to the stage registry, skip them for v1-v4 through protocol-specific active stages, and run paired AdaBlock plus frozen q500 seed on the rollout split.

- [ ] **Step 5: Implement grouped refit**

Rebuild compact features from serialized native-loop observations, keep prompt groups intact for validation, fit harm and normalized-saving heads, and save a provenance-rich advantage manifest.

- [ ] **Step 6: Enforce v5 screening headroom**

Use disjoint v5 tuning rows and require 8% empirical NFE reduction per model before calibration. Preserve the 5% calibration null.

- [ ] **Step 7: Run focused orchestrator tests**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py -q`

### Task 4: Runtime loading and safe v4 artifact reuse

**Files:**
- Modify: `src/pag/experiments/rc_pag_runtime.py`
- Modify: `src/pag/experiments/rc_pag_orchestrator.py`
- Modify: `scripts/run_rc_pag.py`
- Modify: `tests/experiments/test_rc_pag_runtime.py`
- Modify: `tests/experiments/test_rc_pag_orchestrator.py`

- [ ] **Step 1: Add failing runtime tests**

Assert v5 candidates require the advantage harm and gain files, use normalized gain thresholds, and enable exact agreement; v4 still loads only its local estimator.

- [ ] **Step 2: Add failing reuse tests**

Assert a stopped v4 run with complete paired q500/AdaBlock screen rows can seed v5 rollout/refit, while incomplete or observation-free rows fail before fitting.

- [ ] **Step 3: Implement v5 runtime loading**

Load `MODEL_rc_pag_advantage_harm.joblib` and `MODEL_rc_pag_advantage_gain.joblib`, validate their feature schema, and construct the verified policy.

- [ ] **Step 4: Implement surgical reuse**

Copy compatible exact-loop collect rows and paired v4 screen rows into v5 `rollout`; never copy v4 estimator decisions, readiness, calibration, or confirmation artifacts.

- [ ] **Step 5: Update CLI reuse help and run focused tests**

Run: `uv run pytest tests/experiments/test_rc_pag_runtime.py tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_run_rc_pag.py -q`

### Task 5: One-command launcher and concise run guide

**Files:**
- Modify: `scripts/slurm/submit_rc_pag_all.sh`
- Modify: `docs/rc_pag_one_command.md`
- Modify: `tests/experiments/test_submit_rc_pag_all.py`
- Modify: `tests/experiments/test_run_rc_pag.py`

- [ ] **Step 1: Add failing launcher assertions**

Assert v5 is the default config and `RC_PAG_REUSE_FROM` is forwarded unchanged to the one Python command.

- [ ] **Step 2: Run focused launcher tests and confirm failure**

Run: `uv run pytest tests/experiments/test_submit_rc_pag_all.py tests/experiments/test_run_rc_pag.py -q`

- [ ] **Step 3: Update launcher and documentation**

Keep the single `sbatch scripts/slurm/submit_rc_pag_all.sh` interface. Document optional reuse from the failed v4 run, resume behavior, stage outputs, and the v5 controlled-stop meaning.

- [ ] **Step 4: Run focused launcher tests and shell parsing**

Run: `uv run pytest tests/experiments/test_submit_rc_pag_all.py tests/experiments/test_run_rc_pag.py -q`

Run: `bash -n scripts/slurm/submit_rc_pag_all.sh scripts/slurm/rc_pag_a100.sbatch`

### Task 6: End-to-end verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run all experiment tests**

Run: `uv run pytest tests/experiments -q`

- [ ] **Step 2: Run the complete suite**

Run: `make test`

- [ ] **Step 3: Run scoped lint and formatting checks**

Run: `uv run ruff check src/pag/experiments tests/experiments scripts/run_rc_pag.py`

Run: `uv run ruff format --check src/pag/experiments tests/experiments scripts/run_rc_pag.py`

- [ ] **Step 4: Run the v5 mock pipeline**

Run: `uv run python scripts/run_rc_pag.py all --config configs/experiments/rc_pag_neurips_workshop_v5.yaml --mock --allow-confirmatory --output-root <temporary-directory>`

- [ ] **Step 5: Inspect the patch**

Run: `git diff --check` and review `git diff --stat` plus the launcher/config diff for unrelated changes.

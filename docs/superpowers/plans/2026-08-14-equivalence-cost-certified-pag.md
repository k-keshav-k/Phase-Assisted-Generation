# Equivalence- and Cost-Certified PAG v9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable v9 experiment that calibrates batch-shape numerical drift, runs only guarded depth-one/two speculation, and permits confirmation only after exact A100 trajectory parity and a greater-than-5% paired latency lower bound.

**Architecture:** A new model-independent equivalence module owns numerical margins, hardware envelopes, guarded verification, state digests, and a deterministic shallow cost policy. Dream and LLaDA retain their v8 paths and add a separate v9 branch that can audit a batched root against canonical batch-size-one execution or run production guarded speculation. The orchestrator freezes audit-derived policy artifacts, validates them on a disjoint pilot, and reports latency and evaluated-row evidence instead of treating batched rows as one NFE.

**Tech Stack:** Python 3.11, PyTorch, NumPy/SciPy, YAML, pytest, Ruff, Bash/Slurm.

---

## File structure

- Create `src/pag/experiments/rc_pag_equivalence.py`: pure v9 math, serialization, policy, verifier, and trajectory hashing.
- Modify `src/pag/experiments/rc_pag_config.py`: v9 schema and frozen audit/pilot/confirmation settings.
- Modify `src/pag/experiments/rc_pag_runtime.py`: v9 policy routing, execution fingerprint, and honest prompt-level counters.
- Modify `AdaBlock-dLLM/llada/generate_adablock.py`: canonical audit and guarded v9 branch for LLaDA.
- Modify `AdaBlock-dLLM/dream/model/generation_utils_adablock.py`: canonical audit and guarded v9 branch for Dream.
- Modify `src/pag/experiments/rc_pag_orchestrator.py`: audit fitting, held-out pilot gate, v9 screen/calibration/confirmation, and reuse.
- Modify `src/pag/experiments/rc_pag_report.py`: latency, trajectory, guard, and evaluated-row summaries/gates.
- Create `configs/experiments/rc_pag_neurips_workshop_v9.yaml`: frozen v9 protocol.
- Modify `scripts/run_rc_pag.py`, `scripts/slurm/submit_rc_pag_all.sh`, and `docs/rc_pag_one_command.md`: v9 defaults and one-command handoff.
- Modify focused tests under `tests/experiments/`, `tests/dream/`, and `tests/llada/`.

### Task 1: Pure equivalence and cost core

**Files:**
- Create: `src/pag/experiments/rc_pag_equivalence.py`
- Create: `tests/experiments/test_rc_pag_equivalence.py`

- [ ] **Step 1: Write failing numerical-margin and envelope tests**

Create tests that import the following public API and cover stable, threshold-boundary, rank-boundary,
and batch-sensitive examples:

```python
from pag.experiments.rc_pag_equivalence import (
    EquivalenceEnvelope,
    EquivalenceCostPolicy,
    decision_margins,
    fit_equivalence_artifact,
    guard_transition,
)

def test_guard_requires_all_three_decision_margins():
    envelope = EquivalenceEnvelope(logit_epsilon=0.1, probability_epsilon=0.01)
    margins = decision_margins(logits, state, mask_token_id=9, threshold=0.9)
    assert margins.token_margin > 0.2
    assert not guard_transition(margins, envelope).passed
```

Add tests that `fit_equivalence_artifact` applies the frozen `1.25` inflation, rejects empty or
non-finite events, prefers depth one, and maps sparse/unseen bins to depth zero.

- [ ] **Step 2: Run the focused test and verify import failure**

Run: `uv run pytest tests/experiments/test_rc_pag_equivalence.py -q`

Expected: collection fails because `pag.experiments.rc_pag_equivalence` does not exist.

- [ ] **Step 3: Implement immutable core types and pure functions**

Implement these stable signatures:

```python
@dataclass(frozen=True, slots=True)
class DecisionMargins:
    token_margin: float
    threshold_margin: float
    forced_rank_margin: float

@dataclass(frozen=True, slots=True)
class EquivalenceEnvelope:
    logit_epsilon: float
    probability_epsilon: float
    safety_inflation: float = 1.25

@dataclass(frozen=True, slots=True)
class GuardDecision:
    passed: bool
    margins: DecisionMargins
    reason: str

@dataclass(frozen=True, slots=True)
class GuardedSpeculationResult:
    tokens: torch.Tensor
    accepted_draft_edges: int
    reference_equivalent_transitions: int
    evaluated_nodes: int
    canonical_fallback_rows: int
    guard_passed: bool
    reference_checked: bool
    successor_equal_when_checked: bool | None
    transition_states: tuple[torch.Tensor, ...]
    reason: str
```

Add `decision_margins`, `guard_transition`, `verify_guarded_draft`, `state_digest`,
`fit_equivalence_artifact`, and `EquivalenceCostPolicy`. The production verifier must call a lazy
canonical fallback only for an unsafe root; an unsafe deeper row returns the last guarded state.
Evaluated-node counters include rejected nodes and fallback rows.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/experiments/test_rc_pag_equivalence.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the pure core**

```bash
git add src/pag/experiments/rc_pag_equivalence.py tests/experiments/test_rc_pag_equivalence.py
git commit -m "feat: add equivalence cost guard core"
```

### Task 2: Frozen v9 configuration

**Files:**
- Modify: `src/pag/experiments/rc_pag_config.py`
- Create: `configs/experiments/rc_pag_neurips_workshop_v9.yaml`
- Modify: `tests/conftest.py`
- Modify: `tests/experiments/test_rc_pag_config.py`

- [ ] **Step 1: Add failing v9 parsing tests**

Add a `v9_config` fixture and assertions equivalent to:

```python
def test_v9_config_freezes_disjoint_equivalence_funnel(v9_config):
    assert v9_config.protocol_version == "v9"
    assert v9_config.stage_sizes.audit_per_model == 32
    assert v9_config.stage_sizes.pilot_per_model == 64
    assert len(v9_config.candidates) == 1
    assert v9_config.candidates[0].variant == "ec_pag"
    assert v9_config.equivalence.maximum_depth == 2
    assert v9_config.equivalence.minimum_latency_reduction == 0.05
    assert v9_config.equivalence.require_evaluated_row_nonincrease
```

Also mutate each frozen value and require validation failure. Explicitly verify that audit, pilot,
tuning, and calibration indices are disjoint.

- [ ] **Step 2: Run config tests and observe failures**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py -q`

Expected: v9 is rejected as an unknown protocol.

- [ ] **Step 3: Add v9 dataclasses and validation**

Extend `StageSizes` with `audit_per_model: int = 0` and add:

```python
@dataclass(frozen=True, slots=True)
class EquivalenceSpec:
    maximum_depth: int
    safety_inflation: float
    minimum_bin_count: int
    minimum_acceptance_lcb: float
    minimum_latency_reduction: float
    require_evaluated_row_nonincrease: bool
    source_run: str
```

Register v9 with exactly one `ec_pag_v9` candidate, no estimator kinds, 32 audit rows, 64 held-out
pilot rows, 150 tuning rows, 500 calibration rows, and the existing fresh workshop confirmation
complement. Preserve byte-for-byte validation behavior for v1--v8.

- [ ] **Step 4: Add the frozen YAML**

Copy pinned model/dataset/decoding revisions from v8. Use the audit ranges `450--465`, `350--357`,
`250--257`; pilot ranges `500--531`, `360--375`, `258--273`; existing tuning/calibration ranges;
and one candidate:

```yaml
policy:
  estimator_kinds: []
  history_window: 4
  candidates:
    - {name: ec_pag_v9, variant: ec_pag, threshold: 1.0, min_steps: 1, patience: 1,
       max_speculation_depth: 2, medium_speculation_depth: 1,
       draft_width_multiplier: 1.0}
```

- [ ] **Step 5: Run config tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py -q`

Expected: all tests pass.

```bash
git add src/pag/experiments/rc_pag_config.py configs/experiments/rc_pag_neurips_workshop_v9.yaml tests/conftest.py tests/experiments/test_rc_pag_config.py
git commit -m "feat: register ec pag v9 protocol"
```

### Task 3: Runtime and Dream/LLaDA guarded execution

**Files:**
- Modify: `src/pag/experiments/rc_pag_runtime.py`
- Modify: `AdaBlock-dLLM/llada/generate_adablock.py`
- Modify: `AdaBlock-dLLM/dream/model/generation_utils_adablock.py`
- Modify: `tests/experiments/test_rc_pag_runtime.py`
- Modify: `tests/llada/test_adablock_policy_hook.py`
- Modify: `tests/dream/test_generation_utils_pag.py`

- [ ] **Step 1: Write failing routing and batch-sensitive adapter tests**

Require runtime methods `ec_pag_audit_d1`, `ec_pag_audit_d2`, and a candidate with variant `ec_pag`
to produce `EquivalenceCostPolicy` instances. In each fake adapter, make batch-size-two logits differ
at a low-margin root. Assert audit uses the canonical successor, production falls back at the root,
and a high-margin case advances through a matching child without fallback.

```python
assert actual.generated_ids == reference.generated_ids
assert step["guard_passed"] is False
assert step["canonical_fallback_rows"] == 1
assert step["evaluated_rows"] == step["evaluated_nodes"] + 1
```

- [ ] **Step 2: Run focused tests and verify failures**

Run: `uv run pytest tests/experiments/test_rc_pag_runtime.py tests/llada/test_adablock_policy_hook.py tests/dream/test_generation_utils_pag.py -q`

Expected: v9 methods are unsupported and adapters use the v8 verifier.

- [ ] **Step 3: Implement v9 runtime routing and fingerprinting**

Load `equivalence/<model>.json` for production candidates and construct audit policies without an
artifact. Add an execution fingerprint containing model revision, bundled AdaBlock commit, GPU
name/capability, CUDA/PyTorch/Transformers versions, dtype, and attention backend. Reject a
production artifact whose fingerprint differs. Extend modern-protocol routing to v9.

- [ ] **Step 4: Implement separate v9 branches in both adapters**

Keep `RiskAdaptiveSpeculationPolicy` on the unchanged v8 branch. For `EquivalenceCostPolicy`:

1. choose depth zero/one/two before batching;
2. use ordinary AdaBlock immediately for depth zero;
3. evaluate the shallow batch and charge every row;
4. in audit mode, also evaluate the root canonically and return that canonical transition;
5. in production, apply `verify_guarded_draft` and lazily rerun the root at batch size one only when
   its guard fails; and
6. serialize margins, guard/fallback reason, accepted edges, serial calls, evaluated rows, reference
   checks, and ordered state digests.

The returned prompt payload must expose `serial_forward_calls`, `evaluated_rows`,
`reference_equivalent_transitions`, `state_trajectory_digest`, and `model_time_sec`.

- [ ] **Step 5: Run adapter/runtime tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_runtime.py tests/llada/test_adablock_policy_hook.py tests/dream/test_generation_utils_pag.py -q`

Expected: all tests pass, including v8 regressions.

```bash
git add src/pag/experiments/rc_pag_runtime.py AdaBlock-dLLM/llada/generate_adablock.py AdaBlock-dLLM/dream/model/generation_utils_adablock.py tests/experiments/test_rc_pag_runtime.py tests/llada/test_adablock_policy_hook.py tests/dream/test_generation_utils_pag.py
git commit -m "feat: add guarded ec pag model execution"
```

### Task 4: Audit fitting, reuse, and held-out pilot gate

**Files:**
- Modify: `src/pag/experiments/rc_pag_orchestrator.py`
- Modify: `tests/experiments/test_rc_pag_orchestrator.py`

- [ ] **Step 1: Add failing v9 funnel tests**

Use a v9-aware mock runtime to require:

```python
runner.run_through("pilot")
assert (run_dir / "equivalence" / "llada.json").is_file()
assert (run_dir / "equivalence" / "dream.json").is_file()
gate = json.loads((run_dir / "equivalence_pilot.json").read_text())
assert gate["passed"]
assert gate["models"]["llada"]["trajectory_mismatches"] == 0
assert gate["models"]["dream"]["latency_reduction"]["lower"] > 0.05
```

Add negative mocks for a single trajectory mismatch, a nonpositive latency lower bound, excess
evaluated rows, and fingerprint mismatch. Each must write a failed manifest and raise
`ControlledStop`, not a generic exception. Add a reuse test that imports only compatible v8
AdaBlock audit records with provenance and hashes.

- [ ] **Step 2: Run orchestrator tests and observe failures**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py -q`

Expected: v9 uses the v8 pilot path or cannot resolve the audit split.

- [ ] **Step 3: Implement `_run_v9_pilot`**

Run audit methods on `self._refs("audit")`, aggregate their schedule events, call
`fit_equivalence_artifact`, and atomically write one artifact per model. Freeze the artifact hash
before held-out generation. Run paired `adablock` and `ec_pag_v9` on `self._refs("pilot")`, compute
paired latency intervals with the registered bootstrap count, compare generated IDs and trajectory
digests, sum evaluated rows, and emit `equivalence_audit.json`, `equivalence_pilot.json`, and
`compute_projection.json`.

- [ ] **Step 4: Implement narrow v8 reference reuse**

Accept `--reuse-development-from` when the source identity has identical revisions and decoding.
Copy/link only matching `pilot/<model>/adablock` audit rows into `audit/<model>/adablock`, record
source/content hashes, and regenerate all logits/audit/candidate rows. A missing or incompatible
source falls back to generating the 32 small reference sets.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py -q`

Expected: all v9 and existing v1--v8 orchestration tests pass.

```bash
git add src/pag/experiments/rc_pag_orchestrator.py tests/experiments/test_rc_pag_orchestrator.py
git commit -m "feat: add ec pag audit and heldout gate"
```

### Task 5: Screen, calibration, confirmation, and reporting

**Files:**
- Modify: `src/pag/experiments/rc_pag_orchestrator.py`
- Modify: `src/pag/experiments/rc_pag_report.py`
- Modify: `tests/experiments/test_rc_pag_orchestrator.py`
- Modify: `tests/experiments/test_rc_pag_report.py`

- [ ] **Step 1: Add failing v9 certificate/report tests**

Require v9 screening to keep the single frozen policy, calibration to stop on any ID/trajectory
mismatch, and the report headline to depend on latency and evaluated rows rather than old NFE:

```python
assert certificate["certificate_mode"] == "hardware_scoped_execution_equivalence"
assert audit["gates"]["exact_trajectory_equivalence"]
assert audit["gates"]["model_latency_reduction_lower_ci"]
assert audit["gates"]["evaluated_row_nonincrease"]
assert "model_nfe_reduction_lower_ci" not in audit["gates"]
```

- [ ] **Step 2: Run focused tests and observe failures**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_report.py -q`

Expected: existing modern gates select by NFE and expect `verified_sequence_safe`.

- [ ] **Step 3: Implement v9 screen and calibration branches**

Make collect/fit explicit no-op manifests because v9 has no fitted risk head. Screen the single
frozen candidate on 150 tuning prompts and require exact trajectories, a latency lower bound above
5%, and evaluated-row nonincrease. Calibration repeats the paired test on 500 untouched prompts and
writes a custom certificate containing exact Clopper--Pearson mismatch upper bounds, paired latency
intervals, row totals, and guard/fallback counts.

- [ ] **Step 4: Implement v9 confirmation and report gates**

Skip the old NFE futility check for v9. Run `adablock`, `best_nonlearned` (resolved to AdaBlock), and
`rc_pag_selected`. Extend summaries with serial calls, evaluated rows, model time, trajectory
disagreements, guard passes, and fallback counts. For v9, headline eligibility requires zero ID and
trajectory mismatches, latency-reduction LCB above 5% for each model, nonincreasing rows, non-mock
evidence, and complete guard diagnostics.

- [ ] **Step 5: Run report/orchestration tests and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_report.py -q`

Expected: all tests pass.

```bash
git add src/pag/experiments/rc_pag_orchestrator.py src/pag/experiments/rc_pag_report.py tests/experiments/test_rc_pag_orchestrator.py tests/experiments/test_rc_pag_report.py
git commit -m "feat: certify ec pag latency and equivalence"
```

### Task 6: One-command v9 launcher and concise documentation

**Files:**
- Modify: `scripts/run_rc_pag.py`
- Modify: `scripts/slurm/submit_rc_pag_all.sh`
- Modify: `docs/rc_pag_one_command.md`
- Modify: `tests/experiments/test_run_rc_pag.py`
- Modify: `tests/experiments/test_submit_rc_pag_all.py`

- [ ] **Step 1: Add failing launcher tests**

Assert the default config is v9, the known v8 run `rc-pag-d36b982c2388` is preferred for narrow
reuse, output mentions the 32-row audit and 64-row held-out gate, and the one-command document begins
with:

```bash
bash scripts/slurm/submit_rc_pag_all.sh
```

- [ ] **Step 2: Run launcher tests and observe v8 defaults**

Run: `uv run pytest tests/experiments/test_run_rc_pag.py tests/experiments/test_submit_rc_pag_all.py -q`

Expected: assertions find `rc_pag_neurips_workshop_v8.yaml`.

- [ ] **Step 3: Update CLI, Slurm wrapper, and runbook**

Set the default config to v9, describe v9 reuse scope accurately, prefer the supplied completed v8
artifact, retain automatic Hugging Face cache bootstrap, and print that later GPU stages are blocked
unless exact parity, latency, and row-work gates pass. Document the new config hash/run directory,
resume behavior, artifacts to inspect, and conservative audit/pilot versus confirmation A100-hour
estimates.

- [ ] **Step 4: Run tests and shell syntax checks**

Run: `uv run pytest tests/experiments/test_run_rc_pag.py tests/experiments/test_submit_rc_pag_all.py -q`

Run: `bash -n scripts/slurm/submit_rc_pag.sh scripts/slurm/submit_rc_pag_all.sh scripts/slurm/rc_pag_a100.sbatch`

Expected: all tests pass and shell syntax produces no output.

- [ ] **Step 5: Commit launcher and docs**

```bash
git add scripts/run_rc_pag.py scripts/slurm/submit_rc_pag_all.sh docs/rc_pag_one_command.md tests/experiments/test_run_rc_pag.py tests/experiments/test_submit_rc_pag_all.py
git commit -m "docs: make ec pag v9 the one command default"
```

### Task 7: Full verification and handoff

**Files:**
- Modify only files required by verification failures.

- [ ] **Step 1: Run the focused experiment and adapter suites**

Run: `uv run pytest tests/experiments tests/dream tests/llada -q`

Expected: all tests pass.

- [ ] **Step 2: Run the mock v9 funnel**

Run:

```bash
uv run python scripts/run_rc_pag.py all \
  --config configs/experiments/rc_pag_neurips_workshop_v9.yaml \
  --mock --allow-confirmatory \
  --output-root /tmp/rc-pag-v9-mock
```

Expected: all stages complete, or a deliberately configured gate writes a controlled-stop artifact;
no unhandled exception occurs. Passing mock artifacts contain equivalence envelopes, pilot gates,
custom certificate, latency/row report fields, and v9 paper manifest.

- [ ] **Step 3: Run lint before formatting**

Run: `uv run ruff check src tests scripts phase_cpd phase_predict`

Expected: no lint errors.

- [ ] **Step 4: Format and rerun focused verification**

Run: `uv run ruff format src tests scripts phase_cpd phase_predict`

Run: `uv run pytest tests/experiments tests/dream tests/llada -q`

Expected: all tests pass after formatting.

- [ ] **Step 5: Check shell, diff, and repository state**

Run: `bash -n scripts/slurm/submit_rc_pag.sh scripts/slurm/submit_rc_pag_all.sh scripts/slurm/rc_pag_a100.sbatch`

Run: `git diff --check && git status --short`

Expected: no whitespace errors; status lists only intentional implementation changes if the final
verification fix has not already been committed.

- [ ] **Step 6: Commit final verification fixes**

```bash
git add src tests scripts configs docs
git commit -m "test: verify ec pag v9 funnel"
```

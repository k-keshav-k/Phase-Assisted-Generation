# NeurIPS Strategy 1 Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one resumable, budget-aware command that runs the approved PAG ablations, full and untouched GSM8K evaluations, MATH-500 transfer evaluation, synchronized timing benchmark, paired statistics, and paper-ready artifacts.

**Architecture:** A small `pag.experiments` package owns frozen configuration, datasets, grading, atomic artifacts, cost control, statistics, and reporting. LLaDA-specific scheduler policies and budget enforcement remain beside the LLaDA generation code. A thin script loads the model once and orchestrates preflight, development, promotion, final evaluation, timing, and reporting from a hashed YAML configuration.

**Tech Stack:** Python 3.11, PyTorch, Transformers, Hugging Face Datasets, PyYAML, NumPy, SciPy, scikit-learn, Math-Verify, Matplotlib, pytest, Ruff.

---

## File Map

- `configs/experiments/neurips_strategy1.yaml`: immutable research protocol and dataset revisions.
- `src/pag/experiments/config.py`: typed config parsing, validation, and hashing.
- `src/pag/experiments/datasets.py`: GSM8K/MATH-500 materialization and split assertions.
- `src/pag/experiments/grading.py`: strict GSM8K and symbolic MATH grading.
- `src/pag/experiments/records.py`: atomic per-prompt records, manifests, quarantine, and resume.
- `src/pag/experiments/budget.py`: wall-time cost tracking and stage admission.
- `src/pag/experiments/statistics.py`: paired bootstrap, Wilson intervals, and exact McNemar test.
- `src/pag/experiments/report.py`: CSV, JSON, LaTeX, and PDF report assets.
- `src/pag/experiments/runtime.py`: method execution against one loaded LLaDA model.
- `src/pag/experiments/orchestrator.py`: stage state machine and controlled-stop behavior.
- `AdaBlock-dLLM/llada/scheduler_variants.py`: non-Transformer schedulers and trace-derived models.
- `AdaBlock-dLLM/llada/generate_pag.py`: shared hard-cap/soft-gate enforcement flags and exit reasons.
- `scripts/run_neurips_strategy1.py`: one-command entrypoint.
- `tests/experiments/`: unit and mocked end-to-end coverage.
- `tests/llada/test_scheduler_variants.py`: scheduler policy tests.
- `tests/llada/test_generate_pag.py`: enforcement-mode regression tests.

### Task 1: Declare dependencies and freeze the protocol

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `configs/experiments/neurips_strategy1.yaml`
- Create: `src/pag/experiments/__init__.py`
- Create: `src/pag/experiments/config.py`
- Test: `tests/experiments/test_config.py`

- [ ] **Step 1: Write failing config tests**

```python
def test_loads_frozen_protocol_and_hashes_it(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text("schema_version: 1\nseed: 20260710\n", encoding="utf-8")
    config = load_experiment_config(path)
    assert config.schema_version == 1
    assert len(config.config_hash) == 64


def test_rejects_nonzero_temperature(strategy_config):
    strategy_config["decoding"]["temperature"] = 0.1
    with pytest.raises(ValueError, match="temperature must be 0"):
        validate_experiment_config(strategy_config)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `uv run pytest tests/experiments/test_config.py -v`

Expected: FAIL because `pag.experiments.config` does not exist.

- [ ] **Step 3: Add pinned analysis dependencies**

Add to `[project].dependencies`:

```toml
  "antlr4-python3-runtime==4.13.2",
  "math-verify==0.9.0",
  "matplotlib>=3.9",
  "numpy>=1.26",
  "scikit-learn>=1.5",
  "scipy>=1.12",
```

Run: `uv lock`

Expected: `uv.lock` resolves with Python 3.11.

- [ ] **Step 4: Add the frozen YAML protocol**

The YAML must encode the approved revisions and values directly:

```yaml
schema_version: 1
seed: 20260710
datasets:
  gsm8k:
    path: openai/gsm8k
    revision: 740312add88f781978c0658806c59bc2815b9866
    config: main
    development_indices: [6000, 6199]
    confirmatory_indices: [400, 1318]
  math500:
    path: math-ai/math500
    revision: 91b8f0024070e42ff83b949d6ca29da311fd3371
    sample_size: 300
decoding:
  temperature: 0.0
  gen_length: 256
  steps: 64
  threshold: 0.9
  delimiter_threshold: 0.3
  delimiter_ids: [198]
  use_cache: true
  dual_cache: true
  tau_commit: 0.80
  tau_stable_steps: 2
methods:
  development: [adablock, gates_only, constant_budget, size_lookup, previous_nfe, random_forest, pag_hard_cap, pag]
  final_required: [adablock, gates_only, pag]
  math500: [adablock, pag]
promotion:
  candidates: [constant_budget, size_lookup]
  max_correct_loss: 3
timing:
  warmups: 5
  prompts: 50
  repetitions: 3
statistics:
  bootstrap_samples: 10000
budget:
  reserve_fraction: 0.10
```

- [ ] **Step 5: Implement typed loading and invariant checks**

Implement immutable dataclasses, SHA-256 over canonical JSON, inclusive index-range helpers, and
validation for deterministic decoding, disjoint development/test splits, required methods,
positive timing counts, and a reserve in `[0, 1)`.

- [ ] **Step 6: Run config tests**

Run: `uv run pytest tests/experiments/test_config.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock configs/experiments/neurips_strategy1.yaml \
  src/pag/experiments/__init__.py src/pag/experiments/config.py \
  tests/experiments/test_config.py
git commit -m "feat: freeze NeurIPS experiment protocol"
```

### Task 2: Materialize clean datasets and strict graders

**Files:**
- Create: `src/pag/experiments/datasets.py`
- Create: `src/pag/experiments/grading.py`
- Test: `tests/experiments/test_datasets.py`
- Test: `tests/experiments/test_grading.py`

- [ ] **Step 1: Write failing dataset tests**

```python
def test_development_and_confirmatory_ids_are_disjoint(fake_gsm8k):
    splits = materialize_gsm8k(fake_gsm8k, development=range(6000, 6200), confirmatory=range(400, 1319))
    assert len(splits.development) == 200
    assert len(splits.full_test) == 1319
    assert len(splits.confirmatory) == 919
    assert set(splits.development_ids).isdisjoint(splits.full_test_ids)


def test_math_subset_is_reproducible_and_stratified(fake_math500):
    first = stratified_math500(fake_math500, sample_size=30, seed=20260710)
    second = stratified_math500(fake_math500, sample_size=30, seed=20260710)
    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert {(row.subject, row.level) for row in first} == {(row.subject, row.level) for row in fake_math500}
```

- [ ] **Step 2: Write failing grading tests**

```python
@pytest.mark.parametrize(("text", "gold", "correct"), [
    ("Final answer: 1,234", "1234", True),
    ("I considered 72. Final answer: 772", "72", False),
    ("Final answer: -3/2", "-1.5", True),
    ("The arithmetic contains 42 but no final marker", "42", False),
])
def test_grade_gsm8k(text, gold, correct):
    assert grade_gsm8k(text, gold).is_correct is correct
```

- [ ] **Step 3: Run tests and verify they fail**

Run: `uv run pytest tests/experiments/test_datasets.py tests/experiments/test_grading.py -v`

Expected: FAIL because dataset and grading modules do not exist.

- [ ] **Step 4: Implement dataset records and materialization**

Use a shared frozen record:

```python
@dataclass(frozen=True, slots=True)
class ExperimentSample:
    sample_id: str
    dataset: str
    prompt: str
    gold_answer: str
    subject: str | None = None
    level: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
```

Load pinned revisions with `datasets.load_dataset`, extract GSM8K gold text after `####`, build
development indices 6000--6199, full test 0--1318, confirmatory test 400--1318, and implement
largest-remainder proportional sampling over `(subject, level)` for MATH-500.

- [ ] **Step 5: Implement strict graders**

Return a structured `GradeResult(is_correct, extracted_answer, gold_answer, error)`. GSM8K must
require the last `Final answer:` marker and compare normalized `fractions.Fraction` values.
MATH-500 must call `math_verify.parse` and `math_verify.verify`, catch parse exceptions, and retain
the exception string.

- [ ] **Step 6: Run dataset and grader tests**

Run: `uv run pytest tests/experiments/test_datasets.py tests/experiments/test_grading.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pag/experiments/datasets.py src/pag/experiments/grading.py \
  tests/experiments/test_datasets.py tests/experiments/test_grading.py
git commit -m "feat: add clean evaluation splits and strict grading"
```

### Task 3: Add atomic artifacts and budget control

**Files:**
- Create: `src/pag/experiments/records.py`
- Create: `src/pag/experiments/budget.py`
- Test: `tests/experiments/test_records.py`
- Test: `tests/experiments/test_budget.py`

- [ ] **Step 1: Write failing atomic-write and resume tests**

```python
def test_record_round_trip_and_resume(tmp_path):
    store = RecordStore(tmp_path, identity={"config_hash": "abc"})
    store.write("development", "pag", "sample-1", {"total_nfe": 9})
    assert store.is_complete("development", "pag", "sample-1")
    assert store.read("development", "pag", "sample-1")["total_nfe"] == 9


def test_mismatched_record_is_quarantined(tmp_path):
    first = RecordStore(tmp_path, identity={"config_hash": "old"})
    first.write("development", "pag", "sample-1", {"total_nfe": 9})
    second = RecordStore(tmp_path, identity={"config_hash": "new"})
    assert not second.is_complete("development", "pag", "sample-1")
    assert list((tmp_path / "quarantine").glob("*.json"))
```

- [ ] **Step 2: Write failing budget tests**

```python
def test_stage_rejected_when_projection_crosses_reserve(fake_clock):
    guard = BudgetGuard(budget_usd=20, hourly_rate=0.35, reserve_fraction=0.10, clock=fake_clock)
    fake_clock.advance(hours=50)
    decision = guard.can_start(stage="final", projected_seconds=3 * 3600)
    assert not decision.allowed
    assert decision.reason == "budget_reserve"
```

- [ ] **Step 3: Implement atomic storage**

Write JSON to a sibling `.tmp`, flush, `os.fsync`, and `os.replace`. Embed identity fields in every
record. Quarantine invalid JSON and identity mismatches. Write manifests atomically with stage
status, controlled-stop reason, environment metadata, selected IDs, and spend estimate.

- [ ] **Step 4: Implement cost admission**

Use `time.monotonic`, calculate total instance time, estimated USD, usable ceiling after reserve,
and a structured decision. Stage projections use observed seconds per completed prompt-method plus
a 25% variance multiplier.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/experiments/test_records.py tests/experiments/test_budget.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pag/experiments/records.py src/pag/experiments/budget.py \
  tests/experiments/test_records.py tests/experiments/test_budget.py
git commit -m "feat: add resumable artifacts and cost guard"
```

### Task 4: Implement scheduler ablation policies

**Files:**
- Create: `AdaBlock-dLLM/llada/scheduler_variants.py`
- Test: `tests/llada/test_scheduler_variants.py`

- [ ] **Step 1: Write failing scheduler tests**

```python
def test_constant_scheduler_uses_trace_median_after_seed(trace_stats):
    scheduler = ConstantBudgetScheduler(seed_budget=7, content_budget=5, delimiter_budget=1)
    assert scheduler.next_schedule(block_size=16, remaining_tokens=64, max_block_length=64, max_refinement_steps=64).budgeted_refinement_steps == 7
    scheduler.record_realized(16, 6)
    assert scheduler.next_schedule(block_size=16, remaining_tokens=48, max_block_length=64, max_refinement_steps=64).budgeted_refinement_steps == 5


def test_previous_nfe_uses_last_content_block():
    scheduler = PreviousNFEScheduler(seed_budget=7)
    scheduler.next_schedule(block_size=16, remaining_tokens=64, max_block_length=64, max_refinement_steps=64)
    scheduler.record_realized(16, 9)
    assert scheduler.next_schedule(block_size=1, remaining_tokens=48, max_block_length=64, max_refinement_steps=64).budgeted_refinement_steps == 1
    scheduler.record_realized(1, 1)
    assert scheduler.next_schedule(block_size=12, remaining_tokens=47, max_block_length=64, max_refinement_steps=64).budgeted_refinement_steps == 9
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/llada/test_scheduler_variants.py -v`

Expected: FAIL because `scheduler_variants` does not exist.

- [ ] **Step 3: Implement shared schedule records and simple policies**

Expose `ScheduledBlock`, `GatesOnlyScheduler`, `ConstantBudgetScheduler`,
`SizeLookupBudgetScheduler`, and `PreviousNFEScheduler`. Each scheduler implements `reset`,
`next_schedule`, `record_realized`, `prediction_trace`, and `scheduler_predict_time_sec` with the
same call shape used by `generate_pag.py`.

- [ ] **Step 4: Implement trace statistics and RF policy**

Read the 5,000 training JSONL traces, separate delimiter blocks (`block_size == 1` or
`delimiter_fraction >= 0.5`), calculate constant medians and per-size medians with global fallback,
and build rolling RF examples from the previous eight blocks. Use deterministic
`RandomForestRegressor(n_estimators=200, max_depth=15, random_state=20260710, n_jobs=-1)` over
last/mean/std/min/max/trend for NFE and block size plus confidence and token-type summaries. Clamp
rounded predictions to `[1, max_refinement_steps]`.

- [ ] **Step 5: Run scheduler tests**

Run: `uv run pytest tests/llada/test_scheduler_variants.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add AdaBlock-dLLM/llada/scheduler_variants.py tests/llada/test_scheduler_variants.py
git commit -m "feat: add scheduler attribution baselines"
```

### Task 5: Add explicit budget-enforcement modes

**Files:**
- Modify: `AdaBlock-dLLM/llada/generate_pag.py`
- Modify: `tests/llada/test_generate_pag.py`

- [ ] **Step 1: Add failing enforcement tests**

Add tests proving:

```python
_, _, _, schedule_history = generate_pag(
    model,
    input_ids,
    scheduler,
    steps=4,
    gen_length=2,
    threshold=0.8,
    max_block_length=2,
    max_refinement_steps=4,
    enforcement_mode="hard_cap",
)
assert schedule_history[0]["exit_reason"] == "hard_budget"
assert schedule_history[0]["actual_nfe_used"] == 2

_, _, _, schedule_history = generate_pag(
    model,
    input_ids,
    scheduler,
    steps=4,
    gen_length=2,
    threshold=0.8,
    max_block_length=2,
    max_refinement_steps=4,
    enforcement_mode="soft_gate",
)
assert schedule_history[0]["exit_reason"] in {"complete", "confidence_gate", "stability_gate", "hard_max"}
```

Also assert invalid modes raise `ValueError` and all existing default behavior remains unchanged.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/llada/test_generate_pag.py -v`

Expected: FAIL because `enforcement_mode` and `exit_reason` are absent.

- [ ] **Step 3: Extract a pure enforcement decision helper**

Add:

```python
@dataclass(frozen=True, slots=True)
class EnforcementDecision:
    force_commit: bool
    reason: str | None


def decide_budget_enforcement(*, mode, nfe, budget, max_steps, confident, stable, complete):
    if complete:
        return EnforcementDecision(False, "complete")
    if nfe >= max_steps:
        return EnforcementDecision(True, "hard_max")
    if nfe < budget:
        return EnforcementDecision(False, None)
    if mode == "hard_cap":
        return EnforcementDecision(True, "hard_budget")
    if mode != "soft_gate":
        raise ValueError(f"Unsupported enforcement mode: {mode}")
    if confident:
        return EnforcementDecision(True, "confidence_gate")
    if stable:
        return EnforcementDecision(True, "stability_gate")
    return EnforcementDecision(False, None)
```

`hard_cap` commits at `nfe >= budget`; `soft_gate` commits after budget only when confidence or
stability passes; `hard_max` always commits at the global cap; completed blocks report `complete`.

- [ ] **Step 4: Apply the helper to uncached, prefix-cache, and dual-cache paths**

Preserve their cache-specific model calls. Add `enforcement_mode="soft_gate"` to all three public
signatures and record `exit_reason` in every schedule row. Make the default reproduce current PAG
behavior.

- [ ] **Step 5: Run LLaDA generation tests**

Run: `uv run pytest tests/llada/test_generate_pag.py tests/llada/test_run_pag_vs_adablock_eval.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add AdaBlock-dLLM/llada/generate_pag.py tests/llada/test_generate_pag.py
git commit -m "feat: expose PAG budget enforcement modes"
```

### Task 6: Build the model runtime and stage orchestrator

**Files:**
- Create: `src/pag/experiments/runtime.py`
- Create: `src/pag/experiments/orchestrator.py`
- Create: `scripts/run_neurips_strategy1.py`
- Test: `tests/experiments/test_runtime.py`
- Test: `tests/experiments/test_orchestrator.py`

- [ ] **Step 1: Write failing mocked runtime tests**

```python
def test_runtime_runs_baseline_once_and_reuses_its_seed(fake_backend, sample):
    runtime = ExperimentRuntime(fake_backend)
    baseline = runtime.run(sample, "adablock")
    pag = runtime.run(sample, "pag", baseline_seed=baseline.first_block_seed)
    assert pag.method == "pag"
    assert pag.seed == baseline.first_block_seed


def test_orchestrator_resumes_completed_records(fake_runtime, store, config_with_adablock_only):
    store.write("development", "adablock", "sample-1", fake_record("sample-1"))
    run_development(
        config_with_adablock_only,
        fake_runtime,
        store,
        [sample("sample-1"), sample("sample-2")],
    )
    assert fake_runtime.calls == [("adablock", "sample-2")]
```

- [ ] **Step 2: Implement the runtime adapter**

Load LLaDA modules by adding `AdaBlock-dLLM/llada` to `sys.path`. Precompute digit and delimiter
token tensors once. Dispatch AdaBlock to `generate_adablock_dual_cache` and controlled methods to
`generate_pag_dual_cache` with the correct scheduler and enforcement mode. Synchronize CUDA around
timed regions, reset peak memory for timing trials, decode only generated tokens, grade immediately,
and return a structured record with raw text, block histories, NFE, seed, exit reasons, latency,
predictor time, and memory.

- [ ] **Step 3: Implement promotion**

Calculate correct-count loss and mean NFE for constant and size lookup. Enforce the approved
eligibility and tie-break rules and write `selection.json` before any final evaluation.

- [ ] **Step 4: Implement stage order and controlled stopping**

Stages are `preflight`, `development`, `promotion`, `gsm8k_test`, `math500`, `timing`, and `report`.
Before each stage, update the manifest and ask `BudgetGuard` for admission. Handle SIGINT/SIGTERM by
finishing the active atomic record and setting `controlled_stop`. Retry one failed generation after
`torch.cuda.empty_cache`; stop after a second failure.

- [ ] **Step 5: Implement the CLI**

Parse the approved arguments plus `--config`, `--run-id`, `--output-root`, and `--resume`. Resolve
relative paths from repository root, load the model once, and print the artifact path, current
stage, completed/total count, elapsed hours, and estimated USD.

- [ ] **Step 6: Run mocked tests**

Run: `uv run pytest tests/experiments/test_runtime.py tests/experiments/test_orchestrator.py -v`

Expected: PASS without CUDA or model downloads.

- [ ] **Step 7: Commit**

```bash
git add src/pag/experiments/runtime.py src/pag/experiments/orchestrator.py \
  scripts/run_neurips_strategy1.py tests/experiments/test_runtime.py \
  tests/experiments/test_orchestrator.py
git commit -m "feat: orchestrate resumable NeurIPS experiments"
```

### Task 7: Produce paired statistics and paper assets

**Files:**
- Create: `src/pag/experiments/statistics.py`
- Create: `src/pag/experiments/report.py`
- Test: `tests/experiments/test_statistics.py`
- Test: `tests/experiments/test_report.py`

- [ ] **Step 1: Write failing statistics tests**

```python
def test_correctness_matrix_and_mcnemar():
    matrix = correctness_matrix([True, True, False, False], [True, False, True, False])
    assert asdict(matrix) == {"both_correct": 1, "left_only": 1, "right_only": 1, "both_wrong": 1}
    assert exact_mcnemar(matrix).pvalue == 1.0


def test_paired_bootstrap_is_reproducible():
    first = paired_bootstrap([1, 2, 3], [2, 4, 6], samples=1000, seed=20260710)
    second = paired_bootstrap([1, 2, 3], [2, 4, 6], samples=1000, seed=20260710)
    assert first == second
```

- [ ] **Step 2: Implement statistical functions**

Use NumPy's seeded generator for paired resampling, SciPy's beta/binomial utilities for Wilson and
exact two-sided McNemar results, and explicit sample-ID joins that reject missing or duplicate
records. Calculate all metrics separately for full GSM8K and confirmatory IDs 400--1318.

- [ ] **Step 3: Implement report outputs**

Write stable-column CSVs with `csv.DictWriter`, canonical JSON, escaped LaTeX tables, and
Matplotlib PDF figures for NFE delta histogram, NFE parity, accuracy/NFE trade-off, and paired
latency distribution. Report parse failures and forbid a speedup claim unless the latency CI
excludes zero in the favorable direction.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/experiments/test_statistics.py tests/experiments/test_report.py -v`

Expected: PASS and byte-stable JSON/CSV/TeX fixtures.

- [ ] **Step 5: Commit**

```bash
git add src/pag/experiments/statistics.py src/pag/experiments/report.py \
  tests/experiments/test_statistics.py tests/experiments/test_report.py
git commit -m "feat: generate paired workshop evidence"
```

### Task 8: Add preflight and live smoke gating

**Files:**
- Modify: `src/pag/experiments/orchestrator.py`
- Modify: `src/pag/experiments/runtime.py`
- Test: `tests/experiments/test_preflight.py`

- [ ] **Step 1: Write failing preflight tests**

```python
def test_preflight_fails_before_generation_on_checkpoint_hash_mismatch(fake_context):
    result = run_preflight(fake_context.with_bad_checkpoint())
    assert not result.ok
    assert "checkpoint" in result.errors[0]
    assert fake_context.runtime.calls == []


def test_preflight_requires_ten_gib_free(fake_context):
    result = run_preflight(fake_context.with_free_bytes(9 * 1024**3))
    assert not result.ok
    assert any("10 GiB" in error for error in result.errors)
```

- [ ] **Step 2: Implement cheap checks before model loading**

Validate config, checkpoint existence/SHA-256/schema, trace files and IDs, writable output, disk
space, dependencies, dataset schemas, Math-Verify fixtures, CUDA presence, and 48 GB-class memory.

- [ ] **Step 3: Implement paid live smoke checks**

After model loading, run two development prompts through AdaBlock and full PAG, verify finite logits,
nonempty blocks, complete generation length, valid grades, dual-cache operation, and compatible
checkpoint features. Use observed duration to project every remaining stage with a 25% margin.

- [ ] **Step 4: Run preflight tests**

Run: `uv run pytest tests/experiments/test_preflight.py -v`

Expected: PASS without a real GPU through injected system probes.

- [ ] **Step 5: Commit**

```bash
git add src/pag/experiments/orchestrator.py src/pag/experiments/runtime.py \
  tests/experiments/test_preflight.py
git commit -m "feat: gate paid experiments with preflight"
```

### Task 9: Verify the one-command handoff

**Files:**
- Modify: `README.md`
- Modify: `Makefile`
- Test: all affected tests

- [ ] **Step 1: Add a dry-run CLI test**

Run:

```bash
uv run python scripts/run_neurips_strategy1.py \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --predictor-ckpt output/ablations/medium_ws8_d64_h4_l4_dp10_lr0.5_bestval=2.216957.pt \
  --device cpu --budget-usd 20 --gpu-rate 0.35 --dry-run
```

Expected: exit 0 after printing seven stages, eight development methods, four final methods, 300
MATH examples, the 90% usable budget, and no model load.

- [ ] **Step 2: Document the exact GPU command and resume behavior**

Add the final command, expected artifact path, controlled-stop exit behavior, `--run-id` resume
example, and a warning to keep the Thunder instance alive only while the runner is active.

- [ ] **Step 3: Add focused Make targets**

```makefile
test-neurips:
	uv run pytest tests/experiments tests/llada/test_scheduler_variants.py tests/llada/test_generate_pag.py -v

run-neurips-dry:
	uv run python scripts/run_neurips_strategy1.py --dry-run
```

- [ ] **Step 4: Run focused verification**

Run: `make test-neurips`

Expected: PASS.

- [ ] **Step 5: Run regression verification**

Run: `make test-integration && make lint && make format && git diff --check`

Expected: all commands exit 0; formatting creates no semantic changes.

- [ ] **Step 6: Run the final dry run**

Run the command from Step 1.

Expected: exit 0 with the complete protocol summary and projected artifact directory.

- [ ] **Step 7: Commit**

```bash
git add README.md Makefile scripts/run_neurips_strategy1.py
git commit -m "docs: hand off one-command NeurIPS run"
```

## GPU Execution Command

After copying the repository and the predictor checkpoint to the Thunder Compute instance, run:

```bash
uv run python scripts/run_neurips_strategy1.py --model-path GSAI-ML/LLaDA-8B-Instruct --predictor-ckpt output/ablations/medium_ws8_d64_h4_l4_dp10_lr0.5_bestval=2.216957.pt --device cuda --budget-usd 20 --gpu-rate 0.35
```

Use the same command with the emitted `--run-id` if the process is interrupted.

# Cross-Model Residual PAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally verify a resumable, budget-capped LLaDA/Dream experiment that evaluates risk-controlled residual PAG on frozen GSM8K and MATH-500 splits.

**Architecture:** A shared residual scheduler starts from the size-conditioned trace median and uses a random-forest quantile to predict a bounded NFE correction from rolling realized phase features. LLaDA and Dream retain AdaBlock boundaries and expose the same scheduler protocol; a cross-model orchestrator separates calibration, policy freezing, confirmatory generation, and claim-gated reporting.

**Tech Stack:** Python 3.11, PyTorch, scikit-learn, Hugging Face datasets/transformers, NumPy, pytest, Ruff, YAML, LaTeX.

---

## File Map

- Create `src/pag/experiments/residual.py`: residual examples, estimator persistence, quantile inference, and shared scheduler.
- Create `src/pag/experiments/cross_model_config.py`: schema and validation for the frozen protocol.
- Create `src/pag/experiments/cross_model_runtime.py`: lazy LLaDA/Dream runtime adapters and normalized generation records.
- Create `src/pag/experiments/cross_model_orchestrator.py`: staged calibration, freezing, confirmatory execution, resume, and cost control.
- Create `src/pag/experiments/claim_audit.py`: pure success-gate evaluation.
- Create `scripts/run_neurips_cross_model.py`: one-command paid-run entrypoint.
- Create `configs/experiments/neurips_cross_model.yaml`: immutable sample ranges, candidates, models, and thresholds.
- Modify `src/pag/experiments/datasets.py`: fresh GSM8K-train split and complement MATH-500 materialization.
- Modify `src/pag/experiments/report.py`: cross-model summaries, tables, plots, and claim audit output.
- Modify `AdaBlock-dLLM/dream/model/generation_utils_pag.py`: pass rich realized features to the scheduler.
- Modify `AdaBlock-dLLM/dream/eval_dream_pag.py`: construct tokenizer-specific digit/delimiter tensors.
- Modify `README.md` and `writeup/final_report.tex`: concise command/protocol documentation and removal of invalidated claims.
- Add focused tests under `tests/experiments/` and `tests/dream/`.

### Task 1: Land Corrected Grading and Immutable Dataset Splits

**Files:**
- Modify: `src/pag/experiments/grading.py`
- Modify: `src/pag/experiments/datasets.py`
- Create: `scripts/regrade_neurips_results.py`
- Modify: `tests/experiments/test_grading.py`
- Modify: `tests/experiments/test_datasets.py`

- [ ] **Step 1: Add failing split tests**

```python
def test_materialize_fresh_gsm8k_uses_train_for_both_splits():
    rows = [{"question": f"q{i}", "answer": f"work #### {i}"} for i in range(8)]
    splits = materialize_fresh_gsm8k(rows, calibration=range(2, 4), test=range(4, 8))
    assert [row.metadata["index"] for row in splits.calibration] == [2, 3]
    assert [row.metadata["index"] for row in splits.test] == [4, 5, 6, 7]
    assert set(splits.calibration_ids).isdisjoint(splits.test_ids)


def test_math500_complement_excludes_prior_selection():
    rows = make_math_rows(10)
    selected = stratified_math500(rows, sample_size=6, seed=7)
    complement = complement_math500(rows, excluded_ids={row.sample_id for row in selected})
    assert len(complement) == 4
    assert {row.sample_id for row in complement}.isdisjoint(row.sample_id for row in selected)
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `uv run pytest tests/experiments/test_datasets.py tests/experiments/test_grading.py -q`

Expected: failure because `materialize_fresh_gsm8k` and `complement_math500` do not exist.

- [ ] **Step 3: Implement immutable split helpers**

```python
@dataclass(frozen=True, slots=True)
class FreshGSM8KSplits:
    calibration: tuple[ExperimentSample, ...]
    test: tuple[ExperimentSample, ...]

    @property
    def calibration_ids(self) -> tuple[str, ...]:
        return tuple(row.sample_id for row in self.calibration)

    @property
    def test_ids(self) -> tuple[str, ...]:
        return tuple(row.sample_id for row in self.test)


def materialize_fresh_gsm8k(train_rows, *, calibration, test):
    calibration_rows = tuple(_gsm8k_sample(train_rows[i], split="train", index=i) for i in calibration)
    test_rows = tuple(_gsm8k_sample(train_rows[i], split="train", index=i) for i in test)
    if set(row.sample_id for row in calibration_rows) & set(row.sample_id for row in test_rows):
        raise ValueError("fresh GSM8K calibration and test IDs overlap")
    return FreshGSM8KSplits(calibration_rows, test_rows)


def complement_math500(rows, *, excluded_ids):
    all_rows = stratified_math500(rows, sample_size=len(rows), seed=0)
    return tuple(row for row in all_rows if row.sample_id not in excluded_ids)
```

- [ ] **Step 4: Verify focused grading and dataset tests**

Run: `uv run pytest tests/experiments/test_datasets.py tests/experiments/test_grading.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pag/experiments/grading.py src/pag/experiments/datasets.py \
  scripts/regrade_neurips_results.py tests/experiments/test_grading.py \
  tests/experiments/test_datasets.py
git commit -m "fix: grade and freeze confirmatory datasets"
```

### Task 2: Implement the Shared Residual Scheduler

**Files:**
- Create: `src/pag/experiments/residual.py`
- Create: `tests/experiments/test_residual.py`

- [ ] **Step 1: Write failing scheduler tests**

```python
def test_residual_scheduler_uses_lower_tree_quantile_and_clamps_correction():
    estimator = FakeEstimator(tree_predictions=[-4.0, -2.0, 1.0, 2.0])
    scheduler = ResidualBudgetScheduler(
        seed_budget=8,
        stats=TraceBudgetStats(8, 1, {16: 8}),
        estimator=estimator,
        quantile=0.25,
        max_abs_correction=2,
    )
    scheduler.record_realized(16, 7, 0.9, 0.7, 0.2, 0.0)
    block = scheduler.next_schedule(
        block_size=16, remaining_tokens=64, max_block_length=64, max_refinement_steps=32
    )
    assert block.budgeted_refinement_steps == 6


def test_residual_estimator_round_trip(tmp_path):
    stats = TraceBudgetStats(8, 1, {16: 8})
    estimator = ResidualEstimator.fit(example_sequences(), stats, seed=11, n_estimators=8)
    before = estimator.tree_predictions(history_rows(), block_size=16)
    path = tmp_path / "residual.joblib"
    estimator.save(path)
    after = ResidualEstimator.load(path).tree_predictions(history_rows(), block_size=16)
    np.testing.assert_allclose(after, before)
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run: `uv run pytest tests/experiments/test_residual.py -q`

Expected: collection fails because `pag.experiments.residual` does not exist.

- [ ] **Step 3: Implement residual targets and estimator**

```python
@dataclass(frozen=True, slots=True)
class ResidualPolicyConfig:
    quantile: float
    max_abs_correction: int
    window_size: int = 8


class ResidualEstimator:
    @classmethod
    def fit(cls, sequences, stats, *, seed, n_estimators=200):
        features, targets = residual_training_matrix(sequences, stats)
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=15, random_state=seed, n_jobs=-1
        )
        model.fit(features, targets)
        return cls(model=model, stats=stats)

    def tree_predictions(self, history, *, block_size):
        row = residual_features(history, block_size=block_size)[None, :]
        return np.asarray([tree.predict(row)[0] for tree in self.model.estimators_])
```

- [ ] **Step 4: Implement scheduler protocol and traceability**

```python
residual = float(np.quantile(self.estimator.tree_predictions(self._history, block_size=size), self.quantile))
correction = int(np.clip(round(residual), -self.max_abs_correction, self.max_abs_correction))
prior = size_lookup_budget(self.stats, size)
budget = min(max_refinement_steps, max(1, prior + correction))
self.prediction_trace.append({
    "source": "residual_pag",
    "prior_budget": prior,
    "residual_quantile": residual,
    "applied_correction": correction,
    "budgeted_refinement_steps": budget,
})
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/experiments/test_residual.py tests/llada/test_scheduler_variants.py -q`

Expected: all tests pass.

```bash
git add src/pag/experiments/residual.py tests/experiments/test_residual.py
git commit -m "feat: add risk-controlled residual scheduler"
```

### Task 3: Bring Dream to Scheduler Feature Parity

**Files:**
- Modify: `AdaBlock-dLLM/dream/model/generation_utils_pag.py`
- Modify: `AdaBlock-dLLM/dream/eval_dream_pag.py`
- Modify: `tests/dream/test_generation_utils_pag.py`
- Modify: `tests/dream/test_eval_dream_pag.py`

- [ ] **Step 1: Extend the Dream test scheduler to capture six realized fields**

```python
def record_realized(
    self, block_size, nfe, mean_confidence=1.0, min_confidence=1.0,
    digit_fraction=0.0, delimiter_fraction=0.0,
):
    self.recorded.append((
        block_size, nfe, mean_confidence, min_confidence,
        digit_fraction, delimiter_fraction,
    ))
```

Assert that the first record has finite confidences in `[0, 1]` and fractions in `[0, 1]`.

- [ ] **Step 2: Run Dream tests and confirm the old two-argument calls fail assertions**

Run: `uv run pytest tests/dream/test_generation_utils_pag.py tests/dream/test_eval_dream_pag.py -q`

Expected: failure because Dream supplies only block size and NFE.

- [ ] **Step 3: Compute realized features before each Dream scheduler update**

```python
block_confidence = confidence[:, block_start:block_end]
mean_conf = float(block_confidence.mean().item())
min_conf = float(block_confidence.min().item())
block_tokens = output_ids[:, block_start:block_end]
digit_frac = token_fraction(block_tokens, self.pag_digit_ids)
delim_frac = token_fraction(block_tokens, self.pag_delimiter_ids)
self.pag_scheduler.record_realized(
    schedule.applied_block_size, nfe, mean_conf, min_conf, digit_frac, delim_frac
)
```

- [ ] **Step 4: Initialize tokenizer-specific Dream token tensors**

Use decoded vocabulary entries to populate `pag_digit_ids` and Dream delimiter token IDs on the
model device, matching the LLaDA runtime's token-class definitions.

- [ ] **Step 5: Verify both cached and uncached Dream paths and commit**

Run: `uv run pytest tests/dream -q`

Expected: all Dream tests pass.

```bash
git add AdaBlock-dLLM/dream/model/generation_utils_pag.py \
  AdaBlock-dLLM/dream/eval_dream_pag.py tests/dream
git commit -m "feat: record rich Dream phase features"
```

### Task 4: Add and Validate the Frozen Cross-Model Configuration

**Files:**
- Create: `src/pag/experiments/cross_model_config.py`
- Create: `configs/experiments/neurips_cross_model.yaml`
- Create: `tests/experiments/test_cross_model_config.py`

- [ ] **Step 1: Write validation tests**

```python
def test_config_rejects_overlap_and_wrong_headline_threshold(tmp_path):
    payload = valid_payload()
    payload["datasets"]["gsm8k"]["test_indices"] = [6250, 6350]
    with pytest.raises(ValueError, match="disjoint"):
        validate_cross_model_config(payload)
    payload = valid_payload()
    payload["claim_gates"]["minimum_nfe_reduction"] = 0.09
    with pytest.raises(ValueError, match="0.10"):
        validate_cross_model_config(payload)
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/experiments/test_cross_model_config.py -q`

Expected: collection fails because the configuration module is absent.

- [ ] **Step 3: Implement strict schema and YAML**

The YAML must encode exactly:

```yaml
schema_version: 1
seed: 20260712
models:
  llada: GSAI-ML/LLaDA-8B-Instruct
  dream: Dream-org/Dream-v0-Base-7B
datasets:
  gsm8k:
    calibration_indices: [6200, 6299]
    test_indices: [6300, 6699]
  math500:
    prior_selection_manifest: artifacts/neurips_strategy1/strategy1-b36c0c38-fd7a9aaa-5a6f6c2a/selected_samples.json
    expected_complement: 200
policy:
  quantiles: [0.15, 0.25, 0.35]
  max_abs_corrections: [1, 2, 3]
  n_estimators: 200
methods:
  confirmatory: [adablock, size_lookup, residual_pag]
claim_gates:
  minimum_nfe_reduction: 0.10
  minimum_lookup_reduction: 0.03
  minimum_accuracy_ci: -0.02
budget:
  usd: 19.0
  gpu_rate: 0.35
  reserve_fraction: 0.05
```

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/experiments/test_cross_model_config.py -q`

Expected: all tests pass.

```bash
git add src/pag/experiments/cross_model_config.py \
  configs/experiments/neurips_cross_model.yaml tests/experiments/test_cross_model_config.py
git commit -m "feat: freeze cross-model experiment protocol"
```

### Task 5: Implement Runtime Adapters and Resumable Orchestration

**Files:**
- Create: `src/pag/experiments/cross_model_runtime.py`
- Create: `src/pag/experiments/cross_model_orchestrator.py`
- Create: `scripts/run_neurips_cross_model.py`
- Create: `tests/experiments/test_cross_model_runtime.py`
- Create: `tests/experiments/test_cross_model_orchestrator.py`

- [ ] **Step 1: Write fake-runtime resume and leakage tests**

```python
def test_resume_runs_only_missing_keys(tmp_path):
    runtime = FakeCrossModelRuntime()
    runner = make_runner(tmp_path, runtime=runtime)
    runner.store.write("test_gsm8k/llada", "adablock", "gsm8k_train_6300", record())
    runner.run_confirmatory()
    assert ("llada", "adablock", "gsm8k_train_6300") not in runtime.calls


def test_selection_reads_only_calibration_records(tmp_path):
    runner = make_runner(tmp_path)
    runner.store.write_named("confirmatory/poison.json", {"quantile": 0.15, "accuracy": 1.0})
    assert runner.select_policy(calibration_records()) == runner.select_policy(calibration_records())
```

- [ ] **Step 2: Run and confirm missing modules**

Run: `uv run pytest tests/experiments/test_cross_model_runtime.py tests/experiments/test_cross_model_orchestrator.py -q`

Expected: collection fails because runtime and orchestrator modules are absent.

- [ ] **Step 3: Implement a normalized runtime protocol**

```python
class CrossModelRuntime(Protocol):
    def load_model(self, model_name: str) -> None: ...
    def unload_model(self) -> None: ...
    def run(self, sample, method, *, policy_path=None, baseline_seed=None) -> GenerationRecord: ...
```

LLaDA delegates to the existing `ExperimentRuntime`. Dream lazily imports the existing evaluation
classes, runs at batch size one, and returns the same `GenerationRecord` fields. Neither adapter may
load both model weights simultaneously.

- [ ] **Step 4: Implement explicit stages**

```python
STAGES = (
    "freeze_samples",
    "calibration_adablock",
    "fit_estimators",
    "calibration_candidates",
    "freeze_policy",
    "test_gsm8k",
    "test_math500",
    "report",
)
```

Before `test_gsm8k`, require a frozen policy JSON containing the selected quantile, correction bound,
estimator hashes, calibration record hash, configuration hash, and `frozen_at`. Confirmatory methods
must read this artifact and must not import selection functions.

- [ ] **Step 5: Implement atomic resume and budget admission**

Write each record via the existing `RecordStore` atomic path, using `stage/model` as the stage key.
The resulting path is `stage/model/method/sample_id`; reject duplicate mismatched payloads and use observed per-model,
per-method medians for projected remaining cost. Raise `ControlledStop` before an inadmissible stage.

- [ ] **Step 6: Add the one-line CLI**

```python
parser.add_argument("--config", default="configs/experiments/neurips_cross_model.yaml")
parser.add_argument("--device", default="cuda")
parser.add_argument("--output-root", default="artifacts/neurips_cross_model")
args = parser.parse_args()
raise SystemExit(run_from_args(args))
```

- [ ] **Step 7: Verify mock orchestration and commit**

Run: `uv run pytest tests/experiments/test_cross_model_runtime.py tests/experiments/test_cross_model_orchestrator.py -q`

Expected: all tests pass without loading a real model.

```bash
git add src/pag/experiments/cross_model_runtime.py \
  src/pag/experiments/cross_model_orchestrator.py scripts/run_neurips_cross_model.py \
  tests/experiments/test_cross_model_runtime.py \
  tests/experiments/test_cross_model_orchestrator.py
git commit -m "feat: orchestrate resumable cross-model evaluation"
```

### Task 6: Add Claim-Gated Cross-Model Reporting

**Files:**
- Create: `src/pag/experiments/claim_audit.py`
- Modify: `src/pag/experiments/report.py`
- Create: `tests/experiments/test_claim_audit.py`
- Modify: `tests/experiments/test_report.py`

- [ ] **Step 1: Write failing claim-gate tests**

```python
def test_headline_requires_every_model_and_dataset_gate():
    summary = passing_cross_model_summary()
    thresholds = ClaimThresholds(
        minimum_nfe_reduction=0.10,
        minimum_lookup_reduction=0.03,
        minimum_accuracy_ci=-0.02,
    )
    audit = audit_claims(summary, thresholds=thresholds)
    assert audit["headline_eligible"] is True
    summary["dream"]["math500"]["residual_pag_vs_adablock"]["nfe_reduction"] = -0.01
    audit = audit_claims(summary, thresholds=thresholds)
    assert audit["headline_eligible"] is False
    assert "dream/math500/nfe_direction" in audit["failed_gates"]
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/experiments/test_claim_audit.py tests/experiments/test_report.py -q`

Expected: failure because `audit_claims` is absent.

- [ ] **Step 3: Implement pure gate evaluation**

```python
@dataclass(frozen=True, slots=True)
class ClaimThresholds:
    minimum_nfe_reduction: float
    minimum_lookup_reduction: float
    minimum_accuracy_ci: float


headline_eligible = all(gate["passed"] for gate in gates)
return {
    "headline_eligible": headline_eligible,
    "gates": gates,
    "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
}
```

Accuracy passes only when the paired-bootstrap lower bound is at least `-0.02`. NFE reductions use
paired mean totals and positive values mean fewer candidate NFEs.

- [ ] **Step 4: Extend report outputs**

Emit `summary.json`, `claim_audit.json`, model/dataset LaTeX tables, pooled and per-model Pareto PDFs,
NFE delta plots, correctness matrices, and failure-category CSVs. Emit `headline.tex` with neutral
language unless `headline_eligible` is true.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/experiments/test_claim_audit.py tests/experiments/test_report.py -q`

Expected: all tests pass.

```bash
git add src/pag/experiments/claim_audit.py src/pag/experiments/report.py \
  tests/experiments/test_claim_audit.py tests/experiments/test_report.py
git commit -m "feat: gate cross-model PAG claims"
```

### Task 7: Update Concise Run Documentation and Correct the Paper

**Files:**
- Modify: `README.md`
- Modify: `writeup/final_report.tex`
- Create: `tests/experiments/test_paper_claims.py`

- [ ] **Step 1: Add a regression test for invalidated claims**

```python
def test_paper_does_not_retain_invalidated_nfe_claim():
    paper = Path("writeup/final_report.tex").read_text(encoding="utf-8")
    assert "21.4\\%" not in paper
    assert "matches AdaBlock answer accuracy while reducing" not in paper
```

- [ ] **Step 2: Run and confirm the test fails against the current paper**

Run: `uv run pytest tests/experiments/test_paper_claims.py -q`

Expected: failure because the old 21.4% claim remains.

- [ ] **Step 3: Replace claims with audited exploratory results and protocol text**

State the corrected prior evidence: PAG reduces NFE relative to AdaBlock but does not dominate
`size_lookup`; cross-model residual PAG is a preregistered confirmatory experiment whose result is
inserted only from generated report fragments. Do not write anticipated numbers into the paper.

- [ ] **Step 4: Add a concise README section**

Document setup, preflight, one command, resume behavior, artifact paths, $19 cap, and the rule that
`claim_audit.json` controls headline eligibility. Keep the section under 35 lines.

- [ ] **Step 5: Build the paper and commit**

Run: `uv run pytest tests/experiments/test_paper_claims.py -q`

Run: `latexmk -pdf -interaction=nonstopmode -halt-on-error final_report.tex` from `writeup/`.

Expected: test passes and `latexmk` exits 0.

```bash
git add README.md writeup/final_report.tex tests/experiments/test_paper_claims.py
git commit -m "docs: align PAG paper with audited evidence"
```

### Task 8: Full Local Verification and GPU Handoff

**Files:**
- Modify only if verification exposes defects in files from Tasks 1--7.

- [ ] **Step 1: Run focused experiment and model-integration tests**

Run: `uv run pytest tests/experiments tests/llada tests/dream -q`

Expected: all tests pass.

- [ ] **Step 2: Run integration tests**

Run: `make test-integration`

Expected: all integration tests pass.

- [ ] **Step 3: Run targeted lint and formatting checks**

```bash
uv run ruff check src/pag/experiments scripts/run_neurips_cross_model.py \
  tests/experiments tests/llada tests/dream AdaBlock-dLLM/dream/eval_dream_pag.py \
  AdaBlock-dLLM/dream/model/generation_utils_pag.py
uv run ruff format --check src/pag/experiments scripts/run_neurips_cross_model.py \
  tests/experiments tests/llada tests/dream AdaBlock-dLLM/dream/eval_dream_pag.py \
  AdaBlock-dLLM/dream/model/generation_utils_pag.py
```

Expected: both commands exit 0. Repository-wide legacy lint failures outside this scope are reported
separately and are not silently fixed.

- [ ] **Step 4: Run CPU-only preflight and mock end-to-end execution**

Run: `uv run python scripts/run_neurips_cross_model.py --preflight-only --device cpu`

Expected: dataset/config/artifact checks pass; CUDA and remote-model checks are explicitly reported as
skipped in CPU preflight.

Run: `uv run pytest tests/experiments/test_cross_model_orchestrator.py -q`

Expected: mock run reaches report generation and resume performs zero duplicate generations.

- [ ] **Step 5: Verify the exact paid command parses without starting GPU work**

Run: `uv run python scripts/run_neurips_cross_model.py --help`

Expected: exits 0 and documents config, device, output root, budget, rate, resume, and preflight flags.

- [ ] **Step 6: Hand off this one-line GPU command**

```bash
TOKENIZERS_PARALLELISM=false uv run python scripts/run_neurips_cross_model.py --config configs/experiments/neurips_cross_model.yaml --device cuda --budget-usd 19 --gpu-rate 0.35
```

Do not claim main-conference headline eligibility before the completed run's
`report/claim_audit.json` says `headline_eligible: true`.

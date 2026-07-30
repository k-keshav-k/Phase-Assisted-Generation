# Risk-Calibrated PAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally verified, one-A100 workflow for training, calibrating, evaluating, and
writing up risk-calibrated phase-adaptive stopping for LLaDA and Dream.

**Architecture:** Pure feature, policy, and risk-control modules sit under `pag.experiments`; model
adapters translate LLaDA and Dream refinement states into the shared schema. A staged orchestrator
freezes policy families and data manifests, writes atomic per-sample artifacts, and generates
claim-gated reports. One CLI is used locally and from a resumable Slurm wrapper.

**Tech Stack:** Python 3.11, PyTorch, scikit-learn, SciPy, NumPy, Hugging Face
datasets/transformers, pytest, Ruff, YAML, Bash/Slurm, LaTeX.

---

## File Map

- Create `src/pag/experiments/rc_pag_features.py`: validated step observations and pure features.
- Create `src/pag/experiments/rc_pag_policy.py`: estimator and stateful stop policy.
- Create `src/pag/experiments/risk_control.py`: exact LTT tests and certificates.
- Create `src/pag/experiments/rc_pag_config.py`: frozen protocol loader.
- Create `configs/experiments/rc_pag_neurips.yaml`: model, data, policy, and claim settings.
- Create `src/pag/experiments/rc_pag_orchestrator.py`: staged CPU/mock execution and manifests.
- Create `src/pag/experiments/rc_pag_report.py`: audit, tables, and plots.
- Create `scripts/run_rc_pag.py`: stage CLI.
- Create `scripts/slurm/rc_pag_a100.sbatch`: A100 worker.
- Create `scripts/slurm/submit_rc_pag.sh`: NYU submission wrapper.
- Modify LLaDA and Dream PAG generation files to expose shared online observations.
- Create `writeup/rc_pag_workshop.tex`: anonymized eight-page workshop manuscript.
- Add focused tests under `tests/experiments/`, `tests/llada/`, and `tests/dream/`.

### Task 1: Shared Online Feature Schema

**Files:**
- Create: `src/pag/experiments/rc_pag_features.py`
- Create: `tests/experiments/test_rc_pag_features.py`

- [ ] **Step 1: Write failing validation and feature tests**

```python
def test_step_observation_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="top1 probabilities"):
        StepObservation.from_arrays(
            step_index=1,
            block_size=4,
            masked=[True, True, False, False],
            top1_probs=[1.2, 0.5, 0.9, 0.8],
            top2_probs=[0.1, 0.2, 0.1, 0.1],
            entropies=[0.1, 0.2, 0.3, 0.4],
            token_ids=[1, 2, 3, 4],
        )


def test_feature_vector_contains_local_and_history_fields():
    previous = observation(step=1, token_ids=[1, 2, 3, 4])
    current = observation(step=2, token_ids=[1, 9, 3, 4])
    history = [RealizedBlock(4, 3, 0.9, 0.6, 0.25, 0.0)]
    features = extract_features(current, previous=previous, history=history, history_window=4)
    assert features["local.token_churn"] == 0.25
    assert features["local.remaining_fraction"] == 0.5
    assert features["history.length"] == 1.0
    assert features["history.nfe_last"] == 3.0
    assert all(math.isfinite(value) for value in features.values())
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `uv run pytest tests/experiments/test_rc_pag_features.py -q`

Expected: collection fails with `ModuleNotFoundError: pag.experiments.rc_pag_features`.

- [ ] **Step 3: Implement immutable observations and feature extraction**

```python
@dataclass(frozen=True, slots=True)
class StepObservation:
    step_index: int
    block_size: int
    masked: tuple[bool, ...]
    top1_probs: tuple[float, ...]
    top2_probs: tuple[float, ...]
    entropies: tuple[float, ...]
    token_ids: tuple[int, ...]
    digit_ids: frozenset[int] = frozenset()
    delimiter_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        lengths = {
            len(self.masked), len(self.top1_probs), len(self.top2_probs),
            len(self.entropies), len(self.token_ids),
        }
        if self.step_index < 1 or self.block_size < 1 or lengths != {self.block_size}:
            raise ValueError("step observation arrays must match a positive block size")
        if any(not 0.0 <= value <= 1.0 for value in self.top1_probs):
            raise ValueError("top1 probabilities must be in [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in self.top2_probs):
            raise ValueError("top2 probabilities must be in [0, 1]")
        if any(not math.isfinite(value) or value < 0 for value in self.entropies):
            raise ValueError("entropies must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RealizedBlock:
    block_size: int
    nfe: int
    mean_confidence: float
    min_confidence: float
    digit_fraction: float
    delimiter_fraction: float
```

Implement quantiles at 0.1, 0.5, and 0.9; masked-token entropy statistics; provisional-token
fractions; token churn; one-step trends; and last/mean/std/trend history summaries. Return a
deterministically ordered `dict[str, float]`. `vectorize_features` must accept a fixed ordered field
tuple and return a finite `numpy.float64` row.

- [ ] **Step 4: Verify feature tests**

Run: `uv run pytest tests/experiments/test_rc_pag_features.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pag/experiments/rc_pag_features.py tests/experiments/test_rc_pag_features.py
git commit -m "feat: add RC-PAG online features"
```

### Task 2: Risk Score and Stateful Stop Policy

**Files:**
- Create: `src/pag/experiments/rc_pag_policy.py`
- Create: `tests/experiments/test_rc_pag_policy.py`

- [ ] **Step 1: Write failing policy tests**

```python
def test_policy_requires_minimum_steps_and_patience():
    policy = RiskStoppingPolicy(FixedScorer(0.02), threshold=0.05, min_steps=2, patience=2)
    assert not policy.observe(observation(step=1)).should_stop
    assert not policy.observe(observation(step=2)).should_stop
    decision = policy.observe(observation(step=3))
    assert decision.should_stop
    assert decision.reason == "risk_certified_candidate"


def test_policy_fallback_never_stops_early():
    policy = RiskStoppingPolicy.full_budget()
    for step in range(1, 9):
        assert not policy.observe(observation(step=step)).should_stop
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/experiments/test_rc_pag_policy.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement estimator and policy**

```python
@dataclass(frozen=True, slots=True)
class StopDecision:
    should_stop: bool
    risk_score: float
    safe_streak: int
    reason: str


class RiskStoppingPolicy:
    def observe(self, observation: StepObservation) -> StopDecision:
        features = extract_features(
            observation,
            previous=self._previous,
            history=self._history if self.include_history else (),
            history_window=self.history_window,
        )
        score = float(self.scorer.predict_risk(features))
        if not 0.0 <= score <= 1.0:
            raise ValueError("risk scorer must return a probability in [0, 1]")
        eligible = observation.step_index >= self.min_steps and score <= self.threshold
        self._safe_streak = self._safe_streak + 1 if eligible else 0
        self._previous = observation
        stop = self._safe_streak >= self.patience
        return StopDecision(stop, score, self._safe_streak,
                            "risk_certified_candidate" if stop else "continue")
```

`RiskEstimator.fit` must train either `HistGradientBoostingClassifier` or `LogisticRegression`, retain
the exact feature order, expose `predict_risk`, and persist with joblib plus a SHA-256 metadata file.
Single-class labels use a deterministic constant-probability scorer rather than failing.

- [ ] **Step 4: Verify persistence and policy behavior**

Run: `uv run pytest tests/experiments/test_rc_pag_policy.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pag/experiments/rc_pag_policy.py tests/experiments/test_rc_pag_policy.py
git commit -m "feat: add RC-PAG stopping policy"
```

### Task 3: Learn-then-Test Risk Certificates

**Files:**
- Create: `src/pag/experiments/risk_control.py`
- Create: `tests/experiments/test_risk_control.py`

- [ ] **Step 1: Write exact-binomial and selection tests**

```python
def test_invalid_candidate_is_not_certified():
    candidate = CandidateRisk("fast", losses=(1,) * 20 + (0,) * 80, mean_nfe=40.0)
    result = certify_candidates((candidate,), alpha=0.05, delta=0.05)
    assert result.selected == "full_budget"
    assert not result.candidates[0].certified


def test_lowest_compute_certified_candidate_wins():
    candidates = (
        CandidateRisk("safe", losses=(0,) * 200, mean_nfe=50.0),
        CandidateRisk("safer", losses=(0,) * 200, mean_nfe=60.0),
    )
    result = certify_candidates(candidates, alpha=0.05, delta=0.05)
    assert result.selected == "safe"
    assert result.familywise_delta == 0.05
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/experiments/test_risk_control.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement predeclared-family certification**

```python
def binomial_null_pvalue(losses: Sequence[int], *, alpha: float) -> float:
    failures = int(sum(losses))
    return float(binomtest(failures, len(losses), p=alpha, alternative="less").pvalue)


def certify_candidates(candidates, *, alpha, delta):
    ordered = tuple(sorted(candidates, key=lambda item: item.name))
    cutoff = delta / len(ordered)
    audits = tuple(
        CandidateAudit(
            name=item.name,
            failures=sum(item.losses),
            count=len(item.losses),
            empirical_risk=sum(item.losses) / len(item.losses),
            pvalue=binomial_null_pvalue(item.losses, alpha=alpha),
            corrected_cutoff=cutoff,
            certified=binomial_null_pvalue(item.losses, alpha=alpha) <= cutoff,
            mean_nfe=item.mean_nfe,
        )
        for item in ordered
    )
    certified = [item for item in audits if item.certified]
    selected = min(certified, key=lambda item: (item.mean_nfe, item.name)).name \
        if certified else "full_budget"
    return RiskCertificate(alpha, delta, selected, audits)
```

Validate binary prompt losses, nonempty equal protocol identity, unique names, and parameters in
`(0, 1)`. Serialize all p-values, cutoffs, empirical risks, and fallback state.

- [ ] **Step 4: Add a seeded null simulation**

Generate 2,000 synthetic calibration families with true risk above `alpha`; assert the observed rate
of selecting any invalid policy is no more than `delta + 0.02` with the fixed seed.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/experiments/test_risk_control.py -q`

```bash
git add src/pag/experiments/risk_control.py tests/experiments/test_risk_control.py
git commit -m "feat: certify RC-PAG policy risk"
```

### Task 4: Frozen RC-PAG Protocol

**Files:**
- Create: `src/pag/experiments/rc_pag_config.py`
- Create: `configs/experiments/rc_pag_neurips.yaml`
- Create: `tests/experiments/test_rc_pag_config.py`

- [ ] **Step 1: Write configuration rejection tests**

```python
def test_config_rejects_overlap_and_unfrozen_risk(tmp_path):
    payload = valid_payload()
    payload["splits"]["calibration"]["gsm8k"] = [0, 99]
    payload["splits"]["training"]["gsm8k"] = [50, 149]
    with pytest.raises(ValueError, match="overlap"):
        validate_rc_pag_config(payload)
    payload = valid_payload()
    payload["risk"]["alpha"] = 0.1
    with pytest.raises(ValueError, match="alpha must remain 0.05"):
        validate_rc_pag_config(payload)
```

- [ ] **Step 2: Implement dataclasses and strict loader**

The YAML must freeze model names, dataset revisions, train/tune/calibration/development ranges,
policy thresholds, minimum steps, patience values, estimator kinds, `alpha=0.05`, `delta=0.05`,
bootstrap count, pilot/trace/calibration sizes, and claim gates. Compute `config_hash` with the existing
canonical hash helper.

- [ ] **Step 3: Materialize the initial protocol**

Use 32 pilot prompts/model, 600 trace prompts/model, 300 calibration prompts/model, six policy
candidates, and 10,000 bootstrap samples. Keep confirmatory benchmark counts explicit and allow a
`--limit` override only in mock/development stages; confirmation rejects limits.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_config.py -q`

```bash
git add src/pag/experiments/rc_pag_config.py configs/experiments/rc_pag_neurips.yaml \
  tests/experiments/test_rc_pag_config.py
git commit -m "feat: freeze RC-PAG protocol"
```

### Task 5: Staged Orchestrator and Mock Runtime

**Files:**
- Create: `src/pag/experiments/rc_pag_orchestrator.py`
- Create: `tests/experiments/test_rc_pag_orchestrator.py`

- [ ] **Step 1: Write resume, freeze, and futility tests**

```python
def test_mock_all_stages_resume_without_duplicate_runs(tmp_path, config):
    runtime = MockRCPAGRuntime()
    runner = RCPAGOrchestrator(config, tmp_path, runtime_factory=lambda model: runtime)
    runner.run_through("report")
    first_calls = tuple(runtime.calls)
    runner.run_through("report")
    assert tuple(runtime.calls) == first_calls


def test_confirmation_requires_certificate(tmp_path, config):
    runner = RCPAGOrchestrator(config, tmp_path, runtime_factory=unsafe_runtime)
    runner.run_through("calibrate")
    with pytest.raises(ControlledStop, match="no certified policy"):
        runner.run_stage("confirm")
```

- [ ] **Step 2: Implement stages and manifests**

Define the ordered stages `preflight`, `pilot`, `collect`, `fit`, `screen`, `calibrate`, `confirm`,
`report`, and `paper`. Each stage reads only outputs of earlier stages, checks config/model/data hashes,
writes a running/completed manifest, and uses `RecordStore` for per-sample records. Freeze
`policy_family.json` before calibration and `risk_certificate.json` afterward.

- [ ] **Step 3: Implement pilot estimates and futility**

Compute seconds/sample and bytes/sample from pilot records. Before each GPU stage, emit projected
A100-hours and storage. After calibration, stop before confirmation if no nonfallback policy is
certified or if the selected candidate has no development NFE improvement over the strongest eligible
heuristic.

- [ ] **Step 4: Verify mock end to end**

Run: `uv run pytest tests/experiments/test_rc_pag_orchestrator.py -q`

Expected: complete mock artifacts, idempotent resume, controlled unsafe fallback.

- [ ] **Step 5: Commit**

```bash
git add src/pag/experiments/rc_pag_orchestrator.py \
  tests/experiments/test_rc_pag_orchestrator.py
git commit -m "feat: orchestrate RC-PAG experiments"
```

### Task 6: Claim-Gated Reporting

**Files:**
- Create: `src/pag/experiments/rc_pag_report.py`
- Create: `tests/experiments/test_rc_pag_report.py`

- [ ] **Step 1: Write positive and negative headline tests**

```python
def test_failed_risk_certificate_blocks_headline(tmp_path):
    audit = write_rc_pag_report(tmp_path, synthetic_records(certified=False), bootstrap=1000, seed=7)
    assert not audit["headline_eligible"]
    assert "risk_certificate" in audit["failed_gates"]
    assert "passed" not in (tmp_path / "report" / "headline.tex").read_text().lower()
```

- [ ] **Step 2: Implement paired summaries and gates**

Generate JSON summaries for each model/dataset/method, paired NFE/accuracy/latency intervals,
normalized cross-model aggregation, risk reliability data, failure taxonomy, and history-local
comparison. The audit must require the exact five gates from the design specification.

- [ ] **Step 3: Generate publication artifacts**

Write `main_results.tex`, `calibration.tex`, `ablations.tex`, `headline.tex`, `nfe_accuracy.pdf`,
`risk_compute.pdf`, and `reliability.pdf`. Every table cell comes from JSON; missing coverage is an
error.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/experiments/test_rc_pag_report.py -q`

```bash
git add src/pag/experiments/rc_pag_report.py tests/experiments/test_rc_pag_report.py
git commit -m "feat: audit RC-PAG paper claims"
```

### Task 7: CLI and NYU A100 Slurm Workflow

**Files:**
- Create: `scripts/run_rc_pag.py`
- Create: `scripts/slurm/rc_pag_a100.sbatch`
- Create: `scripts/slurm/submit_rc_pag.sh`
- Create: `tests/experiments/test_run_rc_pag.py`

- [ ] **Step 1: Test CLI parsing and preflight-only mock run**

```python
def test_cli_mock_preflight(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/run_rc_pag.py", "preflight", "--mock", "--output-root", str(tmp_path)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Preflight complete" in result.stdout
```

- [ ] **Step 2: Implement stage CLI**

Expose one positional stage plus `all`, `--config`, `--output-root`, `--device`, `--mock`, `--resume`,
`--limit`, and `--allow-confirmatory`. Confirmation requires the explicit flag and rejects `--limit`.
Print run ID, config hash, projected A100-hours, paths, and exact resume command.

- [ ] **Step 3: Implement Slurm worker**

Request one A100, BF16, one task, requeue, and `USR1@120`. Activate the existing Singularity overlay
and uv environment, set scratch Hugging Face/uv caches, trap `USR1`/`TERM`, run the requested stage,
and preserve exit code 2 for controlled stops. Do not embed tokens or credentials.

- [ ] **Step 4: Implement submission wrapper**

Read `RC_PAG_ACCOUNT`, `RC_PAG_PARTITION`, `RC_PAG_TIME`, `OVERLAY_PATH`, `SIF_PATH`, and
`PROJECT_DIR`; validate required paths; and invoke `sbatch --account ... --partition ... --time ...`
with exported stage/config/output variables. Default to the existing A100 partition only when the user
does not provide an override.

- [ ] **Step 5: Verify shell syntax and commit**

Run:

```bash
uv run pytest tests/experiments/test_run_rc_pag.py -q
bash -n scripts/slurm/submit_rc_pag.sh
bash -n scripts/slurm/rc_pag_a100.sbatch
```

```bash
git add scripts/run_rc_pag.py scripts/slurm tests/experiments/test_run_rc_pag.py
git commit -m "feat: launch RC-PAG on one A100"
```

### Task 8: LLaDA and Dream Observation Adapters

**Files:**
- Modify: `AdaBlock-dLLM/llada/generate_pag.py`
- Modify: `AdaBlock-dLLM/dream/model/generation_utils_pag.py`
- Modify: `src/pag/experiments/cross_model_runtime.py`
- Modify: `tests/llada/test_generate_pag.py`
- Modify: `tests/dream/test_generation_utils_pag.py`

- [ ] **Step 1: Add adapter parity tests**

Use dummy logits and tokens to assert both integrations pass the same fields to
`policy.observe_step`: masked positions, top-1/top-2 probability, entropy, token IDs, block size, and
step index. Assert no full-vocabulary logits are retained.

- [ ] **Step 2: Add a shared tensor-to-observation helper**

Compute `log_softmax`, top-2 probabilities, and entropy on device under `torch.no_grad`; slice the
active block; transfer only compact arrays to CPU; and construct `StepObservation`.

- [ ] **Step 3: Wire LLaDA**

Call the policy after each active-block forward pass, preserve existing NFE accounting, and record
the decision trace. When shadow mode is enabled, clone the current token tensor and local cache state,
continue the clone to the hard ceiling, label the proposed stop, then continue the policy branch.

- [ ] **Step 4: Wire Dream**

Add the identical call to cached and uncached Dream refinement paths. Preserve Dream's tokenizer and
generation return types. Keep token-class IDs model-specific but output fractions schema-identical.

- [ ] **Step 5: Verify model stubs and commit**

Run:

```bash
uv run pytest tests/llada/test_generate_pag.py tests/dream/test_generation_utils_pag.py -q
uv run pytest tests/llada tests/dream -q
```

```bash
git add AdaBlock-dLLM/llada/generate_pag.py \
  AdaBlock-dLLM/dream/model/generation_utils_pag.py \
  src/pag/experiments/cross_model_runtime.py tests/llada tests/dream
git commit -m "feat: observe RC-PAG refinement states"
```

### Task 9: Workshop Manuscript and Reproducibility Guide

**Files:**
- Create: `writeup/rc_pag_workshop.tex`
- Create: `writeup/rc_pag_references.bib`
- Modify: `README.md`
- Create: `docs/rc_pag_experiments.md`

- [ ] **Step 1: Write the mathematical manuscript**

Create an anonymized NeurIPS-style paper with abstract, constrained-stopping formulation, shadow
loss, LTT algorithm, proposition and proof sketch, model/dataset protocol, generated result inputs,
limitations, and reproducibility statement. Use conditional LaTeX inputs: when audited result files
are absent, compile a clearly marked protocol draft with no invented numbers.

- [ ] **Step 2: Add current primary literature**

Include LLaDA, Dream, MDLM, SEDD, Block Diffusion, AdaBlock, Fast-dLLM, APD, SOAR, SchED, DiCo,
confidence-decoding theory, PAPL, Conformal Risk Control, and Learn-then-Test using primary-source
metadata.

- [ ] **Step 3: Document exact workflows**

Document local mock validation, NYU submission environment variables, stage commands, resume,
artifact layout, claim audit interpretation, report compilation, and OpenReview submission checklist.

- [ ] **Step 4: Compile and commit**

Run:

```bash
cd writeup
latexmk -pdf -interaction=nonstopmode -halt-on-error rc_pag_workshop.tex
```

If `latexmk` is unavailable, run `pdflatex` twice and record that BibTeX was not locally verified.

```bash
git add writeup/rc_pag_workshop.tex writeup/rc_pag_references.bib README.md \
  docs/rc_pag_experiments.md
git commit -m "docs: draft DiffuLM RC-PAG paper"
```

### Task 10: Full Local Verification

**Files:**
- Modify only files required by verification failures.

- [ ] **Step 1: Run focused experiment tests**

Run: `uv run pytest tests/experiments -q`

Expected: all tests pass.

- [ ] **Step 2: Run model adapter tests**

Run: `uv run pytest tests/llada tests/dream -q`

Expected: all tests pass without loading real model weights.

- [ ] **Step 3: Run integration and full core tests**

Run:

```bash
make test-integration
uv run pytest tests/contracts tests/baselines tests/phases tests/scheduler -q
```

Expected: all tests pass.

- [ ] **Step 4: Run mock pipeline and inspect audit**

Run:

```bash
uv run python scripts/run_rc_pag.py all --mock --allow-confirmatory \
  --output-root artifacts/rc_pag_mock
```

Expected: exit zero, complete stage manifests, report JSON/LaTeX/figures, and an explicitly synthetic
headline status.

- [ ] **Step 5: Run lint and formatting**

Run:

```bash
make lint
make format
make lint
```

Expected: final lint exits zero and formatting leaves no unreviewed semantic changes.

- [ ] **Step 6: Verify repository diff and GPU preflight command**

Run:

```bash
git diff --check
git status --short
uv run python scripts/run_rc_pag.py preflight --device cpu --output-root artifacts/rc_pag_preflight
```

Record all remaining limitations: real CUDA/model execution and confirmatory numerical results cannot
be claimed until the A100 job completes.

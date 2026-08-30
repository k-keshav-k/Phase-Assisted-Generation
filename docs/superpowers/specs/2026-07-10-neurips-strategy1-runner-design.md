# NeurIPS Strategy 1 Experiment Runner Design

**Date:** 2026-07-10

**Objective:** Build one restartable command that runs the evidence package needed to turn
Phase-Adaptive Generation (PAG) into a credible NeurIPS 2026 workshop submission while keeping
Thunder Compute usage below a user-configured dollar ceiling.

## Research Goal

The experiment package must isolate whether PAG's NFE reduction comes from its learned,
history-conditioned budget predictor or from its hand-designed soft gates. It must also replace
the current 200-prompt result with a full GSM8K test-set evaluation, distinguish previously used
examples from an untouched confirmatory subset, establish limited cross-benchmark generalization
on MATH-500, and support statistically defensible claims about accuracy, NFE, latency, and memory.

The work deliberately remains a one-model study using `GSAI-ML/LLaDA-8B-Instruct`. Adding Dream
would require model-specific traces and predictor validation and is outside the available compute
budget. The second benchmark tests task generalization without introducing that confound.

## User Interface

The complete workflow is launched with one command:

```bash
uv run python scripts/run_neurips_strategy1.py \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --predictor-ckpt output/ablations/medium_ws8_d64_h4_l4_dp10_lr0.5_bestval=2.216957.pt \
  --device cuda \
  --budget-usd 20 \
  --gpu-rate 0.35
```

The command uses `configs/experiments/neurips_strategy1.yaml` for all frozen research choices.
CLI arguments identify machine-specific resources and override only the output run ID, model,
checkpoint, device, and cost ceiling. Research settings cannot be silently overridden.

The runner is non-interactive after launch. It exits successfully after producing the complete
report, exits with a distinct budget-exhausted status after saving all completed work, and exits
with an error before expensive stages when preflight validation fails.

## Compute Envelope

The target machine is one RTX A6000 with 48 GB VRAM billed at USD 0.35 per hour. The user permits
at most USD 20, equivalent to approximately 57.14 instance-hours. The expected workflow is 25--40
GPU-hours, leaving capacity for model download, smoke tests, variance in throughput, and recovery.

Budget accounting starts when the runner process starts because the remote instance is billed even
while CPU-side analysis runs. After the live smoke test, the runner extrapolates remaining cost
from observed per-method throughput. It refuses to start a stage whose conservative estimate would
cross 90% of the configured ceiling. The remaining 10% is a shutdown and variance reserve. The
manifest records both measured wall time and estimated dollar cost; the Thunder Compute invoice
remains the authoritative billing record.

## Data Splits

### Development screen

The scheduler screen uses GSM8K training indices 6000--6199. The runner asserts that these sample
IDs are absent from:

- predictor training traces (`gsm8k-train-0000` through `gsm8k-train-4999`);
- the 200-example predictor validation trace file, which contains GSM8K test indices 0--199;
- the GSM8K test split.

If the trace metadata does not establish this disjointness, preflight fails instead of guessing.
No result from the 200 GSM8K test examples used in the original report participates in method
selection.

### Final GSM8K evaluation

The final evaluation runs all 1,319 examples in the official GSM8K test split at dataset revision
`740312add88f781978c0658806c59bc2815b9866`. Prior work in this
repository used test indices 0--199 for predictor validation and indices 200--399 for the earlier
report. Consequently, indices 400--1318 form the 919-example untouched confirmatory subset. The
paper reports the 919-example confirmatory result as its primary statistical test and the complete
1,319-example result as a transparent full-benchmark summary, with the previously used 400 examples
identified explicitly. No test example is used for scheduler promotion or threshold selection.
Results are paired by immutable dataset sample ID.

### MATH-500 evaluation

The cross-benchmark evaluation uses 300 examples from `math-ai/math500` at dataset revision
`91b8f0024070e42ff83b949d6ca29da311fd3371`. The subset is selected deterministically with seed
`20260710` and stratified jointly by subject and difficulty level, with proportional allocation and
deterministic remainder assignment. The selected unique IDs are written to the run manifest before
generation. Only AdaBlock and full PAG run on this subset.

## Frozen Decoding Protocol

All quality and NFE evaluations use:

- temperature 0;
- generation length 256;
- 64 diffusion steps;
- threshold 0.9;
- delimiter threshold 0.3;
- dual-cache decoding;
- identical prompts and generation settings across methods;
- LLaDA-8B-Instruct in the precision selected by preflight for the A6000;
- AdaBlock's reactive delimiter-based block boundaries for every scheduler variant.

Temperature-zero decoding is deterministic, so repeated decoding seeds would duplicate outputs and
are not used as evidence of quality variance. Prompt-level uncertainty is quantified using paired
bootstrap intervals. Repetition is reserved for wall-clock timing, where system noise is real.

## Scheduler Ablation Matrix

All eight methods run on the 200-example development screen.

| Method | Boundary source | Refinement-budget source | Soft gates |
|---|---|---|---|
| AdaBlock | AdaBlock | AdaBlock confidence stopping | No |
| Gates only | AdaBlock | Gate checks begin after NFE 1 | Yes |
| Constant budget | AdaBlock | Median content-block NFE in training traces | Yes |
| Size lookup | AdaBlock | Training median NFE for current reactive block-size bucket | Yes |
| Previous NFE | AdaBlock | Previous realized content-block NFE | Yes |
| Random forest | AdaBlock | RF over realized history statistics | Yes |
| PAG without gates | AdaBlock | Transformer prediction enforced as a hard cap | No |
| Full PAG | AdaBlock | Transformer history prediction | Yes |

Delimiter-only blocks and content blocks are summarized separately when deriving training medians.
The constant and size-lookup baselines never consume prior block history. The previous-NFE baseline
uses history without learning. The random forest tests whether the effect is specific to the
Transformer architecture. PAG without gates isolates the predictor from soft-gate behavior. Gates
only isolates the hand-written gate from learned budgeting.

The soft gate retains the existing confidence and stability semantics, with the exact thresholds
stored in the frozen YAML configuration. The hard-budget variant force-commits remaining masked
tokens exactly when the predicted budget is exhausted. Every forced commitment and gate exit is
recorded with a machine-readable reason.

## Promotion Rule

Four methods run on the full GSM8K test split:

1. AdaBlock;
2. gates only;
3. full PAG;
4. the selected history-free baseline.

The selected history-free baseline is chosen between constant budget and size lookup. A candidate
is eligible if it loses no more than three correct answers relative to AdaBlock on the 200-example
development screen. Among eligible candidates, the one with the lowest mean total NFE is selected;
ties are broken by higher accuracy and then lexicographically by method name. If neither candidate
is eligible, constant budget is selected and the failed eligibility result is reported explicitly.

Development results for all eight variants and the complete deterministic selection record are
published in the artifact package. This keeps the promotion decision auditable.

## Answer Grading

GSM8K grading extracts the response's final marked answer, normalizes signs, commas, currency
symbols, decimal forms, and simple numeric fractions, and compares it with the dataset gold answer.
The grader does not accept a number merely because it appears somewhere in the reasoning. Grader
tests include negative values, embedded digits, equivalent decimals, fractions, missing final
markers, and multiple candidate answers.

MATH-500 grading uses the `answer` field from `math-ai/math500`, `math-verify==0.9.0`, and
`antlr4-python3-runtime==4.13.2`. The generated prompt requests a final answer in `\\boxed{}` form.
Parse exceptions are stored with the raw prediction and gold answer and count as incorrect; the
report lists their count and IDs so failures cannot disappear silently.

## Timing and Memory Protocol

Quality-run timestamps are diagnostic only. Publication latency results come from a separate
benchmark after all generation stages:

- five warm-up prompts per method, excluded from analysis;
- 50 deterministic GSM8K test prompts selected across AdaBlock NFE quintiles;
- AdaBlock and full PAG only;
- three measured repetitions per prompt and method;
- alternating AB/BA order by prompt and repetition;
- `torch.cuda.synchronize()` immediately before and after the timed region;
- end-to-end latency including predictor calls and Python scheduler logic;
- predictor time also recorded separately;
- peak allocated and reserved CUDA memory reset and recorded per trial;
- model, dtype, CUDA, driver, PyTorch, Transformers, and GPU metadata recorded.

The report includes per-method mean, median, and 95th-percentile latency, paired latency deltas, and
the complete distribution. A speedup claim is permitted only when the paired 95% bootstrap interval
for end-to-end latency improvement excludes zero. Otherwise the paper reports NFE reduction and
states that wall-clock speedup was not established.

## Statistical Analysis

The analysis stage produces each applicable GSM8K statistic for both the untouched 919-example
confirmatory subset and the complete 1,319-example benchmark. It includes:

- accuracy and 95% Wilson intervals per method;
- paired accuracy differences with 10,000 prompt-level bootstrap resamples;
- both-correct, both-wrong, PAG-only, and comparator-only counts;
- an exact two-sided McNemar test;
- mean and median total NFE per method;
- paired mean NFE differences, ratios, and 95% bootstrap intervals;
- per-prompt NFE win, tie, and loss counts;
- timing and memory summaries under the dedicated timing protocol;
- development and final tables that clearly distinguish selected from unselected methods and
  untouched from previously used test examples.

Bootstrap resampling uses seed `20260710`. Analysis operates from immutable per-prompt records and
can be rerun without loading the model.

## Implementation Boundaries

### User-facing orchestration

`scripts/run_neurips_strategy1.py` owns CLI parsing, stage transitions, resumability, budget checks,
and progress reporting. It does not contain model generation, grading, or statistical formulas.

`configs/experiments/neurips_strategy1.yaml` owns all research settings, split definitions, method
parameters, selection criteria, timing design, bootstrap settings, and the safety reserve.

### LLaDA generation and scheduler policies

`AdaBlock-dLLM/llada/scheduler_variants.py` contains a shared scheduler protocol plus constant,
size-lookup, previous-NFE, random-forest, and Transformer adapters. Each policy returns a refinement
budget and a structured explanation. Boundary choice remains outside these policies.

`AdaBlock-dLLM/llada/generate_controlled.py` contains the common controlled-decoding loop. It uses
AdaBlock boundary selection and accepts a scheduler policy plus one of three enforcement modes:
AdaBlock stopping, hard budget, or soft gates. Existing public PAG and AdaBlock entrypoints delegate
to this implementation so tests can verify behavioral compatibility.

### Experiment support package

`src/pag/experiments/` contains focused modules for configuration, dataset materialization,
grading, atomic records, budget accounting, experiment execution, statistics, and report assets.
These modules depend on the existing `pag.contracts` types where compatible and keep experiment-
specific schemas local to the experiment package.

## Resumability and Artifacts

Each method-prompt result is written to a temporary file, flushed and fsynced, then atomically
renamed to its final JSON path. A record is skipped only when its schema version, configuration
hash, model identifier, checkpoint hash, method name, and sample ID match the active run. Invalid or
partial records are quarantined and recomputed.

Artifacts are stored under:

```text
artifacts/neurips_strategy1/<run-id>/
├── manifest.json
├── environment.json
├── preflight/
├── development/<method>/
├── selection.json
├── gsm8k_test/<method>/
├── math500/<method>/
├── timing/<method>/
└── report/
    ├── summary.json
    ├── ablations.csv
    ├── gsm8k_results.csv
    ├── math500_results.csv
    ├── timing.csv
    ├── paired_statistics.json
    ├── tables/*.tex
    └── figures/*.pdf
```

The manifest records the git commit, dirty-tree state, complete resolved configuration, config
hash, model ID, checkpoint SHA-256, dataset revisions, selected IDs, dependency versions, hardware
metadata, stage state, elapsed instance time, estimated spend, and any controlled stop reason.

SIGINT and SIGTERM handlers stop after the active prompt record is safely finalized. Rerunning the
same command with the same run ID continues from the next missing record. A different resolved
configuration requires a new run ID.

## Preflight and Failure Handling

Preflight runs before the development screen and validates:

1. configuration schema and research-setting invariants;
2. CUDA availability, A6000 memory, selected dtype, and dual-cache support;
3. model and tokenizer loading;
4. checkpoint existence, hash, feature schema, and predictor compatibility;
5. required trace files and disjoint development IDs;
6. GSM8K and MATH-500 revisions and expected schemas;
7. Math-Verify availability and known-answer fixtures;
8. writable artifact directory and at least 10 GB free space;
9. one mocked and two live prompt-method smoke paths;
10. conservative projected cost under the configured ceiling.

A failed preflight produces a diagnostic JSON report and does not begin the development screen.
An individual generation failure is recorded with its exception and retried once after clearing the
CUDA cache. A second failure stops the stage to prevent biased missingness. Analysis refuses to
produce final tables when paired method coverage is incomplete.

## Verification

Unit tests cover:

- each scheduler's initial and subsequent budgets;
- delimiter versus content-block statistics;
- soft-gate, hard-cap, and AdaBlock stopping semantics;
- structured exit reasons;
- split disjointness and deterministic MATH-500 stratification;
- GSM8K and MATH answer grading fixtures;
- atomic write, quarantine, and resume behavior;
- promotion eligibility and deterministic tie-breaking;
- budget projection and controlled stopping;
- bootstrap reproducibility and exact McNemar counts;
- timing-order construction and warm-up exclusion;
- report generation from a fixed synthetic result set.

Integration tests use dummy model and tokenizer implementations to run the complete staged workflow
without a GPU. Existing LLaDA and Dream tests must continue to pass. The live two-prompt smoke test
is the final gate before paid execution.

## Paper-Facing Deliverables

The runner produces LaTeX-ready tables for the development ablation, untouched and full GSM8K
comparisons, MATH-500 transfer result, paired correctness matrix, and synchronized timing
benchmark. It also produces vector figures for NFE distributions, per-prompt NFE parity,
accuracy/NFE trade-offs, and latency distributions.

The manuscript should make the following claims only when supported by generated intervals:

- learned history-conditioned budgeting improves over gates-only and history-free budgeting;
- PAG reduces NFE on the untouched GSM8K subset without a statistically established accuracy loss,
  with the complete test-set result reported separately;
- PAG's NFE behavior transfers to the selected MATH-500 subset;
- PAG improves wall-clock latency only if the paired timing interval excludes zero.

The title and framing describe PAG as history-conditioned refinement budgeting. AdaBlock remains
responsible for deployed block boundaries; block-size prediction is presented as an auxiliary
training objective rather than a deployed control.

## Out of Scope

- Dream or a second diffusion model;
- stochastic decoding or seed sweeps;
- predictor retraining or a new hyperparameter search;
- learning gate thresholds;
- full MATH-500 evaluation beyond the fixed 300-example subset;
- automatic manuscript rewriting or workshop submission.

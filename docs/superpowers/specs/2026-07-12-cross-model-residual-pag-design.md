# Cross-Model Risk-Controlled Residual PAG Design

## Objective

Turn the current workshop-scale PAG evaluation into a main-conference-grade experiment without
changing the base language models or retraining them. The new method must demonstrate that learned
phase history adds value beyond AdaBlock and a size-only lookup rule. Positive headline claims are
permitted only when a frozen, held-out evaluation satisfies predeclared thresholds.

The implementation and protocol cannot guarantee a positive result or acceptance. If the thresholds
below are not met, the report must state that outcome and must not describe PAG as a leap.

## Primary Claim and Success Gates

The claim under test is:

> A model-agnostic, risk-controlled phase scheduler learns residual compute allocation on top of
> semantic block boundaries, reducing NFE across two diffusion language models while satisfying a
> predeclared accuracy-retention constraint.

All of the following gates must pass on the untouched confirmatory data before the paper can use the
headline claim:

1. Residual PAG reduces paired mean NFE by at least 10% relative to AdaBlock on both LLaDA and Dream.
2. On both models, the lower end of the paired 95% bootstrap confidence interval for the accuracy
   difference is no worse than -2 percentage points.
3. Residual PAG reduces paired mean NFE by at least 3% relative to `size_lookup` while satisfying the
   same accuracy constraint.
4. The direction of the NFE result is consistent on GSM8K and MATH-500.

Wall-clock latency, peak memory, and prediction overhead are secondary outcomes. They will be
reported with uncertainty but are not headline gates because Python and provider overhead can mask
model-compute savings.

## Data Protocol

The previous GSM8K test run is retained as audit evidence but is not used to select or validate the
new method.

- Calibration: GSM8K train indices 6200--6299 (100 prompts).
- Primary confirmatory test: GSM8K train indices 6300--6699 (400 prompts).
- OOD confirmatory test: the 200 MATH-500 records excluded from the prior stratified 300-record run.
- Models: GSAI-ML/LLaDA-8B-Instruct and Dream-org/Dream-v0-Base-7B.
- Confirmatory methods: AdaBlock, `size_lookup`, and residual PAG.
- Development-only methods may include the existing PAG, gates-only, and residual-policy candidates.

Samples, method order, and random seeds are materialized before generation. Calibration selects one
global risk quantile jointly across both models. Model-specific residual estimators are allowed
because raw confidence scales differ, but their architecture, features, and training procedure must
be identical. Confirmatory records are never read by selection code.

## Scheduler Architecture

Both model integrations retain AdaBlock delimiter-based block boundaries. The new scheduler controls
only the earliest safe refinement exit.

For a proposed block of size `b`, the scheduler obtains a prior budget from the existing size lookup.
A regression-tree ensemble predicts the residual between the next block's realized refinement NFE
and that prior. Features are computed from a fixed rolling history and include block size, realized
NFE, mean and minimum top-1 confidence, digit fraction, delimiter fraction, and recent summary/trend
statistics. The ensemble's calibrated lower quantile produces an aggressive residual estimate. The
sum of prior and residual is rounded and clamped to the valid refinement range and to a bounded
maximum correction.

The proposed budget is a soft threshold, not a forced commitment. Existing confidence, stability,
and few-remaining gates may exit only after the proposed budget; otherwise decoding continues until
the block is safe or reaches the existing hard ceiling. This gives the learned policy an opportunity
to save compute while keeping an online safety mechanism.

Each model's residual estimator is trained only from its calibration AdaBlock traces. Candidate risk
quantiles and correction bounds are fixed in configuration. Selection chooses the eligible candidate
with lowest joint mean normalized NFE, subject to the development accuracy constraint. The selected
configuration and hashes of both estimators are persisted before confirmatory generation starts.

## Dream Feature Parity

Dream currently records only applied block size and NFE when updating PAG history. Its generation
path will compute and pass the same realized features as LLaDA: mean and minimum top-1 confidence,
digit fraction, and delimiter fraction. Token classes remain tokenizer-specific. Feature definitions
and serialization are shared so both integrations feed the same scheduler interface.

## Experiment Orchestration and Cost Control

A new cross-model runner owns preflight, dataset freezing, calibration trace collection, estimator
training, candidate selection, confirmatory generation, and report creation. Every generation is
written atomically as a self-contained JSON record. Resume logic schedules only missing
sample/method/model keys and validates existing keys before reuse.

The paid run has a hard budget of $19 at $0.35 per GPU-hour. Before each stage, the runner projects
the remaining cost from observed throughput. It stops cleanly rather than beginning a stage whose
projected cost exceeds the remaining budget. Model changes occur at explicit stage boundaries so
only one base model occupies GPU memory. Provider failures leave completed records reusable.

Preflight verifies model access, checkpoints, rich traces, dataset disjointness, expected sample
counts, CUDA availability, output writability, and package versions. It also records code revision,
configuration, input hashes, GPU metadata, and environment information.

## Analysis and Claim Audit

Reporting computes per-model and pooled paired results for accuracy, NFE, latency, and memory. It
includes paired bootstrap confidence intervals, exact McNemar tests, correctness matrices, Pareto
plots, and a failure taxonomy. Missing or duplicate records make a stage incomplete; they are never
silently dropped.

A machine-readable claim audit evaluates each success gate and emits `headline_eligible: true` only
when every gate passes. The LaTeX tables and prose snippets consume this audit so a failed gate cannot
accidentally produce a positive headline.

## Code Boundaries

- Shared scheduler policy: a focused module under `src/pag/experiments/` implementing residual
  feature extraction, estimator fitting, prediction, serialization, and the scheduler protocol.
- LLaDA/Dream adapters: retain model-specific token handling and generation mechanics while passing
  the shared realized-feature schema.
- Cross-model orchestration: new configuration, dataset manifest, resumable stage runner, and
  one-line entry script; the existing Strategy 1 runner remains intact.
- Analysis: extend paired statistics and report generation with cross-model aggregation and claim
  gates rather than embedding statistics in the runner.

## Validation

Local validation includes:

1. Unit tests for residual targets, rolling features, quantile prediction, clamping, reset, and model
   persistence.
2. LLaDA and Dream stub tests showing identical scheduler calls and realized-feature fields.
3. Selection tests proving confirmatory records cannot influence calibration and ineligible policies
   cannot win on NFE alone.
4. Resume, duplicate detection, atomic-write, budget-stop, and coverage tests using mock runtimes.
5. Grading regression tests for numeric units, percentages, bold answers, fractions, and unboxed
   MATH gold answers.
6. CPU-only preflight and mock end-to-end report generation before the paid command is handed off.

## Paper Changes

The invalidated 21.4% NFE claim and any assertion of statistically significant speedup are removed
before submission. The paper will distinguish the prior exploratory run from the confirmatory
cross-model protocol, describe all baselines and selection rules, and include limitations and compute
cost. New headline results are inserted only after the claim audit passes. Otherwise the paper reports
the measured trade-off and is not represented as demonstrating a leap.

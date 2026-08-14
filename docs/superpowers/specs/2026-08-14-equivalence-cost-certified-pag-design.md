# Equivalence- and Cost-Certified PAG (EC-PAG) v9 Design

## Decision

Add a frozen `v9` protocol beside v1--v8. V9 keeps AdaBlock as the decoding authority and replaces
v8's unconditional depth-four same-model tree with one guarded method whose maximum depth is two.
The method may use a batched transition only when a hardware-calibrated numerical-stability guard
passes and an online cost rule predicts positive A100 latency benefit. It otherwise executes the
ordinary batch-size-one AdaBlock transition.

The paper claim is deliberately hardware-scoped: on the pinned A100 software/model stack, EC-PAG
must reproduce the complete AdaBlock state trajectory on the held-out pilot and all confirmation
prompts while reducing synchronized end-to-end latency. This is an empirical execution-equivalence
certificate, not a universal bitwise guarantee for arbitrary hardware or kernels.

## Evidence motivating the change

The completed v8 pilot contains 32 prompts per model and gives three decisive results:

1. `full_budget_shadow` exactly matches AdaBlock on all 64 paired prompts, so passive
   instrumentation is not the source of the failure.
2. `verified_fixed_d4` changes 8/32 LLaDA outputs and 14/32 Dream outputs. Its internal
   `sequence_safe` flag is therefore invalid: it checks consistency with the batched root row, not
   equivalence to the canonical batch-size-one AdaBlock transition.
3. The method reduces the repository's old logical-NFE counter by 14.7% on LLaDA and 21.3% on
   Dream, but is 24.1% and 7.2% slower, respectively. Counting every evaluated row shows about
   three times the AdaBlock model-equivalent work. Roughly 73% of speculative calls accept no
   draft edge, while only 8--9% accept at least two.

The result rules out a depth-only retuning of v8. It also exposes two distinct requirements that a
replacement must satisfy: the batched numerical transition must agree with the reference execution,
and saved serial launches must outweigh all extra batch rows in actual wall-clock time.

## Positioning and alternatives

Recent self-speculative dLLM methods already cover fixed or learned draft structures, including
[Self-Speculative Decoding](https://arxiv.org/abs/2510.04147),
[Spiffy](https://arxiv.org/abs/2509.18085), and
[Free Draft-and-Verification](https://arxiv.org/abs/2510.00294). A larger tree or another acceptance
predictor would therefore be a crowded contribution. Separately,
[Batch Speculative Decoding Done Right](https://arxiv.org/abs/2510.22876) documents that practical
batched implementations can violate claimed output equivalence.

Three designs were considered:

1. **Patch the depth-four v8 verifier.** Rejected because it does not address poor A100 latency,
   most rows are rejected, and the novelty is weak relative to existing self-speculative trees.
2. **Recompute every proposed transition at batch size one.** Rejected as the main method because
   it provides universal reference equivalence but removes the serial-launch reduction that could
   produce speedup. It remains the fallback and diagnostic oracle.
3. **Numerically guarded, cost-gated depth-one/two speculation.** Selected because it directly
   addresses both observed failures, has a compact mathematical condition, and can be evaluated
   honestly with a small fail-fast pilot.

## Reference semantics

Let `F_1(x)` denote one deterministic AdaBlock transition from active state `x` when the model is
evaluated with batch size one on the pinned stack. Let `z_1(x)` be its logits and let `z_B(x)` be the
logits for the same state when it is one row of a speculative batch of size `B`.

AdaBlock's discrete decision consists of:

- the top-one token at every masked position;
- whether each top-one probability crosses the fixed transfer threshold; and
- the identity of the highest-confidence position, which is transferred when no additional token
  is eligible.

EC-PAG treats `F_1`, rather than the batched row transition, as the reference semantics. The
ordinary non-speculative code path remains unchanged.

## Numerical-stability guard

For each evaluated state, define three observable batched margins:

\[
m_{\mathrm{tok}}(x)=\min_{i\in M(x)}
\left(z_{B,i,(1)}-z_{B,i,(2)}\right),
\]

\[
m_{\mathrm{thr}}(x)=\min_{i\in M(x)}
\left|p_{B,i,(1)}-\tau\right|,
\]

and `m_rank(x)`, the gap between the largest and second-largest top-one probabilities over masked
positions. Here `M(x)` is the masked set and `tau` is AdaBlock's frozen transfer threshold.

During the audit, every state is evaluated both canonically and in speculative batch shapes two
and three. The audit records:

- the maximum relevant logit discrepancy `eps_tok`;
- the maximum top-one-probability discrepancy `eps_prob`;
- top-one, transfer-mask, forced-position, and complete-successor agreement; and
- cache shape, active length, batch shape, dtype, kernel determinism settings, and software/hardware
  fingerprint.

A batched transition is guard-eligible only when

\[
m_{\mathrm{tok}}>2\epsilon_{\mathrm{tok}},\qquad
m_{\mathrm{thr}}>\epsilon_{\mathrm{prob}},\qquad
m_{\mathrm{rank}}>2\epsilon_{\mathrm{prob}}.
\]

These inequalities are sufficient for the token argmax, threshold decisions, and forced-position
ranking to remain unchanged under perturbations bounded by the calibrated envelope. V9 uses the
larger of the observed maximum and the predeclared high quantile for each batch shape, followed by a
fixed 25% safety inflation. Thresholds are never relaxed after the held-out pilot begins.

The condition is hardware-calibrated rather than universally deterministic. Consequently, v9 also
requires direct trajectory equality on held-out prompts and invalidates the calibration whenever
the execution fingerprint changes.

## Guarded verifier behavior

V9 constructs only a linear batch of two or three states:

- batch size two represents the root and one proposed successor;
- batch size three represents the root and two proposed successors; and
- no branching and no depth greater than two are allowed.

The verifier processes rows in parent order. A row may affect generation only if its guard passes.
An edge is accepted only if the guarded parent transition exactly equals the proposed child tensor.
If a deeper row is unsafe or disagrees, traversal stops at the last guarded state. If the root row is
unsafe, the verifier discards the entire speculative result, runs the unchanged batch-size-one
AdaBlock transition, and records the wasted batch work. No unguarded batched row can commit a token.

`sequence_safe=True` is removed as a default assertion. Diagnostics instead record
`guard_passed`, `reference_checked`, `successor_equal_when_checked`, and the reason for fallback.

## Cost-aware activation

Speculation has two decisions: whether to invoke it and whether the maximum depth is one or two.
V9 uses a deterministic lookup table rather than another high-capacity learned model. Audit events
are binned by:

- remaining-mask fraction;
- previous transfer count;
- minimum token margin;
- minimum threshold margin; and
- active block length.

For each bin and batch shape, the table stores the full-acceptance rate and paired synchronized GPU
time. A shape is eligible only when its one-sided 95% lower confidence bound on full acceptance
exceeds the frozen minimum and its paired latency-benefit lower bound is positive. Depth one is
preferred. Depth two is used only when it passes both tests independently. Unseen and sparse bins
fall back to AdaBlock.

This controller is intentionally myopic: it predicts whether one shallow batch is worthwhile, not
whether changing a token is safe. Safety remains solely with the numerical guard and exact child
comparison.

## Honest work and timing accounting

V9 distinguishes four quantities:

- **serial forward calls:** the number of Python/model invocations;
- **evaluated rows:** the sum of batch sizes across model invocations;
- **reference-equivalent transitions:** canonical AdaBlock transitions represented by the returned
  state; and
- **latency:** synchronized CUDA-event model time and synchronized end-to-end prompt time.

For a speculative call with `B` rows, evaluated work increases by `B`, not one. Canonical fallback
after an unsafe root adds another evaluated row and another serial call. All rejected and fallback
work remains in aggregates.

The strict pilot work gate requires total candidate evaluated rows not to exceed the paired AdaBlock
total. This is attainable only when selected shallow batches are fully useful; otherwise the method
fails closed. The main positive efficiency claim is based on paired end-to-end latency, not logical
NFE.

## Development, reuse, and held-out data

The existing run `artifacts/rc_pag/rc-pag-d36b982c2388` is development evidence. V9 may reuse:

- its complete AdaBlock final outputs for the original 32-prompt/model audit rows;
- its v8 speculative schedule statistics as an initialization prior; and
- compatible downloaded model and dataset caches.

It may not reuse v8 parity conclusions, cost projections, selected policies, or any positive
certificate. Raw logits were not persisted, so the numerical audit itself must run again.

V9 has two pilot roles:

| Role | GSM8K train | MATH train | MBPP train | Total/model |
|---|---:|---:|---:|---:|
| Development audit | 450--465 | 350--357 | 250--257 | 32 |
| Held-out gate | 500--531 | 360--375 | 258--273 | 64 |

The held-out gate is disjoint from every v4--v8 training, tuning, calibration, and pilot range.
After the gate passes, v9 may reuse compatible native AdaBlock confirmation records by exact prompt
ID, revision, decoding configuration, and backend fingerprint. It reruns every EC-PAG record.

## Fail-fast experimental funnel

### 1. Audit

Run paired canonical and instrumented batch-shape evaluations on the 32 development prompts per
model. Produce numerical envelopes, decision-disagreement tables, acceptance/cost bins, and depth
zero/one/two ablations. No paper claim is made from this split.

### 2. Held-out pilot

Freeze one EC-PAG controller per model before running 64 new prompts. Continue only when both models
satisfy all conditions:

- 64/64 exact generated-ID equality with AdaBlock;
- exact equality of the ordered post-transition state-digest trajectory;
- no internal guard violation or unrecorded fallback;
- paired end-to-end latency-reduction lower 95% bootstrap bound greater than 5%;
- paired synchronized model-time reduction lower bound greater than zero; and
- total evaluated rows no greater than AdaBlock.

Failure writes a complete diagnostic artifact and stops before screening without raising a generic
exception. It is a valid negative result, not permission to weaken thresholds.

### 3. Screen and confirmation

If the pilot passes, run the single frozen EC-PAG method rather than a post-hoc candidate grid. The
workshop confirmation profile remains 500 GSM8K, the untouched 200-example MATH-500 complement,
100 sanitized MBPP, and the untouched 64-example HumanEval complement per model. Compatible
AdaBlock records may be linked rather than regenerated.

The final positive claim requires, for both models:

- zero generated-ID and state-trajectory mismatches over all confirmation prompts;
- a one-sided exact upper confidence bound on prompt mismatch rate reported explicitly;
- aggregate paired end-to-end latency-reduction lower 95% bound above 5%;
- nonpositive total evaluated-row difference;
- no benchmark cell with an accuracy-difference lower bound below zero; and
- lower aggregate latency than every retained nonlearned baseline.

Because exact outputs imply exact task accuracy, accuracy is reported as a consistency check rather
than an independent optimization target.

## Artifact and configuration identity

The v9 artifact identity includes model/dataset revisions, decoding parameters, controller and
envelope hashes, PyTorch/Transformers/CUDA versions, GPU name and capability, dtype, attention
backend, determinism flags, and bundled AdaBlock commit. A mismatch starts a new audit directory;
it never silently resumes an old hardware certificate.

Every stage is append-only and resumable. Imported records retain their source run, content hash,
and compatibility decision. A new v9 configuration hash prevents the failed v8 pilot records from
satisfying v9 stage completeness.

## Components

- `rc_pag_speculation.py`: numerical margins, envelope types, guarded verification, evaluated-row
  ledger, and deterministic cost lookup.
- Dream/LLaDA AdaBlock adapters: audit dual execution, state digests, synchronized timing, guarded
  root fallback, and depth-one/two batches.
- `rc_pag_config.py`: frozen v9 schema, audit/held-out splits, envelope and cost-gate parameters,
  execution-fingerprint requirements, and one EC-PAG method.
- `rc_pag_runtime.py`: construct audit and production policies and reject incompatible envelope or
  cost artifacts.
- `rc_pag_orchestrator.py`: import compatible reference rows, run audit then held-out gate, freeze
  policies, stop cleanly, and aggregate exactness/work/latency evidence.
- `rc_pag_report.py`: parity diagnostics, depth ablations, acceptance histograms, row-normalized
  work, latency confidence intervals, and explicit failed-gate explanations.
- Slurm launcher and concise runbook: one command, automatic reuse discovery, a unique v9 run
  directory, and an audit/pilot estimate before confirmation.

## Failure behavior

- Missing or incompatible v8 source: rerun the small audit baseline instead of trusting it.
- Missing raw logits: expected; regenerate only the audit instrumentation.
- Execution-fingerprint change: invalidate envelopes and cost tables, then rerun audit.
- Unsafe root: canonical AdaBlock fallback with all wasted work charged.
- Unsafe deeper row: stop before that transition; never commit it.
- Any held-out mismatch: mark the pilot failed and block later GPU stages.
- Nonpositive or noisy speedup: mark the pilot failed and preserve the diagnostic frontier.
- CUDA out-of-memory at batch size three: disable depth two, recalibrate, and create a new config
  hash; never mutate a frozen run in place.

## Verification

Tests must cover:

- margin calculations for argmax, threshold, and forced-position decisions;
- exact boundary behavior at each envelope threshold;
- unsafe-root canonical fallback and unsafe-child early return;
- no default `sequence_safe=True` assertion;
- evaluated-row accounting for acceptance, rejection, and fallback;
- deterministic sparse-bin fallback and depth-one preference;
- state-digest equality for compressed versus sequential transitions;
- execution-fingerprint invalidation;
- exact prompt/revision/config checks for reused AdaBlock rows;
- disjoint audit and held-out splits;
- clean controlled stopping for parity, latency, or work-gate failure;
- mock end-to-end resume and one-command launcher behavior;
- Dream/LLaDA fake-model cases with deliberately batch-sensitive logits; and
- preservation of every v1--v8 code path.

Before handoff, run the focused experiment, Dream, and LLaDA tests; the complete relevant suite;
Ruff check before format; shell syntax checks; a mock v9 audit/pilot; and `git diff --check`.

Real A100 equivalence and speed remain empirical. The implementation and paper must not promise
workshop acceptance, universal numerical equivalence, or a latency improvement before the frozen
held-out and confirmation gates pass.

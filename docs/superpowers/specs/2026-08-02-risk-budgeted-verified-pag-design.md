# Risk-Budgeted Verified PAG v6 Design

## Decision

Add a frozen `v6` protocol beside v1--v5. V6 abandons v5's prompt-outcome harm and normalized-
advantage heads. It reuses the strong local full-trajectory disagreement signal from v4, calibrates
that score on held-out prompts, predicts remaining block NFE from the same traces, and treats early
stopping as a sequential prompt-level resource-allocation problem.

V6 formally certifies end-to-end harmful-regression risk. Compute is evaluated with a predeclared
paired bootstrap lower confidence bound on raw NFE reduction because a path-changing decoder can
occasionally use more total NFE than AdaBlock. V6 makes no bounded-mean compute claim.

## Evidence motivating the change

V4's local estimators achieved AUROC about 0.98, but applying an independent threshold at every
block allowed small local risks to accumulate across a prompt. V5 attempted to learn prompt harm
from paired q500 rollouts, but its held-out AUROC was 0.456 on Dream and 0.372 on LLaDA. Its best
screen candidates caused 29/150 and 17/150 harmful regressions, respectively, against an allowance
of three. The failure is consistent with multiple-instance label noise and seed-policy distribution
shift. V5 is retained as negative development evidence and is never promoted to calibration.

## Alternatives considered

1. **Retune or invert v5.** Rejected because reversed AUROC remains weak and the observed accuracy
   losses are 10--31 percentage points. Post-hoc threshold search on the completed screen would also
   invalidate the frozen protocol.
2. **Retain a joint bounded compute certificate.** Rejected for v6 because normalized paired savings
   can be negative when an early commitment changes later block boundaries. Treating those values as
   observations in `[0,1]` is invalid; clipping them would overstate net compute savings.
3. **Risk-budgeted local stopping with raw paired compute inference.** Selected because it uses the
   reliable training target, explicitly limits accumulation, remains simple enough to audit, and
   reports compute without a false boundedness assumption.

## Reuse and data boundaries

V6 may reuse only complete native-loop full-budget collection traces from a v4 or v5 run with the
same model and dataset revisions and complete temporal-JS observations. It refits all v6 estimator
artifacts. It does not reuse v4/v5 policy choices, screen outcomes, readiness decisions,
certificates, or confirmation results.

The v6 splits are disjoint from all v4/v5 development rows:

| Role | GSM8K train | MATH train | MBPP train | Total/model |
|---|---:|---:|---:|---:|
| Training traces (reused) | 0--299 | 0--199 | 0--99 | 600 |
| Parity pilot (same registered rows) | 450--465 | 350--357 | 250--257 | 32 |
| Fresh v6 tuning | 400--449 | 300--349 | 200--249 | 150 |
| Fresh v6 calibration | 854--1241 | 433--507 | 295--331 | 500 |

The v6 calibration rows are untouched by v4/v5 fitting, rollout, tuning, and calibration. The
confirmation profile remains the previously untouched deterministic benchmark complement: 500
GSM8K, 200 MATH-500, 100 sanitized MBPP, and 64 HumanEval prompts per model.

## Estimator fitting

For each model, the 600 full-budget prompt traces are deterministically partitioned by prompt group:

- 480 prompts fit a local histogram-gradient-boosting disagreement classifier;
- 120 prompts calibrate its raw scores with an isotonic map; and
- all 600 prompts fit a histogram-gradient-boosting remaining-block-NFE regressor.

The local label remains the observable full-trajectory event that a state's proposed tokens differ
from the eventual native AdaBlock block tokens. The benefit target is

$$
R_{b,t}=\max\{0,T_b-t\},
$$

where `T_b` is the realized AdaBlock NFE for the block. It is a prioritization signal, not a
guarantee of realized prompt-level savings. Estimator metadata records prompt partitions, feature
schema, validation AUROC/Brier/MAE, hashes, and target definitions.

The calibrated risk value is an online scheduling score. It is not itself claimed to be a formal
prompt-risk bound; the untouched end-to-end calibration stage supplies that guarantee.

## Online controller

For prompt `X`, the controller maintains cumulative charged risk `B_spent` and executed-stop count
`K_spent`. At state `s_t`, let calibrated local risk be `p_t` and predicted remaining block NFE be
`g_t`. A proposal is eligible when

$$
t\ge2,\qquad B_{spent}+p_t\le B,\qquad K_{spent}<K,\qquad g_t\ge g.
$$

Eligibility is revocable. The first eligible state stores the proposed top-one tokens. The next
eligible native refinement step must agree at every still-masked position. Only then does the
controller commit, increment `K_spent`, and charge the current verified state's `p_t` to
`B_spent`. A changed or ineligible
proposal clears pending verification without charging the ledger. Pending state resets at block
boundaries; the prompt ledger resets only at prompt boundaries.

The frozen family is:

| Candidate | Total risk budget `B` | Maximum stops `K` | Minimum predicted remaining NFE `g` |
|---|---:|---:|---:|
| `ledger_b020_k1_g4_v2` | 0.02 | 1 | 4 |
| `ledger_b050_k2_g3_v2` | 0.05 | 2 | 3 |
| `ledger_b100_k3_g2_v2` | 0.10 | 3 | 2 |

All candidates use minimum step two, two-step exact agreement, the same two estimator artifacts,
and no task-specific parser. No candidate is added after seeing v6 tuning results.

## Screening

Screening runs AdaBlock, the two transparent nonlearned controls, and all three v6 candidates on
the fresh 150-prompt mixture per model. A candidate is accuracy-eligible only with at most three
AdaBlock-correct/candidate-wrong regressions. Among eligible candidates, selection minimizes mean
raw NFE, breaking ties by accuracy and name.

Calibration launches only when the selected candidate for each model:

- reduces mean raw NFE by at least 8% versus paired AdaBlock; and
- has lower mean NFE than the best accuracy-eligible nonlearned control.

Failure is a controlled AdaBlock fallback, not an exception and not a positive result.

## Calibration and confirmation inference

Calibration freezes one selected candidate per model and runs paired AdaBlock/candidate generations
on 500 fresh prompts per model. The binary harm loss is

$$
H(X)=1\{A_{AdaBlock}(X)=1,\ A_{v6}(X)=0\}.
$$

The exact lower-tail binomial test evaluates `H0: E[H] >= 0.02`. Bonferroni correction covers the
two frozen model policies, so each test uses cutoff `0.05/2`. Both models must certify before
confirmation. Calibration also reports raw paired NFE deltas and their preregistered bootstrap
interval, but does not turn that interval into a finite-sample distribution-free certificate.

Confirmation reports paired accuracy and raw NFE bootstrap intervals on all registered benchmark
cells. A positive compute headline requires:

- lower 95% paired-bootstrap NFE-reduction bound above 5% for each model in aggregate;
- lower accuracy-difference bound at least -2 percentage points on every in-domain cell;
- lower mean NFE than AdaBlock for both models and lower overall mean NFE than the best nonlearned
  control; and
- both model harm certificates.

Negative prompt-level savings remain in every raw computation. No clipping is allowed in screening,
calibration compute summaries, confirmation, or paper tables.

## Components

- `rc_pag_config.py`: register v6 fields, methods, profile, exact split counts, candidate family, and
  harm-only certificate mode.
- `rc_pag_policy.py`: add calibrated-risk persistence and prompt-ledger state without changing
  v1--v5 behavior.
- `rc_pag_orchestrator.py`: fit calibrated local-risk and remaining-NFE heads, reuse only raw traces,
  screen v6, create harm-only calibration certificates, and emit raw paired-compute intervals.
- `rc_pag_runtime.py`: load v6 calibrated local-risk and remaining-NFE artifacts and instantiate the
  ledger controller.
- Reporting/paper: identify v4 accumulation and v5 label-shift failures as development evidence;
  distinguish the formal harm guarantee from bootstrap compute evidence.
- Slurm/docs: make v6 the one-command default and accept the completed v5 directory as raw-trace
  reuse input.

## Failure behavior

- Missing or observation-incomplete reused traces: stop before fitting.
- Single-class calibration labels: use an explicit constant calibration map and record it.
- Missing estimator schema or hash mismatch: stop before GPU screening.
- No accuracy-eligible candidate or failed 8% readiness: controlled AdaBlock fallback.
- Either harm certificate fails: block confirmation.
- Negative paired NFE saving: retain it and continue; it is valid raw compute evidence.

## Verification

Tests cover v1--v5 preservation, v6 config identity and split disjointness, prompt-group estimator
partitioning, calibrated-score persistence, ledger charging only after verified commits, maximum
stop count, prompt/block resets, remaining-NFE gating, v4/v5 raw-trace reuse, harm-only calibration,
negative NFE savings, readiness, confirmation gates, launcher forwarding, mock end-to-end resume,
shell syntax, scoped Ruff, the full suite, and a results-pending paper build.

Real accuracy and NFE improvements remain empirical. The implementation must not claim that the
controller guarantees an 8% saving or workshop acceptance before non-mock evidence passes every
frozen gate.

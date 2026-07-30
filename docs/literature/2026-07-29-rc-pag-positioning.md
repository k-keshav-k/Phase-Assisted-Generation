# RC-PAG literature positioning (cutoff: 2026-07-29)

This review uses primary paper or venue pages and is frozen for the DiffuLM workshop draft.
The field moved quickly in June--July 2026, so the final camera-ready search should repeat the
queries for newly posted adaptive decoding work.

## Closest methods

| Work | Adaptation signal | Extra learned component | Deployment verification | Statistical guarantee | Relationship to RC-PAG |
|---|---|---:|---:|---:|---|
| [AdaBlock-dLLM](https://arxiv.org/abs/2509.26432) | delimiter confidence and volatility band; adaptive block boundary | no | no | no | Primary same-model blockwise baseline; RC-PAG instead controls when refinement stops. |
| [Fast-dLLM](https://arxiv.org/abs/2505.22618) | token confidence, parallel transfer, KV cache | no | no | no | Important efficient-inference reference; the harness labels its local rule as a style reproduction. |
| [APD](https://openreview.net/forum?id=xwqTt26NJf) | learned token-wise denoising paths | yes | no | no | Strong learned scheduling baseline, but a materially different decoder/training stack. |
| [SchED](https://arxiv.org/abs/2512.02892) | inter-step similarity | no | no | no | Closest early-stopping comparator; no selection-safe prompt-risk certificate. |
| [OSDT](https://arxiv.org/abs/2511.02077) | one-sequence confidence trajectory | no | no | no | Shows reusable task-level schedules; RC-PAG uses a held-out risk calibration set. |
| [FreeDave](https://arxiv.org/abs/2510.00294) | draft and verification | no | yes, integrated | sequence reproduction target | Closest lossless goal; RC-PAG pays shadow cost only during calibration and accepts user-set risk. |
| [WINO](https://arxiv.org/abs/2507.18578) | confidence-based revocation/remasking | no | online remasking | no | Revises commitments rather than certifying a stopping rule. |
| [SOAR](https://arxiv.org/abs/2602.10953) | confidence switches search vs. parallel commit | no | beam search | no | Searches alternatives under uncertainty; complementary but more deployment compute. |
| [DiCo](https://arxiv.org/abs/2602.23792) | consistency distillation | yes | no | no | Changes training/model; outside RC-PAG's frozen-model scope. |
| [SemBlock](https://arxiv.org/abs/2606.04964) | predicted discourse/reasoning/code boundaries | yes | no | no | Very close learned block construction; add as discussion and future official baseline, while RC-PAG's novelty is risk control. |
| [LESS](https://arxiv.org/abs/2606.16908) | confidence, top-1 persistence, top-$K$ distribution stability | no | no | no | Closest recent training-free stopping rule. The broad screen includes analogous stability features but must not call them an official LESS reproduction. |
| [Mean-Field Parallel Decoding](https://arxiv.org/abs/2606.15805) | pairwise compatibility between token commits | no | no | no | Orthogonal within-step coordination that could be composed with RC-PAG stopping. |
| [Confidence decoding theory](https://arxiv.org/abs/2603.22248) | theoretical confidence conditions | no | no | asymptotic/structural efficiency results | Complements RC-PAG's black-box finite-sample operational-risk guarantee. |

The June 2026 [re-evaluation of confidence remasking](https://arxiv.org/abs/2606.12232)
reports strong setting dependence for WINO-style remasking. That negative evidence directly
motivates RC-PAG's paired multi-setting evaluation and refusal to infer latency or quality from
confidence alone.

## Statistical foundation

- [Learn then Test](https://arxiv.org/abs/2110.01052) provides the core reduction from a
  finite predictor family to valid risk-controlling selection using hypothesis tests.
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814) generalizes calibration to
  bounded losses. RC-PAG uses the simpler exact Bernoulli case because its strict prompt loss
  is binary.

## Positioning decision

The original PAG result is not a compelling efficiency headline: audited GSM8K results were
only 4.0% lower NFE than AdaBlock with a 1.6 percentage-point accuracy loss, and a simple
size-lookup rule was at least as competitive. The best path is therefore not a larger phase
predictor. It is a sharper research question that the crowded scheduling literature has not
answered: finite-sample control of premature-commitment risk after threshold selection.

The paper should lead with three distinctions:

1. **Counterfactual target:** same-state full-refinement block disagreement, not confidence
   calibration and not semantic correctness.
2. **Selection-safe evidence:** 12 model-policy hypotheses are tested simultaneously on data
   untouched by training and threshold definition.
3. **Transparent fallback:** failure to certify produces full refinement rather than an
   unqualified speed claim.

## Required experiments and honest exclusions

The high-value experiment set is: LLaDA and Dream; AdaBlock; the best broad nonlearned rule;
history RC-PAG in workshop confirmation; local vs. history RC-PAG during development and
calibration; HGB vs. logistic estimator; full calibration diagnostics; paired
NFE/accuracy/latency; and HumanEval transfer. Development-only Fast-dLLM/SchED-style rules
must be called reproductions. SemBlock, APD, SOAR, and LESS should be discussed as highly
relevant contemporaneous work. Unless their official code is integrated under identical
models and accounting, the paper must not imply empirical superiority over them.

If time remains after the registered confirmation, the most valuable extension is an
explicit composition study: use LESS-like mutual-stability or mean-field token selection as
the base decoder, then calibrate RC-PAG's block stop on top. This should be labeled exploratory
and must use a fresh calibration split; it must not alter the frozen headline experiment.

# Counterfactual Advantage-Gated RC-PAG Design

## Decision

Add a frozen `v5` protocol beside v4. V4 remains reproducible and unchanged. V5 reuses v4's
native AdaBlock hook and joint harm/compute certificate, but replaces the local token-disagreement
gate with an on-policy, prompt-outcome advantage gate and verifies a proposed stop before any
remaining token is irreversibly committed.

The implementation is deliberately narrow: no model fine-tuning, hidden-state probe, new decoding
backend, or large candidate sweep. It adds one development rollout/refit stage, one paired estimator
artifact per model, and policy state for a one-step agreement check.

## Alternatives considered

1. **Retune v4 thresholds.** Rejected because Dream has no scalar threshold satisfying both the 2%
   harm screen and 5% NFE gate. Moving the gates after seeing the results would also weaken the
   experimental protocol.
2. **Add only a hard stability rule.** Low-risk and simple, but it cannot correct the mismatch
   between v4's token-disagreement labels and the certified end-to-end harmful-regression loss.
3. **Counterfactual advantage plus verified commit.** Selected because it changes both the learning
   target and the irreversible action that produced the Dream knee while preserving v4's decoder,
   certificate, datasets, and one-command runner.

## Data protocol

The v5 development funnel is:

```text
pilot -> collect -> fit_seed -> rollout -> fit_advantage -> screen
      -> calibrate -> confirm -> report -> paper
```

- `collect` and `fit_seed` retain v4's full-budget exact-loop traces and local disagreement model.
- `rollout` runs paired native AdaBlock and one frozen aggressive seed policy on 150 prompts per
  model. The seed is v4 `local_q500_p2`; it is used only to expose states produced by stopping.
- Each stopped rollout supplies its compact decision features. Its prompt-level labels are

  $$H=1\{A_{AdaBlock}=1,A_{seed}=0\},\qquad
    G=\max(0,1-C_{seed}/C_{AdaBlock}).$$

  All stop decisions from one prompt stay in the same train/validation fold. A harmful prompt labels
  every executed stop conservatively; safe prompts label their executed stops safe. This is an
  explicit multiple-instance approximation rather than a claim of causal attribution to a single
  block.
- `fit_advantage` trains a histogram-gradient-boosting harm classifier and bounded savings
  regressor per model. Metadata records the seed policy, source split, prompt grouping, feature
  schema, and approximation above.
- `screen` uses a disjoint 150-prompt v5 tuning split. Calibration and confirmation retain their
  untouched splits. V5 expands calibration from 300 to 500 prompts per model using 200 additional,
  predeclared training-pool indices that do not overlap pilot, collection, rollout, or screening.
  This is the only sample-size increase and gives the 2% harm test useful power when one harmful
  regression is observed.

An incomplete stopped v4 run may be supplied through `--reuse-development-from`. V5 may reuse its
compatible exact-loop collection traces and complete paired q500/AdaBlock tuning rows as rollout
training data. It never reuses the v4 tuning decision as v5 screening evidence, and it always refits
the v5 advantage estimator. Missing pairs or missing serialized observations trigger a clear error.

## Online policy

For state $s_t$, the fitted heads predict harmful-regression risk $\widehat h(s_t)$ and normalized
NFE benefit $\widehat g(s_t)$. Candidate $\lambda=(q,g)$ is eligible when

$$
t\ge2,\qquad \widehat h(s_t)\le q,\qquad \widehat g(s_t)\ge g.
$$

Eligibility creates a **pending proposal**; it does not commit tokens. The next native AdaBlock
refinement step verifies it. The policy commits only when it is still eligible and every token that
remains masked has the same top-one proposal in the pending and current steps. Otherwise the pending
proposal is discarded and normal AdaBlock continues. This pays at most the already-required next
refinement step and never commits before verification.

The predeclared family contains exactly three operating points, all using the same two fitted heads
and exact agreement verifier:

| Candidate | Maximum harm score | Minimum predicted NFE reduction |
|---|---:|---:|
| `adv_h020_g050_v2` | 0.02 | 0.05 |
| `adv_h050_g080_v2` | 0.05 | 0.08 |
| `adv_h100_g100_v2` | 0.10 | 0.10 |

Candidate fields are `max_harm_score` and `min_predicted_nfe_reduction`; names encode both values and
the mandatory two-step verifier. No post-hoc candidate is added.

## Screening and certification

Screening keeps AdaBlock and the two v4 nonlearned controls. A v5 candidate is accuracy-eligible
only with at most 2% harmful regressions. Selection minimizes mean NFE among eligible candidates.
The workshop-readiness gate requires at least 8% empirical tuning reduction per model, rather than
5%, to provide headroom for the unchanged 5% confirmatory compute certificate.

Calibration freezes one candidate per model and retains v4's paired tests on 500 prompts per model:

$$E[H]\le0.02,\qquad E[S]>0.05.$$

Bonferroni still covers harm and compute for both models. Failure of any test activates exact
AdaBlock fallback and blocks confirmation. No v5 development label or estimator is treated as a
certificate.

## Components and surgical boundaries

- `rc_pag_config.py`: register v5, the rollout split/size, paired-head candidate fields, and 8%
  readiness gate while preserving v1-v4 validation.
- `rc_pag_policy.py`: add a bounded savings estimator and verified advantage policy. Existing v4
  policy behavior remains unchanged.
- `rc_pag_runtime.py`: load v5 paired heads and serialize executed/pending/verified decision states.
- Native LLaDA and Dream AdaBlock loops: no new decoding algorithm; they continue to honor only
  `decision.should_stop`. Verification occurs inside policy state before that flag becomes true.
- `rc_pag_orchestrator.py`: add rollout/refit handlers, grouped counterfactual label construction,
  safe reuse validation, and v5 screening.
- Launcher/docs/paper: make v5 the one-command default only after mock and full verification pass;
  describe v4 results as the motivating failure, not confirmatory evidence.

## Failure behavior

- No executed seed stops: stop before fitting because counterfactual labels do not exist.
- Missing paired AdaBlock/seed prompt: invalidate the rollout stage.
- Candidate NFE exceeds paired AdaBlock NFE: retain the raw value for screening/reporting, but label
  its bounded advantage-training target as zero, as defined by $G=\max(0,1-C_{seed}/C_{AdaBlock})$.
- Single-class harm labels: save a constant harm head with explicit metadata; screening still
  decides whether the policy has enough compute value.
- Failed v5 readiness or certificate: controlled AdaBlock fallback, exactly as in v4.

## Verification

Tests must cover grouped counterfactual labels, bounded savings fitting, pending-proposal agreement,
pending reset at block/prompt boundaries, v4 behavior preservation, v5 artifact reuse rejection,
screening headroom, joint fallback, mock end-to-end resume, shell syntax, and the one-command config.
Focused tests run first, followed by the complete suite, scoped Ruff checks, `git diff --check`, and a
mock `all` pipeline. Real A100 performance remains an empirical question and is not claimed by unit
tests.

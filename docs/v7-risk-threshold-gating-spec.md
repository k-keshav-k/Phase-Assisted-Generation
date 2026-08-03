# v7 Spec: Risk-Threshold Gating (RTG)

**Goal:** get past the v6 screen gate (≥8% NFE reduction per model, ≤2% harmful risk) and into final calibration (≤2% harm, >5% NFE savings), so we can claim the NeurIPS DiffuLM workshop result.

## Why v6 fails (from the partial screen)

| Symptom | Root cause |
|---|---|
| NFE saved 0.00–0.75% across all candidates (gate needs ≥8%) | `threshold: 1.0` means the risk score **never** acts as a gate |
| `Harm=0, Recover=0, ΔAcc=+0.00pp` everywhere | the gate almost never fires → candidates ≈ AdaBlock (trivial parity, not a win) |
| `advantage_manifest.json not found` warning | stale v5 check — refit was removed in v6. **Red herring, not the bug.** |

The v6 ledger spends **raw risk probability** against `total_risk_budget` (0.02–0.10),
while null risk is ~0.19–0.23. So only the bottom ~2–10% of prompts can ever be
eligible to stop → gate throttled into uselessness, despite the risk estimator being
excellent (AUC 0.98, Brier 0.04).

## The v7 idea (one sentence)

Stop generation early **only when the calibrated harm-risk score is below a real
threshold for `patience` consecutive steps** — i.e., let the AUC-0.98 estimator
actually gate, instead of burying it under a probability-spending budget.

## The beautiful part

`RiskStoppingPolicy` already implements exactly this path. When
`require_exact_agreement=false` and `threshold < 1.0`, the policy uses:

```python
self._safe_streak = self._safe_streak + 1 if eligible else 0
should_stop = self._safe_streak >= self.patience   # patience consecutive low-risk steps
```

So **v7 needs zero behavioral change to the policy code** — it is config +
protocol registration only. (We add one optional flag only if we later want
count-based budgets; not needed for the first v7 run.)

## New candidate config

`configs/experiments/rc_pag_neurips_workshop_v7.yaml` — same as v6 except:

```yaml
protocol_version: v7
confirmation_profile: workshop_v7_fresh   # new profile key, same counts as v6

policy:
  estimator_kinds: [hist_gradient_boosting]
  history_window: 4
  candidates:
    - {name: rgate_t10_p3_v7, variant: rc_pag_budgeted, threshold: 0.10, min_steps: 2, patience: 3,
       min_predicted_nfe_savings: 0.0, require_exact_agreement: false,
       total_risk_budget: 1.0, max_prompt_stops: 3}
    - {name: rgate_t15_p3_v7, variant: rc_pag_budgeted, threshold: 0.15, min_steps: 2, patience: 3,
       min_predicted_nfe_savings: 0.0, require_exact_agreement: false,
       total_risk_budget: 1.0, max_prompt_stops: 3}
    - {name: rgate_t20_p3_v7, variant: rc_pag_budgeted, threshold: 0.20, min_steps: 2, patience: 3,
       min_predicted_nfe_savings: 0.0, require_exact_agreement: false,
       total_risk_budget: 1.0, max_prompt_stops: 3}
```

Rationale per field:

- `threshold: 0.10 / 0.15 / 0.20` — the actual gate. Sweep across the three
  candidates; the screen picks the best accuracy-eligible one. (If 0.20 is too
  aggressive, the harm gate will reject it — that's what the ≤2% harm check is for.)
- `patience: 3` — need 3 consecutive below-threshold steps before stopping
  (slightly stronger than v6's 2, since we're now actually gating on risk).
- `min_predicted_nfe_savings: 0.0` — stop requiring the benefit estimator to
  clear 2–4 units; it was an extra throttle. The benefit/remaining-NFE estimator
  stays loaded (it's part of the v6/v7 budgeted head path) but no longer blocks.
- `require_exact_agreement: false` — v5/v6 used the token-agreement verifier as
  a safety net; with threshold gating the calibrated risk score IS the safety net.
  (If reviewers push back, we can re-enable it as an ablation: "agreement adds
  safety but costs X% savings".)
- `total_risk_budget: 1.0, max_prompt_stops: 3` — constructor requires them
  together. Budget 1.0 can't bind (3 stops × score ≤ 0.2 each = 0.6 max), so the
  real cap is 3 stops per prompt. Keeps the ledger machinery intact, no code change.

Everything else (models, datasets, splits, decoding, risk alpha/delta, statistics,
confirmatory, readiness 0.08) is frozen identical to v6.

## Code changes (registration only, ~5 small edits)

### `src/pag/experiments/rc_pag_config.py`

1. **Line ~243:** allow `"v7"` in the protocol set and error message.
2. **Line ~268 / ~308:** v7 uses the v6 stage/size profile
   (`_EXPECTED_V6_STAGES`, `calibration_per_model: 500`).
3. **Line ~334:** v7 in the `{"v4","v5","v6"}` estimator-kinds set.
4. **Line ~342:** `expected_candidates` → add `"v7": 3`.
5. **Line ~356:** v7 uses `rc_pag_budgeted` variants (like v6).
6. **Lines ~390–422 (candidate validation):** add a `v7` branch —
   * threshold in `(0, 0.5]` (not forced to 1.0),
   * `require_exact_agreement` may be false,
   * `patience` may be 3 (not frozen to 2),
   * `total_risk_budget` / `max_prompt_stops` allowed (update the
     `protocol_version != "v6"` guard at ~407 to `not in {"v6","v7"}`),
   * skip the v6 `ledger_family` frozen check (~412).
7. **Line ~496:** add `_REQUIRED_V7_DEVELOPMENT_METHODS` = same as v6
   (`adablock`, `stability_weighted_style`, `token_convergence_style`,
   `rc_pag_budgeted`).
8. **Confirmation map (~line 452):** add `"workshop_v7_fresh": _WORKSHOP_V2_CONFIRMATORY`.
9. **Readiness (~line 526–529):** v7 uses the 0.08 screen gate like v5/v6.

### `src/pag/experiments/rc_pag_runtime.py`

10. **Line ~534:** `protocol_version == "v6"` → `in {"v6", "v7"}` for the
    `uses_budgeted_heads` path (loads `{model}_rc_pag_budgeted_risk` +
    `{model}_remaining_nfe`).
11. **Lines ~446, ~681:** include `"v7"` in the modern-protocol sets.

### `src/pag/experiments/rc_pag_orchestrator.py`

12. **Line ~41:** `_MODERN_PROTOCOLS` add `"v7"`.
13. **Line ~907 / `_prepare_v6_reuse` (~1129):** let v7 reuse native full-budget
    traces the same way v6 does (v6's reuse path is exactly what v7 wants —
    no refit, estimators fit from native observations).
14. **Lines ~831, ~1258, ~1281, ~1866, ~2115:** extend the v6-inclusive sets to
    `{"v4","v5","v6","v7"}` where they gate estimator fitting / reporting.

### Cosmetic (optional)

The screen/report code still prints the v5 `advantage_manifest.json not found`
warning. Leave it for now; fix the message when we're sure v7 passes the gate.

## Expected outcome

- With `threshold: 0.15–0.20` and AUC 0.98, the bottom ~15–20% of prompts should
  become stop-eligible → NFE reduction in the low-to-mid double digits, well past
  the 8% gate, at ≤2% harm (the estimator's ranking is what protects accuracy).
- `t10` is the conservative member (less savings, near-zero harm); `t20` is the
  aggressive member. The screen selects the best eligible one per model.

## Validation plan (cheap first)

1. **Config-only sanity:** run `validate_rc_pag_config` on the v7 YAML (unit test
   or a tiny script) — no GPU needed.
2. **Gate-firing check on the tuning set (150 prompts):** instrument
   `RiskStoppingPolicy.observe` to log the fraction of observations where
   `eligible` is true and the distribution of `score` vs thresholds. Confirms the
   gate fires before spending GPU on the full screen.
3. **Full screen (1,800 records):** same pipeline as the v6 partial screen.
   Gate: ≥8% NFE reduction per model, harm ≤ 3 regressions / 150.
4. **Calibration + confirmatory:** existing v6 flow, unchanged.

## Risks / fallbacks

- **t20 harms too much** (regressions > 2%): drop it, keep t10/t15; or raise
  patience to 4 for the aggressive member.
- **Still <8% savings:** the remaining bottleneck would be `remaining_fraction`
  / `min_steps` / block structure — check the gate-firing log first, then relax
  `min_steps` to 1 or `max_remaining_fraction` below 1.0 (code already supports it).
- **Reviewer concern about dropping exact agreement:** keep one v6-ledger
  candidate in the confirmatory comparison as a "verified ledger" baseline and
  frame RTG as "calibrated risk-threshold gating vs frozen conservative ledger" —
  that contrast is itself a workshop-appropriate contribution.

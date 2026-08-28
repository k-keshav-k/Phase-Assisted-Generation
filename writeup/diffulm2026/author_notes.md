# DiffuLM 2026 author notes

## Submission target

- Track: short paper (8 pages of main text; references and appendices excluded).
- Format: official NeurIPS 2026 `dblblindworkshop` style.
- Review: anonymous, double-blind, non-archival.
- Workshop: *Diffusion Language Models: Foundations, Efficiency, and Reasoning*.

## Paper story

**Thesis.** Diffusion-LM traces contain useful phase information, but a history predictor is a forecasting instrument rather than a safe token-acceptance rule. Reliable adaptive decoding should preserve a strong reactive boundary prior, benchmark against history-free controls, count every model evaluation, and reserve acceptance authority for an audited online verifier.

### Main-text outline

1. **Introduction:** adaptive compute is a natural dLM control problem, but offline predictiveness and online safety are different questions.
2. **Related work:** masked/block diffusion, adaptive decoding, early stopping, and verified speculation.
3. **PAG:** token-level phase analysis, block-native traces, categorical/ordinal prediction, and hybrid online scheduling.
4. **Evidence protocol:** corrected NFE accounting, paired evaluation, cross-model validation, and explicit promotion gates.
5. **Results:** a causal progression from the 6.7% corrected PAG reduction through transfer,
   output-preserving routing, and verified decoding.
6. **Discussion and conclusion:** phase signals predict effort; they do not by themselves justify commitment.

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| dLM traces exhibit localized phase changes | Dream/LLaDA stabilization features and PELT visualizations | Supported qualitatively; no population-level CPD claim |
| Block history predicts future compute | 138,901 block tuples; RF and Transformer held-out results; risk-model AUROC/Brier results | Supported |
| Initial PAG reduces corrected NFE on LLaDA | GSM8K: 87.81 to 84.26 NFE; MATH-500: 119.65 to 111.67 | Supported |
| Initial PAG dominates AdaBlock | Accuracy decreases by 1.60 and 1.33 points; size lookup is stronger on GSM8K | Unsupported and excluded |
| RC-PAG transfers safely across LLaDA and Dream | Accuracy noninferiority gate failed; minimum lower CI was -21.2 points | Unsupported and explicitly rejected |
| Learned history is universally valuable | Conservative v6/v7 policies save below 1.3%; v5 advantage router has poor harm AUROC | Unsupported |
| Batched macro-verification is exactly equivalent in the tested implementation | v8 has 22/64 sequence disagreements; v9 shape audit fails | Unsupported and explicitly rejected |
| Forecasting and acceptance should be separated | Convergent evidence from the size-lookup comparison, cross-model accuracy failures, and equivalence audits | Supported as an empirical design recommendation |

## Evidence hierarchy

1. Audited, complete, corrected Strategy-1 and RC-PAG v1 reports.
2. Completed 150-prompt/model screens with frozen configs (v4--v7).
3. Pilot/equivalence audits (v8--v9), reported only as diagnostic evidence.
4. Legacy 200-prompt result with inconsistent NFE accounting, excluded from claims.
5. Protocols v2--v3, which have configs but no materialized run artifacts, excluded from empirical tables.

## Reviewer-risk checklist

- [x] State the negative result in the abstract rather than hiding it.
- [x] Distinguish model-call NFE from latency, FLOPs, and evaluated batch rows.
- [x] Mark the legacy 20.6% NFE result as invalid under corrected accounting.
- [x] Avoid claiming official reproductions of related methods when only style controls were implemented.
- [x] Keep CPD as interpretability evidence, not a scheduler-label oracle.
- [x] Keep all author-identifying links and names out of the anonymous submission.
- [x] Check every headline number against the artifact path recorded in the appendix.

## Final validation

- Main text ends on page 7; the workshop limit is 8 pages, excluding references and appendix.
- Official NeurIPS 2026 `dblblindworkshop` style compiles successfully.
- Bibliography: 20 cited entries, 0 unresolved citations, 0 unused entries, and 0 duplicates.
- Layout: no overfull or underfull boxes, unresolved references, or multiply defined labels in the
  final source log.
- Grammar: no rule-based errors detected.
- De-AI/readability scan: no visible-prose trace findings after revision. Remaining scanner hits are
  NeurIPS checklist boilerplate and TikZ color syntax, neither of which is manuscript prose.
- Visual inspection completed for all seven main-text pages; the phase plot and all tables are
  legible at normal zoom.
- NeurIPS checklist completed with evidence-linked justifications. The compute question is answered
  `Yes` using an estimate of approximately 160 A100-equivalent GPU-hours: 40 hours for trace
  collection and the original PAG study, plus six materialized protocols at about 20 hours each.
  The LLM-usage question is `N/A` because generative AI assistance was limited to writing, editing,
  and formatting and did not supply the research, experiments, or analysis.
- Reviewer-style submission gate: **PASS** with no critical or major blockers. The principal
  acceptance risk is empirical scope (two 7--8B model families) rather than presentation,
  accounting, or claim support; the paper states this limitation and frames the contribution as a
  controlled stress test rather than a universal scheduler claim.

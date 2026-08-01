# RC-PAG v4 literature positioning — 2026-08-01

This audit uses primary paper/project pages available on 2026-08-01.

| Work | Main mechanism | Formal harm control | Formal compute floor | Consequence for RC-PAG |
|---|---|---:|---:|---|
| [Prophet](https://arxiv.org/abs/2508.19982) | answer-region confidence early commit | no | no | Early commit itself is not novel. |
| [SWD](https://arxiv.org/abs/2604.17068) | consecutive-distribution stability | no | no | Temporal JS must be an input, not the claimed contribution. |
| [STDec](https://arxiv.org/abs/2604.06330) | spatial and token-ID temporal stability | no | no | Token stability is a control, not novelty. |
| [Just on Time](https://arxiv.org/abs/2602.11133) | independent token convergence | no | no | Token-level stopping is occupied. |
| [LATCH](https://arxiv.org/abs/2607.28166) | candidate-span stability and local/global commit | no | no | Candidate-aware stopping is strong but task-format-specific. |
| [RC-Jot](https://analemma.ai/papers/808c134d-cfc5-4140-993a-632cfdc4dc0b/) | conformal calibration of an exit threshold | yes | no | Risk-only calibration is not enough for novelty. The report is explicitly automated, but is still disclosed as prior art. |
| [Adaptive LTT](https://arxiv.org/abs/2409.15844) | selection-safe sequential risk testing | generic | generic | LTT is methodology, not a novelty claim. |

## Frozen decision

V4 uses one local HGB risk score and three thresholds. It does not add a second benefit model,
a separate temporal-JS threshold, a tail grid, or a task-specific parser. Its differentiator is a
paired joint certificate on the exact native AdaBlock loop:

- exact binomial test of harmful regression at `alpha = 0.02`;
- bounded Hoeffding--Bentkus test of mean paired NFE saving above `kappa = 0.05`;
- Bonferroni familywise control over two claims for each of two frozen model policies;
- AdaBlock fallback unless all four nulls are rejected.

This choice is less ambitious than learning a semantic controller, but it has the clearest theorem,
the smallest policy family, the lowest screening cost, and the strongest protection against claiming
a statistically safe but practically negligible speedup.

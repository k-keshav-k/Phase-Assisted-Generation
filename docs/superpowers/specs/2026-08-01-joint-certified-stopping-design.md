# Joint-Certified Stable Stopping Design

## Decision

Replace the v3 decoder's separate risk and remaining-NFE estimators, hard temporal-JS gate,
tail-fraction grid, and nine candidates with one local risk estimator and three predeclared score
thresholds. Temporal stability remains an input to the risk estimator, not an independently tuned
mechanism. A candidate is deployable only when held-out calibration jointly rejects both an
excess-harm null and an insufficient-compute-saving null for each model.

This is the best novelty/reliability/simplicity compromise for the workshop. Prophet, SWD, STDec,
LATCH, and RC-Jot make confidence/stability stopping or risk-only threshold calibration weak as a
standalone novelty claim. The paper's differentiator is instead a selection-safe, paired certificate
that simultaneously covers end-to-end harmful regression and a useful NFE reduction on the native
AdaBlock decoder.

## Method

For decoder state $s_t$, a histogram gradient-boosting classifier estimates
$\widehat r(s_t)=P(\hat y_t\ne\hat y_T\mid s_t)$. Its local features contain progress, confidence,
entropy, top-two margin, token churn, and normalized temporal Jensen--Shannon summaries. There is
no benefit regressor and no manually imposed JS or tail gate.

For threshold $q\in\{0.05,0.20,0.50\}$, stop when

$$
  t\ge2,\qquad \widehat r(s_t)\le q
$$

holds for two consecutive refinement steps. Tuning freezes one threshold per model using paired
AdaBlock task correctness and NFE. Calibration never reselects thresholds.

For prompt $i$, define harmful regression

$$H_i=\mathbf 1\{A_i=1,\widehat A_i=0\}$$

and paired normalized saving

$$S_i=1-N_i^{\mathrm{candidate}}/N_i^{\mathrm{AdaBlock}}\in[0,1].$$

Each frozen model policy must reject both

$$H_0^H:E[H]\ge0.02,\qquad H_0^C:E[S]\le0.05.$$

The harm test uses the exact binomial lower-tail p-value. The compute test uses a finite-sample
Hoeffding--Bentkus upper-tail p-value for bounded observations. Bonferroni correction covers both
tests for both frozen model policies. The certificate falls back to exact AdaBlock unless every
model passes both tests.

## Pipeline and artifact policy

The existing staged pipeline remains `preflight -> pilot -> collect -> fit -> screen -> calibrate ->
confirm -> report -> paper`. Exact no-stop parity remains mandatory. Old result tables are retained
as prior evidence, but an estimator or trace is reusable only when its feature schema contains the
temporal-JS fields and its native decoder identity matches. Older v1 estimators are intentionally not
silently reused because doing so would make model feature protocols inconsistent.

The v4 screen contains AdaBlock, two transparent stability controls, and the three learned
thresholds. Confirmation contains only AdaBlock, the tuning-selected nonlearned control, and the
single jointly certified policy. The fresh workshop confirmation split remains unchanged.

## Failure behavior

- Missing or incompatible reuse artifacts trigger a clear error before scientific execution.
- Incomplete paired calibration invalidates the certificate.
- Any negative NFE saving violates the early-stop invariant and aborts calibration.
- Failure of either statistical null test for either model activates AdaBlock fallback and blocks
  confirmation.
- Mock certificates remain watermarked and cannot populate the manuscript.

## Verification

Unit tests cover temporal-JS features, the one-score stopping rule, Hoeffding--Bentkus p-values,
joint multiplicity, configuration freezing, orchestration, fallback, and the one-command launcher.
Focused experiment tests, the complete test suite, Ruff, shell syntax checks, and the paper build are
required before handoff.

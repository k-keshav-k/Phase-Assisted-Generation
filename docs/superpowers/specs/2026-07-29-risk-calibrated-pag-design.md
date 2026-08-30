# Risk-Calibrated Phase-Adaptive Generation Design

## Objective

Turn the existing Phase-Adaptive Generation (PAG) project into an eight-page DiffuLM workshop
submission centered on a defensible new contribution: finite-sample risk control for adaptive
stopping in block diffusion language models. The implementation must support LLaDA and Dream on a
single A100, reuse existing traces and adapters where valid, minimize GPU waste, and prevent claims
from outrunning the evidence.

The implementation cannot guarantee acceptance or positive results. It must make a strong paper
possible when the evidence supports it and generate an honest negative or qualified report when it
does not.

## Primary Contribution

Risk-Calibrated PAG (RC-PAG) treats refinement as a constrained stopping problem. For a policy
\(\pi\), let \(C_\pi\) be its model-forward cost and let \(L(\pi)\in[0,1]\) measure premature
commitment relative to a shadow continuation of the same block. RC-PAG solves the empirical analogue
of

\[
\min_{\pi\in\Pi}\;\mathbb E[C_\pi]
\quad\text{subject to}\quad
R(\pi)=\mathbb E[L(\pi)]\leq\alpha.
\]

The primary strict loss for a prompt is one when any block committed by the policy differs from the
block obtained by continuing refinement from the identical state. It is zero otherwise. Secondary
losses include changed-token fraction and task-correctness regression. The risk certificate is about
premature commitment, not semantic correctness; task accuracy is measured separately.

A finite, predeclared policy family is calibrated with Learn-then-Test (LTT). Each model--policy pair
is a separate hypothesis (six policies times two models), and invalid-policy selection is controlled
simultaneously over all 12 pairs at family-wise level \(\delta\). If either headline variant lacks a
certificate for either model, RC-PAG uses a full-budget fallback. Under exchangeability of calibration
and deployment prompts, this provides the paper's finite-sample guarantee: with probability at least
\(1-\delta\) over calibration, every selected model-specific policy is either the fallback or has risk
at most \(\alpha\).

## Runtime State and Policy

At refinement step \(t\) of block \(b\), the runtime records only online-observable, compact features:

- step index and block size;
- remaining-mask count and fraction;
- mean, maximum, and selected quantiles of token entropy;
- selected quantiles of top-1 probability and top-1/top-2 margin;
- provisional-token churn from the previous refinement step;
- entropy and confidence trends;
- provisional digit and delimiter fractions;
- recent block history: size, realized NFE, confidence, stability, token type, and trends.

The learned risk score uses a small CPU-friendly estimator. The default is histogram gradient
boosting, with logistic regression as a transparent ablation. No base-model fine-tuning is required.
The stopping family varies a score threshold, minimum refinement step, and number of consecutive safe
steps. Decoding stops at the first eligible step and otherwise continues to the existing hard ceiling.

Two headline variants share the same training and calibration procedure:

1. `rc_pag_local` uses current-block state only.
2. `rc_pag_history` adds rolling phase history.

History is called beneficial only if the paired compute-risk comparison improves with uncertainty
excluding zero. Existing Transformer PAG, the residual random forest, and shuffled-history controls
remain ablations rather than assumed contributions.

## Shadow Calibration

Counterfactual evaluation from a single baseline trace is invalid after an early commitment changes
future context. Calibration therefore uses an on-policy shadow continuation:

1. Run the candidate policy to a proposed stopping state.
2. Clone the minimal current block state.
3. Continue the clone to the full refinement ceiling.
4. Compare the policy's committed block with the shadow-refined block.
5. Continue the actual policy trajectory and aggregate block labels at prompt level.

Training and offline screening may use full-budget trajectories. The statistical certificate must use
on-policy calibration records. Candidate policy definitions, calibration sample IDs, and risk levels
are frozen before calibration.

## Experiment Protocol

### Models

- `GSAI-ML/LLaDA-8B-Instruct`
- `Dream-org/Dream-v0-Instruct-7B`

Both use BF16, deterministic temperature-zero decoding, one A100, batch size one, and their supported
cache path. Model revisions and tokenizer revisions are recorded.

### Data

- Training/tuning: disjoint samples from GSM8K train, MATH train, and MBPP train.
- Calibration: a disjoint, fixed stratified mixture from the same pools.
- Confirmatory: GSM8K test, MATH-500, and MBPP sanitized.
- Out of distribution: HumanEval, never used for policy selection.

Every split is materialized with IDs and content hashes. Calibration and test overlap is a hard error.
The certificate applies only to the declared calibration/deployment distribution; OOD results are
reported as empirical transfer, not covered by the guarantee.

### Baselines and Funnel

The development screen includes fixed-block decoding, AdaBlock, Fast-dLLM confidence decoding,
SchED-style progress-aware stopping, entropy-sum stopping, confidence-only and stability-only gates,
constant budget, size lookup, existing PAG, residual PAG, the two RC-PAG variants, and an oracle
stabilization lower bound.

All methods run on a small disjoint development slice. Full confirmatory generation includes
AdaBlock, the strongest eligible nonlearned baseline, `rc_pag_local`, and `rc_pag_history`. Dominated
development methods are reported in an appendix table rather than rerun over the full test set.

### Claim Gates

A positive headline requires all of the following:

1. The selected policy receives an \((\alpha=0.05,\delta=0.05)\) risk certificate.
2. Mean paired NFE is lower than AdaBlock on LLaDA and Dream.
3. Mean paired NFE is lower than the strongest eligible nonlearned baseline.
4. The lower paired 95% confidence bound for task-accuracy change is at least -0.02.
5. History is credited only if `rc_pag_history` improves the paired compute-risk frontier over
   `rc_pag_local` with an interval excluding zero.

Failed gates remain visible in machine-readable JSON and generated LaTeX. Reporting code may not emit
a positive headline when any required gate fails.

### Statistics

- exact binomial/LTT policy tests with family-wise error control;
- prompt-level paired bootstrap intervals for accuracy, NFE, latency, and risk differences;
- exact McNemar tests for paired correctness;
- multiplicity correction for secondary comparisons;
- empirical risk, score AUROC, Brier score, and reliability curves;
- normalized per-model NFE aggregation rather than pooling raw model costs;
- accuracy-NFE and risk-NFE Pareto frontiers.

## GPU-Efficient Execution

The runner uses a staged funnel:

1. `preflight`: validate CUDA, model access, data, storage, and revisions.
2. `pilot`: run 32 prompts per model, exercise a forced-stop same-state shadow, and estimate time,
   storage, and remaining GPU-hours with the complete screening matrix counted.
3. `collect`: create compact full-budget refinement traces for about 600 prompts per model.
4. `fit`: train and evaluate risk scores on CPU.
5. `screen`: evaluate broad policy and feature families offline and on a small development slice.
6. `calibrate`: run only six frozen on-policy shadow candidates on about 300 prompts per model.
7. `confirm`: run the frozen method matrix on untouched tests.
8. `report`: create audits, tables, plots, and failure taxonomies on CPU.
9. `paper`: populate an anonymized NeurIPS workshop manuscript without inventing results.

Records are atomic and keyed by model, stage, method, dataset, sample ID, seed, and configuration hash.
Resume logic schedules only missing keys. The pilot estimate is displayed before expensive work, and a
futility gate prevents confirmatory expansion when no policy improves on the best heuristic.

## Software Architecture

The implementation extends the existing `pag.experiments` boundaries:

- `rc_pag_features.py`: pure online feature extraction and compact step-state schemas.
- `rc_pag_policy.py`: estimator fitting, persistence, local/history score models, policy state, and
  stopping decisions.
- `risk_control.py`: bounded losses, exact binomial p-values, LTT correction, selection, and
  certificate serialization.
- `rc_pag_config.py`: immutable protocol schema and YAML validation.
- `rc_pag_records.py`: typed trace/calibration records and validation helpers if the existing record
  store cannot express step traces cleanly.
- `rc_pag_orchestrator.py`: stages, frozen manifests, resume, futility, and report handoff.
- model adapters: expose identical step observations and optional shadow continuation for LLaDA and
  Dream without moving statistical logic into third-party model code.
- `rc_pag_report.py`: paired summaries, claim audit, publication tables, and figures.
- `scripts/run_rc_pag.py`: one Python entry point for every stage.
- `scripts/slurm/rc_pag_a100.sbatch`: one-A100 resumable worker.
- `scripts/slurm/submit_rc_pag.sh`: cluster account/partition/overlay wrapper.

The existing cross-model residual runner remains reproducible. RC-PAG is additive and may reuse its
grading, dataset, record-store, statistics, and environment helpers.

## Failure Handling

- Missing CUDA or inaccessible checkpoints fail in preflight before a job allocation is consumed.
- NaN/inf features, impossible probabilities, block/step mismatches, duplicate records, and split
  overlap are hard errors.
- Interrupted jobs leave completed per-sample records reusable and never treat partial JSON as valid.
- SIGUSR1/SIGTERM writes the current manifest before Slurm requeue.
- Estimator and policy hashes are checked on resume.
- A calibration family mismatch invalidates the certificate rather than silently recalibrating.
- If no policy is certified, reporting selects the full-budget fallback and records the negative
  result.
- Code benchmark execution uses isolated subprocesses with timeouts and no network access.

## Validation

Local validation includes:

1. Feature tests for entropy, margins, churn, histories, missing masks, and finite outputs.
2. Policy tests for threshold direction, patience, minimum steps, reset, persistence, and fallback.
3. Risk-control tests with exact known binomial cases, family-wise correction, unsafe candidates, and
   no-certified-candidate behavior.
4. Synthetic coverage simulations showing selection frequency respects the declared error level.
5. Adapter parity tests proving LLaDA and Dream emit the same observation schema.
6. Shadow-branch tests proving labels compare states with identical prefixes.
7. Dataset disjointness, manifest hashing, atomic write, resume, signal, and futility tests.
8. Mock end-to-end runs producing a complete report without a model or GPU.
9. Regression tests ensuring failed gates cannot generate positive LaTeX.
10. Focused tests, integration tests, Ruff check, Ruff format, and LaTeX compilation before completion.

## Paper Structure

The submission uses the NeurIPS 2026 workshop template and limits the main text to eight pages,
excluding references and appendix:

1. Introduction: heuristic stopping lacks auditable risk control.
2. Related work: masked diffusion inference, AdaBlock, Fast-dLLM, APD, SOAR, SchED, and risk control.
3. Problem formulation: constrained stopping, shadow loss, and compute objective.
4. Method: online state, local/history score, LTT calibration, fallback, and proposition.
5. Experiments: two models, four benchmarks, baselines, metrics, and frozen gates.
6. Results: certificate, efficiency, accuracy, phase-history value, and OOD transfer.
7. Limitations: exchangeability, surrogate risk, task-specific calibration distribution, sequential
   generation effects, and single-GPU timing.

The proof, complete ablations, calibration diagnostics, compute accounting, prompts, and negative
results go in the appendix. All numerical macros are generated from audited artifacts.

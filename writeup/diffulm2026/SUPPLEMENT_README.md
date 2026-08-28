# Anonymous supplement: Phase-Adaptive Generation

This repository is the anonymous source supplement for the DiffuLM workshop submission. It
contains the implementation, frozen experiment configurations, atomic run records, and report
inputs used in the paper. Model checkpoints and benchmark datasets are not redistributed.

## Environment

Use Python 3.11 and install the locked project environment from the repository root:

```bash
uv sync
```

The artifact-backed cross-model runs used PyTorch 2.5.1+cu121, Transformers 4.46.3,
Datasets 4.8.5, scikit-learn 1.9.0, and math-verify 0.9.0. Model and dataset revisions are
pinned in `configs/experiments/` and repeated in each run's `environment.json` and preflight
manifest.

## Initial PAG audit

The complete Strategy 1 run evaluates the eight development policies, all 1,319 GSM8K test
examples, the 300-example MATH-500 sample, and synchronized latency trials. From the repository
root, run:

```bash
uv run python scripts/run_neurips_strategy1.py \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --predictor-ckpt output/ablations/medium_ws8_d64_h4_l4_dp10_lr0.5_bestval=2.216957.pt \
  --device cuda --budget-usd 20 --gpu-rate 0.35
```

The frozen configuration is `configs/experiments/neurips_strategy1.yaml`. The command is
resumable because records are written atomically under `artifacts/neurips_strategy1/`.

## Cross-model stress-test protocols

The artifact-backed configurations reported in the paper are:

| Protocol | Configuration | Run hash |
|---|---|---|
| v1 | `rc_pag_neurips_workshop.yaml` | `5197d981c0bf` |
| v4 | `rc_pag_neurips_workshop_v4.yaml` | `8cce1af8e35b` |
| v5 | `rc_pag_neurips_workshop_v5.yaml` | `7688c5235bd4` |
| v6 | `rc_pag_neurips_workshop_v6.yaml` | `58e76c3eb9b1` |
| v7 | `rc_pag_neurips_workshop_v7.yaml` | `c1eda289fb08` |
| v8 | `rc_pag_neurips_workshop_v8.yaml` | `d36b982c2388` |
| v9 | `rc_pag_neurips_workshop_v9.yaml` | `588204cfb482` |

Protocols v2 and v3 were superseded before GPU evidence was materialized and are not reported as
experiments. To run a protocol locally, replace `<CONFIG>` below with one of the paths above:

```bash
uv run python scripts/run_rc_pag.py all \
  --config configs/experiments/<CONFIG> \
  --device cuda --allow-confirmatory
```

Use `--resume` with the printed run ID after a controlled stop. The v9 cluster wrapper is
`scripts/slurm/submit_rc_pag_all.sh`; it reuses only the eligible v8 AdaBlock audit references and
recomputes every numerical audit, policy rule, tuning, calibration, and confirmation decision.

## Compute estimate

The completed experiments total approximately 160 A100-equivalent GPU-hours: about 40 hours for
trace collection and the original PAG study, plus about 20 hours for each of the six materialized
protocols v1 and v4--v8. Protocols v2--v3 were superseded before full runs, and v9 stopped at its
equivalence preflight before benchmark rollout. Incomplete attempts and CPU-only analysis are not
included in this estimate.

## Claim audit and provenance

Paper numbers should be checked against the files named in Appendix Table 5, not against partial
screen logs. A result is ineligible for a positive claim if its manifest is incomplete, a record is
mocked, paired prompt coverage is incomplete, or an automatic promotion gate fails. NFE includes
the initial boundary proposal and every refinement call. Batched invocations additionally report
evaluated rows; NFE reductions are not automatically interpreted as latency or FLOP reductions.

The paper source is in `writeup/diffulm2026/main.tex`. Compile it with the official NeurIPS 2026
style file in that directory.

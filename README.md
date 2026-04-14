# Phase-Adaptive Generation (PAG)

This repository is a modular skeleton for studying phase-adaptive decoding in diffusion language models. The scaffold is intentionally light on internal constraints: each teammate can implement their component however they want, as long as the shared contracts, public stage entrypoints, and tests continue to pass.

## Design goals

- Keep module boundaries explicit and small.
- Share data through typed artifacts rather than inheritance-heavy services.
- Allow each team to replace the stub implementation inside their own package without touching orchestration.
- Make test subsets runnable per teammate.

## Public stage entrypoints

- `pag.baselines.run_baseline(...)`
- `pag.phases.run_phase_analysis(...)`
- `pag.scheduler.run_adaptive_decoding(...)`
- `pag.evaluation.evaluate_runs(...)`

These are the only hard code-level boundaries the scaffold enforces. Teams are free to use functions, classes, registries, or other internal structure behind them.

## Quick start

```bash
uv sync
uv run python scripts/run_pipeline.py --config configs/runs/adaptive_mock.yaml
make test
```

`uv sync` creates the project-local virtual environment at `.venv/` and installs the default `dev`
dependency group automatically.

## Repo layout

- `src/pag/contracts`: shared dataclasses, typing protocols, serialization helpers
- `src/pag/baselines`: baseline stage entrypoint and stub implementation
- `src/pag/phases`: phase analysis stage entrypoint and stub implementation
- `src/pag/scheduler`: adaptive scheduling stage entrypoint and stub implementation
- `src/pag/evaluation`: comparison/evaluation entrypoint and stub implementation
- `src/pag/orchestration`: registry, pipeline wiring, CLI
- `src/pag/config`: YAML config loading and artifact path helpers
- `src/pag/utils`: artifact persistence helpers and run/request ids
- `configs`: example YAML configs for runs, model, decoding, predictor, scheduler, evaluation, dataset
- `tests`: contract, stage, and end-to-end wiring tests
- `docs`: architecture and teammate workflow documentation

## Artifact layout

Artifacts are written under `artifacts/<run_id>/` by stage:

- `baseline/requests.jsonl`
- `baseline/traces.jsonl`
- `baseline/token_signals.jsonl`
- `baseline/completions.jsonl`
- `baseline/run_summary.json`
- `phases/phase_annotations.jsonl`
- `phases/predictor_dataset.jsonl`
- `phases/predictions.jsonl`
- `phases/predictor_metadata.json`
- `phases/run_summary.json`
- `scheduler/schedule_decisions.jsonl`
- `scheduler/schedule_plans.jsonl`
- `scheduler/adaptive_results.jsonl`
- `scheduler/comparison_metrics.json`
- `scheduler/run_summary.json`
- `evaluation/records.jsonl`
- `evaluation/run_summary.json`

## Team ownership

- Baselines / adapters / inference / eval: `src/pag/baselines`, `src/pag/evaluation`
- Phase analysis / signals / predictor: `src/pag/phases`
- Scheduler / adaptive decode / orchestration policy: `src/pag/scheduler`

The orchestration layer only wires stage outputs into stage inputs. It should stay thin.

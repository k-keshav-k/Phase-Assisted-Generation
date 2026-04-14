# Module I/O Contracts

Artifacts are stored under `artifacts/<run_id>/`.

## Baseline Stage

- Reads:
  - `run_config.yaml`
  - dataset file referenced by `RunConfig.dataset_path`
- Writes:
  - `baseline/requests.jsonl`
  - `baseline/traces.jsonl`
  - `baseline/token_signals.jsonl`
  - `baseline/completions.jsonl`
  - `baseline/run_summary.json`
- Downstream consumers:
  - phase analysis reads traces and token signals
  - scheduler and evaluation read baseline completions

## Phase Stage

- Reads:
  - `baseline/traces.jsonl`
  - `baseline/token_signals.jsonl`
  - optional external hidden-state features
- Writes:
  - `phases/phase_annotations.jsonl`
  - `phases/predictor_dataset.jsonl`
  - `phases/predictions.jsonl`
  - `phases/predictor_metadata.json`
  - `phases/run_summary.json`
- Downstream consumers:
  - scheduler reads predictions and optionally annotations

## Scheduler Stage

- Reads:
  - `baseline/completions.jsonl`
  - `baseline/traces.jsonl`
  - `phases/predictions.jsonl`
- Writes:
  - `scheduler/schedule_decisions.jsonl`
  - `scheduler/schedule_plans.jsonl`
  - `scheduler/adaptive_results.jsonl`
  - `scheduler/comparison_metrics.json`
  - `scheduler/run_summary.json`
- Downstream consumers:
  - evaluation reads adaptive results and comparison metrics

## Evaluation Stage

- Reads:
  - `baseline/completions.jsonl`
  - `scheduler/adaptive_results.jsonl`
  - `scheduler/comparison_metrics.json`
- Writes:
  - `evaluation/records.jsonl`
  - `evaluation/run_summary.json`


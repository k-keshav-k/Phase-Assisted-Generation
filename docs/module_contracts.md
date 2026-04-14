# Module Contracts

## Baselines

- Purpose: Produce fixed-decoding baseline outputs and token-level traces/signals.
- Owner teammate: Baselines / model adapters / inference / eval.
- Required inputs: `RunConfig`, `list[SampleRecord]`.
- Optional inputs: custom implementation callable passed directly to the public entrypoint.
- Returned outputs: `BaselineRunArtifacts`.
- Downstream consumers: `pag.phases`, `pag.scheduler`, `pag.evaluation`.
- Unit tests required: contract-compliant outputs, aligned sample ids, serializable artifacts.
- TODO extension points: model adapter integration, batching, dataset readers, real trace extraction.

## Phases

- Purpose: Convert traces and token signals into phase labels, predictor items, and phase predictions.
- Owner teammate: Phase analysis / signal extraction / predictor.
- Required inputs: `RunConfig`, `BaselineRunArtifacts`.
- Optional inputs: hidden-state features and custom implementation callable.
- Returned outputs: `PhaseArtifacts`.
- Downstream consumers: `pag.scheduler`.
- Unit tests required: predictor dataset creation, phase prediction compatibility, serializable outputs.
- TODO extension points: richer features, learned predictors, offline labeling pipelines.

## Scheduler

- Purpose: Use phase predictions to create schedule decisions and adaptive decode outputs.
- Owner teammate: Scheduler / adaptive decoding / orchestration.
- Required inputs: `RunConfig`, `BaselineRunArtifacts`, `PhaseArtifacts`.
- Optional inputs: custom implementation callable.
- Returned outputs: `AdaptiveRunArtifacts`.
- Downstream consumers: `pag.evaluation`.
- Unit tests required: schedule plan creation, adaptive result compatibility, metric serialization.
- TODO extension points: policy search, chunk/refinement tuning, latency-aware scheduling.

## Evaluation

- Purpose: Compare baseline and adaptive outputs and emit comparison-ready records.
- Owner teammate: Baselines / eval.
- Required inputs: `RunConfig`, `BaselineRunArtifacts`, `AdaptiveRunArtifacts`.
- Optional inputs: custom implementation callable.
- Returned outputs: `EvaluationArtifacts`.
- Downstream consumers: analysis notebooks, reports, future dashboards.
- Unit tests required: evaluation record compatibility in integration coverage.
- TODO extension points: real quality metrics, benchmark suites, report generation.

## Orchestration

- Purpose: Wire stages together, resolve implementations, persist artifacts, expose CLI.
- Owner teammate: Shared utility layer, with scheduler team as closest downstream owner.
- Required inputs: resolved configs and samples.
- Optional inputs: registered alternative implementations.
- Returned outputs: `PipelineArtifacts` and on-disk artifacts.
- Downstream consumers: scripts, CI, teammate workflows.
- Unit tests required: mock end-to-end pipeline wiring.
- TODO extension points: experiment grids, caching, distributed execution.

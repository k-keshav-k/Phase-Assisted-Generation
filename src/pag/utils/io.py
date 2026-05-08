from __future__ import annotations

from pathlib import Path

import yaml

from pag.config.paths import (
    adaptive_paths,
    baseline_paths,
    evaluation_paths,
    phase_paths,
    run_directory,
)
from pag.contracts.artifacts import (
    AdaptiveRunArtifacts,
    BaselineRunArtifacts,
    EvaluationArtifacts,
    PhaseArtifacts,
    PipelineArtifacts,
)
from pag.contracts.schemas import (
    DecodingResult,
    EvaluationRecord,
    GenerationRequest,
    GenerationTrace,
    PhasePrediction,
    PhaseSpan,
    PredictorDatasetItem,
    RunConfig,
    RunSummary,
    SampleRecord,
    ScheduleDecision,
    SchedulePlan,
    TokenSignal,
)
from pag.contracts.serialization import dump_json, dump_jsonl, load_json, load_jsonl, to_dict


def snapshot_run_config(run_config: RunConfig) -> Path:
    target = run_directory(run_config) / "run_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(to_dict(run_config), sort_keys=False), encoding="utf-8")
    return target


def write_baseline_artifacts(artifacts: BaselineRunArtifacts) -> dict[str, Path]:
    snapshot_run_config(artifacts.run_config)
    paths = baseline_paths(artifacts.run_config)
    artifacts.summary.artifact_paths = {key: str(value) for key, value in paths.items()}
    dump_jsonl(paths["requests"], artifacts.requests)
    dump_jsonl(paths["traces"], artifacts.traces)
    dump_jsonl(paths["token_signals"], artifacts.token_signals)
    dump_jsonl(paths["completions"], artifacts.completions)
    dump_json(paths["summary"], artifacts.summary)
    return paths


def write_phase_artifacts(artifacts: PhaseArtifacts) -> dict[str, Path]:
    snapshot_run_config(artifacts.run_config)
    paths = phase_paths(artifacts.run_config)
    artifacts.summary.artifact_paths = {key: str(value) for key, value in paths.items()}
    dump_jsonl(paths["phase_annotations"], artifacts.phase_annotations)
    dump_jsonl(paths["predictor_dataset"], artifacts.predictor_dataset)
    dump_jsonl(paths["predictions"], artifacts.predictions)
    dump_json(paths["predictor_metadata"], artifacts.predictor_metadata)
    dump_json(paths["summary"], artifacts.summary)
    return paths


def write_adaptive_artifacts(artifacts: AdaptiveRunArtifacts) -> dict[str, Path]:
    snapshot_run_config(artifacts.run_config)
    paths = adaptive_paths(artifacts.run_config)
    artifacts.summary.artifact_paths = {key: str(value) for key, value in paths.items()}
    dump_jsonl(paths["schedule_decisions"], artifacts.schedule_decisions)
    dump_jsonl(paths["schedule_plans"], artifacts.schedule_plans)
    dump_jsonl(paths["adaptive_results"], artifacts.adaptive_results)
    dump_json(paths["comparison_metrics"], artifacts.comparison_metrics)
    dump_json(paths["summary"], artifacts.summary)
    return paths


def write_evaluation_artifacts(artifacts: EvaluationArtifacts) -> dict[str, Path]:
    snapshot_run_config(artifacts.run_config)
    paths = evaluation_paths(artifacts.run_config)
    artifacts.summary.artifact_paths = {key: str(value) for key, value in paths.items()}
    dump_jsonl(paths["records"], artifacts.records)
    dump_json(paths["summary"], artifacts.summary)
    return paths


def persist_pipeline_artifacts(artifacts: PipelineArtifacts) -> None:
    write_baseline_artifacts(artifacts.baseline)
    write_phase_artifacts(artifacts.phases)
    write_adaptive_artifacts(artifacts.adaptive)
    if artifacts.evaluation is not None:
        write_evaluation_artifacts(artifacts.evaluation)


def read_baseline_artifacts(run_config: RunConfig) -> BaselineRunArtifacts:
    paths = baseline_paths(run_config)
    return BaselineRunArtifacts(
        run_config=run_config,
        samples=load_samples_from_config(run_config),
        requests=load_jsonl(paths["requests"], GenerationRequest),
        traces=load_jsonl(paths["traces"], GenerationTrace),
        token_signals=load_jsonl(paths["token_signals"], TokenSignal),
        completions=load_jsonl(paths["completions"], DecodingResult),
        summary=load_json(paths["summary"], RunSummary),
    )


def read_phase_artifacts(run_config: RunConfig) -> PhaseArtifacts:
    paths = phase_paths(run_config)
    return PhaseArtifacts(
        run_config=run_config,
        phase_annotations=load_jsonl(paths["phase_annotations"], PhaseSpan),
        predictor_dataset=load_jsonl(paths["predictor_dataset"], PredictorDatasetItem),
        predictions=load_jsonl(paths["predictions"], PhasePrediction),
        predictor_metadata=load_json(paths["predictor_metadata"], dict),
        summary=load_json(paths["summary"], RunSummary),
    )


def read_adaptive_artifacts(run_config: RunConfig) -> AdaptiveRunArtifacts:
    paths = adaptive_paths(run_config)
    return AdaptiveRunArtifacts(
        run_config=run_config,
        schedule_decisions=load_jsonl(paths["schedule_decisions"], ScheduleDecision),
        schedule_plans=load_jsonl(paths["schedule_plans"], SchedulePlan),
        adaptive_results=load_jsonl(paths["adaptive_results"], DecodingResult),
        comparison_metrics=load_json(paths["comparison_metrics"], dict),
        summary=load_json(paths["summary"], RunSummary),
    )


def read_evaluation_artifacts(run_config: RunConfig) -> EvaluationArtifacts:
    paths = evaluation_paths(run_config)
    return EvaluationArtifacts(
        run_config=run_config,
        records=load_jsonl(paths["records"], EvaluationRecord),
        summary=load_json(paths["summary"], RunSummary),
    )


def load_samples_from_config(run_config: RunConfig) -> list[SampleRecord]:
    payload = yaml.safe_load(Path(run_config.dataset_path).read_text(encoding="utf-8"))
    raw_items = (
        payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    )
    return [
        SampleRecord(
            sample_id=item["sample_id"],
            prompt=item["prompt"],
            reference=item.get("reference"),
            metadata=item.get("metadata", {}),
        )
        for item in raw_items
    ]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass(slots=True)
class BaselineRunArtifacts:
    run_config: RunConfig
    samples: list[SampleRecord]
    requests: list[GenerationRequest]
    traces: list[GenerationTrace]
    token_signals: list[TokenSignal]
    completions: list[DecodingResult]
    summary: RunSummary


@dataclass(slots=True)
class PhaseArtifacts:
    run_config: RunConfig
    phase_annotations: list[PhaseSpan]
    predictor_dataset: list[PredictorDatasetItem]
    predictions: list[PhasePrediction]
    predictor_metadata: dict[str, Any]
    summary: RunSummary


@dataclass(slots=True)
class AdaptiveRunArtifacts:
    run_config: RunConfig
    schedule_decisions: list[ScheduleDecision]
    schedule_plans: list[SchedulePlan]
    adaptive_results: list[DecodingResult]
    comparison_metrics: dict[str, float]
    summary: RunSummary


@dataclass(slots=True)
class EvaluationArtifacts:
    run_config: RunConfig
    records: list[EvaluationRecord]
    summary: RunSummary


@dataclass(slots=True)
class PipelineArtifacts:
    run_config: RunConfig
    baseline: BaselineRunArtifacts
    phases: PhaseArtifacts
    adaptive: AdaptiveRunArtifacts
    evaluation: EvaluationArtifacts | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


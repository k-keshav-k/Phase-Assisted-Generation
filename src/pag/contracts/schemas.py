from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        msg = f"{field_name} must be non-empty"
        raise ValueError(msg)


def _require_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        msg = f"{field_name} must be non-negative"
        raise ValueError(msg)


@dataclass(slots=True)
class ModelConfig:
    name: str
    revision: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, "ModelConfig.name")


@dataclass(slots=True)
class DecodingConfig:
    strategy: str
    max_tokens: int = 32
    chunk_size: int = 1
    refinement_steps: int = 1
    temperature: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.strategy, "DecodingConfig.strategy")
        _require_non_negative(self.max_tokens, "DecodingConfig.max_tokens")
        _require_non_negative(self.chunk_size, "DecodingConfig.chunk_size")
        _require_non_negative(self.refinement_steps, "DecodingConfig.refinement_steps")


@dataclass(slots=True)
class PredictorConfig:
    name: str
    label_space: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, "PredictorConfig.name")


@dataclass(slots=True)
class SchedulerConfig:
    name: str
    default_chunk_size: int = 1
    default_refinement_steps: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, "SchedulerConfig.name")
        _require_non_negative(self.default_chunk_size, "SchedulerConfig.default_chunk_size")
        _require_non_negative(
            self.default_refinement_steps,
            "SchedulerConfig.default_refinement_steps",
        )


@dataclass(slots=True)
class EvaluationConfig:
    name: str
    metrics: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, "EvaluationConfig.name")


@dataclass(slots=True)
class RunConfig:
    run_id: str
    output_root: str
    dataset_path: str
    seed: int = 0
    enabled_stages: list[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="mock-model"))
    decoding: DecodingConfig = field(default_factory=lambda: DecodingConfig(strategy="fixed"))
    predictor: PredictorConfig = field(
        default_factory=lambda: PredictorConfig(name="mock-predictor")
    )
    scheduler: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(name="mock-scheduler")
    )
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig(name="mock-eval"))
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "RunConfig.run_id")
        _require_text(self.output_root, "RunConfig.output_root")
        _require_text(self.dataset_path, "RunConfig.dataset_path")


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    prompt: str
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "SampleRecord.sample_id")
        _require_text(self.prompt, "SampleRecord.prompt")


@dataclass(slots=True)
class GenerationRequest:
    run_id: str
    request_id: str
    sample: SampleRecord
    model: ModelConfig
    decoding: DecodingConfig
    requested_artifacts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "GenerationRequest.run_id")
        _require_text(self.request_id, "GenerationRequest.request_id")


@dataclass(slots=True)
class TraceStep:
    step_index: int
    chunk_size: int
    refinement_steps: int
    emitted_tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_negative(self.step_index, "TraceStep.step_index")
        _require_non_negative(self.chunk_size, "TraceStep.chunk_size")
        _require_non_negative(self.refinement_steps, "TraceStep.refinement_steps")


@dataclass(slots=True)
class TokenSignal:
    sample_id: str
    token_index: int
    token_text: str
    step_index: int
    values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "TokenSignal.sample_id")
        _require_non_negative(self.token_index, "TokenSignal.token_index")
        _require_non_negative(self.step_index, "TokenSignal.step_index")


@dataclass(slots=True)
class GenerationTrace:
    sample_id: str
    request_id: str
    steps: list[TraceStep] = field(default_factory=list)
    final_tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "GenerationTrace.sample_id")
        _require_text(self.request_id, "GenerationTrace.request_id")


@dataclass(slots=True)
class PhaseSpan:
    sample_id: str
    start_token: int
    end_token: int
    label: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "PhaseSpan.sample_id")
        _require_text(self.label, "PhaseSpan.label")
        _require_non_negative(self.start_token, "PhaseSpan.start_token")
        _require_non_negative(self.end_token, "PhaseSpan.end_token")
        if self.end_token < self.start_token:
            msg = "PhaseSpan.end_token must be >= start_token"
            raise ValueError(msg)


@dataclass(slots=True)
class PhasePrediction:
    sample_id: str
    predictor_name: str
    spans: list[PhaseSpan] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "PhasePrediction.sample_id")
        _require_text(self.predictor_name, "PhasePrediction.predictor_name")


@dataclass(slots=True)
class PredictorDatasetItem:
    sample_id: str
    token_index: int
    features: dict[str, float] = field(default_factory=dict)
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "PredictorDatasetItem.sample_id")
        _require_non_negative(self.token_index, "PredictorDatasetItem.token_index")


@dataclass(slots=True)
class ScheduleDecision:
    sample_id: str
    step_index: int
    chunk_size: int
    refinement_steps: int
    reason: str
    phase_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "ScheduleDecision.sample_id")
        _require_text(self.reason, "ScheduleDecision.reason")
        _require_non_negative(self.step_index, "ScheduleDecision.step_index")
        _require_non_negative(self.chunk_size, "ScheduleDecision.chunk_size")
        _require_non_negative(self.refinement_steps, "ScheduleDecision.refinement_steps")


@dataclass(slots=True)
class SchedulePlan:
    sample_id: str
    planner_name: str
    decisions: list[ScheduleDecision] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "SchedulePlan.sample_id")
        _require_text(self.planner_name, "SchedulePlan.planner_name")


@dataclass(slots=True)
class DecodingResult:
    sample_id: str
    request_id: str
    completion: str
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "DecodingResult.sample_id")
        _require_text(self.request_id, "DecodingResult.request_id")
        _require_text(self.completion, "DecodingResult.completion")


@dataclass(slots=True)
class EvaluationRecord:
    sample_id: str
    baseline_completion: str
    adaptive_completion: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "EvaluationRecord.sample_id")
        _require_text(self.baseline_completion, "EvaluationRecord.baseline_completion")
        _require_text(self.adaptive_completion, "EvaluationRecord.adaptive_completion")


@dataclass(slots=True)
class RunSummary:
    run_id: str
    stage: str
    num_samples: int
    artifact_paths: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "RunSummary.run_id")
        _require_text(self.stage, "RunSummary.stage")
        _require_non_negative(self.num_samples, "RunSummary.num_samples")

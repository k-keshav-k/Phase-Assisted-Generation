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
class TokenStepObservation:
    step_index: int
    token_id: int | None = None
    token_text: str | None = None
    top1_prob: float | None = None
    selected_logit: float | None = None
    top2_prob: float | None = None
    extras: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_negative(self.step_index, "TokenStepObservation.step_index")
        if self.token_id is not None:
            _require_non_negative(self.token_id, "TokenStepObservation.token_id")


@dataclass(slots=True)
class TraceToken:
    token_index: int
    token_text: str
    char_start: int
    char_end: int
    observations: list[TokenStepObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_non_negative(self.token_index, "TraceToken.token_index")
        _require_non_negative(self.char_start, "TraceToken.char_start")
        _require_non_negative(self.char_end, "TraceToken.char_end")
        if self.char_end < self.char_start:
            msg = "TraceToken.char_end must be >= char_start"
            raise ValueError(msg)


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    backend: str
    model_name: str
    prompt: str
    final_text: str
    tokens: list[TraceToken] = field(default_factory=list)
    decoding_metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source_path: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.trace_id, "TraceRecord.trace_id")
        _require_text(self.backend, "TraceRecord.backend")
        _require_text(self.model_name, "TraceRecord.model_name")
        _require_text(self.prompt, "TraceRecord.prompt")
        _require_text(self.final_text, "TraceRecord.final_text")
        if self.tokens and len(self.tokens) != len({token.token_index for token in self.tokens}):
            msg = "TraceRecord.tokens must have unique token_index values"
            raise ValueError(msg)


@dataclass(slots=True)
class FeatureSeries:
    feature_name: str
    token_indices: list[int]
    values: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.feature_name, "FeatureSeries.feature_name")
        if len(self.token_indices) != len(self.values):
            msg = "FeatureSeries.token_indices and values must have the same length"
            raise ValueError(msg)


@dataclass(slots=True)
class SegmentSummary:
    start_token: int
    end_token: int
    length: int
    text: str
    mean: float
    std: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        _require_non_negative(self.start_token, "SegmentSummary.start_token")
        _require_non_negative(self.end_token, "SegmentSummary.end_token")
        _require_non_negative(self.length, "SegmentSummary.length")
        if self.end_token <= self.start_token:
            msg = "SegmentSummary.end_token must be greater than start_token"
            raise ValueError(msg)

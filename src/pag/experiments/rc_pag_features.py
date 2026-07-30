from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StepObservation:
    step_index: int
    block_size: int
    masked: tuple[bool, ...]
    top1_probs: tuple[float, ...]
    top2_probs: tuple[float, ...]
    entropies: tuple[float, ...]
    token_ids: tuple[int, ...]
    digit_ids: frozenset[int] = frozenset()
    delimiter_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if self.step_index < 1:
            raise ValueError("step_index must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        arrays = (
            self.masked,
            self.top1_probs,
            self.top2_probs,
            self.entropies,
            self.token_ids,
        )
        if any(len(values) != self.block_size for values in arrays):
            raise ValueError("step observation arrays must match block_size")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.top1_probs):
            raise ValueError("top1 probabilities must be finite and in [0, 1]")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.top2_probs):
            raise ValueError("top2 probabilities must be finite and in [0, 1]")
        probability_pairs = zip(self.top1_probs, self.top2_probs, strict=True)
        if any(second > first for first, second in probability_pairs):
            raise ValueError("top2 probabilities cannot exceed top1")
        if any(not math.isfinite(value) or value < 0.0 for value in self.entropies):
            raise ValueError("entropies must be finite and non-negative")

    @classmethod
    def from_arrays(
        cls,
        *,
        step_index: int,
        block_size: int,
        masked: Sequence[bool],
        top1_probs: Sequence[float],
        top2_probs: Sequence[float],
        entropies: Sequence[float],
        token_ids: Sequence[int],
        digit_ids: Iterable[int] = (),
        delimiter_ids: Iterable[int] = (),
    ) -> StepObservation:
        return cls(
            step_index=int(step_index),
            block_size=int(block_size),
            masked=tuple(bool(value) for value in masked),
            top1_probs=tuple(float(value) for value in top1_probs),
            top2_probs=tuple(float(value) for value in top2_probs),
            entropies=tuple(float(value) for value in entropies),
            token_ids=tuple(int(value) for value in token_ids),
            digit_ids=frozenset(int(value) for value in digit_ids),
            delimiter_ids=frozenset(int(value) for value in delimiter_ids),
        )


@dataclass(frozen=True, slots=True)
class RealizedBlock:
    block_size: int
    nfe: int
    mean_confidence: float
    min_confidence: float
    digit_fraction: float
    delimiter_fraction: float

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.nfe < 1:
            raise ValueError("realized block size and NFE must be positive")
        bounded = (
            self.mean_confidence,
            self.min_confidence,
            self.digit_fraction,
            self.delimiter_fraction,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("realized block probabilities and fractions must be in [0, 1]")
        if self.min_confidence > self.mean_confidence:
            raise ValueError("minimum confidence cannot exceed mean confidence")


_LOCAL_FEATURE_NAMES = (
    "local.step_index",
    "local.block_size",
    "local.remaining_count",
    "local.remaining_fraction",
    "local.entropy_mean",
    "local.entropy_max",
    "local.entropy_q10",
    "local.entropy_q50",
    "local.entropy_q90",
    "local.top1_mean",
    "local.top1_min",
    "local.top1_max",
    "local.top1_q10",
    "local.top1_q50",
    "local.top1_q90",
    "local.margin_mean",
    "local.margin_min",
    "local.margin_max",
    "local.margin_q10",
    "local.margin_q50",
    "local.margin_q90",
    "local.token_churn",
    "local.entropy_delta",
    "local.top1_delta",
    "local.digit_fraction",
    "local.delimiter_fraction",
)

_HISTORY_FIELDS = (
    "block_size",
    "nfe",
    "mean_confidence",
    "min_confidence",
    "digit_fraction",
    "delimiter_fraction",
)

_HISTORY_FEATURE_NAMES = ("history.length",) + tuple(
    f"history.{field}_{stat}"
    for field in _HISTORY_FIELDS
    for stat in ("last", "mean", "std", "trend")
)


def feature_names(*, include_history: bool) -> tuple[str, ...]:
    if include_history:
        return _LOCAL_FEATURE_NAMES + _HISTORY_FEATURE_NAMES
    return _LOCAL_FEATURE_NAMES


def _selected(values: Sequence[float], masked: Sequence[bool]) -> np.ndarray:
    return np.asarray(
        [value for value, is_masked in zip(values, masked, strict=True) if is_masked],
        dtype=np.float64,
    )


def _summary(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    if not values.size:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q10": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q90": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_q10": float(np.quantile(values, 0.1)),
        f"{prefix}_q50": float(np.quantile(values, 0.5)),
        f"{prefix}_q90": float(np.quantile(values, 0.9)),
    }


def _masked_mean(values: Sequence[float], masked: Sequence[bool]) -> float:
    selected = _selected(values, masked)
    return float(selected.mean()) if selected.size else 0.0


def _token_fraction(token_ids: Sequence[int], selected_ids: frozenset[int]) -> float:
    if not token_ids or not selected_ids:
        return 0.0
    return sum(token_id in selected_ids for token_id in token_ids) / len(token_ids)


def _history_features(
    history: Sequence[RealizedBlock],
    *,
    history_window: int,
) -> dict[str, float]:
    if history_window < 1:
        raise ValueError("history_window must be positive")
    recent = history[-history_window:]
    features: dict[str, float] = {"history.length": float(len(recent))}
    for field in _HISTORY_FIELDS:
        values = np.asarray([float(getattr(block, field)) for block in recent], dtype=np.float64)
        if not values.size:
            last = mean = std = trend = 0.0
        else:
            last = float(values[-1])
            mean = float(values.mean())
            std = float(values.std())
            trend = float(values[-1] - values[0]) if values.size > 1 else 0.0
        features.update(
            {
                f"history.{field}_last": last,
                f"history.{field}_mean": mean,
                f"history.{field}_std": std,
                f"history.{field}_trend": trend,
            }
        )
    return features


def extract_features(
    observation: StepObservation,
    *,
    previous: StepObservation | None,
    history: Sequence[RealizedBlock],
    history_window: int,
) -> dict[str, float]:
    if previous is not None and previous.block_size != observation.block_size:
        raise ValueError("previous observation must describe the same block size")

    masked_entropy = _selected(observation.entropies, observation.masked)
    masked_top1 = _selected(observation.top1_probs, observation.masked)
    margins = tuple(
        first - second
        for first, second in zip(observation.top1_probs, observation.top2_probs, strict=True)
    )
    masked_margins = _selected(margins, observation.masked)
    remaining_count = sum(observation.masked)
    churn = 0.0
    entropy_delta = 0.0
    top1_delta = 0.0
    if previous is not None:
        churn = sum(
            left != right
            for left, right in zip(observation.token_ids, previous.token_ids, strict=True)
        ) / observation.block_size
        entropy_delta = _masked_mean(observation.entropies, observation.masked) - _masked_mean(
            previous.entropies, previous.masked
        )
        top1_delta = _masked_mean(observation.top1_probs, observation.masked) - _masked_mean(
            previous.top1_probs, previous.masked
        )

    entropy_summary = _summary(masked_entropy, prefix="local.entropy")
    entropy_summary.pop("local.entropy_min")
    top1_summary = _summary(masked_top1, prefix="local.top1")
    margin_summary = _summary(masked_margins, prefix="local.margin")
    features = {
        "local.step_index": float(observation.step_index),
        "local.block_size": float(observation.block_size),
        "local.remaining_count": float(remaining_count),
        "local.remaining_fraction": remaining_count / observation.block_size,
        **entropy_summary,
        **top1_summary,
        **margin_summary,
        "local.token_churn": float(churn),
        "local.entropy_delta": float(entropy_delta),
        "local.top1_delta": float(top1_delta),
        "local.digit_fraction": _token_fraction(observation.token_ids, observation.digit_ids),
        "local.delimiter_fraction": _token_fraction(
            observation.token_ids, observation.delimiter_ids
        ),
        **_history_features(history, history_window=history_window),
    }
    missing = set(feature_names(include_history=True)) - features.keys()
    unexpected = features.keys() - set(feature_names(include_history=True))
    if missing or unexpected:
        raise RuntimeError(f"feature schema mismatch: missing={missing}, unexpected={unexpected}")
    if any(not math.isfinite(value) for value in features.values()):
        raise ValueError("feature extraction produced a non-finite value")
    return features


def vectorize_features(
    features: Mapping[str, float],
    names: Sequence[str],
) -> np.ndarray:
    missing = [name for name in names if name not in features]
    if missing:
        raise ValueError(f"feature mapping is missing fields: {missing}")
    row = np.asarray([float(features[name]) for name in names], dtype=np.float64)
    if not np.isfinite(row).all():
        raise ValueError("feature vector must be finite")
    return row

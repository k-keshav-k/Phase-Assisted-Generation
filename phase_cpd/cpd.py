from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import ruptures as rpt

from phase_cpd.segments import normalize_breakpoints


@dataclass(slots=True)
class CPDParameters:
    cost: str = "l2"
    penalty: float = 0.1
    min_segment_length: int = 2
    smoothing_window: int = 1

    def __post_init__(self) -> None:
        if self.cost not in {"l2", "normal"}:
            msg = "CPDParameters.cost must be one of: l2, normal"
            raise ValueError(msg)
        if self.penalty < 0:
            msg = "CPDParameters.penalty must be non-negative"
            raise ValueError(msg)
        if self.min_segment_length < 1:
            msg = "CPDParameters.min_segment_length must be at least 1"
            raise ValueError(msg)
        if self.smoothing_window < 1:
            msg = "CPDParameters.smoothing_window must be at least 1"
            raise ValueError(msg)


class ChangePointDetector(Protocol):
    name: str

    def detect(self, values: Sequence[float], params: CPDParameters) -> list[int]: ...


class PeltDetector:
    name = "pelt"

    def detect(self, values: Sequence[float], params: CPDParameters) -> list[int]:
        signal = np.asarray(values, dtype=float)
        token_count = int(signal.shape[0])
        if token_count < max(2, params.min_segment_length * 2):
            return []

        smoothed = _moving_average(signal, params.smoothing_window)
        standardized = _standardize(smoothed)
        if np.allclose(standardized, 0.0):
            return []

        raw_breakpoints = (
            rpt.Pelt(
                model=params.cost,
                min_size=params.min_segment_length,
                jump=1,
            )
            .fit(standardized.reshape(-1, 1))
            .predict(pen=params.penalty)
        )
        return normalize_breakpoints(
            raw_breakpoints,
            token_count,
            min_segment_length=params.min_segment_length,
        )


class KernelCPDDetector:
    name = "kernel_cpd"

    def __init__(self, *, kernel: str = "rbf") -> None:
        if kernel not in {"linear", "rbf", "cosine"}:
            msg = "KernelCPDDetector.kernel must be one of: linear, rbf, cosine"
            raise ValueError(msg)
        self.kernel = kernel

    def detect(self, values: Sequence[float], params: CPDParameters) -> list[int]:
        signal = np.asarray(values, dtype=float)
        token_count = int(signal.shape[0])
        if token_count < max(2, params.min_segment_length * 2):
            return []

        smoothed = _moving_average(signal, params.smoothing_window)
        standardized = _standardize(smoothed)
        if np.allclose(standardized, 0.0):
            return []

        raw_breakpoints = (
            rpt.KernelCPD(
                kernel=self.kernel,
                min_size=params.min_segment_length,
            )
            .fit(standardized.reshape(-1, 1))
            .predict(pen=params.penalty)
        )
        return normalize_breakpoints(
            raw_breakpoints,
            token_count,
            min_segment_length=params.min_segment_length,
        )


DETECTORS: dict[str, ChangePointDetector] = {
    PeltDetector.name: PeltDetector(),
}


def get_detector(name: str, *, kernel: str = "rbf") -> ChangePointDetector:
    if name == KernelCPDDetector.name:
        return KernelCPDDetector(kernel=kernel)
    try:
        return DETECTORS[name]
    except KeyError as error:
        available = ", ".join(sorted([*DETECTORS, KernelCPDDetector.name]))
        msg = f"Unknown detector '{name}'. Available: {available}"
        raise KeyError(msg) from error


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values

    radius = window // 2
    smoothed = np.empty_like(values, dtype=float)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        smoothed[index] = float(values[start:end].mean())
    return smoothed


def _standardize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = float(values.mean())
    std = float(values.std())
    if std <= 1e-8:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std

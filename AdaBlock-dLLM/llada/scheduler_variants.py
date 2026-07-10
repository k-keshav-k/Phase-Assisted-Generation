from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from phase_predict.schema import PhaseTuple


@dataclass(slots=True)
class ScheduledBlock:
    predicted_tuple: PhaseTuple
    applied_block_size: int
    budgeted_refinement_steps: int


class SchedulerPolicy(Protocol):
    prediction_trace: list[dict[str, object]]
    scheduler_predict_time_sec: float

    def reset(self) -> None: ...

    def next_schedule(
        self,
        *,
        block_size: int | None,
        remaining_tokens: int,
        max_block_length: int,
        max_refinement_steps: int,
    ) -> ScheduledBlock: ...

    def record_realized(
        self,
        applied_block_size: int,
        actual_nfe_used: int,
        mean_confidence: float = 1.0,
        min_confidence: float = 1.0,
        digit_fraction: float = 0.0,
        delimiter_fraction: float = 0.0,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TraceBudgetStats:
    content_median: int
    delimiter_median: int
    by_size: dict[int, int]


def load_trace_sequences(path: str | Path) -> list[list[dict[str, float]]]:
    sequences: list[list[dict[str, float]]] = []
    with Path(path).open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            payload = json.loads(line)
            tuples = payload.get("tuples")
            if not isinstance(tuples, list):
                raise ValueError("trace row is missing a tuples list")
            sequences.append(
                [{str(key): float(value) for key, value in row.items()} for row in tuples]
            )
    if not sequences:
        raise ValueError("trace file contains no sequences")
    return sequences


def derive_trace_budget_stats(sequences: list[list[dict[str, float]]]) -> TraceBudgetStats:
    content: list[int] = []
    delimiters: list[int] = []
    by_size: dict[int, list[int]] = {}
    for sequence in sequences:
        for row in sequence:
            size = max(1, int(row["block_size"]))
            nfe = max(1, int(row["nfe"]))
            is_delimiter = size == 1 or row.get("delimiter_fraction", 0.0) >= 0.5
            (delimiters if is_delimiter else content).append(nfe)
            by_size.setdefault(size, []).append(nfe)
    if not content or not delimiters:
        raise ValueError("trace statistics require content and delimiter blocks")
    return TraceBudgetStats(
        content_median=max(1, round(statistics.median(content))),
        delimiter_median=max(1, round(statistics.median(delimiters))),
        by_size={
            size: max(1, round(statistics.median(values))) for size, values in by_size.items()
        },
    )


class BaseBudgetScheduler:
    source = "base"

    def __init__(self, *, seed_budget: int) -> None:
        self.seed_budget = max(1, int(seed_budget))
        self.reset()

    def reset(self) -> None:
        self._block_index = 0
        self._history: list[dict[str, float]] = []
        self.prediction_trace: list[dict[str, object]] = []
        self.scheduler_predict_time_sec = 0.0

    def _budget(self, block_size: int, max_steps: int) -> int:
        del block_size, max_steps
        return self.seed_budget

    def next_schedule(
        self,
        *,
        block_size: int | None = None,
        remaining_tokens: int,
        max_block_length: int,
        max_refinement_steps: int,
    ) -> ScheduledBlock:
        if remaining_tokens < 1:
            raise ValueError("remaining_tokens must be positive")
        size = min(max(1, int(block_size or 1)), int(max_block_length), int(remaining_tokens))
        started = time.perf_counter()
        budget = (
            self.seed_budget
            if self._block_index == 0
            else self._budget(size, int(max_refinement_steps))
        )
        predict_time = time.perf_counter() - started
        self.scheduler_predict_time_sec += predict_time
        budget = min(max(1, int(budget)), int(max_refinement_steps))
        predicted = PhaseTuple(size, budget)
        self.prediction_trace.append(
            {
                "block_index": self._block_index,
                "source": self.source,
                "predicted_tuple": {"block_size": size, "refinement_steps": budget},
                "history_length": len(self._history),
                "predict_time_sec": predict_time,
            }
        )
        self._block_index += 1
        return ScheduledBlock(predicted, size, budget)

    def record_realized(
        self,
        applied_block_size: int,
        actual_nfe_used: int,
        mean_confidence: float = 1.0,
        min_confidence: float = 1.0,
        digit_fraction: float = 0.0,
        delimiter_fraction: float = 0.0,
    ) -> None:
        row = {
            "block_size": float(applied_block_size),
            "nfe": float(actual_nfe_used),
            "mean_top1_confidence": float(mean_confidence),
            "min_top1_confidence": float(min_confidence),
            "digit_fraction": float(digit_fraction),
            "delimiter_fraction": float(delimiter_fraction),
        }
        self._history.append(row)
        if self.prediction_trace:
            self.prediction_trace[-1]["realized_tuple"] = row


class GatesOnlyScheduler(BaseBudgetScheduler):
    source = "gates_only"

    def __init__(self) -> None:
        super().__init__(seed_budget=1)

    def _budget(self, block_size: int, max_steps: int) -> int:
        del block_size, max_steps
        return 1


class ConstantBudgetScheduler(BaseBudgetScheduler):
    source = "constant_budget"

    def __init__(self, *, seed_budget: int, content_budget: int, delimiter_budget: int = 1) -> None:
        self.content_budget = max(1, int(content_budget))
        self.delimiter_budget = max(1, int(delimiter_budget))
        super().__init__(seed_budget=seed_budget)

    def _budget(self, block_size: int, max_steps: int) -> int:
        del max_steps
        return self.delimiter_budget if block_size == 1 else self.content_budget


class SizeLookupBudgetScheduler(ConstantBudgetScheduler):
    source = "size_lookup"

    def __init__(self, *, seed_budget: int, stats: TraceBudgetStats) -> None:
        self.by_size = dict(stats.by_size)
        super().__init__(
            seed_budget=seed_budget,
            content_budget=stats.content_median,
            delimiter_budget=stats.delimiter_median,
        )

    def _budget(self, block_size: int, max_steps: int) -> int:
        del max_steps
        return self.by_size.get(
            block_size,
            self.delimiter_budget if block_size == 1 else self.content_budget,
        )


class PreviousNFEScheduler(BaseBudgetScheduler):
    source = "previous_nfe"

    def _budget(self, block_size: int, max_steps: int) -> int:
        del max_steps
        if block_size == 1:
            return 1
        for row in reversed(self._history):
            if int(row["block_size"]) > 1 and row.get("delimiter_fraction", 0.0) < 0.5:
                return max(1, int(row["nfe"]))
        return self.seed_budget


def _history_features(history: list[dict[str, float]], window_size: int = 8) -> np.ndarray:
    fields = (
        "nfe",
        "block_size",
        "mean_top1_confidence",
        "min_top1_confidence",
        "digit_fraction",
        "delimiter_fraction",
    )
    recent = history[-window_size:]
    features: list[float] = []
    for field in fields:
        values = np.asarray([row.get(field, 0.0) for row in recent], dtype=np.float64)
        if not values.size:
            values = np.zeros(1, dtype=np.float64)
        trend = float(values[-1] - values[0]) if values.size > 1 else 0.0
        features.extend(
            [
                float(values[-1]),
                float(values.mean()),
                float(values.std()),
                float(values.min()),
                float(values.max()),
                trend,
            ]
        )
    features.append(float(len(recent)))
    return np.asarray(features, dtype=np.float64)


class RandomForestBudgetScheduler(BaseBudgetScheduler):
    source = "random_forest"

    def __init__(self, *, seed_budget: int, model: Any) -> None:
        self.model = model
        super().__init__(seed_budget=seed_budget)

    @classmethod
    def fit(
        cls,
        sequences: list[list[dict[str, float]]],
        *,
        seed_budget: int,
        seed: int = 20260710,
    ) -> RandomForestBudgetScheduler:
        from sklearn.ensemble import RandomForestRegressor

        features: list[np.ndarray] = []
        targets: list[float] = []
        for sequence in sequences:
            for index in range(1, len(sequence)):
                features.append(_history_features(sequence[:index]))
                targets.append(float(sequence[index]["nfe"]))
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(np.stack(features), np.asarray(targets))
        return cls(seed_budget=seed_budget, model=model)

    def _budget(self, block_size: int, max_steps: int) -> int:
        if block_size == 1:
            return 1
        prediction = float(self.model.predict(_history_features(self._history)[None, :])[0])
        return min(max_steps, max(1, round(prediction)))

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from phase_predict.schema import PhaseTuple


@dataclass(frozen=True, slots=True)
class TraceBudgetStats:
    content_median: int
    delimiter_median: int
    by_size: dict[int, int]


@dataclass(slots=True)
class ScheduledBlock:
    predicted_tuple: PhaseTuple
    applied_block_size: int
    budgeted_refinement_steps: int


_HISTORY_FIELDS = (
    "nfe",
    "block_size",
    "mean_top1_confidence",
    "min_top1_confidence",
    "digit_fraction",
    "delimiter_fraction",
)


def size_lookup_budget(stats: TraceBudgetStats, block_size: int) -> int:
    fallback = stats.delimiter_median if block_size == 1 else stats.content_median
    return max(1, int(stats.by_size.get(block_size, fallback)))


def residual_features(
    history: list[dict[str, float]],
    *,
    block_size: int,
    stats: TraceBudgetStats,
    window_size: int = 8,
) -> np.ndarray:
    recent = history[-window_size:]
    features: list[float] = []
    for field in _HISTORY_FIELDS:
        values = np.asarray([row.get(field, 0.0) for row in recent], dtype=np.float64)
        if not values.size:
            values = np.zeros(1, dtype=np.float64)
        trend = float(values[-1] - values[0]) if values.size > 1 else 0.0
        features.extend(
            (
                float(values[-1]),
                float(values.mean()),
                float(values.std()),
                float(values.min()),
                float(values.max()),
                trend,
            )
        )
    features.extend(
        (
            float(len(recent)),
            float(block_size),
            float(size_lookup_budget(stats, block_size)),
        )
    )
    return np.asarray(features, dtype=np.float64)


def residual_training_matrix(
    sequences: list[list[dict[str, float]]],
    stats: TraceBudgetStats,
    *,
    window_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    for sequence in sequences:
        for index in range(1, len(sequence)):
            row = sequence[index]
            block_size = max(1, int(row["block_size"]))
            features.append(
                residual_features(
                    sequence[:index],
                    block_size=block_size,
                    stats=stats,
                    window_size=window_size,
                )
            )
            targets.append(float(row["nfe"]) - size_lookup_budget(stats, block_size))
    if not features:
        raise ValueError("residual estimator requires training examples")
    return np.stack(features), np.asarray(targets, dtype=np.float64)


class ResidualEstimator:
    def __init__(
        self,
        *,
        model: RandomForestRegressor,
        stats: TraceBudgetStats,
        window_size: int = 8,
    ) -> None:
        self.model = model
        self.stats = stats
        self.window_size = int(window_size)

    @classmethod
    def fit(
        cls,
        sequences: list[list[dict[str, float]]],
        stats: TraceBudgetStats,
        *,
        seed: int,
        n_estimators: int = 200,
        window_size: int = 8,
    ) -> ResidualEstimator:
        features, targets = residual_training_matrix(
            sequences,
            stats,
            window_size=window_size,
        )
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(features, targets)
        return cls(model=model, stats=stats, window_size=window_size)

    def tree_predictions(
        self,
        history: list[dict[str, float]],
        *,
        block_size: int,
    ) -> np.ndarray:
        row = residual_features(
            history,
            block_size=block_size,
            stats=self.stats,
            window_size=self.window_size,
        )[None, :]
        return np.asarray(
            [float(tree.predict(row)[0]) for tree in self.model.estimators_],
            dtype=np.float64,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "stats": asdict(self.stats),
                "window_size": self.window_size,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> ResidualEstimator:
        payload: dict[str, Any] = joblib.load(Path(path))
        raw_stats = payload["stats"]
        stats = TraceBudgetStats(
            content_median=int(raw_stats["content_median"]),
            delimiter_median=int(raw_stats["delimiter_median"]),
            by_size={int(key): int(value) for key, value in raw_stats["by_size"].items()},
        )
        return cls(
            model=payload["model"],
            stats=stats,
            window_size=int(payload["window_size"]),
        )


class ResidualBudgetScheduler:
    source = "residual_pag"

    def __init__(
        self,
        *,
        seed_budget: int,
        stats: TraceBudgetStats,
        estimator: ResidualEstimator,
        quantile: float,
        max_abs_correction: int,
    ) -> None:
        if not 0 <= quantile <= 1:
            raise ValueError("quantile must be in [0, 1]")
        if max_abs_correction < 0:
            raise ValueError("max_abs_correction must be non-negative")
        self.seed_budget = max(1, int(seed_budget))
        self.stats = stats
        self.estimator = estimator
        self.quantile = float(quantile)
        self.max_abs_correction = int(max_abs_correction)
        self.reset()

    def reset(self) -> None:
        self._block_index = 0
        self._history: list[dict[str, float]] = []
        self.prediction_trace: list[dict[str, object]] = []
        self.scheduler_predict_time_sec = 0.0

    def next_schedule(
        self,
        *,
        block_size: int | None,
        remaining_tokens: int,
        max_block_length: int,
        max_refinement_steps: int,
    ) -> ScheduledBlock:
        if remaining_tokens < 1:
            raise ValueError("remaining_tokens must be positive")
        size = min(max(1, int(block_size or 1)), max_block_length, remaining_tokens)
        started = time.perf_counter()
        prior = size_lookup_budget(self.stats, size)
        residual = 0.0
        correction = 0
        if self._block_index == 0:
            budget = self.seed_budget
        else:
            predictions = self.estimator.tree_predictions(self._history, block_size=size)
            residual = float(np.quantile(predictions, self.quantile))
            correction = int(
                np.clip(
                    round(residual),
                    -self.max_abs_correction,
                    self.max_abs_correction,
                )
            )
            budget = prior + correction
        budget = min(max_refinement_steps, max(1, int(budget)))
        elapsed = time.perf_counter() - started
        self.scheduler_predict_time_sec += elapsed
        self.prediction_trace.append(
            {
                "block_index": self._block_index,
                "source": self.source,
                "prior_budget": prior,
                "residual_quantile": residual,
                "applied_correction": correction,
                "budgeted_refinement_steps": budget,
                "history_length": len(self._history),
                "predict_time_sec": elapsed,
            }
        )
        self._block_index += 1
        return ScheduledBlock(PhaseTuple(size, budget), size, budget)

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

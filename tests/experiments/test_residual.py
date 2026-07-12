from __future__ import annotations

import numpy as np

from pag.experiments.residual import (
    ResidualBudgetScheduler,
    ResidualEstimator,
    TraceBudgetStats,
)


class FakeEstimator:
    def __init__(self, tree_predictions: list[float]) -> None:
        self.predictions = np.asarray(tree_predictions, dtype=np.float64)

    def tree_predictions(
        self,
        history: list[dict[str, float]],
        *,
        block_size: int,
    ) -> np.ndarray:
        assert history
        assert block_size == 16
        return self.predictions


def _sequences() -> list[list[dict[str, float]]]:
    return [
        [
            {
                "block_size": 8.0,
                "nfe": 5.0,
                "mean_top1_confidence": 0.9,
                "min_top1_confidence": 0.7,
                "digit_fraction": 0.0,
                "delimiter_fraction": 0.0,
            },
            {
                "block_size": 16.0,
                "nfe": 6.0,
                "mean_top1_confidence": 0.8,
                "min_top1_confidence": 0.5,
                "digit_fraction": 0.2,
                "delimiter_fraction": 0.0,
            },
            {
                "block_size": 1.0,
                "nfe": 1.0,
                "mean_top1_confidence": 1.0,
                "min_top1_confidence": 1.0,
                "digit_fraction": 0.0,
                "delimiter_fraction": 1.0,
            },
        ],
        [
            {
                "block_size": 4.0,
                "nfe": 4.0,
                "mean_top1_confidence": 0.85,
                "min_top1_confidence": 0.6,
                "digit_fraction": 0.0,
                "delimiter_fraction": 0.0,
            },
            {
                "block_size": 16.0,
                "nfe": 9.0,
                "mean_top1_confidence": 0.7,
                "min_top1_confidence": 0.3,
                "digit_fraction": 0.4,
                "delimiter_fraction": 0.0,
            },
        ],
    ]


def test_residual_scheduler_uses_lower_tree_quantile_and_clamps_correction() -> None:
    estimator = FakeEstimator(tree_predictions=[-4.0, -2.0, 1.0, 2.0])
    scheduler = ResidualBudgetScheduler(
        seed_budget=8,
        stats=TraceBudgetStats(content_median=8, delimiter_median=1, by_size={16: 8}),
        estimator=estimator,
        quantile=0.25,
        max_abs_correction=2,
    )
    first = scheduler.next_schedule(
        block_size=8,
        remaining_tokens=64,
        max_block_length=64,
        max_refinement_steps=32,
    )
    assert first.budgeted_refinement_steps == 8
    scheduler.record_realized(16, 7, 0.9, 0.7, 0.2, 0.0)
    block = scheduler.next_schedule(
        block_size=16,
        remaining_tokens=64,
        max_block_length=64,
        max_refinement_steps=32,
    )
    assert block.budgeted_refinement_steps == 6
    assert scheduler.prediction_trace[-1]["applied_correction"] == -2


def test_residual_estimator_round_trip(tmp_path) -> None:
    stats = TraceBudgetStats(content_median=8, delimiter_median=1, by_size={16: 8})
    estimator = ResidualEstimator.fit(_sequences(), stats, seed=11, n_estimators=8)
    history = _sequences()[0][:1]
    before = estimator.tree_predictions(history, block_size=16)
    path = tmp_path / "residual.joblib"
    estimator.save(path)
    after = ResidualEstimator.load(path).tree_predictions(history, block_size=16)
    np.testing.assert_allclose(after, before)


def test_residual_estimator_rejects_empty_training_data() -> None:
    stats = TraceBudgetStats(content_median=8, delimiter_median=1, by_size={})
    try:
        ResidualEstimator.fit([[]], stats, seed=11, n_estimators=8)
    except ValueError as exc:
        assert "training examples" in str(exc)
    else:
        raise AssertionError("empty training data must fail")

from __future__ import annotations

from pag.experiments.cross_model_runtime import (
    derive_budget_stats,
    records_to_trace_sequences,
)


def _record() -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "block_history": [8, 16, 1],
        "nfe_history": [5, 9, 1],
        "schedule_history": [
            {"mean_top1_confidence": 0.9, "min_top1_confidence": 0.7},
            {"mean_top1_confidence": 0.8, "min_top1_confidence": 0.4},
            {"delimiter_fraction": 1.0},
        ],
    }


def test_records_convert_to_rich_training_sequences() -> None:
    sequences = records_to_trace_sequences([_record()])
    assert len(sequences) == 1
    assert sequences[0][1] == {
        "block_size": 16.0,
        "nfe": 9.0,
        "mean_top1_confidence": 0.8,
        "min_top1_confidence": 0.4,
        "digit_fraction": 0.0,
        "delimiter_fraction": 0.0,
    }


def test_budget_stats_come_only_from_given_records() -> None:
    stats = derive_budget_stats([_record()])
    assert stats.content_median == 7
    assert stats.delimiter_median == 1
    assert stats.by_size == {1: 1, 8: 5, 16: 9}

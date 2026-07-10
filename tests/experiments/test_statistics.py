from __future__ import annotations

from dataclasses import asdict

from pag.experiments.statistics import (
    correctness_matrix,
    exact_mcnemar,
    paired_bootstrap,
    wilson_interval,
)


def test_correctness_matrix_and_mcnemar() -> None:
    matrix = correctness_matrix([True, True, False, False], [True, False, True, False])
    assert asdict(matrix) == {
        "both_correct": 1,
        "left_only": 1,
        "right_only": 1,
        "both_wrong": 1,
    }
    assert exact_mcnemar(matrix).pvalue == 1.0


def test_paired_bootstrap_is_reproducible() -> None:
    first = paired_bootstrap([1, 2, 3], [2, 4, 6], samples=1000, seed=20260710)
    second = paired_bootstrap([1, 2, 3], [2, 4, 6], samples=1000, seed=20260710)
    assert first == second
    assert first.estimate == -2


def test_wilson_interval_contains_observed_accuracy() -> None:
    interval = wilson_interval(90, 100)
    assert interval.lower < 0.9 < interval.upper

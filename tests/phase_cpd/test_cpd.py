from __future__ import annotations

from phase_cpd.cpd import CPDParameters, PeltDetector
from phase_cpd.segments import normalize_breakpoints


def test_normalize_breakpoints_discards_invalid_indices() -> None:
    breakpoints = normalize_breakpoints(
        [0, 3, 3, 4, 9, 99],
        token_count=9,
        min_segment_length=2,
    )

    assert breakpoints == [3]


def test_pelt_detector_returns_interior_sorted_breakpoints() -> None:
    detector = PeltDetector()
    values = [
        0.92,
        0.91,
        0.9,
        0.89,
        0.88,
        0.42,
        0.41,
        0.4,
        0.39,
        0.38,
        0.84,
        0.83,
        0.82,
        0.81,
        0.8,
    ]
    breakpoints = detector.detect(
        values,
        CPDParameters(cost="l2", penalty=0.01, min_segment_length=2, smoothing_window=1),
    )

    assert breakpoints
    assert breakpoints == sorted(set(breakpoints))
    assert all(1 <= breakpoint < len(values) for breakpoint in breakpoints)

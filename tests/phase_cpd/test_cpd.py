from __future__ import annotations

from phase_cpd.cpd import CPDParameters, KernelCPDDetector, PeltDetector
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


def test_kernel_cpd_detector_returns_interior_sorted_breakpoints() -> None:
    detector = KernelCPDDetector(kernel="rbf")
    values = [
        0.1,
        0.12,
        0.11,
        0.09,
        0.08,
        0.7,
        0.74,
        0.72,
        0.71,
        0.73,
        0.2,
        0.19,
        0.18,
        0.22,
        0.21,
    ]
    breakpoints = detector.detect(
        values,
        CPDParameters(cost="l2", penalty=0.01, min_segment_length=2, smoothing_window=1),
    )

    assert breakpoints
    assert breakpoints == sorted(set(breakpoints))
    assert all(1 <= breakpoint < len(values) for breakpoint in breakpoints)

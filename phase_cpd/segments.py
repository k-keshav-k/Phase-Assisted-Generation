from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from phase_cpd.schema import FeatureSeries, SegmentSummary, TraceRecord


def normalize_breakpoints(
    breakpoints: Iterable[int],
    token_count: int,
    *,
    min_segment_length: int = 1,
) -> list[int]:
    if token_count < 2:
        return []

    normalized = sorted(
        {
            int(breakpoint)
            for breakpoint in breakpoints
            if 1 <= int(breakpoint) < token_count
        }
    )

    accepted: list[int] = []
    previous = 0
    for breakpoint in normalized:
        if breakpoint - previous < min_segment_length:
            continue
        if token_count - breakpoint < min_segment_length:
            continue
        accepted.append(breakpoint)
        previous = breakpoint
    return accepted


def segment_ranges(token_count: int, breakpoints: Iterable[int]) -> list[tuple[int, int]]:
    boundaries = [0, *sorted(breakpoints), token_count]
    return [(boundaries[index], boundaries[index + 1]) for index in range(len(boundaries) - 1)]


def build_segment_summaries(
    trace: TraceRecord,
    feature_series: FeatureSeries,
    breakpoints: Iterable[int],
) -> list[SegmentSummary]:
    if len(trace.tokens) != len(feature_series.values):
        msg = "Feature series must have one value per final token"
        raise ValueError(msg)

    summaries: list[SegmentSummary] = []
    ranges = segment_ranges(len(trace.tokens), breakpoints)
    for start_token, end_token in ranges:
        values = np.asarray(feature_series.values[start_token:end_token], dtype=float)
        start_char = trace.tokens[start_token].char_start
        end_char = trace.tokens[end_token - 1].char_end
        summaries.append(
            SegmentSummary(
                start_token=start_token,
                end_token=end_token,
                length=end_token - start_token,
                text=trace.final_text[start_char:end_char],
                mean=float(values.mean()),
                std=float(values.std()),
                minimum=float(values.min()),
                maximum=float(values.max()),
            )
        )
    return summaries


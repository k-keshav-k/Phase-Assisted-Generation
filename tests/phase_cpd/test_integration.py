from __future__ import annotations

from phase_cpd.catalog import default_trace_dir, load_trace_by_id
from phase_cpd.cpd import CPDParameters, PeltDetector
from phase_cpd.features import StabilizingTop1ProbExtractor
from phase_cpd.segments import build_segment_summaries


def test_real_trace_pipeline_smoke() -> None:
    trace = load_trace_by_id("prompt-010", default_trace_dir())
    feature_series = StabilizingTop1ProbExtractor().extract(trace)
    breakpoints = PeltDetector().detect(
        feature_series.values,
        CPDParameters(cost="l2", penalty=0.1, min_segment_length=2, smoothing_window=3),
    )
    segment_summaries = build_segment_summaries(trace, feature_series, breakpoints)

    assert len(segment_summaries) >= 1
    assert "".join(summary.text for summary in segment_summaries) == trace.final_text

from __future__ import annotations

from phase_cpd.cpd import CPDParameters, PeltDetector
from phase_cpd.features import StabilizingTop1ProbExtractor
from phase_cpd.segments import build_segment_summaries
from tests.phase_cpd.trace_fixtures import make_stabilized_trace


def test_trace_pipeline_smoke() -> None:
    trace = make_stabilized_trace(token_texts=["A", "B", "C", "D", "E", "F", "G", "H"])
    feature_series = StabilizingTop1ProbExtractor().extract(trace)
    breakpoints = PeltDetector().detect(
        feature_series.values,
        CPDParameters(cost="l2", penalty=0.1, min_segment_length=2, smoothing_window=3),
    )
    segment_summaries = build_segment_summaries(trace, feature_series, breakpoints)

    assert len(segment_summaries) >= 1
    assert "".join(summary.text for summary in segment_summaries) == trace.final_text

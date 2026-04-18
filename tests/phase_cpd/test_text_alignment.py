from __future__ import annotations

from phase_cpd.catalog import default_trace_dir, load_trace_by_id
from phase_cpd.features import Top1ProbExtractor
from phase_cpd.segments import build_segment_summaries


def test_segment_text_spans_reconstruct_final_text() -> None:
    trace = load_trace_by_id("mock-adaptive-001", default_trace_dir())
    feature_series = Top1ProbExtractor().extract(trace)
    segment_summaries = build_segment_summaries(trace, feature_series, [3, 6])

    assert [summary.length for summary in segment_summaries] == [3, 3, 3]
    assert "".join(summary.text for summary in segment_summaries) == trace.final_text

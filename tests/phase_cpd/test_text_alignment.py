from __future__ import annotations

from phase_cpd.catalog import default_trace_dir, load_trace_by_id
from phase_cpd.features import MeanTop1ProbExtractor
from phase_cpd.segments import build_segment_summaries
from phase_cpd.visualize import render_token_boundary_view_html


def test_segment_text_spans_reconstruct_final_text() -> None:
    trace = load_trace_by_id("prompt-001", default_trace_dir())
    feature_series = MeanTop1ProbExtractor().extract(trace)
    segment_summaries = build_segment_summaries(trace, feature_series, [3, 6])

    assert [summary.length for summary in segment_summaries] == [3, 3, len(trace.tokens) - 6]
    assert "".join(summary.text for summary in segment_summaries) == trace.final_text


def test_boundary_overlay_contains_prompt_and_segment_labels() -> None:
    trace = load_trace_by_id("prompt-001", default_trace_dir())

    rendered = render_token_boundary_view_html(trace, [3, 6])

    assert "Original text with detected response boundaries" in rendered
    assert "Segment 0 [0, 3)" in rendered
    assert "Segment 1 [3, 6)" in rendered
    assert "Explain" in rendered
    assert trace.tokens[0].token_text.strip() in rendered

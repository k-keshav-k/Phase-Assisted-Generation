from __future__ import annotations

from phase_cpd.catalog import default_trace_dir, load_trace_by_id
from phase_cpd.features import StabilizingTop1ProbExtractor
from phase_cpd.schema import FeatureSeries, TraceRecord, TraceToken
from phase_cpd.segments import build_segment_summaries
from phase_cpd.visualize import (
    build_token_feature_table,
    render_segmented_text_html,
    render_token_boundary_view_html,
)


def test_segment_text_spans_reconstruct_final_text() -> None:
    trace = load_trace_by_id("prompt-001", default_trace_dir())
    feature_series = StabilizingTop1ProbExtractor().extract(trace)
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


def test_segmented_text_uses_explicit_dark_text_color() -> None:
    trace = load_trace_by_id("prompt-001", default_trace_dir())
    feature_series = StabilizingTop1ProbExtractor().extract(trace)
    rendered = render_segmented_text_html(build_segment_summaries(trace, feature_series, [3, 6]))

    assert "color:#0f172a" in rendered
    assert "white-space:pre-wrap" in rendered


def test_token_feature_table_includes_token_text_and_scalar_columns() -> None:
    trace = TraceRecord(
        trace_id="feature-table-test",
        backend="dream",
        model_name="dream-test",
        prompt="Prompt",
        final_text="A ",
        tokens=[
            TraceToken(token_index=0, token_text="A", char_start=0, char_end=1),
            TraceToken(token_index=1, token_text=" ", char_start=1, char_end=2),
        ],
    )

    table = build_token_feature_table(
        trace,
        {
            "stabilizing_refinement_step": FeatureSeries(
                feature_name="stabilizing_refinement_step",
                token_indices=[0, 1],
                values=[4.0, 7.0],
            ),
            "stabilizing_entropy": FeatureSeries(
                feature_name="stabilizing_entropy",
                token_indices=[0, 1],
                values=[1.2, 0.8],
            ),
        },
    )

    assert list(table.columns) == [
        "token_index",
        "token_text",
        "stabilizing_refinement_step",
        "stabilizing_entropy",
    ]
    assert table["token_text"].tolist() == ["A", "␠"]

from __future__ import annotations

import html
from collections.abc import Sequence

import altair as alt
import pandas as pd

from phase_cpd.schema import FeatureSeries, SegmentSummary

_SEGMENT_COLORS = ["#eef2ff", "#ecfeff", "#f0fdf4", "#fff7ed"]


def build_feature_chart(feature_series: FeatureSeries, breakpoints: Sequence[int]) -> alt.Chart:
    feature_frame = pd.DataFrame(
        {
            "token_index": feature_series.token_indices,
            "value": feature_series.values,
        }
    )
    line = (
        alt.Chart(feature_frame)
        .mark_line(point=True, color="#2563eb")
        .encode(
            x=alt.X("token_index:Q", title="Token index"),
            y=alt.Y("value:Q", title=feature_series.feature_name),
            tooltip=["token_index", "value"],
        )
    )

    if not breakpoints:
        return line

    rule_frame = pd.DataFrame({"breakpoint": [breakpoint - 0.5 for breakpoint in breakpoints]})
    rules = (
        alt.Chart(rule_frame)
        .mark_rule(color="#dc2626", strokeDash=[6, 4])
        .encode(x="breakpoint:Q")
    )
    return alt.layer(line, rules)


def build_segment_table(segment_summaries: Sequence[SegmentSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": index,
                "span": f"[{summary.start_token}, {summary.end_token})",
                "length": summary.length,
                "text": summary.text,
                "mean": summary.mean,
                "std": summary.std,
                "min": summary.minimum,
                "max": summary.maximum,
            }
            for index, summary in enumerate(segment_summaries)
        ]
    )


def render_segmented_text_html(segment_summaries: Sequence[SegmentSummary]) -> str:
    segments = []
    for index, summary in enumerate(segment_summaries):
        color = _SEGMENT_COLORS[index % len(_SEGMENT_COLORS)]
        label = f"Segment {index} [{summary.start_token}, {summary.end_token})"
        text = html.escape(summary.text)
        segments.append(
            f"<span style='background:{color}; padding:0.35rem 0.45rem; margin:0.2rem;"
            " border-radius:0.45rem; display:inline-block;'>"
            f"<strong style='font-size:0.8rem; color:#334155;'>"
            f"{label}</strong><br>{text}</span>"
        )
    return "".join(segments)

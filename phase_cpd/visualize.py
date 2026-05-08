from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence

import altair as alt
import pandas as pd

from phase_cpd.schema import FeatureSeries, SegmentSummary, TraceRecord
from phase_cpd.segments import segment_ranges

_SEGMENT_STYLES = [
    {"bg": "#FEE2E2", "border": "#DC2626", "token_bg": "#FFFFFF", "label": "#991B1B"},
    {"bg": "#DBEAFE", "border": "#2563EB", "token_bg": "#F8FAFC", "label": "#1D4ED8"},
    {"bg": "#DCFCE7", "border": "#16A34A", "token_bg": "#FFFFFF", "label": "#166534"},
    {"bg": "#FEF3C7", "border": "#D97706", "token_bg": "#FFFFFF", "label": "#B45309"},
    {"bg": "#FCE7F3", "border": "#DB2777", "token_bg": "#FFFFFF", "label": "#9D174D"},
    {"bg": "#E0E7FF", "border": "#4F46E5", "token_bg": "#FFFFFF", "label": "#3730A3"},
]


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
        alt.Chart(rule_frame).mark_rule(color="#dc2626", strokeDash=[6, 4]).encode(x="breakpoint:Q")
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


def build_token_feature_table(
    trace: TraceRecord,
    feature_series_by_name: Mapping[str, FeatureSeries],
) -> pd.DataFrame:
    expected_token_indices = [token.token_index for token in trace.tokens]
    feature_value_maps: dict[str, dict[int, float]] = {}
    for feature_name, series in feature_series_by_name.items():
        if len(series.token_indices) != len(trace.tokens):
            msg = f"Feature series {feature_name!r} must have one value per trace token"
            raise ValueError(msg)
        value_map = {
            token_index: value
            for token_index, value in zip(series.token_indices, series.values, strict=True)
        }
        if set(value_map) != set(expected_token_indices):
            msg = f"Feature series {feature_name!r} token indices do not match the trace"
            raise ValueError(msg)
        feature_value_maps[feature_name] = value_map

    return pd.DataFrame(
        [
            {
                "token_index": token.token_index,
                "token_text": _display_token_text(token.token_text),
                **{
                    feature_name: feature_value_maps[feature_name][token.token_index]
                    for feature_name in feature_series_by_name
                },
            }
            for token in trace.tokens
        ]
    )


def format_breakpoints(breakpoints: Sequence[int]) -> str:
    if not breakpoints:
        return "none"
    return ", ".join(str(breakpoint) for breakpoint in breakpoints)


def render_segmented_text_html(segment_summaries: Sequence[SegmentSummary]) -> str:
    segments = []
    for index, summary in enumerate(segment_summaries):
        style = _segment_style(index)
        label = f"Segment {index} [{summary.start_token}, {summary.end_token})"
        text = html.escape(summary.text)
        segments.append(
            f"<div style='background:{style['bg']}; border:1px solid {style['border']};"
            " padding:0.7rem 0.8rem; border-radius:0.75rem; display:block;"
            " box-shadow:0 1px 2px rgba(15,23,42,0.06);'>"
            f"<strong style='font-size:0.8rem; color:{style['label']};'>"
            f"{label}</strong><br>"
            f"<span style='color:#0f172a; font-weight:500; white-space:pre-wrap;"
            " line-height:1.55;'>"
            f"{text}</span>"
            "</div>"
        )
    return (
        f"<div style='display:flex; flex-direction:column; gap:0.55rem;'>{''.join(segments)}</div>"
    )


def render_token_boundary_view_html(trace: TraceRecord, breakpoints: Sequence[int]) -> str:
    prompt_tokens = _prompt_display_tokens(trace.prompt)
    response_segments = segment_ranges(len(trace.tokens), breakpoints)

    prompt_html = "".join(_prompt_chip(token_text) for token_text in prompt_tokens)
    response_html = "".join(
        _response_segment_box_inline(
            segment_index=segment_index,
            start_token=start_token,
            end_token=end_token,
            token_texts=[token.token_text for token in trace.tokens[start_token:end_token]],
        )
        for segment_index, (start_token, end_token) in enumerate(response_segments)
    )

    return (
        "<div style='border:1px solid #e2e8f0; border-radius:0.9rem; padding:1rem;"
        " background:linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);"
        " box-shadow:0 8px 24px rgba(15,23,42,0.06);'>"
        "<div style='font-size:0.82rem; font-weight:700; letter-spacing:0.02em;"
        " color:#334155; margin-bottom:0.6rem;'>"
        "Original text with detected response boundaries</div>"
        "<div style='display:flex; flex-wrap:wrap; align-items:flex-start; gap:0.35rem;'>"
        f"{prompt_html}"
        "<span style='display:inline-block; padding:0.22rem 0.5rem; border-radius:999px;"
        " background:#0f172a; color:white; font-size:0.78rem; font-weight:700;"
        " margin:0 0.15rem;'>Response</span>"
        f"{response_html}"
        "</div>"
        "</div>"
    )


def _prompt_display_tokens(prompt: str) -> list[str]:
    tokens = re.findall(r"\S+", prompt)
    return tokens if tokens else [prompt]


def _prompt_chip(token_text: str) -> str:
    return (
        "<span style='display:inline-block; padding:0.28rem 0.45rem; border:1px solid #cbd5e1; "
        "border-radius:0.45rem; background:#f8fafc; color:#334155; font-family:ui-monospace, "
        "SFMono-Regular, Menlo, monospace; font-size:0.88rem;'>"
        f"{html.escape(token_text)}"
        "</span>"
    )


def _response_segment_box_inline(
    *,
    segment_index: int,
    start_token: int,
    end_token: int,
    token_texts: Sequence[str],
) -> str:
    style = _segment_style(segment_index)
    label = f"Segment {segment_index} [{start_token}, {end_token})"
    tokens_html = "".join(
        _response_token_chip(token_text, style["token_bg"]) for token_text in token_texts
    )
    return (
        f"<div style='display:inline-flex; flex-wrap:wrap; align-items:flex-start; gap:0.35rem;"
        f" border:2px solid {style['border']}; background:{style['bg']}; border-radius:0.8rem;"
        " padding:0.5rem 0.55rem; box-shadow:0 4px 12px rgba(15,23,42,0.05);'>"
        f"<span style='display:inline-block; padding:0.2rem 0.45rem; border-radius:999px;"
        f" background:{style['border']}; color:white; font-size:0.76rem; font-weight:700;'>"
        f"{label}</span>"
        "<div style='display:flex; flex-wrap:wrap; gap:0.35rem;'>"
        f"{tokens_html}"
        "</div>"
        "</div>"
    )


def _response_token_chip(token_text: str, token_bg: str) -> str:
    display_text = _display_token_text(token_text)
    return (
        "<span style='display:inline-block; padding:0.28rem 0.45rem; border:1px solid #94a3b8; "
        f"border-radius:0.45rem; background:{token_bg}; color:#0f172a; white-space:pre-wrap; "
        "font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:0.88rem;'>"
        f"{html.escape(display_text)}"
        "</span>"
    )


def _display_token_text(token_text: str) -> str:
    display_text = token_text if token_text.strip() else token_text.replace(" ", "␠")
    if not display_text:
        return "∅"
    return display_text


def _segment_style(segment_index: int) -> dict[str, str]:
    return _SEGMENT_STYLES[segment_index % len(_SEGMENT_STYLES)]

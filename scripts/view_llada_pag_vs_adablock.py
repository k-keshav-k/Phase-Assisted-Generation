from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = ROOT / "logs" / "llada_pag_vs_adablock_eval.jsonl"
METHOD_COLORS = ["#0b7285", "#d9480f"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE))
    args, _unknown = parser.parse_known_args()
    return args


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


@st.cache_data(show_spinner=False)
def load_records(path: str) -> list[dict[str, Any]]:
    log_path = _resolve_path(path)
    if not log_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {log_path}") from exc
    return records


def _score(metrics: dict[str, Any]) -> float | None:
    check = metrics.get("substring_check") or {}
    return check.get("score")


def _answer_score(metrics: dict[str, Any]) -> float | None:
    check = metrics.get("answer_check") or {}
    return check.get("score")


def _answer_correct(metrics: dict[str, Any]) -> bool | None:
    check = metrics.get("answer_check") or {}
    return check.get("is_correct")


def _readable(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            labelLimit=180,
            titlePadding=12,
        )
        .configure_axisX(labelAngle=-35, labelOverlap=False)
        .configure_legend(labelFontSize=12, titleFontSize=13)
        .configure_view(strokeWidth=0)
    )


def flatten_methods(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        for method in ("pag", "adablock"):
            payload = record.get(method, {})
            metrics = payload.get("metrics", {})
            rows.append(
                {
                    "record_index": record_index,
                    "run_id": record.get("run_id"),
                    "prompt_id": record.get("prompt_id") or f"record_{record_index}",
                    "category": record.get("prompt_category") or "uncategorized",
                    "created_at": record.get("created_at"),
                    "method": "PAG" if method == "pag" else "AdaBlock",
                    "total_nfe": metrics.get("total_nfe"),
                    "num_blocks": metrics.get("num_blocks"),
                    "avg_block_size": metrics.get("avg_block_size"),
                    "avg_nfe_per_block": metrics.get("avg_nfe_per_block"),
                    "elapsed_sec": metrics.get("elapsed_sec"),
                    "decoded_chars": metrics.get("decoded_chars"),
                    "substring_score": _score(metrics),
                    "answer_score": _answer_score(metrics),
                    "answer_correct": _answer_correct(metrics),
                    "expected_answers": record.get("expected_answers", []),
                    "generated_text": payload.get("generated_text", ""),
                }
            )
    return pd.DataFrame(rows)


def flatten_deltas(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        delta = record.get("delta", {})
        rows.append(
            {
                "record_index": record_index,
                "run_id": record.get("run_id"),
                "prompt_id": record.get("prompt_id") or f"record_{record_index}",
                "category": record.get("prompt_category") or "uncategorized",
                "prompt": record.get("prompt", ""),
                "nfe_delta_pag_minus_adablock": delta.get("nfe_delta_pag_minus_adablock"),
                "nfe_ratio_pag_over_adablock": delta.get("nfe_ratio_pag_over_adablock"),
                "elapsed_delta_sec_pag_minus_adablock": delta.get(
                    "elapsed_delta_sec_pag_minus_adablock"
                ),
                "block_count_delta_pag_minus_adablock": delta.get(
                    "block_count_delta_pag_minus_adablock"
                ),
                "substring_score_delta_pag_minus_adablock": delta.get(
                    "substring_score_delta_pag_minus_adablock"
                ),
                "answer_score_delta_pag_minus_adablock": delta.get(
                    "answer_score_delta_pag_minus_adablock"
                ),
            }
        )
    return pd.DataFrame(rows)


def flatten_pag_blocks(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        for block in record.get("pag", {}).get("block_visualization", []):
            predicted = block.get("predicted_tuple", {})
            rows.append(
                {
                    "record_index": record_index,
                    "prompt_id": record.get("prompt_id") or f"record_{record_index}",
                    "category": record.get("prompt_category") or "uncategorized",
                    "block_index": block.get("block_index"),
                    "predicted_block_size": predicted.get("block_size"),
                    "predicted_refinement_steps": predicted.get("refinement_steps"),
                    "applied_block_size": block.get("applied_block_size"),
                    "budgeted_refinement_steps": block.get("budgeted_refinement_steps"),
                    "actual_nfe_used": block.get("actual_nfe_used"),
                    "block_text": block.get("block_text", ""),
                }
            )
    return pd.DataFrame(rows)


def render_metrics(method_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    pag = method_df[method_df["method"] == "PAG"]
    adablock = method_df[method_df["method"] == "AdaBlock"]
    nfe_ratio = delta_df["nfe_ratio_pag_over_adablock"].dropna()
    elapsed_delta = delta_df["elapsed_delta_sec_pag_minus_adablock"].dropna()
    pag_accuracy = pag["answer_score"].dropna()
    adablock_accuracy = adablock["answer_score"].dropna()

    cols = st.columns(7)
    cols[0].metric("Prompts", int(delta_df.shape[0]))
    cols[1].metric("PAG Avg NFE", f"{pag['total_nfe'].mean():.2f}")
    cols[2].metric("AdaBlock Avg NFE", f"{adablock['total_nfe'].mean():.2f}")
    cols[3].metric("Avg NFE Ratio", f"{nfe_ratio.mean():.2f}" if not nfe_ratio.empty else "n/a")
    cols[4].metric(
        "Avg Runtime Delta",
        f"{elapsed_delta.mean():.2f}s" if not elapsed_delta.empty else "n/a",
    )
    cols[5].metric(
        "PAG Accuracy",
        f"{pag_accuracy.mean() * 100:.1f}%" if not pag_accuracy.empty else "n/a",
    )
    cols[6].metric(
        "AdaBlock Accuracy",
        f"{adablock_accuracy.mean() * 100:.1f}%" if not adablock_accuracy.empty else "n/a",
    )


def render_overview_charts(method_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    nfe_chart = (
        alt.Chart(method_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "prompt_id:N",
                title="Prompt",
                sort=None,
                axis=alt.Axis(labelAngle=-35, labelLimit=170),
            ),
            xOffset=alt.XOffset("method:N"),
            y=alt.Y("total_nfe:Q", title="Total NFE"),
            color=alt.Color("method:N", title="Method", scale=alt.Scale(range=METHOD_COLORS)),
            tooltip=["prompt_id:N", "category:N", "method:N", "total_nfe:Q", "num_blocks:Q"],
        )
        .properties(height=340)
    )
    st.altair_chart(_readable(nfe_chart), use_container_width=True)

    delta_chart = (
        alt.Chart(delta_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "prompt_id:N",
                title="Prompt",
                sort=None,
                axis=alt.Axis(labelAngle=-35, labelLimit=170),
            ),
            y=alt.Y(
                "nfe_delta_pag_minus_adablock:Q",
                title="NFE delta (PAG - AdaBlock)",
            ),
            color=alt.condition(
                alt.datum.nfe_delta_pag_minus_adablock <= 0,
                alt.value("#2f7d32"),
                alt.value("#b3261e"),
            ),
            tooltip=[
                "prompt_id:N",
                "category:N",
                "nfe_delta_pag_minus_adablock:Q",
                "nfe_ratio_pag_over_adablock:Q",
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(_readable(delta_chart), use_container_width=True)

    answer_df = method_df.dropna(subset=["answer_score"])
    if answer_df.empty:
        st.info(
            "No answer accuracy metrics found. "
            "Re-run the comparison eval to populate answer_check."
        )
    else:
        accuracy_chart = (
            alt.Chart(answer_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "prompt_id:N",
                    title="Prompt",
                    sort=None,
                    axis=alt.Axis(labelAngle=-35, labelLimit=170),
                ),
                xOffset=alt.XOffset("method:N"),
                y=alt.Y("answer_score:Q", title="Answer accuracy", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("method:N", title="Method", scale=alt.Scale(range=METHOD_COLORS)),
                tooltip=[
                    "prompt_id:N",
                    "category:N",
                    "method:N",
                    "answer_correct:N",
                    "expected_answers:N",
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(_readable(accuracy_chart), use_container_width=True)

    runtime = (
        alt.Chart(method_df)
        .mark_circle(size=90, opacity=0.8)
        .encode(
            x=alt.X("elapsed_sec:Q", title="Runtime seconds"),
            y=alt.Y("total_nfe:Q", title="Total NFE"),
            color=alt.Color("method:N", title="Method", scale=alt.Scale(range=METHOD_COLORS)),
            shape=alt.Shape("category:N", title="Category"),
            tooltip=[
                "prompt_id:N",
                "category:N",
                "method:N",
                "elapsed_sec:Q",
                "total_nfe:Q",
                "answer_score:Q",
                "substring_score:Q",
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(_readable(runtime), use_container_width=True)


def render_category_summary(method_df: pd.DataFrame) -> None:
    if method_df.empty:
        return
    summary = (
        method_df.groupby(["category", "method"], dropna=False)
        .agg(
            prompts=("prompt_id", "nunique"),
            avg_nfe=("total_nfe", "mean"),
            avg_blocks=("num_blocks", "mean"),
            avg_runtime=("elapsed_sec", "mean"),
            accuracy=("answer_score", "mean"),
            avg_score=("substring_score", "mean"),
        )
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_prompt_detail(records: list[dict[str, Any]], method_df: pd.DataFrame) -> None:
    if not records:
        return

    labels = [
        f"{record.get('prompt_id') or f'record_{index}'} | "
        f"{record.get('prompt_category') or 'uncategorized'}"
        for index, record in enumerate(records)
    ]
    selected = st.selectbox("Prompt", labels)
    record_index = labels.index(selected)
    record = records[record_index]

    st.subheader(record.get("prompt_id") or f"record_{record_index}")
    st.caption(record.get("prompt_category") or "uncategorized")
    st.write(record.get("prompt", ""))

    prompt_metrics = method_df[method_df["record_index"] == record_index]
    st.dataframe(
        prompt_metrics[
            [
                "method",
                "total_nfe",
                "num_blocks",
                "avg_block_size",
                "avg_nfe_per_block",
                "elapsed_sec",
                "answer_correct",
                "answer_score",
                "substring_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    col_pag, col_adablock = st.columns(2)
    with col_pag:
        st.markdown("**PAG Output**")
        st.json(record.get("pag", {}).get("metrics", {}).get("answer_check", {}))
        st.write(record.get("pag", {}).get("generated_text", ""))
    with col_adablock:
        st.markdown("**AdaBlock Output**")
        st.json(record.get("adablock", {}).get("metrics", {}).get("answer_check", {}))
        st.write(record.get("adablock", {}).get("generated_text", ""))

    pag_blocks = record.get("pag", {}).get("block_visualization", [])
    if pag_blocks:
        st.markdown("**PAG Block Trace**")
        for block in pag_blocks:
            predicted = block.get("predicted_tuple", {})
            title = (
                f"Block {block.get('block_index')} | "
                f"pred=({predicted.get('block_size')}, "
                f"{predicted.get('refinement_steps')}) | "
                f"actual_nfe={block.get('actual_nfe_used')}"
            )
            with st.expander(title):
                st.code(block.get("block_text", ""), language="text")
                st.json(
                    {
                        "applied_block_size": block.get("applied_block_size"),
                        "budgeted_refinement_steps": block.get(
                            "budgeted_refinement_steps"
                        ),
                        "actual_nfe_used": block.get("actual_nfe_used"),
                        "predictor_trace": block.get("predictor_trace"),
                    }
                )


def render_pag_block_analysis(block_df: pd.DataFrame) -> None:
    if block_df.empty:
        st.info("No PAG block traces found in this comparison log.")
        return

    scatter = (
        alt.Chart(block_df)
        .mark_circle(size=65, opacity=0.75)
        .encode(
            x=alt.X("applied_block_size:Q", title="Applied block size"),
            y=alt.Y("actual_nfe_used:Q", title="Actual NFE"),
            color=alt.Color("category:N", title="Category"),
            tooltip=[
                "prompt_id:N",
                "block_index:Q",
                "applied_block_size:Q",
                "actual_nfe_used:Q",
                "block_text:N",
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(_readable(scatter), use_container_width=True)
    st.dataframe(block_df, use_container_width=True, hide_index=True)


def _filter_records(records: list[dict[str, Any]], categories: list[str]) -> list[dict[str, Any]]:
    if not categories:
        return records
    return [
        record
        for record in records
        if (record.get("prompt_category") or "uncategorized") in categories
    ]


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="PAG vs AdaBlock", layout="wide")
    st.title("PAG vs AdaBlock")

    log_file = st.sidebar.text_input("Comparison log file", value=str(_resolve_path(args.log_file)))
    if st.sidebar.button("Reload"):
        load_records.clear()

    records = load_records(log_file)
    if not records:
        st.warning(f"No comparison records found at {log_file}")
        return

    all_categories = sorted(
        {
            record.get("prompt_category") or "uncategorized"
            for record in records
        }
    )
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=all_categories,
        default=all_categories,
    )
    records = _filter_records(records, selected_categories)
    method_df = flatten_methods(records)
    delta_df = flatten_deltas(records)
    block_df = flatten_pag_blocks(records)

    tab_overview, tab_categories, tab_prompt, tab_pag_blocks, tab_data = st.tabs(
        ["Overview", "Categories", "Prompt Detail", "PAG Blocks", "Raw Tables"]
    )
    with tab_overview:
        render_metrics(method_df, delta_df)
        render_overview_charts(method_df, delta_df)
    with tab_categories:
        render_category_summary(method_df)
    with tab_prompt:
        render_prompt_detail(records, method_df)
    with tab_pag_blocks:
        render_pag_block_analysis(block_df)
    with tab_data:
        st.markdown("**Method Metrics**")
        st.dataframe(method_df, use_container_width=True, hide_index=True)
        st.markdown("**Deltas**")
        st.dataframe(delta_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

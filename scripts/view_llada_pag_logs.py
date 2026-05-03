from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = ROOT / "logs" / "llada_pag_inference.jsonl"


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


def flatten_blocks(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        blocks = record.get("block_visualization", [])
        summary = record.get("summary", {})
        for block in blocks:
            predicted = block.get("predicted_tuple", {})
            span = block.get("generated_span", {})
            rows.append(
                {
                    "record_index": record_index,
                    "run_id": record.get("run_id"),
                    "prompt_id": record.get("prompt_id") or f"record_{record_index}",
                    "category": record.get("prompt_category") or "uncategorized",
                    "created_at": record.get("created_at"),
                    "block_index": block.get("block_index"),
                    "predicted_block_size": predicted.get("block_size"),
                    "predicted_refinement_steps": predicted.get("refinement_steps"),
                    "applied_block_size": block.get("applied_block_size"),
                    "budgeted_refinement_steps": block.get("budgeted_refinement_steps"),
                    "actual_nfe_used": block.get("actual_nfe_used"),
                    "span_start": span.get("start"),
                    "span_end": span.get("end"),
                    "block_text": block.get("block_text", ""),
                    "total_nfe": summary.get("total_nfe"),
                    "num_blocks": summary.get("num_blocks"),
                }
            )
    return pd.DataFrame(rows)


def summarize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        summary = record.get("summary", {})
        rows.append(
            {
                "record_index": index,
                "run_id": record.get("run_id"),
                "prompt_id": record.get("prompt_id") or f"record_{index}",
                "category": record.get("prompt_category") or "uncategorized",
                "created_at": record.get("created_at"),
                "prompt": record.get("prompt", ""),
                "num_blocks": summary.get("num_blocks", len(record.get("block_history", []))),
                "total_nfe": summary.get("total_nfe", sum(record.get("nfe_history", []))),
                "avg_block_size": summary.get("avg_block_size"),
                "avg_refinement_steps": summary.get("avg_refinement_steps"),
            }
        )
    return pd.DataFrame(rows)


def render_overview(record_df: pd.DataFrame, block_df: pd.DataFrame) -> None:
    total_prompts = len(record_df)
    total_blocks = len(block_df)
    avg_block = block_df["applied_block_size"].mean() if total_blocks else 0
    avg_nfe = block_df["actual_nfe_used"].mean() if total_blocks else 0

    cols = st.columns(4)
    cols[0].metric("Prompts", total_prompts)
    cols[1].metric("Blocks", total_blocks)
    cols[2].metric("Avg Block Size", f"{avg_block:.2f}")
    cols[3].metric("Avg NFE", f"{avg_nfe:.2f}")

    if block_df.empty:
        return

    scatter = (
        alt.Chart(block_df)
        .mark_circle(size=70, opacity=0.75)
        .encode(
            x=alt.X("applied_block_size:Q", title="Applied block size"),
            y=alt.Y("actual_nfe_used:Q", title="Actual NFE"),
            color=alt.Color("category:N", title="Category"),
            tooltip=[
                "prompt_id:N",
                "category:N",
                "block_index:Q",
                "applied_block_size:Q",
                "actual_nfe_used:Q",
                "block_text:N",
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)

    by_category = (
        block_df.groupby("category", dropna=False)
        .agg(
            blocks=("block_index", "count"),
            avg_block_size=("applied_block_size", "mean"),
            avg_nfe=("actual_nfe_used", "mean"),
            avg_budget=("budgeted_refinement_steps", "mean"),
        )
        .reset_index()
    )
    st.dataframe(by_category, use_container_width=True, hide_index=True)


def render_prompt_detail(records: list[dict[str, Any]], block_df: pd.DataFrame) -> None:
    if not records:
        return

    options = [
        f"{record.get('prompt_id') or f'record_{index}'} | "
        f"{record.get('prompt_category') or 'uncategorized'}"
        for index, record in enumerate(records)
    ]
    selected_label = st.selectbox("Prompt", options)
    selected_index = options.index(selected_label)
    record = records[selected_index]
    selected_blocks = block_df[block_df["record_index"] == selected_index].copy()

    st.subheader(record.get("prompt_id") or f"record_{selected_index}")
    st.caption(record.get("prompt_category") or "uncategorized")
    st.write(record.get("prompt", ""))

    with st.expander("Generated Text", expanded=True):
        st.write(record.get("generated_text", ""))

    if not selected_blocks.empty:
        line = (
            alt.Chart(selected_blocks)
            .mark_line(point=True)
            .encode(
                x=alt.X("block_index:Q", title="Block"),
                y=alt.Y("applied_block_size:Q", title="Applied block size"),
                tooltip=[
                    "block_index:Q",
                    "applied_block_size:Q",
                    "actual_nfe_used:Q",
                    "block_text:N",
                ],
            )
            .properties(height=260)
        )
        bars = (
            alt.Chart(selected_blocks)
            .mark_bar(opacity=0.35)
            .encode(
                x=alt.X("block_index:Q", title="Block"),
                y=alt.Y("actual_nfe_used:Q", title="Actual NFE"),
            )
        )
        st.altair_chart(line + bars, use_container_width=True)

    for block in record.get("block_visualization", []):
        predicted = block.get("predicted_tuple", {})
        title = (
            f"Block {block.get('block_index')} | "
            f"pred=({predicted.get('block_size')}, {predicted.get('refinement_steps')}) | "
            f"actual_nfe={block.get('actual_nfe_used')}"
        )
        with st.expander(title):
            st.code(block.get("block_text", ""), language="text")
            st.json(
                {
                    "applied_block_size": block.get("applied_block_size"),
                    "budgeted_refinement_steps": block.get("budgeted_refinement_steps"),
                    "actual_nfe_used": block.get("actual_nfe_used"),
                    "generated_span": block.get("generated_span"),
                    "predictor_trace": block.get("predictor_trace"),
                }
            )


def main() -> None:
    args = _parse_args()

    st.set_page_config(page_title="LLaDA PAG Logs", layout="wide")
    st.title("LLaDA PAG Logs")

    log_file = st.sidebar.text_input("Log file", value=str(_resolve_path(args.log_file)))
    reload_clicked = st.sidebar.button("Reload")
    if reload_clicked:
        load_records.clear()

    records = load_records(log_file)
    if not records:
        st.warning(f"No log records found at {log_file}")
        return

    record_df = summarize_records(records)
    block_df = flatten_blocks(records)

    categories = (
        sorted(block_df["category"].dropna().unique().tolist())
        if not block_df.empty
        else []
    )
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=categories,
        default=categories,
    )
    if selected_categories:
        keep_record_indices = set(
            block_df[block_df["category"].isin(selected_categories)]["record_index"].tolist()
        )
        records = [record for index, record in enumerate(records) if index in keep_record_indices]
        record_df = summarize_records(records)
        block_df = flatten_blocks(records)

    tab_overview, tab_prompts, tab_blocks = st.tabs(["Overview", "Prompt Detail", "Blocks"])
    with tab_overview:
        render_overview(record_df, block_df)
    with tab_prompts:
        render_prompt_detail(records, block_df)
    with tab_blocks:
        st.dataframe(block_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

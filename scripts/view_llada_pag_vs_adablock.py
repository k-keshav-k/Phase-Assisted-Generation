from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = ROOT / "logs" / "llada_pag_vs_adablock_eval.jsonl"
METHOD_COLORS = ["#0b7285", "#d9480f"]
METHOD_DOMAIN = ["PAG", "AdaBlock"]
PAG_BLOCK_COLOR = "rgba(11, 114, 133, 0.78)"
ADABLOCK_COLOR = "rgba(217, 72, 15, 0.76)"


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


def _method_color() -> alt.Color:
    return alt.Color(
        "method:N",
        title="Method",
        scale=alt.Scale(domain=METHOD_DOMAIN, range=METHOD_COLORS),
        sort=METHOD_DOMAIN,
    )


def _method_offset() -> alt.XOffset:
    return alt.XOffset("method:N", sort=METHOD_DOMAIN)


def _run_label(record: dict[str, Any]) -> str:
    return str(record.get("run_id") or "no_run_id")


def _latest_run_label(records: list[dict[str, Any]]) -> str | None:
    latest_by_run: dict[str, str] = {}
    for record in records:
        run_label = _run_label(record)
        created_at = str(record.get("created_at") or "")
        latest_by_run[run_label] = max(created_at, latest_by_run.get(run_label, ""))
    if not latest_by_run:
        return None
    return max(latest_by_run, key=lambda run_label: latest_by_run[run_label])


def _filter_records_by_run(records: list[dict[str, Any]], run_label: str) -> list[dict[str, Any]]:
    if run_label == "All runs":
        return records
    return [record for record in records if _run_label(record) == run_label]


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
                    "run_label": _run_label(record),
                    "prompt_id": record.get("prompt_id") or f"record_{record_index}",
                    "prompt_label": (
                        f"{record.get('prompt_id') or f'record_{record_index}'} "
                        f"#{record_index}"
                    ),
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
                "run_label": _run_label(record),
                "prompt_id": record.get("prompt_id") or f"record_{record_index}",
                "prompt_label": (
                    f"{record.get('prompt_id') or f'record_{record_index}'} "
                    f"#{record_index}"
                ),
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


def validate_graph_data(method_df: pd.DataFrame, delta_df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if method_df.empty:
        return ["No method rows are available for graphing."]

    duplicate_methods = method_df.groupby(["record_index", "method"]).size()
    duplicate_methods = duplicate_methods[duplicate_methods > 1]
    if not duplicate_methods.empty:
        issues.append(
            "Duplicate method rows detected for at least one prompt; "
            "method comparison bars may overlap."
        )

    methods_by_record = method_df.groupby("record_index")["method"].apply(set)
    missing_methods = methods_by_record[
        methods_by_record.apply(lambda methods: set(METHOD_DOMAIN) - methods != set())
    ]
    if not missing_methods.empty:
        issues.append("At least one prompt is missing a PAG or AdaBlock row.")

    if len(delta_df) != method_df["record_index"].nunique():
        issues.append("Delta rows do not align one-to-one with prompts.")

    answer_df = method_df.dropna(subset=["answer_score"])
    if not answer_df.empty:
        inconsistent_answer = answer_df[
            answer_df.apply(
                lambda row: bool(row["answer_correct"]) != bool(row["answer_score"]),
                axis=1,
            )
        ]
        if not inconsistent_answer.empty:
            issues.append("Answer score and answer_correct disagree for at least one row.")

    for record_index, rows in method_df.groupby("record_index"):
        by_method = rows.set_index("method")
        if not set(METHOD_DOMAIN).issubset(by_method.index):
            continue
        delta_rows = delta_df[delta_df["record_index"] == record_index]
        if delta_rows.empty:
            continue
        delta = delta_rows.iloc[0]
        expected_nfe_delta = (
            by_method.loc["PAG", "total_nfe"] - by_method.loc["AdaBlock", "total_nfe"]
        )
        actual_nfe_delta = delta.get("nfe_delta_pag_minus_adablock")
        if pd.notna(actual_nfe_delta) and float(actual_nfe_delta) != float(expected_nfe_delta):
            issues.append(f"NFE delta mismatch for record {record_index}.")

        pag_answer = by_method.loc["PAG", "answer_score"]
        adablock_answer = by_method.loc["AdaBlock", "answer_score"]
        expected_answer_delta = (
            pag_answer - adablock_answer
            if pd.notna(pag_answer) and pd.notna(adablock_answer)
            else None
        )
        actual_answer_delta = delta.get("answer_score_delta_pag_minus_adablock")
        if (
            expected_answer_delta is not None
            and pd.notna(actual_answer_delta)
            and float(actual_answer_delta) != float(expected_answer_delta)
        ):
            issues.append(f"Answer delta mismatch for record {record_index}.")

    return issues


def _block_badge(block: dict[str, Any]) -> str:
    refinement = block.get("budgeted_refinement_steps")
    if refinement is None:
        predicted = block.get("predicted_tuple") or {}
        refinement = predicted.get("refinement_steps")
    if refinement is None:
        refinement = block.get("actual_nfe_used")
    return str(refinement) if refinement is not None else "?"


def _block_title(block: dict[str, Any]) -> str:
    return (
        f"block {block.get('block_index')} | "
        f"size {block.get('applied_block_size')} | "
        f"max refinement {block.get('budgeted_refinement_steps')}"
    )


def _approximate_blocks_from_history(output: dict[str, Any]) -> list[dict[str, Any]]:
    generated_text = str(output.get("generated_text", ""))
    block_history = output.get("block_history") or []
    nfe_history = output.get("nfe_history") or []
    if not generated_text or not block_history:
        return []

    sizes = [max(1, int(size)) for size in block_history]
    total_size = sum(sizes)
    blocks: list[dict[str, Any]] = []
    cursor = 0
    cumulative = 0

    for index, block_size in enumerate(sizes):
        cumulative += block_size
        if index == len(sizes) - 1:
            end = len(generated_text)
        else:
            end = round(len(generated_text) * cumulative / total_size)
            end = max(cursor + 1, min(end, len(generated_text)))
        actual_nfe = int(nfe_history[index]) if index < len(nfe_history) else None
        blocks.append(
            {
                "block_index": index,
                "applied_block_size": block_size,
                "budgeted_refinement_steps": actual_nfe,
                "actual_nfe_used": actual_nfe,
                "block_text": generated_text[cursor:end],
            }
        )
        cursor = end
        if cursor >= len(generated_text):
            break

    return blocks


def _blocks_for_output(output: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    blocks = output.get("block_visualization") or []
    if blocks:
        return blocks, False
    return _approximate_blocks_from_history(output), True


def _render_blocked_output(
    *,
    label: str,
    output: dict[str, Any],
    color: str,
) -> None:
    blocks, is_approximate = _blocks_for_output(output)
    generated_text = output.get("generated_text", "")
    if not blocks:
        st.write(generated_text)
        st.caption("No block history found in this log. Re-run the comparison eval.")
        return

    pieces = []
    for block in blocks:
        block_text = html.escape(str(block.get("block_text", ""))).replace("\n", "<br>")
        if not block_text:
            continue
        title = html.escape(_block_title(block), quote=True)
        badge = html.escape(_block_badge(block), quote=True)
        pieces.append(
            "<span class='generation-block' "
            f"style='--block-color:{color}' "
            f"title='{title}' "
            f"data-refinement='{badge}'>"
            f"{block_text}"
            "</span>"
        )

    if not pieces:
        st.write(generated_text)
        return

    st.markdown(f"**{label} Output**")
    if is_approximate:
        st.caption("Block boundaries are approximate for this older log.")
    st.markdown(
        "<span class='blocked-output'>" + "".join(pieces) + "</span>",
        unsafe_allow_html=True,
    )


def _render_block_css() -> None:
    st.markdown(
        """
        <style>
        .blocked-output {
            white-space: normal;
            font-size: 0.96rem;
            line-height: 2.0;
            color: #ffffff;
        }
        .generation-block {
            position: relative;
            display: inline-block;
            max-width: 100%;
            padding: 0.16rem 0.62rem 0.54rem 0.22rem;
            margin: 0 0.14rem 0.28rem 0;
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 0.28rem;
            background: var(--block-color);
            color: #ffffff !important;
            overflow-wrap: anywhere;
            white-space: normal;
            vertical-align: top;
        }
        .generation-block::after {
            content: attr(data-refinement);
            position: absolute;
            right: 0.16rem;
            bottom: 0.04rem;
            color: rgba(255, 255, 255, 0.86);
            font-size: 0.54rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            pointer-events: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
                "prompt_label:N",
                title="Prompt",
                sort=None,
                axis=alt.Axis(labelAngle=-35, labelLimit=170),
            ),
            xOffset=_method_offset(),
            y=alt.Y("total_nfe:Q", title="Total NFE"),
            color=_method_color(),
            tooltip=[
                "prompt_id:N",
                "run_label:N",
                "category:N",
                "method:N",
                "total_nfe:Q",
                "num_blocks:Q",
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(_readable(nfe_chart), use_container_width=True)

    delta_chart = (
        alt.Chart(delta_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "prompt_label:N",
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
                "run_label:N",
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
                    "prompt_label:N",
                    title="Prompt",
                    sort=None,
                    axis=alt.Axis(labelAngle=-35, labelLimit=170),
                ),
                xOffset=_method_offset(),
                y=alt.Y("answer_score:Q", title="Answer accuracy", scale=alt.Scale(domain=[0, 1])),
                color=_method_color(),
                tooltip=[
                    "prompt_id:N",
                    "run_label:N",
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
            color=_method_color(),
            shape=alt.Shape("category:N", title="Category"),
            tooltip=[
                "prompt_id:N",
                "run_label:N",
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
    _render_block_css()

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
        st.json(record.get("pag", {}).get("metrics", {}).get("answer_check", {}))
        _render_blocked_output(
            label="PAG",
            output=record.get("pag", {}),
            color=PAG_BLOCK_COLOR,
        )
    with col_adablock:
        st.json(record.get("adablock", {}).get("metrics", {}).get("answer_check", {}))
        _render_blocked_output(
            label="AdaBlock",
            output=record.get("adablock", {}),
            color=ADABLOCK_COLOR,
        )

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

    run_options = ["All runs", *sorted({_run_label(record) for record in records})]
    latest_run = _latest_run_label(records)
    default_run_index = run_options.index(latest_run) if latest_run in run_options else 0
    selected_run = st.sidebar.selectbox(
        "Run",
        options=run_options,
        index=default_run_index,
        help="Defaulting to the latest run avoids mixing stale and fresh records.",
    )
    records = _filter_records_by_run(records, selected_run)

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
    graph_issues = validate_graph_data(method_df, delta_df)

    tab_overview, tab_categories, tab_prompt, tab_pag_blocks, tab_data = st.tabs(
        ["Overview", "Categories", "Prompt Detail", "PAG Blocks", "Raw Tables"]
    )
    with tab_overview:
        if graph_issues:
            st.warning("Graph data validation found issues:\n\n" + "\n".join(graph_issues))
        else:
            st.success("Graph data validation passed for the selected run/categories.")
        render_metrics(method_df, delta_df)
        render_overview_charts(method_df, delta_df)
    with tab_categories:
        render_category_summary(method_df)
    with tab_prompt:
        render_prompt_detail(records, method_df)
    with tab_pag_blocks:
        render_pag_block_analysis(block_df)
    with tab_data:
        st.markdown("**Graph Data Validation**")
        if graph_issues:
            st.warning("\n".join(graph_issues))
        else:
            st.success("No graph data issues detected.")
        st.markdown("**Method Metrics**")
        st.dataframe(method_df, use_container_width=True, hide_index=True)
        st.markdown("**Deltas**")
        st.dataframe(delta_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

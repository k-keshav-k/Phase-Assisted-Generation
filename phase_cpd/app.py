from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_cpd.catalog import (  # noqa: E402
    default_trace_dir,
    filter_catalog_entries,
    list_catalog_entries,
    load_trace_by_id,
)
from phase_cpd.cpd import CPDParameters, PeltDetector  # noqa: E402
from phase_cpd.features import FEATURE_EXTRACTORS, get_feature_extractor  # noqa: E402
from phase_cpd.segments import build_segment_summaries  # noqa: E402
from phase_cpd.visualize import (  # noqa: E402
    build_feature_chart,
    build_segment_table,
    render_segmented_text_html,
)

_TRACE_DIR = default_trace_dir()


@st.cache_data
def _catalog(trace_dir: str) -> list:
    return list_catalog_entries(trace_dir)


@st.cache_data
def _trace(trace_dir: str, trace_id: str):
    return load_trace_by_id(trace_id, trace_dir)


def main() -> None:
    st.set_page_config(page_title="PAG Phase CPD", layout="wide")
    st.title("PAG Phase Segmentation Explorer")
    st.caption("Offline change-point analysis over curated diffusion trace files.")

    entries = _catalog(str(_TRACE_DIR))
    if not entries:
        st.error(f"No trace files were found in {_TRACE_DIR}.")
        return

    backends = sorted({entry.backend for entry in entries})
    selected_backend = st.sidebar.selectbox("Backend filter", ["All", *backends])

    backend_filtered = filter_catalog_entries(
        entries,
        backend=None if selected_backend == "All" else selected_backend,
    )

    model_options = sorted({entry.model_name for entry in backend_filtered})
    selected_model = st.sidebar.selectbox("Model filter", ["All", *model_options])

    model_filtered = filter_catalog_entries(
        backend_filtered,
        model_name=None if selected_model == "All" else selected_model,
    )

    tag_options = sorted({tag for entry in model_filtered for tag in entry.tags})
    selected_tags = st.sidebar.multiselect("Tag filter", tag_options)

    run_options = sorted({entry.run_id for entry in model_filtered if entry.run_id})
    selected_run = st.sidebar.selectbox("Run filter", ["All", *run_options])

    filtered_entries = filter_catalog_entries(
        model_filtered,
        required_tags=set(selected_tags),
        run_id=None if selected_run == "All" else selected_run,
    )

    if not filtered_entries:
        st.warning("No traces match the current filters.")
        return

    trace_labels = {entry.label: entry.trace_id for entry in filtered_entries}
    selected_label = st.sidebar.selectbox("Trace", list(trace_labels))
    trace = _trace(str(_TRACE_DIR), trace_labels[selected_label])

    feature_name = st.sidebar.selectbox("Feature", list(sorted(FEATURE_EXTRACTORS)))
    cost = st.sidebar.selectbox("PELT cost", ["l2", "normal"])
    penalty = st.sidebar.number_input("Penalty", min_value=0.0, value=2.0, step=0.5)
    min_segment_length = st.sidebar.number_input(
        "Min segment length",
        min_value=1,
        value=2,
        step=1,
    )
    smoothing_window = st.sidebar.slider("Smoothing window", min_value=1, max_value=7, value=1)

    extractor = get_feature_extractor(feature_name)
    if not extractor.is_available(trace):
        st.error(f"The selected feature '{feature_name}' is not available for this trace.")
        return

    feature_series = extractor.extract(trace)
    detector = PeltDetector()
    breakpoints = detector.detect(
        feature_series.values,
        CPDParameters(
            cost=cost,
            penalty=float(penalty),
            min_segment_length=int(min_segment_length),
            smoothing_window=int(smoothing_window),
        ),
    )
    segment_summaries = build_segment_summaries(trace, feature_series, breakpoints)

    metadata_column, summary_column = st.columns([1.4, 1])
    with metadata_column:
        st.subheader("Trace")
        st.write(f"**Trace ID:** {trace.trace_id}")
        st.write(f"**Backend:** {trace.backend}")
        st.write(f"**Model:** {trace.model_name}")
        st.write(f"**Prompt:** {trace.prompt}")
        if trace.tags:
            st.write(f"**Tags:** {', '.join(trace.tags)}")
    with summary_column:
        st.subheader("Summary")
        st.metric("Tokens", len(trace.tokens))
        st.metric("Breakpoints", len(breakpoints))
        st.metric("Segments", len(segment_summaries))
        st.json(trace.decoding_metadata)

    st.subheader("Segmented text")
    st.markdown(render_segmented_text_html(segment_summaries), unsafe_allow_html=True)

    st.subheader("Feature vs token index")
    st.altair_chart(build_feature_chart(feature_series, breakpoints), use_container_width=True)

    st.subheader("Segments")
    st.dataframe(build_segment_table(segment_summaries), use_container_width=True)


if __name__ == "__main__":
    main()

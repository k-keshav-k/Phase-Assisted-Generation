from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_cpd.catalog import (  # noqa: E402
    filter_catalog_entries,
    list_catalog_entries,
    load_trace_by_id,
    trace_dir_signature,
)
from phase_cpd.cpd import CPDParameters, get_detector  # noqa: E402
from phase_cpd.features import FEATURE_EXTRACTORS, get_feature_extractor  # noqa: E402
from phase_cpd.segments import build_segment_summaries  # noqa: E402
from phase_cpd.visualize import (  # noqa: E402
    build_feature_chart,
    build_segment_table,
    build_token_feature_table,
    format_breakpoints,
    render_token_boundary_view_html,
)


@st.cache_data
def _catalog(_signature) -> list:
    return list_catalog_entries()


@st.cache_data
def _trace(trace_id: str, _signature):
    return load_trace_by_id(trace_id)


def main() -> None:
    st.set_page_config(page_title="PAG Phase CPD", layout="wide")
    st.title("PAG Phase Segmentation Explorer")
    st.caption("Offline change-point analysis over curated diffusion trace files.")

    try:
        signature = trace_dir_signature()
        entries = _catalog(signature)
    except FileNotFoundError as error:
        st.error(str(error))
        return

    backends = sorted({entry.backend for entry in entries})
    selected_backend = st.sidebar.selectbox(
        "Backend filter",
        ["All", *backends],
        help="Restrict the trace catalog to a specific model backend before selecting a trace.",
    )

    backend_filtered = filter_catalog_entries(
        entries,
        backend=None if selected_backend == "All" else selected_backend,
    )

    model_options = sorted({entry.model_name for entry in backend_filtered})
    selected_model = st.sidebar.selectbox(
        "Model filter",
        ["All", *model_options],
        help="Restrict the catalog to a specific model name within the selected backend.",
    )

    model_filtered = filter_catalog_entries(
        backend_filtered,
        model_name=None if selected_model == "All" else selected_model,
    )

    tag_options = sorted({tag for entry in model_filtered for tag in entry.tags})
    selected_tags = st.sidebar.multiselect(
        "Tag filter",
        tag_options,
        help="Keep only traces that include all selected tags, such as `math` or `code`.",
    )

    run_options = sorted({entry.run_id for entry in model_filtered if entry.run_id})
    selected_run = st.sidebar.selectbox(
        "Run filter",
        ["All", *run_options],
        help="Filter traces to a specific collection run id from the stored decoding metadata.",
    )

    run_filtered = filter_catalog_entries(
        model_filtered,
        required_tags=set(selected_tags),
        run_id=None if selected_run == "All" else selected_run,
    )

    profile_options = sorted({entry.profile_label for entry in run_filtered})
    selected_profile = st.sidebar.selectbox(
        "Trace profile",
        ["All", *profile_options],
        help=(
            "Preferred phase-predictor training policy is entropy with generation "
            "temperature=0.0 and alg_temp=0.1. entropy_det and origin_random are "
            "comparison/ablation policies."
        ),
    )

    profile_filtered = filter_catalog_entries(
        run_filtered,
        profile_label=None if selected_profile == "All" else selected_profile,
    )

    seed_options = sorted({entry.seed for entry in profile_filtered if entry.seed is not None})
    selected_seed_label = st.sidebar.selectbox(
        "Seed filter",
        ["All", *[str(seed) for seed in seed_options]],
        help="Restrict traces to a specific collection seed when multiple seeds are present.",
    )

    filtered_entries = filter_catalog_entries(
        profile_filtered,
        seed=None if selected_seed_label == "All" else int(selected_seed_label),
    )

    if not filtered_entries:
        st.warning("No traces match the current filters.")
        return

    trace_labels = {entry.label: entry.trace_id for entry in filtered_entries}
    selected_label = st.sidebar.selectbox(
        "Trace",
        list(trace_labels),
        help="Choose the exact stored trace to analyze after applying the catalog filters.",
    )
    trace = _trace(trace_labels[selected_label], signature)

    available_features = [
        name
        for name, extractor in FEATURE_EXTRACTORS.items()
        if extractor.is_available(trace)
    ]
    if not available_features:
        st.error(
            "No supported scalar features are available for this trace. "
            "Use traces converted from raw Dream step dumps with per-step token identities."
        )
        return
    preferred_feature_order = [
        "stabilizing_entropy",
        "stabilizing_margin",
        "stabilizing_prob",
        "stabilizing_refinement_step",
    ]
    ordered_features = [
        *[name for name in preferred_feature_order if name in available_features],
        *sorted(name for name in available_features if name not in preferred_feature_order),
    ]
    default_feature_name = next(
        name for name in preferred_feature_order if name in available_features
    )
    feature_name = st.sidebar.selectbox(
        "Feature",
        ordered_features,
        index=ordered_features.index(default_feature_name),
        help=(
            "Select which token-level scalar signal to segment. Entropy is usually the best "
            "starting point when probabilities saturate; stabilization step shows when each "
            "token stops changing."
        ),
    )
    detector_name = st.sidebar.selectbox(
        "Detector",
        ["pelt", "kernel_cpd"],
        help=(
            "PELT is the default exact search method. Kernel CPD can be better when "
            "shifts are nonlinear."
        ),
    )
    kernel = "rbf"
    if detector_name == "pelt":
        cost = st.sidebar.selectbox(
            "PELT cost",
            ["l2", "normal"],
            help=(
                "Choose the PELT segment cost model. `l2` is the default; "
                "`normal` can react differently to variance shifts."
            ),
        )
    else:
        cost = "l2"
        kernel = st.sidebar.selectbox(
            "Kernel",
            ["rbf", "linear", "cosine"],
            help=(
                "Kernel CPD compares segments in transformed feature space. "
                "`rbf` is the safest default."
            ),
        )
    penalty = st.sidebar.number_input(
        "Penalty",
        min_value=0.0,
        value=0.1,
        step=0.05,
        format="%.3f",
        help=(
            "Higher penalty yields fewer boundaries; lower penalty makes the "
            "detector more willing to split."
        ),
    )
    min_segment_length = st.sidebar.number_input(
        "Min segment length",
        min_value=1,
        value=2,
        step=1,
        help="Reject boundaries that would create segments shorter than this many tokens.",
    )
    smoothing_window = st.sidebar.slider(
        "Smoothing window",
        min_value=1,
        max_value=7,
        value=1,
        help=(
            "Applies a centered moving-average to the selected feature before change-point "
            "detection. `1` means no smoothing."
        ),
    )

    feature_series_by_name = {
        name: get_feature_extractor(name).extract(trace) for name in ordered_features
    }
    feature_series = feature_series_by_name[feature_name]
    detector = get_detector(detector_name, kernel=kernel)
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
    feature_std = float(np.std(feature_series.values))
    stabilization_steps = feature_series_by_name.get("stabilizing_refinement_step")

    metadata_column, summary_column = st.columns([1.4, 1])
    with metadata_column:
        st.subheader("Trace")
        st.write(f"**Trace ID:** {trace.trace_id}")
        st.write(f"**Backend:** {trace.backend}")
        st.write(f"**Model:** {trace.model_name}")
        st.write(f"**Prompt:** {trace.prompt}")
        if trace.tags:
            st.write(f"**Tags:** {', '.join(trace.tags)}")
        profile_label = selected_profile
        if profile_label == "All":
            profile_label = next(
                entry.profile_label
                for entry in filtered_entries
                if entry.trace_id == trace.trace_id
            )
        st.write(f"**Trace profile:** {profile_label}")
        temperature = trace.decoding_metadata.get("temperature")
        alg_temp = trace.decoding_metadata.get("alg_temp")
        if temperature is not None:
            st.write(f"**Generation temperature:** {temperature}")
        if alg_temp is not None:
            st.write(f"**Refinement alg_temp:** {alg_temp}")
        seed = trace.decoding_metadata.get("seed")
        if seed is not None:
            st.write(f"**Seed:** {seed}")
        trace_profile = trace.decoding_metadata.get("trace_profile")
        if trace_profile == "entropy_det":
            st.caption(
                "Ablation profile: deterministic entropy ordering can make stabilization "
                "steps overly monotone with token index."
            )
        elif trace_profile == "origin_random":
            st.caption(
                "Ablation profile: random refinement order injects policy noise and can "
                "hurt task correctness."
            )
        elif (
            trace.decoding_metadata.get("alg") == "entropy"
            and temperature is not None
            and alg_temp is not None
            and float(temperature) == 0.0
            and float(alg_temp) == 0.1
        ):
            st.caption(
                "Preferred training profile: deterministic token generation with small "
                "refinement-order randomness."
            )
    with summary_column:
        st.subheader("Summary")
        st.metric("Tokens", len(trace.tokens))
        st.metric("Breakpoints", len(breakpoints))
        st.metric("Segments", len(segment_summaries))
        st.metric("Feature Std", f"{feature_std:.4f}")
        st.metric("Feature", feature_series.feature_name)
        st.metric("Detector", detector_name)
        if stabilization_steps is not None:
            st.metric("Mean Stable Step", f"{float(np.mean(stabilization_steps.values)):.1f}")
            st.metric("Max Stable Step", f"{float(np.max(stabilization_steps.values)):.0f}")
        st.json(trace.decoding_metadata)

    if not breakpoints:
        st.warning(
            "No internal boundaries were detected. On Dream traces this often means the "
            f"{feature_series.feature_name} signal is too smooth at the current penalty. "
            "Lowering the penalty or smoothing window can help."
        )

    st.subheader("Boundary overlay")
    st.caption(f"Detected boundary indices: {format_breakpoints(breakpoints)}")
    st.markdown(render_token_boundary_view_html(trace, breakpoints), unsafe_allow_html=True)

    st.subheader("Feature vs token index")
    st.altair_chart(build_feature_chart(feature_series, breakpoints), use_container_width=True)

    st.subheader("Per-token stabilization features")
    st.caption(
        "`stabilizing_refinement_step` is the earliest recorded step_index after which a "
        "token's identity no longer changes."
    )
    st.dataframe(
        build_token_feature_table(trace, feature_series_by_name),
        use_container_width=True,
    )

    st.subheader("Segments")
    st.dataframe(build_segment_table(segment_summaries), use_container_width=True)


if __name__ == "__main__":
    main()

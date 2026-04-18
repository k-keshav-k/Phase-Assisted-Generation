from __future__ import annotations

from phase_cpd.catalog import default_trace_dir, load_trace_by_id
from phase_cpd.features import Top1ProbExtractor


def test_top1_prob_uses_final_refinement_step() -> None:
    trace = load_trace_by_id("mock-adaptive-001", default_trace_dir())
    feature_series = Top1ProbExtractor().extract(trace)

    assert feature_series.feature_name == "top1_prob"
    assert feature_series.values[:4] == [0.91, 0.89, 0.87, 0.54]
    assert len(feature_series.values) == len(trace.tokens)


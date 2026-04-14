from __future__ import annotations

from pag.baselines import run_baseline
from pag.phases import run_phase_analysis
from pag.scheduler import run_adaptive_decoding


def test_run_adaptive_decoding_accepts_baseline_and_phase_outputs(
    run_config,
    sample_records,
) -> None:
    baseline_artifacts = run_baseline(run_config, sample_records)
    phase_artifacts = run_phase_analysis(run_config, baseline_artifacts)
    adaptive_artifacts = run_adaptive_decoding(run_config, baseline_artifacts, phase_artifacts)

    assert len(adaptive_artifacts.schedule_plans) == len(sample_records)
    assert len(adaptive_artifacts.adaptive_results) == len(sample_records)
    assert adaptive_artifacts.comparison_metrics["throughput_proxy"] > 0
    assert all(plan.decisions for plan in adaptive_artifacts.schedule_plans)
    assert adaptive_artifacts.summary.stage == "scheduler"

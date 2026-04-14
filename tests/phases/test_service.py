from __future__ import annotations

from pag.baselines import run_baseline
from pag.phases import run_phase_analysis


def test_run_phase_analysis_accepts_baseline_outputs(run_config, sample_records) -> None:
    baseline_artifacts = run_baseline(run_config, sample_records)
    phase_artifacts = run_phase_analysis(run_config, baseline_artifacts)

    assert len(phase_artifacts.predictions) == len(sample_records)
    assert len(phase_artifacts.predictor_dataset) >= len(sample_records)
    assert phase_artifacts.predictor_metadata["predictor_name"] == run_config.predictor.name
    assert {prediction.sample_id for prediction in phase_artifacts.predictions} == {
        sample.sample_id for sample in sample_records
    }
    assert phase_artifacts.summary.stage == "phases"


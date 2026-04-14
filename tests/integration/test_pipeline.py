from __future__ import annotations

from pathlib import Path

from pag.config import (
    adaptive_paths,
    baseline_paths,
    evaluation_paths,
    load_run_config,
    load_samples,
    phase_paths,
)
from pag.orchestration import run_pipeline
from pag.utils.io import (
    read_adaptive_artifacts,
    read_baseline_artifacts,
    read_evaluation_artifacts,
    read_phase_artifacts,
)


def test_mock_pipeline_wiring_and_persistence(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    run_config = load_run_config(repo_root / "configs" / "runs" / "adaptive_mock.yaml")
    run_config.output_root = str(tmp_path / "artifacts")
    samples = load_samples(run_config.dataset_path)

    pipeline_artifacts = run_pipeline(run_config, samples, persist=True)

    restored_baseline = read_baseline_artifacts(run_config)
    restored_phases = read_phase_artifacts(run_config)
    restored_adaptive = read_adaptive_artifacts(run_config)
    restored_evaluation = read_evaluation_artifacts(run_config)

    assert pipeline_artifacts.evaluation is not None
    assert len(restored_baseline.completions) == len(samples)
    assert len(restored_phases.predictions) == len(samples)
    assert len(restored_adaptive.adaptive_results) == len(samples)
    assert len(restored_evaluation.records) == len(samples)

    expected_paths = [
        *baseline_paths(run_config).values(),
        *phase_paths(run_config).values(),
        *adaptive_paths(run_config).values(),
        *evaluation_paths(run_config).values(),
    ]
    for path in expected_paths:
        assert Path(path).exists()

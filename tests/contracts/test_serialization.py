from __future__ import annotations

from pathlib import Path

from pag.baselines import run_baseline
from pag.contracts.artifacts import BaselineRunArtifacts
from pag.contracts.serialization import from_dict, to_dict
from pag.utils.io import read_baseline_artifacts, write_baseline_artifacts


def test_artifact_dataclasses_round_trip(
    run_config,
    sample_records,
) -> None:
    baseline_artifacts = run_baseline(run_config, sample_records)
    payload = to_dict(baseline_artifacts)
    restored = from_dict(BaselineRunArtifacts, payload)

    assert restored.run_config.run_id == run_config.run_id
    assert len(restored.completions) == len(sample_records)
    assert restored.traces[0].final_tokens == baseline_artifacts.traces[0].final_tokens


def test_baseline_artifacts_persist_and_load(
    run_config,
    sample_records,
) -> None:
    baseline_artifacts = run_baseline(run_config, sample_records)
    paths = write_baseline_artifacts(baseline_artifacts)
    restored = read_baseline_artifacts(run_config)

    for path in paths.values():
        assert Path(path).exists()
    assert restored.summary.artifact_paths["requests"].endswith("requests.jsonl")
    assert len(restored.requests) == len(sample_records)
    assert len(restored.token_signals) == len(baseline_artifacts.token_signals)


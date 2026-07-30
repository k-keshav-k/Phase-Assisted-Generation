from __future__ import annotations

import json
from pathlib import Path

import pytest

from pag.experiments.orchestrator import ControlledStop
from pag.experiments.rc_pag_config import load_rc_pag_config
from pag.experiments.rc_pag_orchestrator import MockRCPAGRuntime, RCPAGOrchestrator


@pytest.fixture
def config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips.yaml"))


def test_mock_all_stages_resume_without_duplicate_runs(tmp_path, config):
    runtime = MockRCPAGRuntime(calibration_repetitions=50)
    runner = RCPAGOrchestrator(
        config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
    )

    runner.run_through("report")
    first_calls = tuple(runtime.calls)
    runner.run_through("report")

    assert tuple(runtime.calls) == first_calls
    assert (tmp_path / "risk_certificate.json").is_file()
    assert (tmp_path / "policy_family.json").is_file()
    assert (tmp_path / "report" / "inputs.json").is_file()
    for stage in runner.STAGES[: runner.STAGES.index("report") + 1]:
        manifest = json.loads((tmp_path / "manifests" / f"{stage}.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["config_hash"] == config.config_hash


def test_confirmation_requires_certificate(tmp_path, config):
    runtime = MockRCPAGRuntime(unsafe=True, calibration_repetitions=50)
    runner = RCPAGOrchestrator(
        config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
    )
    runner.run_through("calibrate")

    with pytest.raises(ControlledStop, match="no certified policy"):
        runner.run_stage("confirm")

    assert not (tmp_path / "manifests" / "confirm.json").is_file()


def test_confirmation_stops_when_certified_policy_loses_compute_futility_gate(
    tmp_path, config
):
    runtime = MockRCPAGRuntime(candidate_nfe_offset=20.0, calibration_repetitions=50)
    runner = RCPAGOrchestrator(
        config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
    )
    runner.run_through("calibrate")

    with pytest.raises(ControlledStop, match="futility gate"):
        runner.run_stage("confirm")


def test_pilot_writes_compute_projection(tmp_path, config):
    runtime = MockRCPAGRuntime()
    runner = RCPAGOrchestrator(
        config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
    )

    runner.run_through("pilot")

    projection = json.loads((tmp_path / "compute_projection.json").read_text())
    assert projection["seconds_per_sample"] > 0
    assert projection["projected_a100_hours"] > 0
    assert projection["projected_storage_bytes"] > 0

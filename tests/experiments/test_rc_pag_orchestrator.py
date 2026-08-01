from __future__ import annotations

import json
from pathlib import Path

import pytest

from pag.experiments.orchestrator import ControlledStop
from pag.experiments.rc_pag_config import load_rc_pag_config
from pag.experiments.rc_pag_orchestrator import (
    MockRCPAGRuntime,
    RCPAGOrchestrator,
    _index_stratified_complement_indices,
    _index_stratified_indices,
)


@pytest.fixture
def config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips.yaml"))


@pytest.fixture
def workshop_config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop.yaml"))


@pytest.fixture
def v2_config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v2.yaml"))


def test_mock_all_stages_resume_without_duplicate_runs(tmp_path, config):
    runtime = MockRCPAGRuntime(calibration_repetitions=60)
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
    assert (tmp_path / "report" / "tables" / "estimator_ablation.tex").is_file()
    estimator_manifest = json.loads((tmp_path / "estimators" / "manifest.json").read_text())
    for model in ("llada", "dream"):
        for variant in ("rc_pag_local", "rc_pag_history"):
            assert set(estimator_manifest["models"][model][variant]["estimators"]) == {
                "hist_gradient_boosting",
                "logistic",
            }
    for stage in runner.STAGES[: runner.STAGES.index("report") + 1]:
        manifest = json.loads((tmp_path / "manifests" / f"{stage}.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["config_hash"] == config.config_hash


def test_confirmation_requires_certificate(tmp_path, config):
    runtime = MockRCPAGRuntime(unsafe=True, calibration_repetitions=60)
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


def test_confirmation_stops_when_certified_policy_loses_compute_futility_gate(tmp_path, config):
    runtime = MockRCPAGRuntime(candidate_nfe_offset=20.0, calibration_repetitions=60)
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
    assert projection["projected_runs_per_model_by_stage"] == {
        "calibrate": 1800,
        "collect": 600,
        "confirm": 8960,
        "screen": 2700,
    }
    assert projection["projected_gpu_runs"] == 28120


def test_workshop_confirmation_is_stratified_and_uses_three_methods(tmp_path, workshop_config):
    runtime = MockRCPAGRuntime(calibration_repetitions=60)
    runner = RCPAGOrchestrator(
        workshop_config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
    )

    gsm8k_refs = runner._confirm_refs("gsm8k_test")
    assert len(gsm8k_refs) == 2
    assert gsm8k_refs[0].index < 131
    assert 131 <= gsm8k_refs[1].index < 263

    runner.run_through("confirm")

    confirm_methods = {method for stage, _, _, method in runtime.calls if stage == "confirm"}
    assert confirm_methods == {"adablock", "best_nonlearned", "rc_pag_history"}
    projection = json.loads((tmp_path / "compute_projection.json").read_text())
    assert projection["projected_runs_per_model_by_stage"]["confirm"] == 3000
    assert projection["projected_gpu_runs"] == 16200


def test_workshop_gsm8k_sample_has_equal_decile_coverage() -> None:
    indices = _index_stratified_indices(
        population=1319,
        count=500,
        strata=10,
        seed=20260729,
        pool="gsm8k_test",
    )

    assert len(indices) == len(set(indices)) == 500
    for stratum in range(10):
        lower = 1319 * stratum // 10
        upper = 1319 * (stratum + 1) // 10
        assert sum(lower <= index < upper for index in indices) == 50


def test_v2_confirmation_is_a_fresh_complement(v2_config) -> None:
    old = set(
        _index_stratified_indices(
            population=1319,
            count=500,
            strata=10,
            seed=v2_config.seed,
            pool="gsm8k_test",
        )
    )
    fresh = set(
        _index_stratified_complement_indices(
            population=1319,
            count=500,
            excluded_count=500,
            strata=10,
            seed=v2_config.seed,
            pool="gsm8k_test",
        )
    )

    assert len(fresh) == 500
    assert old.isdisjoint(fresh)


def test_v2_pipeline_freezes_two_policies_and_calibrates_end_to_end_harm(tmp_path, v2_config):
    runtime = MockRCPAGRuntime(calibration_repetitions=300)
    runner = RCPAGOrchestrator(
        v2_config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
        mock_mode=True,
    )

    runner.run_through("report")

    parity = json.loads((tmp_path / "parity_audit.json").read_text())
    assert parity["passed"]
    certificate = json.loads((tmp_path / "risk_certificate.json").read_text())
    assert certificate["loss"] == "adablock_correct_candidate_wrong"
    assert len(certificate["candidates"]) == 2
    assert set(certificate["selected_by_model"]) == {"llada", "dream"}
    assert all(row["certified"] for row in certificate["candidates"])
    frozen = json.loads((tmp_path / "frozen_confirmatory_policy.json").read_text())
    assert frozen["primary_rc_pag_method"] == "rc_pag_selected"
    assert set(frozen["best_nonlearned"]) == {"llada", "dream"}
    methods = {method for stage, _, _, method in runtime.calls if stage == "confirm"}
    assert methods == {"adablock", "best_nonlearned", "rc_pag_selected"}


def test_v2_reuses_only_validated_llada_local_estimator(tmp_path, config, v2_config):
    source = tmp_path / "v1"
    old_runtime = MockRCPAGRuntime()
    old_runner = RCPAGOrchestrator(
        config,
        source,
        runtime_factory=lambda model: old_runtime,
        development_limit=2,
    )
    old_runner.run_through("fit")

    destination = tmp_path / "v2"
    new_runtime = MockRCPAGRuntime()
    new_runner = RCPAGOrchestrator(
        v2_config,
        destination,
        runtime_factory=lambda model: new_runtime,
        development_limit=2,
        reuse_development_from=source,
    )
    new_runner.run_through("fit")

    reuse = json.loads((destination / "reuse" / "manifest.json").read_text())
    assert reuse["reused_models"] == ["llada"]
    assert reuse["excluded_models"] == ["dream"]
    collect_calls = {model for stage, model, _, _ in new_runtime.calls if stage == "collect"}
    assert collect_calls == {"dream"}
    manifest = json.loads((destination / "estimators" / "manifest.json").read_text())
    assert manifest["models"]["llada"]["rc_pag_local"]["reused"]
    assert set(manifest["models"]["dream"]) == {"rc_pag_local"}


def test_v2_harmful_policy_stops_before_confirmation(tmp_path, v2_config):
    runtime = MockRCPAGRuntime(unsafe=True, calibration_repetitions=300)
    runner = RCPAGOrchestrator(
        v2_config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
        mock_mode=True,
    )
    runner.run_through("calibrate")

    with pytest.raises(ControlledStop, match="end-to-end harm certificate"):
        runner.run_stage("confirm")

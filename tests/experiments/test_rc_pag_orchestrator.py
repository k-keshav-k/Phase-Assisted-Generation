from __future__ import annotations

import json
from pathlib import Path

import pytest

from pag.experiments.orchestrator import ControlledStop
from pag.experiments.rc_pag_config import load_rc_pag_config
from pag.experiments.rc_pag_orchestrator import (
    MockRCPAGRuntime,
    RCPAGOrchestrator,
    _counterfactual_examples_from_pair,
    _index_stratified_complement_indices,
    _index_stratified_indices,
    _mock_observation,
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


@pytest.fixture
def v3_config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v3.yaml"))


@pytest.fixture
def v4_config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v4.yaml"))


@pytest.fixture
def v5_config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v5.yaml"))


@pytest.fixture
def v6_config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v6.yaml"))


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


def test_v3_fits_benefit_models_and_passes_readiness_gate(tmp_path, v3_config):
    runtime = MockRCPAGRuntime(calibration_repetitions=300)
    runner = RCPAGOrchestrator(
        v3_config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
        mock_mode=True,
    )

    runner.run_through("report")

    estimators = json.loads((tmp_path / "estimators" / "manifest.json").read_text())
    assert set(estimators["benefit_models"]) == {"llada", "dream"}
    assert (tmp_path / "report" / "tables" / "screening_ablation.tex").is_file()
    assert (tmp_path / "report" / "tables" / "benefit_ablation.tex").is_file()
    readiness = json.loads((tmp_path / "readiness_audit.json").read_text())
    assert readiness["passed"]
    assert all(row["nfe_reduction"] >= 0.05 for row in readiness["models"].values())
    projection = json.loads((tmp_path / "compute_projection.json").read_text())
    assert projection["projected_plain_runs"] == 2628
    assert projection["projected_instrumented_runs"] == 9156
    screen_methods = {method for stage, _, _, method in runtime.calls if stage == "screen"}
    assert {"stability_weighted_style", "token_convergence_style"} <= screen_methods


def test_v3_stops_before_calibration_when_frontier_is_too_weak(tmp_path, v3_config):
    runtime = MockRCPAGRuntime(candidate_nfe_offset=20.0)
    runner = RCPAGOrchestrator(
        v3_config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
        mock_mode=True,
    )

    with pytest.raises(ControlledStop, match="workshop-readiness gate"):
        runner.run_through("screen")

    readiness = json.loads((tmp_path / "readiness_audit.json").read_text())
    assert not readiness["passed"]
    assert not (tmp_path / "manifests" / "calibrate.json").exists()


def test_v3_reuses_llada_risk_and_trace_artifacts_for_benefit(tmp_path, config, v3_config):
    source = tmp_path / "v1"
    old_runner = RCPAGOrchestrator(
        config,
        source,
        runtime_factory=lambda model: MockRCPAGRuntime(),
        development_limit=2,
    )
    old_runner.run_through("fit")

    destination = tmp_path / "v3"
    runtime = MockRCPAGRuntime()
    runner = RCPAGOrchestrator(
        v3_config,
        destination,
        runtime_factory=lambda model: runtime,
        development_limit=2,
        reuse_development_from=source,
    )
    runner.run_through("fit")

    estimators = json.loads((destination / "estimators" / "manifest.json").read_text())
    reuse = json.loads((destination / "reuse" / "manifest.json").read_text())
    assert estimators["models"]["llada"]["rc_pag_local"]["reused"]
    assert len(reuse["benefit_trace_sha256"]) == 64
    assert estimators["benefit_models"]["llada"]["source"] == "validated_v1_full_budget_traces"
    projection = json.loads((destination / "compute_projection.json").read_text())
    assert projection["projected_instrumented_runs"] == 8556
    collect_models = {model for stage, model, _, _ in runtime.calls if stage == "collect"}
    assert collect_models == {"dream"}


def test_v4_calibration_jointly_certifies_harm_and_compute(tmp_path, v4_config):
    runtime = MockRCPAGRuntime(calibration_repetitions=150)
    runner = RCPAGOrchestrator(
        v4_config,
        tmp_path,
        runtime_factory=lambda model: runtime,
        development_limit=2,
        mock_mode=True,
    )

    runner.run_through("calibrate")

    certificate = json.loads((tmp_path / "risk_certificate.json").read_text())
    assert certificate["certificate_mode"] == "joint_harm_and_compute"
    assert certificate["minimum_nfe_reduction"] == 0.05
    assert certificate["hypotheses"] == 4
    assert not certificate["fallback"]
    assert all(row["harm_certified"] for row in certificate["candidates"])
    assert all(row["compute_certified"] for row in certificate["candidates"])
    assert all(row["lower_nfe_reduction_bound"] > 0.05 for row in certificate["candidates"])
    manifest = json.loads((tmp_path / "estimators" / "manifest.json").read_text())
    for model in ("llada", "dream"):
        assert set(manifest["models"][model]["rc_pag_local"]["estimators"]) == {
            "hist_gradient_boosting"
        }
    screen_methods = {method for stage, _, _, method in runtime.calls if stage == "screen"}
    assert len({name for name in screen_methods if name.startswith("local_q")}) == 3


class ExactTraceMockRuntime(MockRCPAGRuntime):
    def run(self, **kwargs):
        payload = super().run(**kwargs)
        if kwargs["stage"] != "collect":
            return payload
        sample = kwargs["sample"]
        observation = _mock_observation(sample.index, step=2)
        proposed = list(observation.token_ids)
        payload["schedule_history"] = [
            {
                "applied_block_size": observation.block_size,
                "actual_nfe_used": 3,
                "mean_top1_confidence": 0.8,
                "min_top1_confidence": 0.6,
                "digit_fraction": 0.25,
                "delimiter_fraction": 0.25,
                "final_tokens": proposed,
                "risk_steps": [
                    {
                        "observation": {
                            "step_index": observation.step_index,
                            "block_size": observation.block_size,
                            "masked": list(observation.masked),
                            "top1_probs": list(observation.top1_probs),
                            "top2_probs": list(observation.top2_probs),
                            "entropies": list(observation.entropies),
                            "token_ids": proposed,
                            "temporal_js": [0.01] * observation.block_size,
                            "digit_ids": sorted(observation.digit_ids),
                            "delimiter_ids": sorted(observation.delimiter_ids),
                        },
                        "proposed_tokens": proposed,
                    }
                ],
            }
        ]
        return payload


def test_v4_reuses_compatible_raw_traces_but_refits_estimators(tmp_path, v3_config, v4_config):
    source = tmp_path / "v3"
    source_runtime = ExactTraceMockRuntime()
    source_runner = RCPAGOrchestrator(
        v3_config,
        source,
        runtime_factory=lambda model: source_runtime,
        development_limit=2,
        mock_mode=True,
    )
    source_runner.run_through("fit")

    destination = tmp_path / "v4"
    destination_runtime = MockRCPAGRuntime()
    destination_runner = RCPAGOrchestrator(
        v4_config,
        destination,
        runtime_factory=lambda model: destination_runtime,
        development_limit=2,
        mock_mode=True,
        reuse_development_from=source,
    )
    destination_runner.run_through("fit")

    reuse = json.loads((destination / "reuse" / "manifest.json").read_text())
    estimators = json.loads((destination / "estimators" / "manifest.json").read_text())
    assert reuse["reuse_scope"] == "raw_exact_loop_traces_only"
    assert set(reuse["reused_models"]) == {"llada", "dream"}
    assert not any(stage == "collect" for stage, _, _, _ in destination_runtime.calls)
    assert all(
        estimators["models"][model]["rc_pag_local"]["trace_reused"] for model in ("llada", "dream")
    )


def _stopped_rollout_row(*, correct: bool, total_nfe: float) -> dict:
    first = _mock_observation(3, step=2)
    second = _mock_observation(3, step=3)

    def serialized(observation, *, should_stop: bool) -> dict:
        return {
            "should_stop": should_stop,
            "observation": {
                "step_index": observation.step_index,
                "block_size": observation.block_size,
                "masked": list(observation.masked),
                "top1_probs": list(observation.top1_probs),
                "top2_probs": list(observation.top2_probs),
                "entropies": list(observation.entropies),
                "token_ids": list(observation.token_ids),
                "temporal_js": list(observation.temporal_js),
                "digit_ids": sorted(observation.digit_ids),
                "delimiter_ids": sorted(observation.delimiter_ids),
            },
        }

    return {
        "sample_id": "gsm8k_train-00003",
        "is_correct": correct,
        "total_nfe": total_nfe,
        "schedule_history": [
            {
                "applied_block_size": 4,
                "actual_nfe_used": 3,
                "mean_top1_confidence": 0.8,
                "min_top1_confidence": 0.6,
                "digit_fraction": 0.25,
                "delimiter_fraction": 0.25,
                "final_tokens": list(second.token_ids),
                "risk_steps": [
                    serialized(first, should_stop=False),
                    serialized(second, should_stop=True),
                ],
            }
        ],
    }


def test_counterfactual_pair_uses_prompt_harm_and_normalized_saving() -> None:
    baseline = {"sample_id": "gsm8k_train-00003", "is_correct": True, "total_nfe": 100.0}
    seed = _stopped_rollout_row(correct=False, total_nfe=75.0)

    harm, gain = _counterfactual_examples_from_pair(
        baseline,
        seed,
        history_window=4,
    )

    assert len(harm) == len(gain) == 1
    assert harm[0].unsafe
    assert harm[0].prompt_id == "gsm8k_train-00003"
    assert gain[0].nfe_reduction == pytest.approx(0.25)

    _, negative_gain = _counterfactual_examples_from_pair(
        baseline,
        _stopped_rollout_row(correct=True, total_nfe=101.0),
        history_window=4,
    )
    assert negative_gain[0].nfe_reduction == 0.0


def test_counterfactual_pair_skips_prompt_without_executed_stop() -> None:
    baseline = {"sample_id": "gsm8k_train-00003", "is_correct": True, "total_nfe": 100.0}
    seed = _stopped_rollout_row(correct=True, total_nfe=100.0)
    for block in seed["schedule_history"]:
        for step in block["risk_steps"]:
            step["should_stop"] = False

    harm, gain = _counterfactual_examples_from_pair(
        baseline,
        seed,
        history_window=4,
    )

    assert harm == ()
    assert gain == ()


def test_v5_adds_rollout_refit_and_runs_mock_funnel(tmp_path, v4_config, v5_config) -> None:
    old = RCPAGOrchestrator(
        v4_config,
        tmp_path / "v4",
        runtime_factory=lambda model: MockRCPAGRuntime(),
        development_limit=2,
        mock_mode=True,
    )
    new_runtime = MockRCPAGRuntime(calibration_repetitions=500)
    new = RCPAGOrchestrator(
        v5_config,
        tmp_path / "v5",
        runtime_factory=lambda model: new_runtime,
        development_limit=2,
        mock_mode=True,
    )

    assert "rollout" not in old.active_stages
    assert "refit" not in old.active_stages
    assert new.active_stages[:7] == (
        "preflight",
        "pilot",
        "collect",
        "fit",
        "rollout",
        "refit",
        "screen",
    )

    new.run_through("report")

    advantage = json.loads((tmp_path / "v5" / "estimators" / "advantage_manifest.json").read_text())
    assert set(advantage["models"]) == {"llada", "dream"}
    assert {method for stage, _, _, method in new_runtime.calls if stage == "rollout"} == {
        "adablock",
        "seed_local_q500_p2",
    }
    assert json.loads((tmp_path / "v5" / "readiness_audit.json").read_text())["passed"]


def test_v5_reuses_exact_v4_traces_and_paired_q500_rollouts(tmp_path, v4_config, v5_config) -> None:
    source = tmp_path / "v4"
    source_runner = RCPAGOrchestrator(
        v4_config,
        source,
        runtime_factory=lambda model: ExactTraceMockRuntime(),
        development_limit=2,
        mock_mode=True,
    )
    source_runner.run_through("screen")

    destination = tmp_path / "v5"
    runtime = MockRCPAGRuntime()
    runner = RCPAGOrchestrator(
        v5_config,
        destination,
        runtime_factory=lambda model: runtime,
        development_limit=2,
        mock_mode=True,
        reuse_development_from=source,
    )
    runner.run_through("refit")

    reuse = json.loads((destination / "reuse" / "manifest.json").read_text())
    assert reuse["reuse_scope"] == "raw_exact_loop_traces_and_paired_v4_q500_rollouts"
    assert set(reuse["rollout"]) == {"llada", "dream"}
    assert not any(stage in {"collect", "rollout"} for stage, _, _, _ in runtime.calls)
    assert (destination / "estimators" / "advantage_manifest.json").is_file()


def test_v6_refits_calibrated_risk_and_benefit_from_reused_v5_traces(
    tmp_path,
    v5_config,
    v6_config,
) -> None:
    source = tmp_path / "v5"
    source_runner = RCPAGOrchestrator(
        v5_config,
        source,
        runtime_factory=lambda model: ExactTraceMockRuntime(),
        development_limit=2,
        mock_mode=True,
    )
    source_runner.run_through("fit")

    destination = tmp_path / "v6"
    runtime = MockRCPAGRuntime()
    runner = RCPAGOrchestrator(
        v6_config,
        destination,
        runtime_factory=lambda model: runtime,
        development_limit=2,
        mock_mode=True,
        reuse_development_from=source,
    )
    runner.run_through("fit")

    reuse = json.loads((destination / "reuse" / "manifest.json").read_text())
    estimators = json.loads((destination / "estimators" / "manifest.json").read_text())
    assert reuse["reuse_scope"] == "raw_native_exact_loop_traces_only"
    assert set(reuse["reused_models"]) == {"llada", "dream"}
    assert not any(stage == "collect" for stage, _, _, _ in runtime.calls)
    assert "rollout" not in runner.active_stages
    assert "refit" not in runner.active_stages
    for model in ("llada", "dream"):
        fitted = estimators["models"][model]["rc_pag_budgeted"]
        validation = fitted["estimators"]["hist_gradient_boosting"]["validation"]
        assert fitted["trace_reused"]
        assert validation["training_prompts"] == 1
        assert validation["calibration_prompts"] == 1
        assert (destination / "estimators" / f"{model}_rc_pag_budgeted_risk.joblib").is_file()
        assert (destination / "estimators" / f"{model}_remaining_nfe.joblib").is_file()

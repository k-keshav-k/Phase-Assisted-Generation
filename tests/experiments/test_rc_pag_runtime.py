from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pag.experiments.rc_pag_config import PolicyCandidateSpec, load_rc_pag_config
from pag.experiments.rc_pag_equivalence import (
    EquivalenceCostPolicy,
    fit_equivalence_artifact,
)
from pag.experiments.rc_pag_features import feature_names
from pag.experiments.rc_pag_runtime import (
    UnifiedRCPAGRuntime,
    _ensure_llada_config_compatibility,
    _import_llada_model_class_without_compile,
    prompt_loss_from_schedules,
    training_examples_from_schedules,
)
from pag.experiments.rc_pag_speculation import RiskAdaptiveSpeculationPolicy


def _schedule():
    observation = {
        "step_index": 1,
        "block_size": 2,
        "masked": [True, True],
        "top1_probs": [0.8, 0.7],
        "top2_probs": [0.1, 0.2],
        "entropies": [0.5, 0.6],
        "token_ids": [5, 6],
        "digit_ids": [5],
        "delimiter_ids": [6],
    }
    return [
        {
            "applied_block_size": 2,
            "actual_nfe_used": 3,
            "final_tokens": [5, 7],
            "risk_steps": [
                {
                    "observation": observation,
                    "proposed_tokens": [5, 6],
                    "shadow_loss": 1,
                }
            ],
            "shadow_losses": [1],
        }
    ]


def test_training_examples_label_proposals_against_full_trajectory_final_tokens():
    examples = training_examples_from_schedules(_schedule(), history_window=4)

    assert len(examples) == 1
    assert examples[0]["unsafe"]
    assert examples[0]["features"]["local.step_index"] == 1.0
    assert examples[0]["features"]["history.length"] == 0.0
    assert examples[0]["remaining_nfe"] == 2.0


def test_prompt_loss_is_any_on_policy_shadow_disagreement():
    assert prompt_loss_from_schedules(_schedule()) == 1
    schedule = _schedule()
    schedule[0]["shadow_losses"] = [0]
    assert prompt_loss_from_schedules(schedule) == 0


def test_llada_config_compatibility_aliases_missing_training_length() -> None:
    config = SimpleNamespace(max_sequence_length=4096)

    result = _ensure_llada_config_compatibility(config)

    assert result is config
    assert config.train_max_sequence_length == 4096


def test_llada_config_compatibility_preserves_checkpoint_value() -> None:
    config = SimpleNamespace(max_sequence_length=4096, train_max_sequence_length=2048)

    _ensure_llada_config_compatibility(config)

    assert config.train_max_sequence_length == 2048


def test_llada_model_import_disables_compile_and_restores_torch(monkeypatch) -> None:
    original_compile = torch.compile
    observed: dict[str, object] = {}
    sentinel = object()

    def fake_import(name: str):
        observed["module"] = name
        observed["compile_name"] = torch.compile.__name__

        @torch.compile()
        def identity(value):
            return value

        observed["identity"] = identity(3)
        return SimpleNamespace(LLaDAModelLM=sentinel)

    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.importlib.import_module",
        fake_import,
    )

    result = _import_llada_model_class_without_compile()

    assert result is sentinel
    assert observed == {
        "module": "model.modeling_llada",
        "compile_name": "_identity_torch_compile",
        "identity": 3,
    }
    assert torch.compile is original_compile


def test_llada_model_import_restores_compile_after_import_error(monkeypatch) -> None:
    original_compile = torch.compile

    def fail_import(name: str):
        assert name == "model.modeling_llada"
        assert torch.compile.__name__ == "_identity_torch_compile"
        raise RuntimeError("import failed")

    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.importlib.import_module",
        fail_import,
    )

    with pytest.raises(RuntimeError, match="import failed"):
        _import_llada_model_class_without_compile()

    assert torch.compile is original_compile


def test_v4_rejects_an_estimator_without_temporal_js_schema(monkeypatch) -> None:
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v4.yaml"))
    runtime.model_name = "llada"
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.RiskEstimator.load",
        lambda path: SimpleNamespace(
            names=("local.top1_mean",),
            include_history=False,
            kind="hist_gradient_boosting",
        ),
    )

    with pytest.raises(ValueError, match="temporal-JS feature schema"):
        runtime._risk_policy(
            runtime.config.candidates[0],
            {"llada_rc_pag_local": "old-estimator.joblib"},
        )


def test_v5_loads_paired_advantage_heads_and_exact_verifier(monkeypatch) -> None:
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v5.yaml"))
    runtime.model_name = "llada"
    names = feature_names(include_history=False)
    harm = SimpleNamespace(
        names=names,
        include_history=False,
        kind="hist_gradient_boosting",
        predict_risk=lambda features: 0.01,
    )
    gain = SimpleNamespace(
        names=names,
        include_history=False,
        predict_remaining_nfe=lambda features: 0.20,
    )
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.RiskEstimator.load",
        lambda path: harm,
    )
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.NormalizedNFEReductionEstimator.load",
        lambda path: gain,
    )

    policy = runtime._risk_policy(
        runtime.config.candidates[0],
        {
            "llada_rc_pag_advantage_harm": "harm.joblib",
            "llada_rc_pag_advantage_gain": "gain.joblib",
        },
    )

    assert policy.scorer is harm
    assert policy.benefit_scorer is gain
    assert policy.require_exact_agreement
    assert policy.min_predicted_nfe_savings == 0.05


def test_v5_rollout_seed_loads_local_head_before_advantage_refit(monkeypatch) -> None:
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v5.yaml"))
    runtime.model_name = "llada"
    names = feature_names(include_history=False)
    local = SimpleNamespace(
        names=names,
        include_history=False,
        kind="hist_gradient_boosting",
        predict_risk=lambda features: 0.01,
    )
    loaded: list[str] = []

    def load_local(path: str):
        loaded.append(path)
        return local

    monkeypatch.setattr("pag.experiments.rc_pag_runtime.RiskEstimator.load", load_local)
    seed = PolicyCandidateSpec(
        name="seed_local_q500_p2",
        variant="rc_pag_local",
        threshold=0.50,
        min_steps=2,
        patience=2,
    )

    policy = runtime._risk_policy(seed, {"llada_rc_pag_local": "local.joblib"})

    assert loaded == ["local.joblib"]
    assert policy.scorer is local
    assert policy.benefit_scorer is None
    assert not policy.require_exact_agreement


def test_v6_loads_calibrated_risk_benefit_and_prompt_ledger(monkeypatch) -> None:
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v6.yaml"))
    runtime.model_name = "llada"
    names = feature_names(include_history=False)
    risk = SimpleNamespace(
        names=names,
        include_history=False,
        kind="hist_gradient_boosting",
        predict_risk=lambda features: 0.01,
    )
    benefit = SimpleNamespace(
        names=names,
        include_history=False,
        predict_remaining_nfe=lambda features: 4.0,
    )
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.CalibratedRiskEstimator.load",
        lambda path: risk,
    )
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.RemainingNFEEstimator.load",
        lambda path: benefit,
    )

    candidate = runtime.config.candidates[1]
    policy = runtime._risk_policy(
        candidate,
        {
            "llada_rc_pag_budgeted_risk": "risk.joblib",
            "llada_remaining_nfe": "benefit.joblib",
        },
    )

    assert policy.scorer is risk
    assert policy.benefit_scorer is benefit
    assert policy.require_exact_agreement
    assert policy.total_risk_budget == pytest.approx(0.05)
    assert policy.max_prompt_stops == 2
    assert policy.min_predicted_nfe_savings == 3.0


def test_v8_routes_a_local_risk_head_to_verified_speculation(monkeypatch) -> None:
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v8.yaml"))
    runtime.model_name = "llada"
    estimator = SimpleNamespace(
        names=feature_names(include_history=False),
        include_history=False,
        kind="hist_gradient_boosting",
        predict_risk=lambda features: 0.01,
    )
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.RiskEstimator.load",
        lambda path: estimator,
    )

    scheduler, stopping, speculation, enforcement, provenance = runtime._method_components(
        "rc_pag_verified",
        runtime.config.candidates[1],
        {"llada_rc_pag_verified": "verified.joblib"},
    )

    assert scheduler.budget == 64
    assert stopping is None
    assert isinstance(speculation, RiskAdaptiveSpeculationPolicy)
    assert speculation.scorer is estimator
    assert speculation.max_depth == 4
    assert enforcement == "soft_gate"
    assert provenance == "rc_pag_verified"


def test_v8_fixed_depth_ablation_needs_no_learned_estimator() -> None:
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v8.yaml"))
    runtime.model_name = "dream"

    _, stopping, speculation, _, provenance = runtime._method_components(
        "verified_fixed_d4",
        None,
        {},
    )

    assert stopping is None
    assert speculation.max_depth == speculation.medium_depth == 4
    assert speculation.scorer.predict_risk({}) == 0.0
    assert provenance == "verified_fixed_d4"


def test_v9_routes_audit_and_fitted_equivalence_policies(tmp_path) -> None:
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(
        Path("configs/experiments/rc_pag_neurips_workshop_v9.yaml")
    )
    runtime.model_name = "llada"
    runtime.run_dir = tmp_path
    fingerprint = {"gpu_name": "A100", "model_revision": "test"}
    runtime.execution_fingerprint = lambda: fingerprint  # type: ignore[method-assign]
    events = [
        {
            "batch_size": 2,
            "depth": 1,
            "activation_key": "known",
            "max_logit_delta": 0.01,
            "max_probability_delta": 0.001,
            "full_acceptance": True,
            "batched_latency_ms": 1.0,
            "canonical_latency_ms": 1.0,
        }
        for _ in range(8)
    ]
    artifact = fit_equivalence_artifact(
        events,
        fingerprint=fingerprint,
        minimum_acceptance_lcb=0.5,
    )
    path = tmp_path / "equivalence" / "llada.json"
    path.parent.mkdir()
    path.write_text(__import__("json").dumps(artifact), encoding="utf-8")

    _, stopping, audit, _, provenance = runtime._method_components(
        "ec_pag_audit_d1", None, {}
    )
    assert stopping is None
    assert isinstance(audit, EquivalenceCostPolicy)
    assert audit.audit_reference
    assert audit.fixed_depth == 1
    assert provenance == "ec_pag_audit_d1"

    _, stopping, production, _, provenance = runtime._method_components(
        "ec_pag_v9", runtime.config.candidates[0], {}
    )
    assert stopping is None
    assert isinstance(production, EquivalenceCostPolicy)
    assert not production.audit_reference
    assert production.artifact is not None
    assert production.artifact.artifact_hash == artifact["artifact_hash"]
    assert provenance == "ec_pag"


def test_v9_rejects_an_equivalence_artifact_from_another_execution(tmp_path) -> None:
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(
        Path("configs/experiments/rc_pag_neurips_workshop_v9.yaml")
    )
    runtime.model_name = "dream"
    runtime.run_dir = tmp_path
    runtime.execution_fingerprint = lambda: {"gpu_name": "A100"}  # type: ignore[method-assign]
    artifact = fit_equivalence_artifact(
        [
            {
                "batch_size": 2,
                "depth": 1,
                "activation_key": "known",
                "max_logit_delta": 0.01,
                "max_probability_delta": 0.001,
                "full_acceptance": True,
                "batched_latency_ms": 1.0,
                "canonical_latency_ms": 1.0,
            }
        ],
        fingerprint={"gpu_name": "H100"},
        minimum_bin_count=1,
        minimum_acceptance_lcb=0.0,
    )
    path = tmp_path / "equivalence" / "dream.json"
    path.parent.mkdir()
    path.write_text(__import__("json").dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="execution fingerprint"):
        runtime._method_components("ec_pag_v9", runtime.config.candidates[0], {})


def _offline_hub_raises(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "OfflineModeIsEnabled: Cannot reach huggingface.co: "
        "offline mode is enabled. To disable it, please unset the "
        "`HF_HUB_OFFLINE` environment variable."
    )


def _preflight_runtime():
    runtime = object.__new__(UnifiedRCPAGRuntime)
    runtime.config = load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v8.yaml"))
    runtime.model_name = "llada"
    return runtime


def test_preflight_tolerates_offline_hub_when_revision_is_cached(monkeypatch) -> None:
    runtime = _preflight_runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=80 * 1024**3),
    )
    monkeypatch.setattr("huggingface_hub.model_info", _offline_hub_raises)
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime._revision_cached_locally",
        lambda repository, revision: True,
    )

    result = runtime.preflight(
        model="llada",
        spec=runtime.config.models["llada"],
        device="cuda",
    )

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["warnings"] and "local cache" in result["warnings"][0]


def test_preflight_fails_when_offline_and_revision_not_cached(monkeypatch) -> None:
    runtime = _preflight_runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=80 * 1024**3),
    )
    monkeypatch.setattr("huggingface_hub.model_info", _offline_hub_raises)
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime._revision_cached_locally",
        lambda repository, revision: False,
    )

    result = runtime.preflight(
        model="llada",
        spec=runtime.config.models["llada"],
        device="cuda",
    )

    assert result["ok"] is False
    assert any("model revision could not be resolved" in error for error in result["errors"])
    assert result["warnings"] == []


def test_preflight_uses_online_hub_when_not_offline(monkeypatch) -> None:
    runtime = _preflight_runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=80 * 1024**3),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    resolved = SimpleNamespace(sha=runtime.config.models["llada"].revision)
    monkeypatch.setattr("huggingface_hub.model_info", lambda repository, revision: resolved)

    result = runtime.preflight(
        model="llada",
        spec=runtime.config.models["llada"],
        device="cuda",
    )

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["warnings"] == []


def test_preflight_skips_hub_when_offline_env_set_and_uses_cache(monkeypatch) -> None:
    runtime = _preflight_runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=80 * 1024**3),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime._revision_cached_locally",
        lambda repository, revision: True,
    )
    # model_info must never be called in explicit offline mode.
    monkeypatch.setattr("huggingface_hub.model_info", _offline_hub_raises)

    result = runtime.preflight(
        model="llada",
        spec=runtime.config.models["llada"],
        device="cuda",
    )

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["warnings"] and "HF_HUB_OFFLINE is set" in result["warnings"][0]

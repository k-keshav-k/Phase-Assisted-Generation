from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pag.experiments.rc_pag_config import load_rc_pag_config, validate_rc_pag_config

CONFIG_PATH = Path("configs/experiments/rc_pag_neurips.yaml")
WORKSHOP_CONFIG_PATH = Path("configs/experiments/rc_pag_neurips_workshop.yaml")
WORKSHOP_V2_CONFIG_PATH = Path("configs/experiments/rc_pag_neurips_workshop_v2.yaml")
WORKSHOP_V3_CONFIG_PATH = Path("configs/experiments/rc_pag_neurips_workshop_v3.yaml")
WORKSHOP_V4_CONFIG_PATH = Path("configs/experiments/rc_pag_neurips_workshop_v4.yaml")
WORKSHOP_V5_CONFIG_PATH = Path("configs/experiments/rc_pag_neurips_workshop_v5.yaml")
WORKSHOP_V6_CONFIG_PATH = Path("configs/experiments/rc_pag_neurips_workshop_v6.yaml")
WORKSHOP_V7_CONFIG_PATH = Path("configs/experiments/rc_pag_neurips_workshop_v7.yaml")
WORKSHOP_V8_CONFIG_PATH = Path("configs/experiments/rc_pag_neurips_workshop_v8.yaml")


def _valid_payload() -> dict:
    return deepcopy(load_rc_pag_config(CONFIG_PATH).raw)


def test_frozen_config_loads_with_declared_family_and_counts():
    config = load_rc_pag_config(CONFIG_PATH)

    assert config.risk.alpha == 0.05
    assert config.risk.delta == 0.05
    assert config.stage_sizes.pilot_per_model == 32
    assert config.stage_sizes.traces_per_model == 600
    assert config.stage_sizes.calibration_per_model == 300
    assert config.statistics.bootstrap_samples == 10_000
    assert config.decoding.temperature == 0.0
    assert config.decoding.max_refinement_steps == 64
    assert len(config.candidates) == 6
    assert {candidate.variant for candidate in config.candidates} == {
        "rc_pag_local",
        "rc_pag_history",
    }
    assert len(config.config_hash) == 64


def test_workshop_config_reduces_only_confirmation() -> None:
    full = load_rc_pag_config(CONFIG_PATH)
    workshop = load_rc_pag_config(WORKSHOP_CONFIG_PATH)

    assert workshop.confirmation_profile == "workshop_48h"
    assert workshop.confirmatory_counts == {
        "gsm8k_test": 500,
        "math500": 300,
        "mbpp_sanitized": 100,
        "humaneval": 100,
    }
    assert workshop.confirmatory_methods == (
        "adablock",
        "best_nonlearned",
        "rc_pag_history",
    )
    assert workshop.confirmatory_sampling.strategy == "index_stratified"
    assert workshop.confirmatory_sampling.strata == 10
    assert workshop.risk == full.risk
    assert workshop.candidates == full.candidates
    assert workshop.stage_sizes == full.stage_sizes


def test_v2_config_uses_fresh_confirmation_and_end_to_end_harm() -> None:
    config = load_rc_pag_config(WORKSHOP_V2_CONFIG_PATH)

    assert config.protocol_version == "v2"
    assert config.risk.alpha == 0.02
    assert config.risk.loss == "adablock_correct_candidate_wrong"
    assert config.confirmatory_counts == {
        "gsm8k_test": 500,
        "math500": 200,
        "mbpp_sanitized": 100,
        "humaneval": 64,
    }
    assert config.confirmatory_sampling.strategy == "index_stratified_complement"
    assert config.confirmatory_sampling.excluded_counts["gsm8k_test"] == 500
    assert config.confirmatory_methods == (
        "adablock",
        "best_nonlearned",
        "rc_pag_selected",
    )
    assert len(config.candidates) == 3
    assert all(candidate.max_remaining_fraction == 0.25 for candidate in config.candidates)
    assert max(candidate.threshold for candidate in config.candidates) == 0.5


def test_v3_config_registers_stability_benefit_grid_and_readiness_gate() -> None:
    config = load_rc_pag_config(WORKSHOP_V3_CONFIG_PATH)

    assert config.protocol_version == "v3"
    assert len(config.candidates) == 9
    assert {candidate.max_remaining_fraction for candidate in config.candidates} == {
        0.25,
        0.5,
        0.75,
    }
    assert all(candidate.min_predicted_nfe_savings == 2.0 for candidate in config.candidates)
    assert all(candidate.max_temporal_js == 0.05 for candidate in config.candidates)
    assert config.readiness.minimum_tuning_nfe_reduction_per_model == 0.05
    assert "stability_weighted_style" in config.development_methods


def test_v4_config_is_single_estimator_three_threshold_joint_protocol() -> None:
    config = load_rc_pag_config(WORKSHOP_V4_CONFIG_PATH)

    assert config.protocol_version == "v4"
    assert config.estimator_kinds == ("hist_gradient_boosting",)
    assert len(config.candidates) == 3
    assert {candidate.threshold for candidate in config.candidates} == {0.05, 0.20, 0.50}
    assert all(candidate.max_remaining_fraction == 1.0 for candidate in config.candidates)
    assert all(candidate.min_predicted_nfe_savings == 0.0 for candidate in config.candidates)
    assert all(candidate.max_temporal_js == 1.0 for candidate in config.candidates)
    assert config.risk.minimum_nfe_reduction == 0.05
    assert config.development_methods == (
        "adablock",
        "stability_weighted_style",
        "token_convergence_style",
        "rc_pag_local",
    )


def test_v5_config_registers_counterfactual_advantage_protocol() -> None:
    config = load_rc_pag_config(WORKSHOP_V5_CONFIG_PATH)

    assert config.protocol_version == "v5"
    assert config.stage_sizes.rollout_per_model == 150
    assert config.stage_sizes.tuning_per_model == 150
    assert config.stage_sizes.calibration_per_model == 500
    assert set(config.splits) == {"pilot", "training", "rollout", "tuning", "calibration"}
    assert len(config.candidates) == 3
    assert {
        (candidate.threshold, candidate.min_predicted_nfe_savings)
        for candidate in config.candidates
    } == {(0.02, 0.05), (0.05, 0.08), (0.10, 0.10)}
    assert all(candidate.variant == "rc_pag_advantage" for candidate in config.candidates)
    assert all(candidate.require_exact_agreement for candidate in config.candidates)
    assert config.readiness.minimum_tuning_nfe_reduction_per_model == 0.08
    assert config.risk.minimum_nfe_reduction == 0.05


def test_v6_config_freezes_ledger_family_and_fresh_splits() -> None:
    config = load_rc_pag_config(WORKSHOP_V6_CONFIG_PATH)

    assert config.protocol_version == "v6"
    assert set(config.splits) == {"pilot", "training", "tuning", "calibration"}
    assert config.splits["tuning"] == {
        "gsm8k_train": (400, 449),
        "math_train": (300, 349),
        "mbpp_train": (200, 249),
    }
    assert config.splits["calibration"] == {
        "gsm8k_train": (854, 1241),
        "math_train": (433, 507),
        "mbpp_train": (295, 331),
    }
    assert [
        (
            candidate.total_risk_budget,
            candidate.max_prompt_stops,
            candidate.min_predicted_nfe_savings,
        )
        for candidate in config.candidates
    ] == [(0.02, 1, 4.0), (0.05, 2, 3.0), (0.10, 3, 2.0)]
    assert all(candidate.variant == "rc_pag_budgeted" for candidate in config.candidates)
    assert all(candidate.require_exact_agreement for candidate in config.candidates)
    assert config.risk.minimum_nfe_reduction is None
    assert config.readiness.minimum_tuning_nfe_reduction_per_model == 0.08
    assert config.claim_gates.minimum_model_nfe_reduction_lower_ci == 0.05


def test_v7_config_registers_risk_threshold_gating_family() -> None:
    config = load_rc_pag_config(WORKSHOP_V7_CONFIG_PATH)

    assert config.protocol_version == "v7"
    assert set(config.splits) == {"pilot", "training", "tuning", "calibration"}
    assert [
        (candidate.threshold, candidate.patience, candidate.min_predicted_nfe_savings)
        for candidate in config.candidates
    ] == [(0.10, 3, 0.0), (0.15, 3, 0.0), (0.20, 3, 0.0)]
    assert all(candidate.variant == "rc_pag_budgeted" for candidate in config.candidates)
    assert all(not candidate.require_exact_agreement for candidate in config.candidates)
    assert all(candidate.total_risk_budget == 1.0 for candidate in config.candidates)
    assert all(candidate.max_prompt_stops == 3 for candidate in config.candidates)
    assert config.risk.minimum_nfe_reduction is None
    assert config.readiness.minimum_tuning_nfe_reduction_per_model == 0.08
    assert config.claim_gates.minimum_model_nfe_reduction_lower_ci == 0.05


def test_v7_config_rejects_exact_agreement_and_unbounded_thresholds() -> None:
    payload = deepcopy(load_rc_pag_config(WORKSHOP_V7_CONFIG_PATH).raw)
    payload["policy"]["candidates"][0]["require_exact_agreement"] = True

    with pytest.raises(ValueError, match="risk-threshold gating"):
        validate_rc_pag_config(payload)


def test_v8_config_registers_risk_adaptive_verified_speculation() -> None:
    config = load_rc_pag_config(WORKSHOP_V8_CONFIG_PATH)

    assert config.protocol_version == "v8"
    assert [
        (
            candidate.max_speculation_depth,
            candidate.medium_speculation_depth,
            candidate.deep_risk_threshold,
            candidate.medium_risk_threshold,
        )
        for candidate in config.candidates
    ] == [(2, 1, 0.05, 0.15), (4, 2, 0.10, 0.30), (6, 3, 0.20, 0.50)]
    assert all(candidate.variant == "rc_pag_verified" for candidate in config.candidates)
    assert config.development_methods == (
        "adablock",
        "verified_fixed_d2",
        "verified_fixed_d4",
        "rc_pag_verified",
    )
    assert config.readiness.minimum_tuning_nfe_reduction_per_model == 0.05
    assert config.claim_gates.minimum_model_nfe_reduction_lower_ci == 0.05

    payload = deepcopy(load_rc_pag_config(WORKSHOP_V7_CONFIG_PATH).raw)
    payload["policy"]["candidates"][0]["threshold"] = 0.7

    with pytest.raises(ValueError, match="risk-threshold gating"):
        validate_rc_pag_config(payload)


def test_config_rejects_split_overlap():
    payload = _valid_payload()
    payload["splits"]["calibration"]["gsm8k_train"] = [250, 349]

    with pytest.raises(ValueError, match="overlap"):
        validate_rc_pag_config(payload)


def test_config_rejects_unfrozen_risk_levels():
    payload = _valid_payload()
    payload["risk"]["alpha"] = 0.10

    with pytest.raises(ValueError, match="alpha must remain 0.05"):
        validate_rc_pag_config(payload)


def test_config_rejects_wrong_candidate_family_size():
    payload = _valid_payload()
    payload["policy"]["candidates"].pop()

    with pytest.raises(ValueError, match="exactly six"):
        validate_rc_pag_config(payload)


def test_config_rejects_confirmatory_count_drift():
    payload = _valid_payload()
    payload["confirmatory"]["gsm8k_test"] = 100

    with pytest.raises(ValueError, match="confirmatory counts"):
        validate_rc_pag_config(payload)

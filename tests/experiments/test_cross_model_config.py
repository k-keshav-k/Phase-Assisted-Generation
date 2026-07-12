from __future__ import annotations

from copy import deepcopy

import pytest

from pag.experiments.cross_model_config import (
    load_cross_model_config,
    validate_cross_model_config,
)


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "seed": 20260712,
        "models": {
            "llada": "GSAI-ML/LLaDA-8B-Instruct",
            "dream": "Dream-org/Dream-v0-Base-7B",
        },
        "datasets": {
            "gsm8k": {
                "path": "openai/gsm8k",
                "config": "main",
                "revision": "main",
                "calibration_indices": [6200, 6299],
                "test_indices": [6300, 6699],
            },
            "math500": {
                "path": "HuggingFaceH4/MATH-500",
                "revision": "main",
                "prior_selection_manifest": "artifacts/prior/selected_samples.json",
                "expected_complement": 200,
            },
        },
        "policy": {
            "quantiles": [0.15, 0.25, 0.35],
            "max_abs_corrections": [1, 2, 3],
            "n_estimators": 200,
            "window_size": 8,
        },
        "methods": {"confirmatory": ["adablock", "size_lookup", "residual_pag"]},
        "claim_gates": {
            "minimum_nfe_reduction": 0.10,
            "minimum_lookup_reduction": 0.03,
            "minimum_accuracy_ci": -0.02,
        },
        "budget": {"usd": 19.0, "gpu_rate": 0.35, "reserve_fraction": 0.05},
        "statistics": {"bootstrap_samples": 10000},
    }


def test_config_rejects_overlap_and_wrong_headline_threshold() -> None:
    payload = _valid_payload()
    payload["datasets"]["gsm8k"]["test_indices"] = [6250, 6350]
    with pytest.raises(ValueError, match="disjoint"):
        validate_cross_model_config(payload)

    payload = _valid_payload()
    payload["claim_gates"]["minimum_nfe_reduction"] = 0.09
    with pytest.raises(ValueError, match="0.10"):
        validate_cross_model_config(payload)


def test_config_requires_exact_confirmatory_methods() -> None:
    payload = deepcopy(_valid_payload())
    payload["methods"]["confirmatory"].remove("size_lookup")
    with pytest.raises(ValueError, match="confirmatory methods"):
        validate_cross_model_config(payload)


def test_repository_cross_model_config_loads() -> None:
    config = load_cross_model_config("configs/experiments/neurips_cross_model.yaml")
    assert config.gsm8k.calibration_indices == (6200, 6299)
    assert config.gsm8k.test_indices == (6300, 6699)
    assert config.models["dream"] == "Dream-org/Dream-v0-Base-7B"
    assert config.budget_usd == 19.0

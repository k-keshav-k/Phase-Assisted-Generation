from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pag.experiments.config import canonical_config_hash, inclusive_range


@dataclass(frozen=True, slots=True)
class FreshGSM8KConfig:
    path: str
    config: str
    revision: str
    calibration_indices: tuple[int, int]
    test_indices: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ComplementMathConfig:
    path: str
    revision: str
    prior_selection_manifest: str
    expected_complement: int


@dataclass(frozen=True, slots=True)
class ResidualPolicySearch:
    quantiles: tuple[float, ...]
    max_abs_corrections: tuple[int, ...]
    n_estimators: int
    window_size: int


@dataclass(frozen=True, slots=True)
class ClaimGateConfig:
    minimum_nfe_reduction: float
    minimum_lookup_reduction: float
    minimum_accuracy_ci: float


@dataclass(frozen=True, slots=True)
class CrossModelConfig:
    schema_version: int
    seed: int
    models: dict[str, str]
    gsm8k: FreshGSM8KConfig
    math500: ComplementMathConfig
    policy: ResidualPolicySearch
    confirmatory_methods: tuple[str, ...]
    claim_gates: ClaimGateConfig
    budget_usd: float
    gpu_rate: float
    reserve_fraction: float
    bootstrap_samples: int
    config_hash: str
    raw: dict[str, Any]


def _range(payload: object, *, name: str) -> tuple[int, int]:
    values = tuple(int(value) for value in payload)  # type: ignore[union-attr]
    if len(values) != 2 or values[0] < 0 or values[0] > values[1]:
        raise ValueError(f"{name} must contain ordered inclusive bounds")
    return values


def validate_cross_model_config(payload: dict[str, Any]) -> None:
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("schema_version must be 1")
    if int(payload.get("seed", 0)) < 1:
        raise ValueError("seed must be positive")
    models = payload.get("models", {})
    if set(models) != {"llada", "dream"} or not all(str(value) for value in models.values()):
        raise ValueError("models must define non-empty llada and dream paths")
    datasets = payload.get("datasets", {})
    gsm = datasets.get("gsm8k", {})
    calibration = _range(gsm.get("calibration_indices", ()), name="calibration_indices")
    test = _range(gsm.get("test_indices", ()), name="test_indices")
    if set(inclusive_range(calibration)) & set(inclusive_range(test)):
        raise ValueError("GSM8K calibration and test ranges must be disjoint")
    math = datasets.get("math500", {})
    if int(math.get("expected_complement", 0)) != 200:
        raise ValueError("MATH-500 expected_complement must be 200")
    if not str(math.get("prior_selection_manifest", "")):
        raise ValueError("MATH-500 prior_selection_manifest is required")
    policy = payload.get("policy", {})
    quantiles = tuple(float(value) for value in policy.get("quantiles", ()))
    corrections = tuple(int(value) for value in policy.get("max_abs_corrections", ()))
    if not quantiles or any(not 0 < value < 0.5 for value in quantiles):
        raise ValueError("policy quantiles must be non-empty and in (0, 0.5)")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("policy quantiles must be unique")
    if not corrections or any(value < 1 for value in corrections):
        raise ValueError("max_abs_corrections must be positive")
    if int(policy.get("n_estimators", 0)) < 1 or int(policy.get("window_size", 0)) < 1:
        raise ValueError("policy estimator and window counts must be positive")
    methods = tuple(payload.get("methods", {}).get("confirmatory", ()))
    if methods != ("adablock", "size_lookup", "residual_pag"):
        raise ValueError("confirmatory methods must be adablock, size_lookup, residual_pag")
    gates = payload.get("claim_gates", {})
    expected_gates = {
        "minimum_nfe_reduction": 0.10,
        "minimum_lookup_reduction": 0.03,
        "minimum_accuracy_ci": -0.02,
    }
    for name, expected in expected_gates.items():
        if float(gates.get(name, float("nan"))) != expected:
            raise ValueError(f"claim gate {name} must remain {expected:.2f}")
    budget = payload.get("budget", {})
    if not 0 < float(budget.get("usd", 0)) <= 19.0:
        raise ValueError("budget usd must be in (0, 19]")
    if float(budget.get("gpu_rate", 0)) <= 0:
        raise ValueError("gpu_rate must be positive")
    if not 0 <= float(budget.get("reserve_fraction", -1)) < 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    if int(payload.get("statistics", {}).get("bootstrap_samples", 0)) < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")


def load_cross_model_config(path: str | Path) -> CrossModelConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("cross-model config must be a mapping")
    validate_cross_model_config(payload)
    datasets = payload["datasets"]
    gsm = datasets["gsm8k"]
    math = datasets["math500"]
    policy = payload["policy"]
    gates = payload["claim_gates"]
    budget = payload["budget"]
    return CrossModelConfig(
        schema_version=int(payload["schema_version"]),
        seed=int(payload["seed"]),
        models={str(key): str(value) for key, value in payload["models"].items()},
        gsm8k=FreshGSM8KConfig(
            path=str(gsm["path"]),
            config=str(gsm["config"]),
            revision=str(gsm["revision"]),
            calibration_indices=_range(gsm["calibration_indices"], name="calibration_indices"),
            test_indices=_range(gsm["test_indices"], name="test_indices"),
        ),
        math500=ComplementMathConfig(
            path=str(math["path"]),
            revision=str(math["revision"]),
            prior_selection_manifest=str(math["prior_selection_manifest"]),
            expected_complement=int(math["expected_complement"]),
        ),
        policy=ResidualPolicySearch(
            quantiles=tuple(float(value) for value in policy["quantiles"]),
            max_abs_corrections=tuple(int(value) for value in policy["max_abs_corrections"]),
            n_estimators=int(policy["n_estimators"]),
            window_size=int(policy["window_size"]),
        ),
        confirmatory_methods=tuple(payload["methods"]["confirmatory"]),
        claim_gates=ClaimGateConfig(
            minimum_nfe_reduction=float(gates["minimum_nfe_reduction"]),
            minimum_lookup_reduction=float(gates["minimum_lookup_reduction"]),
            minimum_accuracy_ci=float(gates["minimum_accuracy_ci"]),
        ),
        budget_usd=float(budget["usd"]),
        gpu_rate=float(budget["gpu_rate"]),
        reserve_fraction=float(budget["reserve_fraction"]),
        bootstrap_samples=int(payload["statistics"]["bootstrap_samples"]),
        config_hash=canonical_config_hash(payload),
        raw=payload,
    )

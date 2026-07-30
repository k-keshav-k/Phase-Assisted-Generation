from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pag.experiments.config import canonical_config_hash, inclusive_range


@dataclass(frozen=True, slots=True)
class ModelSpec:
    repository: str
    revision: str
    dtype: str


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    path: str
    revision: str
    split: str
    configs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyCandidateSpec:
    name: str
    variant: str
    threshold: float
    min_steps: int
    patience: int


@dataclass(frozen=True, slots=True)
class RiskSpec:
    alpha: float
    delta: float
    loss: str


@dataclass(frozen=True, slots=True)
class StageSizes:
    pilot_per_model: int
    traces_per_model: int
    tuning_per_model: int
    calibration_per_model: int


@dataclass(frozen=True, slots=True)
class DecodingSpec:
    temperature: float
    gen_length: int
    max_refinement_steps: int
    max_block_length: int
    transfer_threshold: float
    delimiter_threshold: float
    use_cache: bool
    dual_cache: bool


@dataclass(frozen=True, slots=True)
class StatisticsSpec:
    bootstrap_samples: int
    confidence: float


@dataclass(frozen=True, slots=True)
class ClaimGateSpec:
    require_risk_certificate: bool
    beat_adablock_both_models: bool
    beat_best_nonlearned: bool
    minimum_accuracy_lower_ci: float
    require_history_frontier_ci: bool


@dataclass(frozen=True, slots=True)
class ConfirmatorySamplingSpec:
    strategy: str
    strata: int
    population_sizes: dict[str, int]


@dataclass(frozen=True, slots=True)
class RCPAGConfig:
    schema_version: int
    seed: int
    models: dict[str, ModelSpec]
    datasets: dict[str, DatasetSpec]
    splits: dict[str, dict[str, tuple[int, int]]]
    stage_sizes: StageSizes
    decoding: DecodingSpec
    estimator_kinds: tuple[str, ...]
    history_window: int
    candidates: tuple[PolicyCandidateSpec, ...]
    risk: RiskSpec
    statistics: StatisticsSpec
    confirmation_profile: str
    confirmatory_counts: dict[str, int]
    confirmatory_sampling: ConfirmatorySamplingSpec
    development_methods: tuple[str, ...]
    confirmatory_methods: tuple[str, ...]
    claim_gates: ClaimGateSpec
    config_hash: str
    raw: dict[str, Any]


_EXPECTED_MODELS = {"llada", "dream"}
_EXPECTED_DATASETS = {
    "gsm8k_train",
    "math_train",
    "mbpp_train",
    "gsm8k_test",
    "math500",
    "mbpp_sanitized",
    "humaneval",
}
_FULL_CONFIRMATORY = {
    "gsm8k_test": 1319,
    "math500": 500,
    "mbpp_sanitized": 257,
    "humaneval": 164,
}
_WORKSHOP_CONFIRMATORY = {
    "gsm8k_test": 500,
    "math500": 300,
    "mbpp_sanitized": 100,
    "humaneval": 100,
}
_EXPECTED_STAGES = {
    "pilot": 32,
    "training": 600,
    "tuning": 150,
    "calibration": 300,
}
_REQUIRED_DEVELOPMENT_METHODS = {
    "fixed",
    "adablock",
    "fast_dllm",
    "sched",
    "entropy_sum",
    "confidence_gate",
    "stability_gate",
    "constant_budget",
    "size_lookup",
    "pag",
    "residual_pag",
    "rc_pag_local",
    "rc_pag_history",
    "oracle",
}


def _bounds(value: object, *, name: str) -> tuple[int, int]:
    values = tuple(int(item) for item in value)  # type: ignore[union-attr]
    if len(values) != 2 or values[0] < 0 or values[0] > values[1]:
        raise ValueError(f"{name} must contain ordered inclusive bounds")
    return values


def _require_sha(value: object, *, name: str) -> None:
    revision = str(value)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"{name} must be a pinned 40-character lowercase commit SHA")


def validate_rc_pag_config(payload: dict[str, Any]) -> None:
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("schema_version must be 1")
    if int(payload.get("seed", 0)) < 1:
        raise ValueError("seed must be positive")

    models = payload.get("models", {})
    if set(models) != _EXPECTED_MODELS:
        raise ValueError("models must define exactly llada and dream")
    for name, model in models.items():
        if not str(model.get("repository", "")):
            raise ValueError(f"model {name} repository is required")
        _require_sha(model.get("revision"), name=f"model {name} revision")
        if model.get("dtype") != "bfloat16":
            raise ValueError("all models must use bfloat16")

    datasets = payload.get("datasets", {})
    if set(datasets) != _EXPECTED_DATASETS:
        raise ValueError("datasets do not match the frozen protocol")
    for name, dataset in datasets.items():
        if not str(dataset.get("path", "")) or not str(dataset.get("split", "")):
            raise ValueError(f"dataset {name} requires path and split")
        _require_sha(dataset.get("revision"), name=f"dataset {name} revision")

    splits = payload.get("splits", {})
    if set(splits) != set(_EXPECTED_STAGES):
        raise ValueError("splits must define pilot, training, tuning, and calibration")
    claimed: dict[str, list[tuple[str, set[int]]]] = {}
    for role, pools in splits.items():
        role_count = 0
        if not isinstance(pools, dict) or not pools:
            raise ValueError(f"split {role} must contain dataset ranges")
        for pool, raw_bounds in pools.items():
            if pool not in {"gsm8k_train", "math_train", "mbpp_train"}:
                raise ValueError(f"split {role} uses non-training pool {pool}")
            bounds = _bounds(raw_bounds, name=f"{role}.{pool}")
            indices = set(inclusive_range(bounds))
            role_count += len(indices)
            for previous_role, previous_indices in claimed.setdefault(pool, []):
                if indices & previous_indices:
                    raise ValueError(f"split overlap for {pool}: {previous_role} and {role}")
            claimed[pool].append((role, indices))
        if role_count != _EXPECTED_STAGES[role]:
            raise ValueError(
                f"split {role} must contain {_EXPECTED_STAGES[role]} prompts, got {role_count}"
            )

    sizes = payload.get("stage_sizes", {})
    expected_sizes = {
        "pilot_per_model": 32,
        "traces_per_model": 600,
        "tuning_per_model": 150,
        "calibration_per_model": 300,
    }
    if {name: int(sizes.get(name, 0)) for name in expected_sizes} != expected_sizes:
        raise ValueError("stage sizes do not match the frozen compute funnel")

    decoding = payload.get("decoding", {})
    expected_decoding = {
        "temperature": 0.0,
        "gen_length": 256,
        "max_refinement_steps": 64,
        "max_block_length": 64,
        "transfer_threshold": 0.9,
        "delimiter_threshold": 0.3,
        "use_cache": True,
        "dual_cache": True,
    }
    if decoding != expected_decoding:
        raise ValueError("decoding settings do not match the frozen deterministic protocol")

    policy = payload.get("policy", {})
    if tuple(policy.get("estimator_kinds", ())) != (
        "hist_gradient_boosting",
        "logistic",
    ):
        raise ValueError("estimator kinds must remain histogram boosting and logistic")
    if int(policy.get("history_window", 0)) != 4:
        raise ValueError("history_window must remain 4")
    candidates = tuple(policy.get("candidates", ()))
    if len(candidates) != 6:
        raise ValueError("policy family must contain exactly six candidates")
    names = tuple(str(item.get("name", "")) for item in candidates)
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("policy candidate names must be non-empty and unique")
    variants = [str(item.get("variant", "")) for item in candidates]
    if variants.count("rc_pag_local") != 3 or variants.count("rc_pag_history") != 3:
        raise ValueError("policy family must contain three local and three history candidates")
    for item in candidates:
        if not 0 < float(item.get("threshold", 0)) <= 0.05:
            raise ValueError("candidate thresholds must be in (0, 0.05]")
        if int(item.get("min_steps", 0)) < 1 or int(item.get("patience", 0)) < 1:
            raise ValueError("candidate min_steps and patience must be positive")

    risk = payload.get("risk", {})
    if float(risk.get("alpha", -1)) != 0.05:
        raise ValueError("risk alpha must remain 0.05")
    if float(risk.get("delta", -1)) != 0.05:
        raise ValueError("risk delta must remain 0.05")
    if risk.get("loss") != "any_shadow_token_disagreement":
        raise ValueError("risk loss must remain the strict prompt-level shadow disagreement")

    statistics = payload.get("statistics", {})
    if int(statistics.get("bootstrap_samples", 0)) != 10_000:
        raise ValueError("bootstrap_samples must remain 10000")
    if float(statistics.get("confidence", 0)) != 0.95:
        raise ValueError("confidence must remain 0.95")

    confirmation_profile = str(payload.get("confirmation_profile", "full"))
    confirmatory = {name: int(value) for name, value in payload.get("confirmatory", {}).items()}
    expected_confirmatory = {
        "full": _FULL_CONFIRMATORY,
        "workshop_48h": _WORKSHOP_CONFIRMATORY,
    }
    if confirmation_profile not in expected_confirmatory:
        raise ValueError("unknown confirmation profile")
    if confirmatory != expected_confirmatory[confirmation_profile]:
        raise ValueError(f"confirmatory counts do not match the {confirmation_profile} profile")

    sampling = payload.get("confirmatory_sampling")
    if confirmation_profile == "full":
        if sampling is not None:
            raise ValueError("the full confirmation profile uses every benchmark row")
    else:
        expected_sampling = {
            "strategy": "index_stratified",
            "strata": 10,
            "population_sizes": _FULL_CONFIRMATORY,
        }
        if sampling != expected_sampling:
            raise ValueError("workshop confirmation must use the frozen stratified sampler")

    methods = payload.get("methods", {})
    development = set(methods.get("development", ()))
    if development != _REQUIRED_DEVELOPMENT_METHODS:
        raise ValueError("development method family does not match the frozen protocol")
    expected_methods = {
        "full": ("adablock", "best_nonlearned", "rc_pag_local", "rc_pag_history"),
        "workshop_48h": ("adablock", "best_nonlearned", "rc_pag_history"),
    }
    if tuple(methods.get("confirmatory", ())) != expected_methods[confirmation_profile]:
        raise ValueError("confirmatory methods do not match the frozen protocol")

    gates = payload.get("claim_gates", {})
    expected_gates = {
        "require_risk_certificate": True,
        "beat_adablock_both_models": True,
        "beat_best_nonlearned": True,
        "minimum_accuracy_lower_ci": -0.02,
        "require_history_frontier_ci": confirmation_profile == "full",
    }
    if gates != expected_gates:
        raise ValueError("claim gates do not match the frozen protocol")


def load_rc_pag_config(path: str | Path) -> RCPAGConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("RC-PAG config must be a mapping")
    validate_rc_pag_config(payload)
    models = {
        name: ModelSpec(
            repository=str(item["repository"]),
            revision=str(item["revision"]),
            dtype=str(item["dtype"]),
        )
        for name, item in payload["models"].items()
    }
    datasets = {
        name: DatasetSpec(
            path=str(item["path"]),
            revision=str(item["revision"]),
            split=str(item["split"]),
            configs=tuple(str(value) for value in item.get("configs", ())),
        )
        for name, item in payload["datasets"].items()
    }
    splits = {
        role: {pool: _bounds(bounds, name=f"{role}.{pool}") for pool, bounds in pools.items()}
        for role, pools in payload["splits"].items()
    }
    stage_sizes = payload["stage_sizes"]
    decoding = payload["decoding"]
    policy = payload["policy"]
    risk = payload["risk"]
    statistics = payload["statistics"]
    gates = payload["claim_gates"]
    confirmation_profile = str(payload.get("confirmation_profile", "full"))
    raw_sampling = payload.get("confirmatory_sampling")
    if raw_sampling is None:
        confirmatory_sampling = ConfirmatorySamplingSpec(
            strategy="full",
            strata=1,
            population_sizes=dict(_FULL_CONFIRMATORY),
        )
    else:
        confirmatory_sampling = ConfirmatorySamplingSpec(
            strategy=str(raw_sampling["strategy"]),
            strata=int(raw_sampling["strata"]),
            population_sizes={
                name: int(value) for name, value in raw_sampling["population_sizes"].items()
            },
        )
    return RCPAGConfig(
        schema_version=int(payload["schema_version"]),
        seed=int(payload["seed"]),
        models=models,
        datasets=datasets,
        splits=splits,
        stage_sizes=StageSizes(**{name: int(value) for name, value in stage_sizes.items()}),
        decoding=DecodingSpec(
            temperature=float(decoding["temperature"]),
            gen_length=int(decoding["gen_length"]),
            max_refinement_steps=int(decoding["max_refinement_steps"]),
            max_block_length=int(decoding["max_block_length"]),
            transfer_threshold=float(decoding["transfer_threshold"]),
            delimiter_threshold=float(decoding["delimiter_threshold"]),
            use_cache=bool(decoding["use_cache"]),
            dual_cache=bool(decoding["dual_cache"]),
        ),
        estimator_kinds=tuple(policy["estimator_kinds"]),
        history_window=int(policy["history_window"]),
        candidates=tuple(
            PolicyCandidateSpec(
                name=str(item["name"]),
                variant=str(item["variant"]),
                threshold=float(item["threshold"]),
                min_steps=int(item["min_steps"]),
                patience=int(item["patience"]),
            )
            for item in policy["candidates"]
        ),
        risk=RiskSpec(
            alpha=float(risk["alpha"]),
            delta=float(risk["delta"]),
            loss=str(risk["loss"]),
        ),
        statistics=StatisticsSpec(
            bootstrap_samples=int(statistics["bootstrap_samples"]),
            confidence=float(statistics["confidence"]),
        ),
        confirmation_profile=confirmation_profile,
        confirmatory_counts={name: int(value) for name, value in payload["confirmatory"].items()},
        confirmatory_sampling=confirmatory_sampling,
        development_methods=tuple(payload["methods"]["development"]),
        confirmatory_methods=tuple(payload["methods"]["confirmatory"]),
        claim_gates=ClaimGateSpec(**gates),
        config_hash=canonical_config_hash(payload),
        raw=payload,
    )

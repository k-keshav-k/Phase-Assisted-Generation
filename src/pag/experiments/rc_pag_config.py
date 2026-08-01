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
    max_remaining_fraction: float = 1.0
    min_predicted_nfe_savings: float = 0.0
    max_temporal_js: float = 1.0


@dataclass(frozen=True, slots=True)
class RiskSpec:
    alpha: float
    delta: float
    loss: str
    minimum_nfe_reduction: float | None


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
class ReadinessSpec:
    minimum_tuning_nfe_reduction_per_model: float
    require_candidate_beats_nonlearned: bool


@dataclass(frozen=True, slots=True)
class ConfirmatorySamplingSpec:
    strategy: str
    strata: int
    population_sizes: dict[str, int]
    excluded_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class RCPAGConfig:
    schema_version: int
    protocol_version: str
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
    readiness: ReadinessSpec
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
_WORKSHOP_V2_CONFIRMATORY = {
    "gsm8k_test": 500,
    "math500": 200,
    "mbpp_sanitized": 100,
    "humaneval": 64,
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
_REQUIRED_V2_DEVELOPMENT_METHODS = {
    "adablock",
    "entropy_sum_gate",
    "mutual_stability_gate",
    "rc_pag_local",
}
_REQUIRED_V3_DEVELOPMENT_METHODS = {
    "adablock",
    "entropy_sum_gate",
    "mutual_stability_gate",
    "stability_weighted_style",
    "token_convergence_style",
    "rc_pag_local",
}
_REQUIRED_V4_DEVELOPMENT_METHODS = {
    "adablock",
    "stability_weighted_style",
    "token_convergence_style",
    "rc_pag_local",
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

    protocol_version = str(payload.get("protocol_version", "v1"))
    if protocol_version not in {"v1", "v2", "v3", "v4"}:
        raise ValueError("protocol_version must be v1, v2, v3, or v4")

    policy = payload.get("policy", {})
    expected_estimators = (
        ("hist_gradient_boosting",)
        if protocol_version == "v4"
        else ("hist_gradient_boosting", "logistic")
    )
    if tuple(policy.get("estimator_kinds", ())) != expected_estimators:
        raise ValueError("estimator kinds do not match the frozen protocol")
    if int(policy.get("history_window", 0)) != 4:
        raise ValueError("history_window must remain 4")
    candidates = tuple(policy.get("candidates", ()))
    expected_candidates = {"v1": 6, "v2": 3, "v3": 9, "v4": 3}[protocol_version]
    if len(candidates) != expected_candidates:
        if protocol_version == "v1":
            raise ValueError("policy family must contain exactly six candidates")
        raise ValueError(f"{protocol_version} policy family has the wrong candidate count")
    names = tuple(str(item.get("name", "")) for item in candidates)
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("policy candidate names must be non-empty and unique")
    variants = [str(item.get("variant", "")) for item in candidates]
    if protocol_version == "v1":
        if variants.count("rc_pag_local") != 3 or variants.count("rc_pag_history") != 3:
            raise ValueError("policy family must contain three local and three history candidates")
    elif set(variants) != {"rc_pag_local"}:
        raise ValueError(f"{protocol_version} policy candidates must use the local estimator")
    for item in candidates:
        threshold_limit = 0.05 if protocol_version == "v1" else 1.0
        if not 0 < float(item.get("threshold", 0)) <= threshold_limit:
            raise ValueError(f"candidate thresholds must be in (0, {threshold_limit}]")
        if int(item.get("min_steps", 0)) < 1 or int(item.get("patience", 0)) < 1:
            raise ValueError("candidate min_steps and patience must be positive")
        remaining = float(item.get("max_remaining_fraction", 1.0))
        if not 0 < remaining <= 1:
            raise ValueError("candidate max_remaining_fraction must be in (0, 1]")
        min_savings = float(item.get("min_predicted_nfe_savings", 0.0))
        if min_savings < 0:
            raise ValueError("candidate min_predicted_nfe_savings must be nonnegative")
        max_temporal_js = float(item.get("max_temporal_js", 1.0))
        if not 0 <= max_temporal_js <= 1:
            raise ValueError("candidate max_temporal_js must be in [0, 1]")
        if protocol_version == "v3" and (min_savings != 2.0 or max_temporal_js != 0.05):
            raise ValueError("v3 candidates require the frozen benefit and stability gates")
        if protocol_version == "v4" and (
            remaining != 1.0 or min_savings != 0.0 or max_temporal_js != 1.0
        ):
            raise ValueError("v4 candidates use only the learned risk score and patience")

    risk = payload.get("risk", {})
    expected_alpha = 0.05 if protocol_version == "v1" else 0.02
    if float(risk.get("alpha", -1)) != expected_alpha:
        raise ValueError(f"risk alpha must remain {expected_alpha}")
    if float(risk.get("delta", -1)) != 0.05:
        raise ValueError("risk delta must remain 0.05")
    expected_loss = (
        "any_shadow_token_disagreement"
        if protocol_version == "v1"
        else "adablock_correct_candidate_wrong"
    )
    if risk.get("loss") != expected_loss:
        raise ValueError(f"risk loss must remain {expected_loss}")
    minimum_nfe_reduction = risk.get("minimum_nfe_reduction")
    if protocol_version == "v4":
        if float(minimum_nfe_reduction or 0.0) != 0.05:
            raise ValueError("v4 minimum NFE reduction must remain 0.05")
    elif minimum_nfe_reduction is not None:
        raise ValueError("minimum NFE reduction is only defined for v4")

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
        "workshop_v2_fresh": _WORKSHOP_V2_CONFIRMATORY,
        "workshop_v3_fresh": _WORKSHOP_V2_CONFIRMATORY,
        "workshop_v4_fresh": _WORKSHOP_V2_CONFIRMATORY,
    }
    if confirmation_profile not in expected_confirmatory:
        raise ValueError("unknown confirmation profile")
    if confirmatory != expected_confirmatory[confirmation_profile]:
        raise ValueError(f"confirmatory counts do not match the {confirmation_profile} profile")

    sampling = payload.get("confirmatory_sampling")
    if confirmation_profile == "full":
        if sampling is not None:
            raise ValueError("the full confirmation profile uses every benchmark row")
    elif confirmation_profile == "workshop_48h":
        expected_sampling = {
            "strategy": "index_stratified",
            "strata": 10,
            "population_sizes": _FULL_CONFIRMATORY,
        }
        if sampling != expected_sampling:
            raise ValueError("workshop confirmation must use the frozen stratified sampler")
    else:
        expected_sampling = {
            "strategy": "index_stratified_complement",
            "strata": 10,
            "population_sizes": _FULL_CONFIRMATORY,
            "excluded_counts": _WORKSHOP_CONFIRMATORY,
        }
        if sampling != expected_sampling:
            raise ValueError("fresh confirmation must use the untouched v1 complement")

    methods = payload.get("methods", {})
    development = set(methods.get("development", ()))
    expected_development = {
        "v1": _REQUIRED_DEVELOPMENT_METHODS,
        "v2": _REQUIRED_V2_DEVELOPMENT_METHODS,
        "v3": _REQUIRED_V3_DEVELOPMENT_METHODS,
        "v4": _REQUIRED_V4_DEVELOPMENT_METHODS,
    }[protocol_version]
    if development != expected_development:
        raise ValueError("development method family does not match the frozen protocol")
    expected_methods = {
        "full": ("adablock", "best_nonlearned", "rc_pag_local", "rc_pag_history"),
        "workshop_48h": ("adablock", "best_nonlearned", "rc_pag_history"),
        "workshop_v2_fresh": ("adablock", "best_nonlearned", "rc_pag_selected"),
        "workshop_v3_fresh": ("adablock", "best_nonlearned", "rc_pag_selected"),
        "workshop_v4_fresh": ("adablock", "best_nonlearned", "rc_pag_selected"),
    }
    if tuple(methods.get("confirmatory", ())) != expected_methods[confirmation_profile]:
        raise ValueError("confirmatory methods do not match the frozen protocol")

    gates = payload.get("claim_gates", {})
    expected_gates = {
        "require_risk_certificate": True,
        "beat_adablock_both_models": True,
        "beat_best_nonlearned": True,
        "minimum_accuracy_lower_ci": -0.02,
        "require_history_frontier_ci": protocol_version == "v1" and confirmation_profile == "full",
    }
    if gates != expected_gates:
        raise ValueError("claim gates do not match the frozen protocol")

    readiness = payload.get("readiness")
    if protocol_version in {"v3", "v4"}:
        expected_readiness = {
            "minimum_tuning_nfe_reduction_per_model": 0.05,
            "require_candidate_beats_nonlearned": True,
        }
        if readiness != expected_readiness:
            raise ValueError("v3 readiness gate does not match the frozen protocol")
    elif readiness is not None:
        raise ValueError("readiness is only defined for the v3 and v4 protocols")


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
    readiness = payload.get("readiness", {})
    confirmation_profile = str(payload.get("confirmation_profile", "full"))
    raw_sampling = payload.get("confirmatory_sampling")
    if raw_sampling is None:
        confirmatory_sampling = ConfirmatorySamplingSpec(
            strategy="full",
            strata=1,
            population_sizes=dict(_FULL_CONFIRMATORY),
            excluded_counts={},
        )
    else:
        confirmatory_sampling = ConfirmatorySamplingSpec(
            strategy=str(raw_sampling["strategy"]),
            strata=int(raw_sampling["strata"]),
            population_sizes={
                name: int(value) for name, value in raw_sampling["population_sizes"].items()
            },
            excluded_counts={
                name: int(value) for name, value in raw_sampling.get("excluded_counts", {}).items()
            },
        )
    return RCPAGConfig(
        schema_version=int(payload["schema_version"]),
        protocol_version=str(payload.get("protocol_version", "v1")),
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
                max_remaining_fraction=float(item.get("max_remaining_fraction", 1.0)),
                min_predicted_nfe_savings=float(item.get("min_predicted_nfe_savings", 0.0)),
                max_temporal_js=float(item.get("max_temporal_js", 1.0)),
            )
            for item in policy["candidates"]
        ),
        risk=RiskSpec(
            alpha=float(risk["alpha"]),
            delta=float(risk["delta"]),
            loss=str(risk["loss"]),
            minimum_nfe_reduction=(
                None
                if risk.get("minimum_nfe_reduction") is None
                else float(risk["minimum_nfe_reduction"])
            ),
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
        readiness=ReadinessSpec(
            minimum_tuning_nfe_reduction_per_model=float(
                readiness.get("minimum_tuning_nfe_reduction_per_model", 0.0)
            ),
            require_candidate_beats_nonlearned=bool(
                readiness.get("require_candidate_beats_nonlearned", False)
            ),
        ),
        config_hash=canonical_config_hash(payload),
        raw=payload,
    )

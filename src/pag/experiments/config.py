from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    path: str
    revision: str
    config: str | None = None
    development_indices: tuple[int, int] | None = None
    confirmatory_indices: tuple[int, int] | None = None
    sample_size: int | None = None


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    temperature: float
    gen_length: int
    steps: int
    threshold: float
    delimiter_threshold: float
    delimiter_ids: tuple[int, ...]
    use_cache: bool
    dual_cache: bool
    tau_commit: float
    tau_stable_steps: int


@dataclass(frozen=True, slots=True)
class MethodsConfig:
    development: tuple[str, ...]
    final_required: tuple[str, ...]
    math500: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    candidates: tuple[str, ...]
    max_correct_loss: int


@dataclass(frozen=True, slots=True)
class TimingConfig:
    warmups: int
    prompts: int
    repetitions: int


@dataclass(frozen=True, slots=True)
class StatisticsConfig:
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    reserve_fraction: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: int
    seed: int
    gsm8k: DatasetConfig
    math500: DatasetConfig
    decoding: DecodingConfig
    methods: MethodsConfig
    promotion: PromotionConfig
    timing: TimingConfig
    statistics: StatisticsConfig
    budget: BudgetConfig
    config_hash: str
    raw: dict[str, Any]


def canonical_config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def inclusive_range(bounds: tuple[int, int]) -> range:
    return range(bounds[0], bounds[1] + 1)


def validate_experiment_config(payload: dict[str, Any]) -> None:
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("schema_version must be 1")
    decoding = payload.get("decoding", {})
    if float(decoding.get("temperature", -1)) != 0.0:
        raise ValueError("decoding temperature must be 0")
    if not decoding.get("use_cache") or not decoding.get("dual_cache"):
        raise ValueError("strategy 1 requires dual-cache decoding")
    datasets = payload.get("datasets", {})
    gsm8k = datasets.get("gsm8k", {})
    dev = tuple(int(value) for value in gsm8k.get("development_indices", ()))
    confirm = tuple(int(value) for value in gsm8k.get("confirmatory_indices", ()))
    if len(dev) != 2 or len(confirm) != 2 or dev[0] > dev[1] or confirm[0] > confirm[1]:
        raise ValueError("dataset index ranges must contain ordered inclusive bounds")
    if set(inclusive_range(dev)).intersection(inclusive_range(confirm)):
        raise ValueError("development and confirmatory indices must be disjoint")
    methods = payload.get("methods", {})
    development = set(methods.get("development", []))
    required = {"adablock", "gates_only", "constant_budget", "size_lookup", "pag"}
    if not required.issubset(development):
        raise ValueError("development methods omit required attribution baselines")
    if not set(methods.get("final_required", [])).issubset(development):
        raise ValueError("final methods must also be development methods")
    timing = payload.get("timing", {})
    if any(int(timing.get(name, 0)) < 1 for name in ("warmups", "prompts", "repetitions")):
        raise ValueError("timing counts must be positive")
    reserve = float(payload.get("budget", {}).get("reserve_fraction", -1))
    if not 0 <= reserve < 1:
        raise ValueError("budget reserve_fraction must be in [0, 1)")


def _dataset_config(payload: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        path=str(payload["path"]),
        revision=str(payload["revision"]),
        config=str(payload["config"]) if payload.get("config") is not None else None,
        development_indices=tuple(payload["development_indices"])
        if payload.get("development_indices") is not None
        else None,
        confirmatory_indices=tuple(payload["confirmatory_indices"])
        if payload.get("confirmatory_indices") is not None
        else None,
        sample_size=int(payload["sample_size"]) if payload.get("sample_size") is not None else None,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("experiment config must be a mapping")
    validate_experiment_config(payload)
    datasets = payload["datasets"]
    decoding = payload["decoding"]
    methods = payload["methods"]
    promotion = payload["promotion"]
    timing = payload["timing"]
    return ExperimentConfig(
        schema_version=int(payload["schema_version"]),
        seed=int(payload["seed"]),
        gsm8k=_dataset_config(datasets["gsm8k"]),
        math500=_dataset_config(datasets["math500"]),
        decoding=DecodingConfig(
            temperature=float(decoding["temperature"]),
            gen_length=int(decoding["gen_length"]),
            steps=int(decoding["steps"]),
            threshold=float(decoding["threshold"]),
            delimiter_threshold=float(decoding["delimiter_threshold"]),
            delimiter_ids=tuple(int(value) for value in decoding["delimiter_ids"]),
            use_cache=bool(decoding["use_cache"]),
            dual_cache=bool(decoding["dual_cache"]),
            tau_commit=float(decoding["tau_commit"]),
            tau_stable_steps=int(decoding["tau_stable_steps"]),
        ),
        methods=MethodsConfig(
            development=tuple(methods["development"]),
            final_required=tuple(methods["final_required"]),
            math500=tuple(methods["math500"]),
        ),
        promotion=PromotionConfig(
            candidates=tuple(promotion["candidates"]),
            max_correct_loss=int(promotion["max_correct_loss"]),
        ),
        timing=TimingConfig(**{key: int(value) for key, value in timing.items()}),
        statistics=StatisticsConfig(
            bootstrap_samples=int(payload["statistics"]["bootstrap_samples"])
        ),
        budget=BudgetConfig(reserve_fraction=float(payload["budget"]["reserve_fraction"])),
        config_hash=canonical_config_hash(payload),
        raw=payload,
    )

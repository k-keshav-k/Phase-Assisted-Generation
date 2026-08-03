from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.metrics import roc_auc_score

from pag.experiments.config import canonical_config_hash, inclusive_range
from pag.experiments.orchestrator import ControlledStop, environment_metadata
from pag.experiments.rc_pag_config import (
    ModelSpec,
    PolicyCandidateSpec,
    RCPAGConfig,
)
from pag.experiments.rc_pag_features import (
    RealizedBlock,
    StepObservation,
    extract_features,
)
from pag.experiments.rc_pag_policy import (
    BenefitExample,
    CalibratedRiskEstimator,
    NormalizedNFEReductionEstimator,
    NormalizedNFEReductionExample,
    RemainingNFEEstimator,
    RiskEstimator,
    TrainingExample,
)
from pag.experiments.records import RecordStore
from pag.experiments.risk_control import CandidateRisk, certify_candidates

_MODERN_PROTOCOLS = {"v2", "v3", "v4", "v5", "v6"}


@dataclass(frozen=True, slots=True)
class SampleRef:
    pool: str
    index: int

    @property
    def sample_id(self) -> str:
        return f"{self.pool}-{self.index:05d}"


def _index_stratified_indices(
    *, population: int, count: int, strata: int, seed: int, pool: str
) -> tuple[int, ...]:
    if population < 1 or not 0 < count <= population:
        raise ValueError("stratified sample count must be within the population")
    if not 0 < strata <= count:
        raise ValueError("strata must be positive and no larger than the sample")
    per_stratum, remainder = divmod(count, strata)
    selected: list[list[int]] = []
    for stratum in range(strata):
        lower = population * stratum // strata
        upper = population * (stratum + 1) // strata
        take = per_stratum + int(stratum < remainder)
        if take > upper - lower:
            raise ValueError("stratified sample exceeds a stratum's population")
        digest = hashlib.sha256(f"{seed}:{pool}:{stratum}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        selected.append(sorted(rng.sample(range(lower, upper), take)))
    return tuple(
        values[position]
        for position in range(max(map(len, selected)))
        for values in selected
        if position < len(values)
    )


def _index_stratified_complement_indices(
    *,
    population: int,
    count: int,
    excluded_count: int,
    strata: int,
    seed: int,
    pool: str,
) -> tuple[int, ...]:
    excluded = set(
        _index_stratified_indices(
            population=population,
            count=excluded_count,
            strata=strata,
            seed=seed,
            pool=pool,
        )
    )
    remaining_by_stratum = [
        [
            index
            for index in range(population * stratum // strata, population * (stratum + 1) // strata)
            if index not in excluded
        ]
        for stratum in range(strata)
    ]
    remaining_count = sum(map(len, remaining_by_stratum))
    if not 0 < count <= remaining_count:
        raise ValueError("complement sample count must be within the unused population")
    quotas = [count * len(values) / remaining_count for values in remaining_by_stratum]
    allocations = [math.floor(quota) for quota in quotas]
    unallocated = count - sum(allocations)
    order = sorted(
        range(strata),
        key=lambda stratum: (-(quotas[stratum] - allocations[stratum]), stratum),
    )
    for stratum in order[:unallocated]:
        allocations[stratum] += 1
    selected: list[list[int]] = []
    for stratum, values in enumerate(remaining_by_stratum):
        take = allocations[stratum]
        digest = hashlib.sha256(f"{seed}:{pool}:v2:{stratum}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        selected.append(sorted(rng.sample(values, take)))
    return tuple(
        values[position]
        for position in range(max(map(len, selected)))
        for values in selected
        if position < len(values)
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_collection_rows(
    root: Path,
    *,
    model: str,
    expected_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    directory = root / "collect" / model / "full_budget_shadow"
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != expected_identity:
            raise ValueError(f"reuse trace identity differs: {path}")
        if payload.get("stage") != f"collect/{model}":
            raise ValueError(f"reuse trace has the wrong stage: {path}")
        if payload.get("method") != "full_budget_shadow":
            raise ValueError(f"reuse trace has the wrong method: {path}")
        rows.append(payload)
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def _raw_stage_rows(
    root: Path,
    *,
    stage: str,
    model: str,
    method: str,
    expected_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    directory = root / stage / model / method
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != expected_identity:
            raise ValueError(f"reuse row identity differs: {path}")
        if payload.get("stage") != f"{stage}/{model}":
            raise ValueError(f"reuse row has the wrong stage: {path}")
        if payload.get("method") != method:
            raise ValueError(f"reuse row has the wrong method: {path}")
        rows.append(payload)
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def _training_payloads(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_examples = (
        row["training_examples"]
        if "training_examples" in row
        else ({"features": row["features"], "unsafe": row["unsafe"]},)
    )
    payloads = [dict(example) for example in raw_examples]
    if all("remaining_nfe" in example for example in payloads):
        return tuple(payloads)
    targets = [
        float(max(0, int(block["actual_nfe_used"]) - int(step["observation"]["step_index"])))
        for block in row.get("schedule_history", ())
        for step in block.get("risk_steps", ())
    ]
    if not targets and "remaining_nfe" in row and len(payloads) == 1:
        targets = [float(row["remaining_nfe"])]
    if len(targets) != len(payloads):
        raise ValueError(
            f"cannot reconstruct remaining-NFE targets for {row.get('sample_id', 'unknown')}"
        )
    for example, target in zip(payloads, targets, strict=True):
        example["remaining_nfe"] = target
    return tuple(payloads)


def _v4_training_payloads(
    row: Mapping[str, Any], *, history_window: int
) -> tuple[dict[str, Any], ...]:
    """Rebuild v4 labels/features from raw exact-loop observations, including temporal JS."""

    schedules = row.get("schedule_history", ())
    if not schedules:
        return _training_payloads(row)
    examples: list[dict[str, Any]] = []
    history: list[RealizedBlock] = []
    for block in schedules:
        previous: StepObservation | None = None
        final_tokens = tuple(int(value) for value in block.get("final_tokens", ()))
        for raw_step in block.get("risk_steps", ()):
            raw_observation = raw_step["observation"]
            observation = StepObservation.from_arrays(
                step_index=int(raw_observation["step_index"]),
                block_size=int(raw_observation["block_size"]),
                masked=raw_observation["masked"],
                top1_probs=raw_observation["top1_probs"],
                top2_probs=raw_observation["top2_probs"],
                entropies=raw_observation["entropies"],
                token_ids=raw_observation["token_ids"],
                temporal_js=raw_observation.get("temporal_js"),
                digit_ids=raw_observation.get("digit_ids", ()),
                delimiter_ids=raw_observation.get("delimiter_ids", ()),
            )
            proposed = tuple(int(value) for value in raw_step["proposed_tokens"])
            if not final_tokens or len(proposed) != len(final_tokens):
                raise ValueError("v4 training trace is missing same-block final tokens")
            examples.append(
                {
                    "features": extract_features(
                        observation,
                        previous=previous,
                        history=history,
                        history_window=history_window,
                    ),
                    "unsafe": proposed != final_tokens,
                    "remaining_nfe": float(
                        max(0, int(block["actual_nfe_used"]) - observation.step_index)
                    ),
                }
            )
            previous = observation
        if final_tokens:
            history.append(
                RealizedBlock(
                    block_size=int(block["applied_block_size"]),
                    nfe=int(block["actual_nfe_used"]),
                    mean_confidence=float(block.get("mean_top1_confidence", 1.0)),
                    min_confidence=float(block.get("min_top1_confidence", 1.0)),
                    digit_fraction=float(block.get("digit_fraction", 0.0)),
                    delimiter_fraction=float(block.get("delimiter_fraction", 0.0)),
                )
            )
    if not examples:
        raise ValueError("v4 generation produced no instrumented training observations")
    return tuple(examples)


def _counterfactual_examples_from_pair(
    baseline: Mapping[str, Any],
    seed: Mapping[str, Any],
    *,
    history_window: int,
) -> tuple[tuple[TrainingExample, ...], tuple[NormalizedNFEReductionExample, ...]]:
    """Label executed seed stops with paired prompt harm and normalized NFE reduction."""

    baseline_id = str(baseline.get("sample_id", ""))
    seed_id = str(seed.get("sample_id", ""))
    if not baseline_id or baseline_id != seed_id:
        raise ValueError("counterfactual rollout requires a matching non-empty sample_id")
    baseline_nfe = float(baseline.get("total_nfe", 0.0))
    seed_nfe = float(seed.get("total_nfe", 0.0))
    if baseline_nfe <= 0.0 or seed_nfe <= 0.0:
        raise ValueError("counterfactual rollout NFE values must be positive")
    harmful = bool(baseline.get("is_correct")) and not bool(seed.get("is_correct"))
    reduction = max(0.0, 1.0 - seed_nfe / baseline_nfe)
    harm_examples: list[TrainingExample] = []
    gain_examples: list[NormalizedNFEReductionExample] = []
    history: list[RealizedBlock] = []
    for block in seed.get("schedule_history", ()):
        previous: StepObservation | None = None
        for raw_step in block.get("risk_steps", ()):
            raw = raw_step.get("observation", {})
            observation = StepObservation.from_arrays(
                step_index=int(raw["step_index"]),
                block_size=int(raw["block_size"]),
                masked=raw["masked"],
                top1_probs=raw["top1_probs"],
                top2_probs=raw["top2_probs"],
                entropies=raw["entropies"],
                token_ids=raw["token_ids"],
                temporal_js=raw.get("temporal_js"),
                digit_ids=raw.get("digit_ids", ()),
                delimiter_ids=raw.get("delimiter_ids", ()),
            )
            features = extract_features(
                observation,
                previous=previous,
                history=history,
                history_window=history_window,
            )
            if bool(raw_step.get("should_stop")):
                harm_examples.append(TrainingExample(features, harmful, baseline_id))
                gain_examples.append(
                    NormalizedNFEReductionExample(features, reduction, baseline_id)
                )
            previous = observation
        final_tokens = block.get("final_tokens", ())
        if final_tokens:
            history.append(
                RealizedBlock(
                    block_size=int(block["applied_block_size"]),
                    nfe=int(block["actual_nfe_used"]),
                    mean_confidence=float(block.get("mean_top1_confidence", 1.0)),
                    min_confidence=float(block.get("min_top1_confidence", 1.0)),
                    digit_fraction=float(block.get("digit_fraction", 0.0)),
                    delimiter_fraction=float(block.get("delimiter_fraction", 0.0)),
                )
            )
    return tuple(harm_examples), tuple(gain_examples)


class RCPAGRuntime(Protocol):
    is_mock: bool

    def preflight(self, *, model: str, spec: ModelSpec, device: str) -> Mapping[str, Any]: ...

    def run(
        self,
        *,
        stage: str,
        model: str,
        sample: SampleRef,
        method: str,
        candidate: PolicyCandidateSpec | None = None,
        estimator_paths: Mapping[str, str] | None = None,
    ) -> dict[str, Any]: ...


def _mock_observation(index: int, *, step: int) -> StepObservation:
    shift = (index % 7) * 0.005
    top1 = [0.55 + shift, 0.65 + shift, 0.75 + shift, 0.85 + shift]
    return StepObservation.from_arrays(
        step_index=step,
        block_size=4,
        masked=[step < 2, False, False, False],
        top1_probs=top1,
        top2_probs=[value - 0.15 for value in top1],
        entropies=[0.8 - value / 2 for value in top1],
        token_ids=[10 + index % 10, 20, 30, 40],
        digit_ids={10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
        delimiter_ids={40},
    )


class MockRCPAGRuntime:
    """Deterministic pipeline exerciser; its outputs are never scientific results."""

    is_mock = True

    def __init__(
        self,
        *,
        unsafe: bool = False,
        candidate_nfe_offset: float = 0.0,
        calibration_repetitions: int = 1,
    ) -> None:
        self.unsafe = bool(unsafe)
        self.candidate_nfe_offset = float(candidate_nfe_offset)
        self.calibration_repetitions = int(calibration_repetitions)
        if self.calibration_repetitions < 1:
            raise ValueError("calibration_repetitions must be positive")
        self.calls: list[tuple[str, str, str, str]] = []

    def preflight(self, *, model: str, spec: ModelSpec, device: str) -> Mapping[str, Any]:
        self.calls.append(("preflight", model, spec.repository, device))
        return {"ok": True, "mock": True, "repository": spec.repository}

    @staticmethod
    def _method_nfe(method: str) -> float:
        values = {
            "fixed": 100.0,
            "adablock": 80.0,
            "fast_dllm": 75.0,
            "sched": 73.0,
            "entropy_sum": 72.0,
            "confidence_gate": 70.0,
            "stability_gate": 69.0,
            "constant_budget": 68.0,
            "size_lookup": 66.0,
            "pag": 67.0,
            "residual_pag": 65.0,
            "oracle": 45.0,
            "best_nonlearned": 66.0,
            "rc_pag_local": 60.0,
            "rc_pag_history": 57.0,
            "entropy_sum_gate": 72.0,
            "mutual_stability_gate": 69.0,
            "stability_weighted_style": 67.0,
            "token_convergence_style": 66.0,
            "rc_pag_selected": 58.0,
        }
        return values.get(method, 80.0)

    def run(
        self,
        *,
        stage: str,
        model: str,
        sample: SampleRef,
        method: str,
        candidate: PolicyCandidateSpec | None = None,
        estimator_paths: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        del estimator_paths
        self.calls.append((stage, model, sample.sample_id, method))
        elapsed = 0.02 + (sample.index % 3) * 0.001
        if stage == "pilot":
            return {
                "elapsed_sec": elapsed,
                "artifact_bytes": 2048,
                "generated_text": f"mock-{sample.sample_id}",
                "generated_ids": [sample.index, 1, 2],
                "nfe_history": [4, 3],
                "block_history": [2, 2],
                "total_nfe": 7,
                "is_correct": True,
                "mock": True,
            }
        if stage == "collect":
            previous = _mock_observation(sample.index, step=1)
            current = _mock_observation(sample.index, step=2)
            history = (RealizedBlock(4, 3, 0.8, 0.6, 0.25, 0.25),)
            return {
                "elapsed_sec": elapsed,
                "features": extract_features(
                    current,
                    previous=previous,
                    history=history,
                    history_window=4,
                ),
                "unsafe": bool(sample.index % 5 == 0),
                "remaining_nfe": 2.0 + sample.index % 4,
                "mock": True,
            }
        if candidate is not None:
            variant_base = 64.0 if candidate.variant == "rc_pag_local" else 61.0
            threshold_rank = -8.0 * candidate.threshold
            nfe = variant_base + threshold_rank + self.candidate_nfe_offset
        else:
            nfe = self._method_nfe(method)
        payload: dict[str, Any] = {
            "elapsed_sec": elapsed,
            "total_nfe": nfe + (sample.index % 3) * 0.1,
            "is_correct": (
                False
                if stage == "calibrate" and candidate is not None and self.unsafe
                else sample.index % 11 != 0
            ),
            "generated_text": f"mock-{sample.sample_id}",
            "generated_ids": [sample.index, 1, 2],
            "mock": True,
        }
        if stage in {"rollout", "screen"} and candidate is not None:
            observation = _mock_observation(sample.index, step=2)
            payload["schedule_history"] = [
                {
                    "applied_block_size": observation.block_size,
                    "actual_nfe_used": 3,
                    "mean_top1_confidence": 0.8,
                    "min_top1_confidence": 0.6,
                    "digit_fraction": 0.25,
                    "delimiter_fraction": 0.25,
                    "final_tokens": list(observation.token_ids),
                    "risk_steps": [
                        {
                            "should_stop": True,
                            "proposed_tokens": list(observation.token_ids),
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
                    ],
                }
            ]
        if stage == "calibrate":
            payload["shadow_losses"] = [int(self.unsafe)] * self.calibration_repetitions
            payload["synthetic_repetitions"] = self.calibration_repetitions
        return payload


class RCPAGOrchestrator:
    STAGES = (
        "preflight",
        "pilot",
        "collect",
        "fit",
        "screen",
        "calibrate",
        "confirm",
        "report",
        "paper",
    )
    V5_STAGES = (
        "preflight",
        "pilot",
        "collect",
        "fit",
        "rollout",
        "refit",
        "screen",
        "calibrate",
        "confirm",
        "report",
        "paper",
    )
    ALL_STAGES = tuple(dict.fromkeys((*STAGES, *V5_STAGES)))

    def __init__(
        self,
        config: RCPAGConfig,
        run_dir: str | Path,
        *,
        device: str = "cpu",
        runtime_factory: Callable[[str], RCPAGRuntime] | None = None,
        development_limit: int | None = None,
        mock_mode: bool | None = None,
        reuse_development_from: str | Path | None = None,
    ) -> None:
        if development_limit is not None and development_limit < 1:
            raise ValueError("development_limit must be positive")
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = device
        self.development_limit = development_limit
        self.mock_mode = mock_mode
        self.reuse_development_from = (
            None if reuse_development_from is None else Path(reuse_development_from).resolve()
        )
        if (
            self.reuse_development_from is not None
            and config.protocol_version not in _MODERN_PROTOCOLS
        ):
            raise ValueError("development artifact reuse is only defined for modern protocols")
        self.runtime_factory = runtime_factory
        if runtime_factory is None:
            raise ValueError("a runtime_factory is required until a model runtime is configured")
        self.store = RecordStore(
            self.run_dir,
            {
                "protocol": f"risk_calibrated_pag_{config.protocol_version}",
                "config_hash": config.config_hash,
                "models": {name: spec.revision for name, spec in sorted(config.models.items())},
                "datasets": {name: spec.revision for name, spec in sorted(config.datasets.items())},
            },
        )
        self._runtimes: dict[str, RCPAGRuntime] = {}

    def _runtime(self, model: str) -> RCPAGRuntime:
        if model not in self._runtimes:
            self._runtimes[model] = self.runtime_factory(model)  # type: ignore[misc]
        return self._runtimes[model]

    def _manifest_path(self, stage: str) -> Path:
        return self.run_dir / "manifests" / f"{stage}.json"

    @property
    def active_stages(self) -> tuple[str, ...]:
        return self.V5_STAGES if self.config.protocol_version == "v5" else self.STAGES

    def _stage_complete(self, stage: str) -> bool:
        path = self._manifest_path(stage)
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("config_hash") != self.config.config_hash:
            raise ValueError(f"stage {stage} manifest has a different config hash")
        return payload.get("status") == "completed"

    def _write_manifest(self, stage: str, status: str, **extra: object) -> None:
        self.store.write_named(
            f"manifests/{stage}.json",
            {
                "schema_version": 1,
                "stage": stage,
                "status": status,
                "config_hash": self.config.config_hash,
                "identity": self.store.identity,
                **extra,
            },
        )

    def _refs(self, role: str) -> tuple[SampleRef, ...]:
        refs = tuple(
            SampleRef(pool, index)
            for pool, bounds in sorted(self.config.splits[role].items())
            for index in inclusive_range(bounds)
        )
        return refs if self.development_limit is None else refs[: self.development_limit]

    def _confirm_refs(self, dataset: str) -> tuple[SampleRef, ...]:
        count = self.config.confirmatory_counts[dataset]
        sampling = self.config.confirmatory_sampling
        if sampling.strategy == "full":
            indices = tuple(range(count))
        elif sampling.strategy == "index_stratified":
            indices = _index_stratified_indices(
                population=sampling.population_sizes[dataset],
                count=count,
                strata=sampling.strata,
                seed=self.config.seed,
                pool=dataset,
            )
        elif sampling.strategy == "index_stratified_complement":
            indices = _index_stratified_complement_indices(
                population=sampling.population_sizes[dataset],
                count=count,
                excluded_count=sampling.excluded_counts[dataset],
                strata=sampling.strata,
                seed=self.config.seed,
                pool=dataset,
            )
        else:  # pragma: no cover - config validation owns the strategy set
            raise ValueError(f"unknown confirmatory sampling strategy: {sampling.strategy}")
        refs = tuple(SampleRef(dataset, index) for index in indices)
        if self.development_limit is None:
            return refs
        is_mock = self.mock_mode
        if is_mock is None:
            is_mock = all(self._runtime(model).is_mock for model in self.config.models)
        if not is_mock:
            raise ValueError("confirmatory execution rejects development limits")
        return refs[: self.development_limit]

    def run_through(self, target: str) -> None:
        stages = self.active_stages
        if target not in stages:
            raise ValueError(f"unknown stage: {target}")
        for stage in stages[: stages.index(target) + 1]:
            self.run_stage(stage)

    def run_stage(self, stage: str) -> None:
        stages = self.active_stages
        if stage not in stages:
            raise ValueError(f"unknown stage: {stage}")
        if self._stage_complete(stage):
            return
        position = stages.index(stage)
        if position and not self._stage_complete(stages[position - 1]):
            raise ValueError(f"stage {stage} requires completed {stages[position - 1]}")
        handler = getattr(self, f"_run_{stage}")
        handler()

    def _run_preflight(self) -> None:
        self._write_manifest("preflight", "running")
        results = {
            model: dict(self._runtime(model).preflight(model=model, spec=spec, device=self.device))
            for model, spec in self.config.models.items()
        }
        if not all(bool(result.get("ok")) for result in results.values()):
            self._write_manifest("preflight", "failed", results=results)
            raise RuntimeError("RC-PAG preflight failed")
        self.store.write_named(
            "environment.json",
            {
                "config_hash": self.config.config_hash,
                "environment": environment_metadata(self.device),
                "models": results,
            },
        )
        self._write_manifest("preflight", "completed", models=results)

    def _run_records(
        self,
        *,
        stage: str,
        refs: Sequence[SampleRef],
        methods: Sequence[tuple[str, PolicyCandidateSpec | None]],
        models: Sequence[str] | None = None,
    ) -> None:
        estimator_paths = {
            path.stem: str(path) for path in sorted((self.run_dir / "estimators").glob("*.joblib"))
        }
        for model in models or tuple(self.config.models):
            runtime = self._runtime(model)
            model_stage = f"{stage}/{model}"
            for sample in refs:
                for method, candidate in methods:
                    if self.store.is_complete(model_stage, method, sample.sample_id):
                        continue
                    payload = runtime.run(
                        stage=stage,
                        model=model,
                        sample=sample,
                        method=method,
                        candidate=candidate,
                        estimator_paths=estimator_paths,
                    )
                    self.store.write(model_stage, method, sample.sample_id, payload)
            if not runtime.is_mock:
                close = getattr(runtime, "close", None)
                if close is not None:
                    close()
                self._runtimes.pop(model, None)

    def _run_pilot(self) -> None:
        self._write_manifest("pilot", "running")
        refs = self._refs("pilot")
        pilot_methods = (
            (("adablock", None), ("full_budget_shadow", None))
            if self.config.protocol_version in _MODERN_PROTOCOLS
            else (("full_budget", None), ("pilot_shadow", None))
        )
        self._run_records(
            stage="pilot",
            refs=refs,
            methods=pilot_methods,
        )
        primary_pilot_method = (
            "adablock" if self.config.protocol_version in _MODERN_PROTOCOLS else "full_budget"
        )
        records = [
            row
            for model in self.config.models
            for row in self.store.records(f"pilot/{model}", primary_pilot_method)
        ]
        if self.config.protocol_version in _MODERN_PROTOCOLS:
            parity_fields = (
                "generated_ids",
                "generated_text",
                "nfe_history",
                "block_history",
                "total_nfe",
            )
            mismatches: list[dict[str, object]] = []
            for model in self.config.models:
                baseline = {
                    row["sample_id"]: row
                    for row in self.store.records(f"pilot/{model}", "adablock")
                }
                instrumented = {
                    row["sample_id"]: row
                    for row in self.store.records(f"pilot/{model}", "full_budget_shadow")
                }
                if set(baseline) != set(instrumented):
                    raise RuntimeError(f"v2 parity coverage differs for {model}")
                for sample_id in sorted(baseline):
                    different = [
                        field
                        for field in parity_fields
                        if baseline[sample_id].get(field) != instrumented[sample_id].get(field)
                    ]
                    if different:
                        mismatches.append(
                            {"model": model, "sample_id": sample_id, "fields": different}
                        )
            parity = {
                "checked_records": len(records),
                "fields": list(parity_fields),
                "mismatches": mismatches,
                "passed": not mismatches,
            }
            self.store.write_named("parity_audit.json", parity)
            if mismatches:
                self._write_manifest("pilot", "failed", parity=parity)
                raise ControlledStop("exact AdaBlock parity gate failed")
        shadow_records = [
            row
            for model in self.config.models
            for row in self.store.records(
                f"pilot/{model}",
                "full_budget_shadow"
                if self.config.protocol_version in _MODERN_PROTOCOLS
                else "pilot_shadow",
            )
        ]
        if len(shadow_records) != len(records):
            raise RuntimeError("pilot shadow coverage does not match full-budget coverage")
        if (
            self.config.protocol_version == "v1"
            and not all(bool(row.get("mock")) for row in shadow_records)
            and any(not row.get("shadow_losses") for row in shadow_records)
        ):
            raise RuntimeError("real pilot did not exercise a same-state shadow continuation")
        seconds = float(np.mean([float(row["elapsed_sec"]) for row in records]))
        instrumented_seconds = float(np.mean([float(row["elapsed_sec"]) for row in shadow_records]))
        artifact_bytes = float(np.mean([float(row.get("artifact_bytes", 0)) for row in records]))
        instrumented_artifact_bytes = float(
            np.mean([float(row.get("artifact_bytes", 0)) for row in shadow_records])
        )
        baseline_screen_methods = sum(
            not method.startswith("rc_pag_") for method in self.config.development_methods
        )
        runs_per_model = {
            "collect": self.config.stage_sizes.traces_per_model,
            "screen": (baseline_screen_methods + len(self.config.candidates))
            * self.config.stage_sizes.tuning_per_model,
            "calibrate": (
                2 * self.config.stage_sizes.calibration_per_model
                if self.config.protocol_version in _MODERN_PROTOCOLS
                else len(self.config.candidates) * self.config.stage_sizes.calibration_per_model
            ),
            "confirm": sum(self.config.confirmatory_counts.values())
            * len(self.config.confirmatory_methods),
        }
        if self.config.protocol_version == "v5":
            runs_per_model["rollout"] = 2 * self.config.stage_sizes.rollout_per_model
        collect_models = (
            0
            if self.config.protocol_version in {"v4", "v5", "v6"}
            and self.reuse_development_from is not None
            else len(self.config.models) - int(self.reuse_development_from is not None)
        )
        full_gpu_runs = collect_models * runs_per_model["collect"] + len(self.config.models) * sum(
            value for stage, value in runs_per_model.items() if stage != "collect"
        )
        if self.config.protocol_version in _MODERN_PROTOCOLS:
            model_count = len(self.config.models)
            confirm_prompts = sum(self.config.confirmatory_counts.values())
            plain_runs = model_count * (
                self.config.stage_sizes.tuning_per_model
                + self.config.stage_sizes.calibration_per_model
                + confirm_prompts
                + (
                    self.config.stage_sizes.rollout_per_model
                    if self.config.protocol_version == "v5"
                    else 0
                )
            )
            instrumented_runs = (
                collect_models * self.config.stage_sizes.traces_per_model
                + model_count
                * (
                    (baseline_screen_methods - 1 + len(self.config.candidates))
                    * self.config.stage_sizes.tuning_per_model
                    + self.config.stage_sizes.calibration_per_model
                    + (len(self.config.confirmatory_methods) - 1) * confirm_prompts
                    + (
                        self.config.stage_sizes.rollout_per_model
                        if self.config.protocol_version == "v5"
                        else 0
                    )
                )
            )
            projected_seconds = plain_runs * seconds + instrumented_runs * instrumented_seconds
            projected_storage = (
                plain_runs * artifact_bytes + instrumented_runs * instrumented_artifact_bytes
            )
        else:
            plain_runs = full_gpu_runs
            instrumented_runs = 0
            projected_seconds = seconds * full_gpu_runs
            projected_storage = artifact_bytes * full_gpu_runs
        projection = {
            "basis": ("paired plain/instrumented pilot wall time; rerun after the real A100 pilot"),
            "seconds_per_sample": seconds,
            "instrumented_seconds_per_sample": instrumented_seconds,
            "bytes_per_sample": artifact_bytes,
            "instrumented_bytes_per_sample": instrumented_artifact_bytes,
            "projected_runs_per_model_by_stage": runs_per_model,
            "projected_collect_models": collect_models,
            "projected_gpu_runs": full_gpu_runs,
            "projected_plain_runs": plain_runs,
            "projected_instrumented_runs": instrumented_runs,
            "projected_a100_hours": projected_seconds / 3600,
            "projected_storage_bytes": int(math.ceil(projected_storage)),
            "mock": any(bool(row.get("mock")) for row in records),
        }
        self.store.write_named("compute_projection.json", projection)
        self._write_manifest("pilot", "completed", projection=projection)

    def _run_collect(self) -> None:
        self._write_manifest("collect", "running")
        reused_models = self._prepare_development_reuse()
        self._run_records(
            stage="collect",
            refs=self._refs("training"),
            methods=(("full_budget_shadow", None),),
            models=tuple(model for model in self.config.models if model not in reused_models),
        )
        self._write_manifest("collect", "completed", reused_models=sorted(reused_models))

    def _prepare_development_reuse(self) -> set[str]:
        if self.reuse_development_from is None:
            return set()
        if self.config.protocol_version == "v6":
            return self._prepare_v6_reuse()
        if self.config.protocol_version == "v5":
            return self._prepare_v5_reuse()
        if self.config.protocol_version == "v4":
            return self._prepare_v4_trace_reuse()
        source = self.reuse_development_from
        if source == self.run_dir.resolve():
            raise ValueError("reuse source must be a different run directory")
        fit_manifest_path = source / "manifests" / "fit.json"
        estimator_manifest_path = source / "estimators" / "manifest.json"
        if not fit_manifest_path.is_file() or not estimator_manifest_path.is_file():
            raise ValueError(
                "reuse source must contain a completed fit stage and estimator manifest"
            )
        fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
        if fit_manifest.get("status") != "completed":
            raise ValueError("reuse source fit stage is not complete")
        source_identity = fit_manifest.get("identity", {})
        if source_identity.get("protocol") != "risk_calibrated_pag_v1":
            raise ValueError("reuse source must be a completed v1 RC-PAG run")
        for field in ("models", "datasets"):
            if source_identity.get(field) != self.store.identity[field]:
                raise ValueError(f"reuse source {field} identity differs from the v2 run")
        estimator_manifest = json.loads(estimator_manifest_path.read_text(encoding="utf-8"))
        trace_rows = _raw_collection_rows(
            source,
            model="llada",
            expected_identity=source_identity,
        )
        expected_traces = self.development_limit or self.config.stage_sizes.traces_per_model
        if len(trace_rows) < expected_traces:
            raise ValueError(
                f"reuse source has {len(trace_rows)} LLaDA traces; expected {expected_traces}"
            )
        trace_digest = hashlib.sha256()
        trace_directory = source / "collect" / "llada" / "full_budget_shadow"
        for path in sorted(trace_directory.glob("*.json")):
            trace_digest.update(path.name.encode("utf-8"))
            trace_digest.update(_sha256_path(path).encode("ascii"))
        try:
            local_metadata = estimator_manifest["models"]["llada"]["rc_pag_local"]
        except KeyError as exc:
            raise ValueError("reuse source has no LLaDA local risk estimator") from exc
        copied: list[dict[str, str]] = []
        for estimator in local_metadata["estimators"].values():
            relative = Path(str(estimator["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("reuse estimator path must be relative to its run directory")
            source_model = source / relative
            source_metadata = source_model.with_suffix(".json")
            if not source_model.is_file() or not source_metadata.is_file():
                raise ValueError(f"reuse estimator is incomplete: {relative}")
            loaded = RiskEstimator.load(source_model)
            if loaded.include_history or loaded.history_window != self.config.history_window:
                raise ValueError("reuse estimator feature protocol differs from v2 local policy")
            if loaded.kind not in self.config.estimator_kinds:
                raise ValueError("reuse estimator kind is not registered in v2")
            destination = self.run_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_model, destination)
            shutil.copy2(source_metadata, destination.with_suffix(".json"))
            if _sha256_path(source_model) != _sha256_path(destination):
                raise RuntimeError(f"copied estimator hash differs: {relative}")
            copied.append({"path": str(relative), "sha256": _sha256_path(destination)})
        evidence_files = (
            "screening_summary.json",
            "risk_certificate.json",
            "frozen_confirmatory_policy.json",
            "report/summary.json",
            "report/claim_audit.json",
        )
        evidence: list[dict[str, str]] = []
        for relative_text in evidence_files:
            relative = Path(relative_text)
            source_path = source / relative
            if not source_path.is_file():
                continue
            destination = self.run_dir / "prior_evidence" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            evidence.append(
                {
                    "path": str(Path("prior_evidence") / relative),
                    "sha256": _sha256_path(destination),
                }
            )
        reuse_payload = {
            "schema_version": 1,
            "source": str(source),
            "source_config_hash": fit_manifest.get("config_hash"),
            "reused_models": ["llada"],
            "reused_variants": ["rc_pag_local"],
            "excluded_models": ["dream"],
            "reason": (
                "Dream changed decoding semantics; only the parity-compatible LLaDA "
                "estimator is reused."
            ),
            "estimators": copied,
            "estimator_metadata": local_metadata,
            "benefit_trace_source": "collect/llada/full_budget_shadow",
            "benefit_trace_count": len(trace_rows),
            "benefit_trace_sha256": trace_digest.hexdigest(),
            "prior_evidence": evidence,
        }
        self.store.write_named("reuse/manifest.json", reuse_payload)
        return {"llada"}

    def _prepare_v4_trace_reuse(
        self,
        *,
        allowed_source_protocols: set[str] | None = None,
        target_protocol: str = "v4",
    ) -> set[str]:
        """Reuse only native exact-loop traces containing the v4 temporal feature evidence."""

        assert self.reuse_development_from is not None
        source = self.reuse_development_from
        if source == self.run_dir.resolve():
            raise ValueError("reuse source must be a different run directory")
        fit_manifest_path = source / "manifests" / "fit.json"
        if not fit_manifest_path.is_file():
            raise ValueError(f"{target_protocol} reuse source must contain a completed fit stage")
        fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
        if fit_manifest.get("status") != "completed":
            raise ValueError("v4 reuse source fit stage is not complete")
        source_identity = fit_manifest.get("identity", {})
        compatible_protocols = allowed_source_protocols or {
            "risk_calibrated_pag_v3",
            "risk_calibrated_pag_v4",
        }
        if source_identity.get("protocol") not in compatible_protocols:
            raise ValueError(
                f"{target_protocol} reuse source does not use a compatible exact-loop protocol"
            )
        for field in ("models", "datasets"):
            if source_identity.get(field) != self.store.identity[field]:
                raise ValueError(
                    f"reuse source {field} identity differs from the {target_protocol} run"
                )

        expected_traces = self.development_limit or self.config.stage_sizes.traces_per_model
        reused_models: list[str] = []
        trace_evidence: dict[str, dict[str, object]] = {}
        record_metadata = {
            "schema_version",
            "identity",
            "stage",
            "method",
            "sample_id",
            "created_at",
        }
        for model in self.config.models:
            rows = _raw_collection_rows(
                source,
                model=model,
                expected_identity=source_identity,
            )
            if not rows:
                continue
            if len(rows) < expected_traces:
                raise ValueError(
                    f"{target_protocol} reuse source has {len(rows)} {model} traces; "
                    f"expected {expected_traces}"
                )
            selected_rows = rows[:expected_traces]
            observation_count = 0
            for row in selected_rows:
                for block in row.get("schedule_history", ()):
                    for step in block.get("risk_steps", ()):
                        observation = step.get("observation", {})
                        temporal_js = observation.get("temporal_js")
                        block_size = int(observation.get("block_size", 0))
                        if not isinstance(temporal_js, list) or len(temporal_js) != block_size:
                            raise ValueError(
                                f"{target_protocol} reuse trace lacks temporal-JS evidence: "
                                f"{model}/"
                                f"{row.get('sample_id', 'unknown')}"
                            )
                        observation_count += 1
                payload = {key: value for key, value in row.items() if key not in record_metadata}
                self.store.write(
                    f"collect/{model}",
                    "full_budget_shadow",
                    str(row["sample_id"]),
                    payload,
                )
            if observation_count < 1:
                raise ValueError(
                    f"{target_protocol} reuse source has no instrumented observations for {model}"
                )
            digest = hashlib.sha256()
            for row in selected_rows:
                digest.update(str(row["sample_id"]).encode("utf-8"))
                digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
            reused_models.append(model)
            trace_evidence[model] = {
                "trace_count": len(selected_rows),
                "observation_count": observation_count,
                "sha256": digest.hexdigest(),
            }
        if not reused_models:
            raise ValueError(
                f"{target_protocol} reuse source contains no compatible collected model traces"
            )
        self.store.write_named(
            "reuse/manifest.json",
            {
                "schema_version": 2,
                "source": str(source),
                "source_config_hash": fit_manifest.get("config_hash"),
                "reused_models": reused_models,
                "reused_variants": [],
                "reuse_scope": "raw_exact_loop_traces_only",
                "reason": (
                    f"{target_protocol} refits its estimators from native exact-loop observations"
                ),
                "traces": trace_evidence,
            },
        )
        return set(reused_models)

    def _prepare_v6_reuse(self) -> set[str]:
        """Reuse raw v4/v5 native traces while discarding their fitted policy heads."""

        reused_models = self._prepare_v4_trace_reuse(
            allowed_source_protocols={
                "risk_calibrated_pag_v4",
                "risk_calibrated_pag_v5",
            },
            target_protocol="v6",
        )
        path = self.run_dir / "reuse" / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.store.write_named(
            "reuse/manifest.json",
            {
                **manifest,
                "schema_version": 4,
                "reuse_scope": "raw_native_exact_loop_traces_only",
                "reused_variants": [],
                "reason": (
                    "v6 reuses only native full-budget observations and refits calibrated "
                    "risk plus remaining-NFE estimators; v4/v5 fitted heads are discarded"
                ),
            },
        )
        return reused_models

    def _prepare_v5_reuse(self) -> set[str]:
        """Reuse exact v4 collection traces and paired q500 rollout rows only."""

        assert self.reuse_development_from is not None
        reused_models = self._prepare_v4_trace_reuse()
        source = self.reuse_development_from
        fit_manifest = json.loads((source / "manifests" / "fit.json").read_text(encoding="utf-8"))
        source_identity = fit_manifest["identity"]
        expected_ids = {ref.sample_id for ref in self._refs("rollout")}
        record_metadata = {
            "schema_version",
            "identity",
            "stage",
            "method",
            "sample_id",
            "created_at",
        }
        rollout_evidence: dict[str, dict[str, object]] = {}
        for model in self.config.models:
            method_rows: dict[str, list[dict[str, Any]]] = {}
            for source_method, target_method in (
                ("adablock", "adablock"),
                ("local_q500_p2", "seed_local_q500_p2"),
            ):
                rows = _raw_stage_rows(
                    source,
                    stage="screen",
                    model=model,
                    method=source_method,
                    expected_identity=source_identity,
                )
                selected = [row for row in rows if str(row["sample_id"]) in expected_ids]
                if {str(row["sample_id"]) for row in selected} != expected_ids:
                    raise ValueError(
                        f"v5 reuse source lacks complete {model}/{source_method} rollout pairs"
                    )
                if source_method == "local_q500_p2":
                    executed_stop_count = 0
                    for row in selected:
                        stopped_steps = [
                            step
                            for block in row.get("schedule_history", ())
                            for step in block.get("risk_steps", ())
                            if bool(step.get("should_stop"))
                        ]
                        executed_stop_count += len(stopped_steps)
                        for step in stopped_steps:
                            observation = step.get("observation", {})
                            if len(observation.get("temporal_js", ())) != int(
                                observation.get("block_size", -1)
                            ):
                                raise ValueError("v5 reuse q500 stop lacks temporal-JS evidence")
                    if executed_stop_count < 1:
                        raise ValueError(
                            f"v5 reuse q500 rollout has no executed serialized stops for {model}"
                        )
                for row in selected:
                    payload = {
                        key: value for key, value in row.items() if key not in record_metadata
                    }
                    self.store.write(
                        f"rollout/{model}",
                        target_method,
                        str(row["sample_id"]),
                        payload,
                    )
                method_rows[target_method] = selected
            digest = hashlib.sha256()
            for method in ("adablock", "seed_local_q500_p2"):
                for row in method_rows[method]:
                    digest.update(method.encode("utf-8"))
                    digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
            rollout_evidence[model] = {
                "paired_prompts": len(expected_ids),
                "sha256": digest.hexdigest(),
            }
        trace_manifest = json.loads(
            (self.run_dir / "reuse" / "manifest.json").read_text(encoding="utf-8")
        )
        self.store.write_named(
            "reuse/manifest.json",
            {
                **trace_manifest,
                "schema_version": 3,
                "reuse_scope": "raw_exact_loop_traces_and_paired_v4_q500_rollouts",
                "reason": (
                    "v5 refits both seed and advantage estimators; no v4 selection or "
                    "certificate is reused"
                ),
                "rollout": rollout_evidence,
            },
        )
        return reused_models

    def _run_fit(self) -> None:
        self._write_manifest("fit", "running")
        metadata: dict[str, dict[str, object]] = {}
        benefit_metadata: dict[str, dict[str, object]] = {}
        reused_models: set[str] = set()
        reuse_manifest_path = self.run_dir / "reuse" / "manifest.json"
        if reuse_manifest_path.is_file():
            reuse_manifest = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
            if self.config.protocol_version in {"v4", "v5", "v6"}:
                reused_trace_models = set(reuse_manifest["reused_models"])
            else:
                reused_trace_models = set()
                reused_models = set(reuse_manifest["reused_models"])
                metadata["llada"] = {
                    "rc_pag_local": {
                        **reuse_manifest["estimator_metadata"],
                        "reused": True,
                        "source_run": reuse_manifest["source"],
                    }
                }
        else:
            reused_trace_models = set()
        variants = (
            (("rc_pag_local", False),)
            if self.config.protocol_version in _MODERN_PROTOCOLS
            else (("rc_pag_local", False), ("rc_pag_history", True))
        )
        for model in (name for name in self.config.models if name not in reused_models):
            rows = self.store.records(f"collect/{model}", "full_budget_shadow")
            if not rows:
                raise ValueError(f"no collected examples for {model}")
            if self.config.protocol_version == "v6":
                payload_groups = tuple(
                    _v4_training_payloads(row, history_window=self.config.history_window)
                    for row in rows
                )
                risk_groups = tuple(
                    tuple(
                        TrainingExample(
                            features=example["features"],
                            unsafe=bool(example["unsafe"]),
                            prompt_id=str(row["sample_id"]),
                        )
                        for example in payloads
                    )
                    for row, payloads in zip(rows, payload_groups, strict=True)
                )
                benefit_groups = tuple(
                    tuple(
                        BenefitExample(
                            features=example["features"],
                            remaining_nfe=float(example["remaining_nfe"]),
                            prompt_id=str(row["sample_id"]),
                        )
                        for example in payloads
                    )
                    for row, payloads in zip(rows, payload_groups, strict=True)
                )
                if len(risk_groups) < 2:
                    raise ValueError(
                        "v6 fitting requires at least two prompts for disjoint calibration"
                    )
                calibration_risk_groups = risk_groups[::5]
                training_risk_groups = tuple(
                    group for index, group in enumerate(risk_groups) if index % 5 != 0
                )
                calibration_benefit_groups = benefit_groups[::5]
                training_benefit_groups = tuple(
                    group for index, group in enumerate(benefit_groups) if index % 5 != 0
                )
                training_risk = tuple(
                    example for group in training_risk_groups for example in group
                )
                calibration_risk = tuple(
                    example for group in calibration_risk_groups for example in group
                )
                kind = self.config.estimator_kinds[0]
                estimator = CalibratedRiskEstimator.fit(
                    training_examples=training_risk,
                    calibration_examples=calibration_risk,
                    kind=kind,
                    include_history=False,
                    history_window=self.config.history_window,
                    seed=self.config.seed,
                )
                scores = np.asarray(
                    [estimator.predict_risk(example.features) for example in calibration_risk],
                    dtype=np.float64,
                )
                labels = np.asarray(
                    [int(example.unsafe) for example in calibration_risk],
                    dtype=np.int64,
                )
                risk_path = self.run_dir / "estimators" / f"{model}_rc_pag_budgeted_risk.joblib"
                saved_risk = estimator.save(risk_path)
                metadata[model] = {
                    "rc_pag_budgeted": {
                        "primary_kind": kind,
                        "estimators": {
                            kind: {
                                **saved_risk,
                                "path": str(risk_path.relative_to(self.run_dir)),
                                "deployment_estimator": True,
                                "validation": {
                                    "split": "prompt_holdout_isotonic_calibration_mod5",
                                    "training_prompts": len(
                                        {example.prompt_id for example in training_risk}
                                    ),
                                    "calibration_prompts": len(
                                        {example.prompt_id for example in calibration_risk}
                                    ),
                                    "examples": len(calibration_risk),
                                    "positive_fraction": float(np.mean(labels)),
                                    "brier": float(np.mean((scores - labels) ** 2)),
                                    "roc_auc": (
                                        float(roc_auc_score(labels, scores))
                                        if np.unique(labels).size == 2
                                        else None
                                    ),
                                },
                            }
                        },
                        "trace_reused": model in reused_trace_models,
                    }
                }

                training_benefit = tuple(
                    example for group in training_benefit_groups for example in group
                )
                calibration_benefit = tuple(
                    example for group in calibration_benefit_groups for example in group
                )
                all_benefit = tuple(example for group in benefit_groups for example in group)
                evaluation_benefit = RemainingNFEEstimator.fit(
                    training_benefit,
                    include_history=False,
                    history_window=self.config.history_window,
                    seed=self.config.seed,
                )
                predictions = np.asarray(
                    [
                        evaluation_benefit.predict_remaining_nfe(example.features)
                        for example in calibration_benefit
                    ],
                    dtype=np.float64,
                )
                targets = np.asarray(
                    [example.remaining_nfe for example in calibration_benefit],
                    dtype=np.float64,
                )
                final_benefit = RemainingNFEEstimator.fit(
                    all_benefit,
                    include_history=False,
                    history_window=self.config.history_window,
                    seed=self.config.seed,
                )
                benefit_path = self.run_dir / "estimators" / f"{model}_remaining_nfe.joblib"
                saved_benefit = final_benefit.save(benefit_path)
                benefit_metadata[model] = {
                    **saved_benefit,
                    "path": str(benefit_path.relative_to(self.run_dir)),
                    "source": "native_full_budget_exact_loop_traces",
                    "validation": {
                        "split": "deterministic_prompt_holdout_mod5",
                        "examples": len(calibration_benefit),
                        "mae": float(np.mean(np.abs(predictions - targets))),
                        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
                    },
                }
                continue
            grouped_examples = tuple(
                tuple(
                    TrainingExample(
                        features=example["features"],
                        unsafe=bool(example["unsafe"]),
                        prompt_id=f"{row['sample_id']}:{example_index}",
                    )
                    for example_index, example in enumerate(
                        _v4_training_payloads(
                            row,
                            history_window=self.config.history_window,
                        )
                        if self.config.protocol_version in {"v4", "v5"}
                        else row["training_examples"]
                        if "training_examples" in row
                        else ({"features": row["features"], "unsafe": row["unsafe"]},)
                    )
                )
                for row in rows
            )
            examples = tuple(example for group in grouped_examples for example in group)
            if len(grouped_examples) > 1:
                validation_groups = grouped_examples[::5]
                training_groups = tuple(
                    group for index, group in enumerate(grouped_examples) if index % 5 != 0
                )
                if not training_groups:
                    training_groups = grouped_examples
                evaluation_split = "deterministic_prompt_holdout_mod5"
            else:
                training_groups = validation_groups = grouped_examples
                evaluation_split = "in_sample_small_run_fallback"
            training_examples = tuple(example for group in training_groups for example in group)
            validation_examples = tuple(example for group in validation_groups for example in group)
            metadata[model] = {}
            for variant, include_history in variants:
                ablations: dict[str, object] = {}
                for kind in self.config.estimator_kinds:
                    evaluation_estimator = RiskEstimator.fit(
                        training_examples,
                        kind=kind,
                        include_history=include_history,
                        history_window=self.config.history_window,
                        seed=self.config.seed,
                    )
                    scores = np.asarray(
                        [
                            evaluation_estimator.predict_risk(example.features)
                            for example in validation_examples
                        ],
                        dtype=np.float64,
                    )
                    labels = np.asarray(
                        [int(example.unsafe) for example in validation_examples],
                        dtype=np.int64,
                    )
                    final_estimator = RiskEstimator.fit(
                        examples,
                        kind=kind,
                        include_history=include_history,
                        history_window=self.config.history_window,
                        seed=self.config.seed,
                    )
                    is_primary = kind == self.config.estimator_kinds[0]
                    suffix = "" if is_primary else f"_{kind}"
                    path = self.run_dir / "estimators" / f"{model}_{variant}{suffix}.joblib"
                    saved = final_estimator.save(path)
                    validation_auc = (
                        float(roc_auc_score(labels, scores))
                        if np.unique(labels).size == 2
                        else None
                    )
                    ablations[kind] = {
                        **saved,
                        "path": str(path.relative_to(self.run_dir)),
                        "deployment_estimator": is_primary,
                        "validation": {
                            "split": evaluation_split,
                            "examples": len(validation_examples),
                            "positive_fraction": float(np.mean(labels)),
                            "brier": float(np.mean((scores - labels) ** 2)),
                            "roc_auc": validation_auc,
                        },
                    }
                metadata[model][variant] = {
                    "primary_kind": self.config.estimator_kinds[0],
                    "estimators": ablations,
                    "trace_reused": model in reused_trace_models,
                }
        if self.config.protocol_version == "v3":
            for model in self.config.models:
                if model in reused_models:
                    assert self.reuse_development_from is not None
                    source_fit = json.loads(
                        (self.reuse_development_from / "manifests" / "fit.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    rows = _raw_collection_rows(
                        self.reuse_development_from,
                        model=model,
                        expected_identity=source_fit["identity"],
                    )
                    rows = rows[: self.development_limit] if self.development_limit else rows
                    source_label = "validated_v1_full_budget_traces"
                else:
                    rows = self.store.records(f"collect/{model}", "full_budget_shadow")
                    source_label = "v3_full_budget_traces"
                payload_groups = tuple(_training_payloads(row) for row in rows)
                benefit_groups = tuple(
                    tuple(
                        BenefitExample(
                            features=example["features"],
                            remaining_nfe=float(example["remaining_nfe"]),
                            prompt_id=f"{row['sample_id']}:{example_index}",
                        )
                        for example_index, example in enumerate(payloads)
                    )
                    for row, payloads in zip(rows, payload_groups, strict=True)
                )
                if len(benefit_groups) > 1:
                    validation_groups = benefit_groups[::5]
                    training_groups = tuple(
                        group for index, group in enumerate(benefit_groups) if index % 5 != 0
                    )
                    if not training_groups:
                        training_groups = benefit_groups
                    split_name = "deterministic_prompt_holdout_mod5"
                else:
                    training_groups = validation_groups = benefit_groups
                    split_name = "in_sample_small_run_fallback"
                all_examples = tuple(example for group in benefit_groups for example in group)
                training_examples = tuple(example for group in training_groups for example in group)
                validation_examples = tuple(
                    example for group in validation_groups for example in group
                )
                evaluation_estimator = RemainingNFEEstimator.fit(
                    training_examples,
                    include_history=False,
                    history_window=self.config.history_window,
                    seed=self.config.seed,
                )
                predictions = np.asarray(
                    [
                        evaluation_estimator.predict_remaining_nfe(example.features)
                        for example in validation_examples
                    ],
                    dtype=np.float64,
                )
                targets = np.asarray(
                    [example.remaining_nfe for example in validation_examples],
                    dtype=np.float64,
                )
                final_estimator = RemainingNFEEstimator.fit(
                    all_examples,
                    include_history=False,
                    history_window=self.config.history_window,
                    seed=self.config.seed,
                )
                path = self.run_dir / "estimators" / f"{model}_remaining_nfe.joblib"
                saved = final_estimator.save(path)
                benefit_metadata[model] = {
                    **saved,
                    "path": str(path.relative_to(self.run_dir)),
                    "source": source_label,
                    "validation": {
                        "split": split_name,
                        "examples": len(validation_examples),
                        "mae": float(np.mean(np.abs(predictions - targets))),
                        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
                    },
                }
        estimator_manifest = {"models": metadata, "benefit_models": benefit_metadata}
        self.store.write_named("estimators/manifest.json", estimator_manifest)
        self._write_manifest(
            "fit",
            "completed",
            estimators=metadata,
            benefit_estimators=benefit_metadata,
        )

    def _run_rollout(self) -> None:
        self._write_manifest("rollout", "running")
        seed = PolicyCandidateSpec(
            name="seed_local_q500_p2",
            variant="rc_pag_local",
            threshold=0.50,
            min_steps=2,
            patience=2,
        )
        self._run_records(
            stage="rollout",
            refs=self._refs("rollout"),
            methods=(("adablock", None), (seed.name, seed)),
        )
        self._write_manifest("rollout", "completed", seed_policy=asdict(seed))

    def _run_refit(self) -> None:
        self._write_manifest("refit", "running")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "label": "paired_prompt_harm_and_normalized_nfe_reduction",
            "attribution": "all_executed_stops_receive_the_paired_prompt_outcome",
            "seed_policy": "seed_local_q500_p2",
            "models": {},
        }
        for model in self.config.models:
            baseline_rows = {
                row["sample_id"]: row for row in self.store.records(f"rollout/{model}", "adablock")
            }
            seed_rows = {
                row["sample_id"]: row
                for row in self.store.records(f"rollout/{model}", "seed_local_q500_p2")
            }
            if not baseline_rows or set(baseline_rows) != set(seed_rows):
                raise ValueError(f"incomplete paired counterfactual rollout for {model}")
            groups = tuple(
                group
                for sample_id in sorted(baseline_rows)
                if (
                    group := _counterfactual_examples_from_pair(
                        baseline_rows[sample_id],
                        seed_rows[sample_id],
                        history_window=self.config.history_window,
                    )
                )[0]
            )
            if not groups:
                raise ValueError(f"counterfactual rollout has no executed seed stops for {model}")
            training_groups = (
                tuple(group for index, group in enumerate(groups) if index % 5 != 0)
                if len(groups) > 1
                else groups
            )
            validation_groups = groups[::5] if len(groups) > 1 else groups
            if not training_groups:
                training_groups = groups

            def flatten_harm(selected):
                return tuple(example for harm, _ in selected for example in harm)

            def flatten_gain(selected):
                return tuple(example for _, gain in selected for example in gain)

            harm_train = flatten_harm(training_groups)
            gain_train = flatten_gain(training_groups)
            harm_validation = flatten_harm(validation_groups)
            gain_validation = flatten_gain(validation_groups)
            harm_all = flatten_harm(groups)
            gain_all = flatten_gain(groups)
            evaluation_harm = RiskEstimator.fit(
                harm_train,
                kind="hist_gradient_boosting",
                include_history=False,
                history_window=self.config.history_window,
                seed=self.config.seed,
            )
            evaluation_gain = NormalizedNFEReductionEstimator.fit(
                gain_train,
                include_history=False,
                history_window=self.config.history_window,
                seed=self.config.seed,
            )
            harm_scores = np.asarray(
                [evaluation_harm.predict_risk(example.features) for example in harm_validation]
            )
            harm_targets = np.asarray([int(example.unsafe) for example in harm_validation])
            gain_scores = np.asarray(
                [
                    evaluation_gain.predict_remaining_nfe(example.features)
                    for example in gain_validation
                ]
            )
            gain_targets = np.asarray([example.nfe_reduction for example in gain_validation])
            final_harm = RiskEstimator.fit(
                harm_all,
                kind="hist_gradient_boosting",
                include_history=False,
                history_window=self.config.history_window,
                seed=self.config.seed,
            )
            final_gain = NormalizedNFEReductionEstimator.fit(
                gain_all,
                include_history=False,
                history_window=self.config.history_window,
                seed=self.config.seed,
            )
            harm_path = self.run_dir / "estimators" / f"{model}_rc_pag_advantage_harm.joblib"
            gain_path = self.run_dir / "estimators" / f"{model}_rc_pag_advantage_gain.joblib"
            harm_saved = final_harm.save(harm_path)
            gain_saved = final_gain.save(gain_path)
            manifest["models"][model] = {
                "prompt_groups": len(groups),
                "executed_stops": len(harm_all),
                "harm": {
                    **harm_saved,
                    "path": str(harm_path.relative_to(self.run_dir)),
                    "validation_brier": float(np.mean((harm_scores - harm_targets) ** 2)),
                    "validation_auc": (
                        float(roc_auc_score(harm_targets, harm_scores))
                        if np.unique(harm_targets).size == 2
                        else None
                    ),
                },
                "gain": {
                    **gain_saved,
                    "path": str(gain_path.relative_to(self.run_dir)),
                    "validation_mae": float(np.mean(np.abs(gain_scores - gain_targets))),
                },
            }
        self.store.write_named("estimators/advantage_manifest.json", manifest)
        self._write_manifest("refit", "completed", advantage_estimators=manifest["models"])

    def _policy_family_payload(self) -> dict[str, Any]:
        core = {
            "schema_version": 1,
            "config_hash": self.config.config_hash,
            "alpha": self.config.risk.alpha,
            "delta": self.config.risk.delta,
            "loss": self.config.risk.loss,
            "multiplicity_unit": "model_policy_pair",
            "models": list(self.config.models),
            "candidates": [asdict(candidate) for candidate in self.config.candidates],
        }
        if self.config.risk.minimum_nfe_reduction is not None:
            core["minimum_nfe_reduction"] = self.config.risk.minimum_nfe_reduction
        return {**core, "protocol_identity": canonical_config_hash(core)}

    def _run_screen(self) -> None:
        self._write_manifest("screen", "running")
        family = self._policy_family_payload()
        self.store.write_named("policy_family.json", family)
        baseline_methods = tuple(
            (method, None)
            for method in self.config.development_methods
            if method not in {"rc_pag_local", "rc_pag_history", "rc_pag_advantage"}
        )
        candidate_methods = tuple(
            (candidate.name, candidate) for candidate in self.config.candidates
        )
        self._run_records(
            stage="screen",
            refs=self._refs("tuning"),
            methods=baseline_methods + candidate_methods,
        )
        if self.config.protocol_version in _MODERN_PROTOCOLS:
            nonlearned = {method for method, candidate in baseline_methods if candidate is None}
            best_nonlearned: dict[str, str] = {}
            best_nonlearned_mean_nfe: dict[str, float] = {}
            selected_candidate: dict[str, str] = {}
            models_summary: dict[str, dict[str, object]] = {}
            candidate_lookup = {candidate.name: candidate for candidate in self.config.candidates}
            for model in self.config.models:
                baseline_rows = self.store.records(f"screen/{model}", "adablock")
                baseline_by_id = {row["sample_id"]: row for row in baseline_rows}
                allowed_harm = max(1, math.floor(0.02 * len(baseline_rows)))

                def method_stats(
                    name: str,
                    *,
                    model: str = model,
                    baseline_by_id: dict[str, dict[str, Any]] = baseline_by_id,
                    allowed_harm: int = allowed_harm,
                ) -> dict[str, object]:
                    rows = self.store.records(f"screen/{model}", name)
                    by_id = {row["sample_id"]: row for row in rows}
                    if set(by_id) != set(baseline_by_id):
                        raise RuntimeError(
                            f"incomplete modern screening coverage for {model}/{name}"
                        )
                    harmful = sum(
                        bool(baseline_by_id[sample_id].get("is_correct"))
                        and not bool(by_id[sample_id].get("is_correct"))
                        for sample_id in baseline_by_id
                    )
                    return {
                        "mean_nfe": float(np.mean([float(row["total_nfe"]) for row in rows])),
                        "correct": sum(bool(row.get("is_correct")) for row in rows),
                        "harmful_regressions": harmful,
                        "accuracy_eligible": harmful <= allowed_harm,
                    }

                nonlearned_stats = {name: method_stats(name) for name in sorted(nonlearned)}
                eligible_nonlearned = [
                    name
                    for name, stats in nonlearned_stats.items()
                    if bool(stats["accuracy_eligible"])
                ]
                if not eligible_nonlearned:
                    raise ControlledStop(
                        f"no accuracy-eligible nonlearned method survived for {model}"
                    )
                best = min(
                    eligible_nonlearned,
                    key=lambda name: (
                        float(nonlearned_stats[name]["mean_nfe"]),
                        -int(nonlearned_stats[name]["correct"]),
                        name,
                    ),
                )
                best_nonlearned[model] = best
                best_nonlearned_mean_nfe[model] = float(nonlearned_stats[best]["mean_nfe"])

                candidate_stats = {
                    candidate.name: method_stats(candidate.name)
                    for candidate in self.config.candidates
                }
                eligible_candidates = [
                    name
                    for name, stats in candidate_stats.items()
                    if bool(stats["accuracy_eligible"])
                ]
                if not eligible_candidates:
                    raise ControlledStop(f"no risk policy survived tuning for {model}")
                selected_candidate[model] = min(
                    eligible_candidates,
                    key=lambda name: (
                        float(candidate_stats[name]["mean_nfe"]),
                        -int(candidate_stats[name]["correct"]),
                        name,
                    ),
                )
                models_summary[model] = {
                    "allowed_harmful_regressions": allowed_harm,
                    "nonlearned": nonlearned_stats,
                    "candidates": candidate_stats,
                }
            summary = {
                "protocol_version": self.config.protocol_version,
                "best_nonlearned": best_nonlearned,
                "best_nonlearned_mean_nfe": best_nonlearned_mean_nfe,
                "selected_candidate": selected_candidate,
                "models": models_summary,
            }
            frozen = {
                "schema_version": 1,
                "config_hash": self.config.config_hash,
                "risk_loss": self.config.risk.loss,
                "selected_by_model": {
                    model: asdict(candidate_lookup[name])
                    for model, name in selected_candidate.items()
                },
            }
            frozen["protocol_identity"] = canonical_config_hash(frozen)
            self.store.write_named("screening_summary.json", summary)
            self.store.write_named("frozen_policy.json", frozen)
            if self.config.protocol_version in {"v3", "v4", "v5"}:
                readiness_models: dict[str, dict[str, object]] = {}
                for model, selected_name in selected_candidate.items():
                    adablock_nfe = float(
                        models_summary[model]["nonlearned"]["adablock"]["mean_nfe"]
                    )
                    selected_nfe = float(
                        models_summary[model]["candidates"][selected_name]["mean_nfe"]
                    )
                    reduction = (adablock_nfe - selected_nfe) / adablock_nfe
                    beats_nonlearned = selected_nfe < best_nonlearned_mean_nfe[model]
                    passed = reduction >= (
                        self.config.readiness.minimum_tuning_nfe_reduction_per_model
                    ) and (
                        not self.config.readiness.require_candidate_beats_nonlearned
                        or beats_nonlearned
                    )
                    readiness_models[model] = {
                        "selected_candidate": selected_name,
                        "adablock_mean_nfe": adablock_nfe,
                        "candidate_mean_nfe": selected_nfe,
                        "nfe_reduction": reduction,
                        "best_nonlearned": best_nonlearned[model],
                        "best_nonlearned_mean_nfe": best_nonlearned_mean_nfe[model],
                        "beats_best_nonlearned": beats_nonlearned,
                        "passed": passed,
                    }
                readiness = {
                    "schema_version": 1,
                    "minimum_tuning_nfe_reduction_per_model": (
                        self.config.readiness.minimum_tuning_nfe_reduction_per_model
                    ),
                    "require_candidate_beats_nonlearned": (
                        self.config.readiness.require_candidate_beats_nonlearned
                    ),
                    "models": readiness_models,
                    "passed": all(bool(row["passed"]) for row in readiness_models.values()),
                }
                self.store.write_named("readiness_audit.json", readiness)
                if not readiness["passed"]:
                    self._write_manifest(
                        "screen",
                        "failed",
                        summary=summary,
                        frozen_policy=frozen,
                        readiness=readiness,
                    )
                    raise ControlledStop(
                        "workshop-readiness gate failed; calibration and confirmation were not run"
                    )
            self._write_manifest("screen", "completed", summary=summary, frozen_policy=frozen)
            return
        nonlearned = {
            "fixed",
            "adablock",
            "fast_dllm",
            "sched",
            "entropy_sum",
            "confidence_gate",
            "stability_gate",
            "constant_budget",
            "size_lookup",
        }
        method_nfe = {
            method: float(
                np.mean(
                    [
                        float(row["total_nfe"])
                        for model in self.config.models
                        for row in self.store.records(f"screen/{model}", method)
                    ]
                )
            )
            for method in sorted(nonlearned)
        }
        method_correct = {
            method: sum(
                bool(row.get("is_correct", row.get("grade", {}).get("is_correct", False)))
                for model in self.config.models
                for row in self.store.records(f"screen/{model}", method)
            )
            for method in sorted(nonlearned)
        }
        count = sum(
            len(self.store.records(f"screen/{model}", "adablock")) for model in self.config.models
        )
        allowed_loss = max(2, math.floor(0.02 * count))
        baseline_correct = method_correct["adablock"]
        eligible = {
            method: baseline_correct - correct <= allowed_loss
            for method, correct in method_correct.items()
        }
        eligible_names = [method for method in sorted(nonlearned) if eligible[method]]
        if not eligible_names:
            raise ControlledStop("no accuracy-eligible nonlearned method survived screening")
        best_nonlearned = min(
            eligible_names,
            key=lambda name: (method_nfe[name], -method_correct[name], name),
        )
        summary = {
            "best_nonlearned": best_nonlearned,
            "best_nonlearned_mean_nfe": method_nfe[best_nonlearned],
            "method_mean_nfe": method_nfe,
            "method_correct": method_correct,
            "accuracy_eligible": eligible,
            "allowed_correct_loss": allowed_loss,
        }
        self.store.write_named("screening_summary.json", summary)
        self._write_manifest("screen", "completed", summary=summary)

    def _run_calibrate(self) -> None:
        self._write_manifest("calibrate", "running")
        if self.config.protocol_version in _MODERN_PROTOCOLS:
            frozen = json.loads((self.run_dir / "frozen_policy.json").read_text(encoding="utf-8"))
            selected = {
                model: PolicyCandidateSpec(**payload)
                for model, payload in frozen["selected_by_model"].items()
            }
            family = {
                "schema_version": 2,
                "config_hash": self.config.config_hash,
                "alpha": self.config.risk.alpha,
                "delta": self.config.risk.delta,
                "loss": self.config.risk.loss,
                "multiplicity_unit": "frozen_model_policy_pair",
                "selected_by_model": {
                    model: asdict(candidate) for model, candidate in selected.items()
                },
                "selection_source": "development_screen_only",
            }
            if self.config.risk.minimum_nfe_reduction is not None:
                family["minimum_nfe_reduction"] = self.config.risk.minimum_nfe_reduction
            family["protocol_identity"] = canonical_config_hash(family)
            self.store.write_named("calibration_family.json", family)
            refs = self._refs("calibration")
            for model, candidate in selected.items():
                self._run_records(
                    stage="calibrate",
                    refs=refs,
                    methods=(("adablock", None), (candidate.name, candidate)),
                    models=(model,),
                )
            candidates: list[CandidateRisk] = []
            diagnostics: dict[str, dict[str, object]] = {}
            for model, candidate in selected.items():
                baseline_rows = {
                    row["sample_id"]: row
                    for row in self.store.records(f"calibrate/{model}", "adablock")
                }
                candidate_rows = {
                    row["sample_id"]: row
                    for row in self.store.records(f"calibrate/{model}", candidate.name)
                }
                if set(baseline_rows) != set(candidate_rows):
                    raise RuntimeError(f"incomplete paired v2 calibration for {model}")
                losses: list[int] = []
                nfe_savings: list[float] = []
                disagreements = 0
                for sample_id in sorted(baseline_rows):
                    baseline = baseline_rows[sample_id]
                    policy = candidate_rows[sample_id]
                    harmful = int(
                        bool(baseline.get("is_correct")) and not bool(policy.get("is_correct"))
                    )
                    repetitions = int(policy.get("synthetic_repetitions", 1))
                    losses.extend([harmful] * repetitions)
                    baseline_nfe = float(baseline["total_nfe"])
                    policy_nfe = float(policy["total_nfe"])
                    if baseline_nfe <= 0.0:
                        raise ValueError("AdaBlock calibration NFE must be positive")
                    saving = 1.0 - policy_nfe / baseline_nfe
                    if self.config.protocol_version in {"v4", "v5"} and not 0.0 <= saving <= 1.0:
                        raise RuntimeError(
                            "joint early-stop NFE invariant failed for "
                            f"{model}/{sample_id}: {saving}"
                        )
                    nfe_savings.extend([saving] * repetitions)
                    disagreements += int(
                        baseline.get("generated_ids") != policy.get("generated_ids")
                    )
                candidates.append(
                    CandidateRisk(
                        f"{model}/{candidate.name}",
                        tuple(losses),
                        mean_nfe=float(
                            np.mean([float(row["total_nfe"]) for row in candidate_rows.values()])
                        ),
                        protocol_identity=str(family["protocol_identity"]),
                        nfe_savings=(
                            tuple(nfe_savings)
                            if self.config.protocol_version in {"v4", "v5"}
                            else ()
                        ),
                    )
                )
                diagnostics[model] = {
                    "candidate": candidate.name,
                    "paired_prompts": len(baseline_rows),
                    "sequence_disagreements": disagreements,
                    "harmful_regressions": sum(losses),
                    "effective_calibration_count": len(losses),
                    "mean_paired_nfe_reduction": float(np.mean(nfe_savings)),
                }
            certificate = certify_candidates(
                candidates,
                alpha=self.config.risk.alpha,
                delta=self.config.risk.delta,
                minimum_nfe_reduction=(
                    self.config.risk.minimum_nfe_reduction
                    if self.config.protocol_version in {"v4", "v5"}
                    else None
                ),
            )
            payload = certificate.to_dict()
            payload.update(
                {
                    "schema_version": 2,
                    "loss": self.config.risk.loss,
                    "selected_by_model": {
                        model: candidate.name for model, candidate in selected.items()
                    },
                    "diagnostics": diagnostics,
                    "certificate_mode": (
                        "joint_harm_and_compute"
                        if self.config.protocol_version in {"v4", "v5"}
                        else "harm_only"
                    ),
                    "mock": (
                        self.mock_mode
                        if self.mock_mode is not None
                        else all(self._runtime(model).is_mock for model in self.config.models)
                    ),
                }
            )
            self.store.write_named("risk_certificate.json", payload)
            self._write_manifest("calibrate", "completed", certificate=payload)
            return
        family = json.loads((self.run_dir / "policy_family.json").read_text(encoding="utf-8"))
        if family != self._policy_family_payload():
            raise ValueError("frozen policy family differs from the current configuration")
        self._run_records(
            stage="calibrate",
            refs=self._refs("calibration"),
            methods=tuple((candidate.name, candidate) for candidate in self.config.candidates),
        )
        candidates: list[CandidateRisk] = []
        for model in self.config.models:
            for candidate in self.config.candidates:
                rows = self.store.records(f"calibrate/{model}", candidate.name)
                losses: tuple[int, ...] = tuple(
                    int(loss)
                    for row in rows
                    for loss in (row["shadow_losses"] if "shadow_losses" in row else (row["loss"],))
                )
                candidates.append(
                    CandidateRisk(
                        f"{model}/{candidate.name}",
                        losses,
                        mean_nfe=float(np.mean([float(row["total_nfe"]) for row in rows])),
                        protocol_identity=str(family["protocol_identity"]),
                    )
                )
        certificate = certify_candidates(
            candidates,
            alpha=self.config.risk.alpha,
            delta=self.config.risk.delta,
        )
        payload = certificate.to_dict()
        payload["mock"] = (
            self.mock_mode
            if self.mock_mode is not None
            else all(self._runtime(model).is_mock for model in self.config.models)
        )
        self.store.write_named("risk_certificate.json", payload)
        self._write_manifest("calibrate", "completed", certificate=payload)

    def _selected_by_variant(self, certificate: Mapping[str, Any], *, model: str) -> dict[str, str]:
        prefix = f"{model}/"
        certified = {
            str(item["name"])[len(prefix) :]: float(item["mean_nfe"])
            for item in certificate["candidates"]
            if bool(item["certified"]) and str(item["name"]).startswith(prefix)
        }
        selected: dict[str, str] = {}
        for variant in ("rc_pag_local", "rc_pag_history"):
            names = [
                candidate.name
                for candidate in self.config.candidates
                if candidate.variant == variant and candidate.name in certified
            ]
            if names:
                selected[variant] = min(names, key=lambda name: (certified[name], name))
        return selected

    def _run_confirm(self) -> None:
        certificate = json.loads(
            (self.run_dir / "risk_certificate.json").read_text(encoding="utf-8")
        )
        if self.config.protocol_version in _MODERN_PROTOCOLS:
            frozen = json.loads((self.run_dir / "frozen_policy.json").read_text(encoding="utf-8"))
            selected = {
                model: PolicyCandidateSpec(**payload)
                for model, payload in frozen["selected_by_model"].items()
            }
            certified_names = {
                str(item["name"]) for item in certificate["candidates"] if bool(item["certified"])
            }
            required = {f"{model}/{candidate.name}" for model, candidate in selected.items()}
            if certified_names & required != required:
                certificate_name = (
                    "joint harm/compute certificate"
                    if self.config.protocol_version in {"v4", "v5"}
                    else "end-to-end harm certificate"
                )
                raise ControlledStop(f"not every frozen model policy has a {certificate_name}")
            screening = json.loads(
                (self.run_dir / "screening_summary.json").read_text(encoding="utf-8")
            )
            if self.config.protocol_version not in {"v4", "v5"}:
                calibrated_nfe = {
                    str(item["name"]): float(item["mean_nfe"]) for item in certificate["candidates"]
                }
                for model, candidate in selected.items():
                    if calibrated_nfe[f"{model}/{candidate.name}"] >= float(
                        screening["best_nonlearned_mean_nfe"][model]
                    ):
                        raise ControlledStop(
                            f"calibration futility gate: {model} policy did not beat its comparator"
                        )
            self._write_manifest("confirm", "running")
            estimator_paths = {
                path.stem: str(path)
                for path in sorted((self.run_dir / "estimators").glob("*.joblib"))
            }
            for model, candidate in selected.items():
                runtime = self._runtime(model)
                methods = (
                    ("adablock", None),
                    ("best_nonlearned", None),
                    ("rc_pag_selected", candidate),
                )
                for dataset in self.config.confirmatory_counts:
                    stage = f"confirm/{dataset}/{model}"
                    for sample in self._confirm_refs(dataset):
                        for method, method_candidate in methods:
                            if self.store.is_complete(stage, method, sample.sample_id):
                                continue
                            payload = runtime.run(
                                stage="confirm",
                                model=model,
                                sample=sample,
                                method=method,
                                candidate=method_candidate,
                                estimator_paths=estimator_paths,
                            )
                            self.store.write(stage, method, sample.sample_id, payload)
                if not runtime.is_mock:
                    close = getattr(runtime, "close", None)
                    if close is not None:
                        close()
                    self._runtimes.pop(model, None)
            policy_payload = {
                "selected_by_model": {
                    model: asdict(candidate) for model, candidate in selected.items()
                },
                "primary_rc_pag_method": "rc_pag_selected",
                "best_nonlearned": screening["best_nonlearned"],
                "config_hash": self.config.config_hash,
                "risk_loss": self.config.risk.loss,
            }
            self.store.write_named("frozen_confirmatory_policy.json", policy_payload)
            self._write_manifest("confirm", "completed", **policy_payload)
            return
        if bool(certificate["fallback"]):
            raise ControlledStop("no certified policy; confirmation was not started")
        selected_by_model = {
            model: self._selected_by_variant(certificate, model=model)
            for model in self.config.models
        }
        required_variants = {"rc_pag_local", "rc_pag_history"}
        if any(set(selected) != required_variants for selected in selected_by_model.values()):
            raise ControlledStop(
                "no simultaneously certified local and history policy for every model"
            )
        screening = json.loads(
            (self.run_dir / "screening_summary.json").read_text(encoding="utf-8")
        )
        primary_method = (
            "rc_pag_history"
            if "rc_pag_history" in self.config.confirmatory_methods
            else "rc_pag_local"
        )
        selected_names = {
            f"{model}/{selected_by_model[model][primary_method]}" for model in self.config.models
        }
        selected_nfe = float(
            np.mean(
                [
                    float(item["mean_nfe"])
                    for item in certificate["candidates"]
                    if item["name"] in selected_names
                ]
            )
        )
        if selected_nfe >= float(screening["best_nonlearned_mean_nfe"]):
            raise ControlledStop(
                "calibration futility gate: certified policy did not beat the best "
                "nonlearned method"
            )
        self._write_manifest("confirm", "running")
        candidate_lookup = {candidate.name: candidate for candidate in self.config.candidates}
        estimator_paths = {
            path.stem: str(path) for path in sorted((self.run_dir / "estimators").glob("*.joblib"))
        }
        for model in self.config.models:
            selected = selected_by_model[model]
            candidate_by_method = {
                "rc_pag_local": candidate_lookup[selected["rc_pag_local"]],
                "rc_pag_history": candidate_lookup[selected["rc_pag_history"]],
            }
            methods = tuple(
                (method, candidate_by_method.get(method))
                for method in self.config.confirmatory_methods
            )
            runtime = self._runtime(model)
            for dataset in self.config.confirmatory_counts:
                stage = f"confirm/{dataset}/{model}"
                for sample in self._confirm_refs(dataset):
                    for method, candidate in methods:
                        if self.store.is_complete(stage, method, sample.sample_id):
                            continue
                        payload = runtime.run(
                            stage="confirm",
                            model=model,
                            sample=sample,
                            method=method,
                            candidate=candidate,
                            estimator_paths=estimator_paths,
                        )
                        self.store.write(stage, method, sample.sample_id, payload)
            if not runtime.is_mock:
                close = getattr(runtime, "close", None)
                if close is not None:
                    close()
                self._runtimes.pop(model, None)
        self.store.write_named(
            "frozen_confirmatory_policy.json",
            {
                "selected_by_model": selected_by_model,
                "primary_rc_pag_method": primary_method,
                "best_nonlearned": screening["best_nonlearned"],
                "config_hash": self.config.config_hash,
            },
        )
        self._write_manifest("confirm", "completed", selected_by_model=selected_by_model)

    def _run_report(self) -> None:
        self._write_manifest("report", "running")
        from pag.experiments.rc_pag_report import write_rc_pag_report

        records = {
            model: {
                dataset: {
                    method: self.store.records(f"confirm/{dataset}/{model}", method)
                    for method in self.config.confirmatory_methods
                }
                for dataset in self.config.confirmatory_counts
            }
            for model in self.config.models
        }
        counts = {
            f"{model}/{dataset}/{method}": len(records[model][dataset][method])
            for model in self.config.models
            for dataset in self.config.confirmatory_counts
            for method in self.config.confirmatory_methods
        }
        certificate = json.loads(
            (self.run_dir / "risk_certificate.json").read_text(encoding="utf-8")
        )
        estimator_manifest = json.loads(
            (self.run_dir / "estimators" / "manifest.json").read_text(encoding="utf-8")
        )
        screening_summary = json.loads(
            (self.run_dir / "screening_summary.json").read_text(encoding="utf-8")
        )
        payload = {
            "config_hash": self.config.config_hash,
            "certificate": certificate,
            "coverage": counts,
            "estimator_manifest": "estimators/manifest.json",
            "mock": bool(certificate.get("mock", False)),
        }
        self.store.write_named("report/inputs.json", payload)
        audit = write_rc_pag_report(
            self.run_dir,
            records=records,
            certificate=certificate,
            bootstrap_samples=self.config.statistics.bootstrap_samples,
            seed=self.config.seed,
            minimum_accuracy_lower_ci=self.config.claim_gates.minimum_accuracy_lower_ci,
            estimator_manifest=estimator_manifest,
            screening_summary=(
                screening_summary if self.config.protocol_version in _MODERN_PROTOCOLS else None
            ),
            methods=self.config.confirmatory_methods,
            primary_method=(
                "rc_pag_selected"
                if self.config.protocol_version in _MODERN_PROTOCOLS
                else "rc_pag_history"
                if "rc_pag_history" in self.config.confirmatory_methods
                else "rc_pag_local"
            ),
            require_history_frontier_ci=(self.config.claim_gates.require_history_frontier_ci),
        )
        self._write_manifest("report", "completed", inputs=payload, claim_audit=audit)

    def _run_paper(self) -> None:
        self._write_manifest("paper", "running")
        self.store.write_named(
            "paper_manifest.json",
            {
                "config_hash": self.config.config_hash,
                "report_inputs": "report/inputs.json",
                "status": "ready_for_rendering",
            },
        )
        self._write_manifest("paper", "completed")

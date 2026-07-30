from __future__ import annotations

import json
import math
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
from pag.experiments.rc_pag_policy import RiskEstimator, TrainingExample
from pag.experiments.records import RecordStore
from pag.experiments.risk_control import CandidateRisk, certify_candidates


@dataclass(frozen=True, slots=True)
class SampleRef:
    pool: str
    index: int

    @property
    def sample_id(self) -> str:
        return f"{self.pool}-{self.index:05d}"


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
            return {"elapsed_sec": elapsed, "artifact_bytes": 2048, "mock": True}
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
                "mock": True,
            }
        if candidate is not None:
            variant_base = 64.0 if candidate.variant == "rc_pag_local" else 61.0
            threshold_rank = {0.010: 0.0, 0.025: -2.0, 0.050: -4.0}[candidate.threshold]
            nfe = variant_base + threshold_rank + self.candidate_nfe_offset
        else:
            nfe = self._method_nfe(method)
        payload: dict[str, Any] = {
            "elapsed_sec": elapsed,
            "total_nfe": nfe + (sample.index % 3) * 0.1,
            "is_correct": sample.index % 11 != 0,
            "mock": True,
        }
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

    def __init__(
        self,
        config: RCPAGConfig,
        run_dir: str | Path,
        *,
        device: str = "cpu",
        runtime_factory: Callable[[str], RCPAGRuntime] | None = None,
        development_limit: int | None = None,
        mock_mode: bool | None = None,
    ) -> None:
        if development_limit is not None and development_limit < 1:
            raise ValueError("development_limit must be positive")
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = device
        self.development_limit = development_limit
        self.mock_mode = mock_mode
        self.runtime_factory = runtime_factory
        if runtime_factory is None:
            raise ValueError("a runtime_factory is required until a model runtime is configured")
        self.store = RecordStore(
            self.run_dir,
            {
                "protocol": "risk_calibrated_pag_v1",
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
        refs = tuple(SampleRef(dataset, index) for index in range(count))
        if self.development_limit is None:
            return refs
        is_mock = self.mock_mode
        if is_mock is None:
            is_mock = all(self._runtime(model).is_mock for model in self.config.models)
        if not is_mock:
            raise ValueError("confirmatory execution rejects development limits")
        return refs[: self.development_limit]

    def run_through(self, target: str) -> None:
        if target not in self.STAGES:
            raise ValueError(f"unknown stage: {target}")
        for stage in self.STAGES[: self.STAGES.index(target) + 1]:
            self.run_stage(stage)

    def run_stage(self, stage: str) -> None:
        if stage not in self.STAGES:
            raise ValueError(f"unknown stage: {stage}")
        if self._stage_complete(stage):
            return
        position = self.STAGES.index(stage)
        if position and not self._stage_complete(self.STAGES[position - 1]):
            raise ValueError(f"stage {stage} requires completed {self.STAGES[position - 1]}")
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
    ) -> None:
        estimator_paths = {
            path.stem: str(path) for path in sorted((self.run_dir / "estimators").glob("*.joblib"))
        }
        for model in self.config.models:
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
        self._run_records(
            stage="pilot",
            refs=refs,
            methods=(("full_budget", None), ("pilot_shadow", None)),
        )
        records = [
            row
            for model in self.config.models
            for row in self.store.records(f"pilot/{model}", "full_budget")
        ]
        shadow_records = [
            row
            for model in self.config.models
            for row in self.store.records(f"pilot/{model}", "pilot_shadow")
        ]
        if len(shadow_records) != len(records):
            raise RuntimeError("pilot shadow coverage does not match full-budget coverage")
        if not all(bool(row.get("mock")) for row in shadow_records) and any(
            not row.get("shadow_losses") for row in shadow_records
        ):
            raise RuntimeError("real pilot did not exercise a same-state shadow continuation")
        seconds = float(np.mean([float(row["elapsed_sec"]) for row in records]))
        artifact_bytes = float(np.mean([float(row.get("artifact_bytes", 0)) for row in records]))
        baseline_screen_methods = sum(
            method not in {"rc_pag_local", "rc_pag_history"}
            for method in self.config.development_methods
        )
        runs_per_model = {
            "collect": self.config.stage_sizes.traces_per_model,
            "screen": (baseline_screen_methods + len(self.config.candidates))
            * self.config.stage_sizes.tuning_per_model,
            "calibrate": len(self.config.candidates)
            * self.config.stage_sizes.calibration_per_model,
            "confirm": sum(self.config.confirmatory_counts.values())
            * len(self.config.confirmatory_methods),
        }
        full_gpu_runs = len(self.config.models) * sum(runs_per_model.values())
        projection = {
            "basis": "pilot wall time; rerun after the 32-prompt real A100 pilot",
            "seconds_per_sample": seconds,
            "bytes_per_sample": artifact_bytes,
            "projected_runs_per_model_by_stage": runs_per_model,
            "projected_gpu_runs": full_gpu_runs,
            "projected_a100_hours": seconds * full_gpu_runs / 3600,
            "projected_storage_bytes": int(math.ceil(artifact_bytes * full_gpu_runs)),
            "mock": any(bool(row.get("mock")) for row in records),
        }
        self.store.write_named("compute_projection.json", projection)
        self._write_manifest("pilot", "completed", projection=projection)

    def _run_collect(self) -> None:
        self._write_manifest("collect", "running")
        self._run_records(
            stage="collect",
            refs=self._refs("training"),
            methods=(("full_budget_shadow", None),),
        )
        self._write_manifest("collect", "completed")

    def _run_fit(self) -> None:
        self._write_manifest("fit", "running")
        metadata: dict[str, dict[str, object]] = {}
        for model in self.config.models:
            rows = self.store.records(f"collect/{model}", "full_budget_shadow")
            if not rows:
                raise ValueError(f"no collected examples for {model}")
            grouped_examples = tuple(
                tuple(
                    TrainingExample(
                        features=example["features"],
                        unsafe=bool(example["unsafe"]),
                        prompt_id=f"{row['sample_id']}:{example_index}",
                    )
                    for example_index, example in enumerate(
                        row["training_examples"]
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
            for variant, include_history in (
                ("rc_pag_local", False),
                ("rc_pag_history", True),
            ):
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
                }
        self.store.write_named("estimators/manifest.json", {"models": metadata})
        self._write_manifest("fit", "completed", estimators=metadata)

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
        return {**core, "protocol_identity": canonical_config_hash(core)}

    def _run_screen(self) -> None:
        self._write_manifest("screen", "running")
        family = self._policy_family_payload()
        self.store.write_named("policy_family.json", family)
        baseline_methods = tuple(
            (method, None)
            for method in self.config.development_methods
            if method not in {"rc_pag_local", "rc_pag_history"}
        )
        candidate_methods = tuple(
            (candidate.name, candidate) for candidate in self.config.candidates
        )
        self._run_records(
            stage="screen",
            refs=self._refs("tuning"),
            methods=baseline_methods + candidate_methods,
        )
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
        selected_names = {
            f"{model}/{selected_by_model[model]['rc_pag_history']}" for model in self.config.models
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
            methods = (
                ("adablock", None),
                ("best_nonlearned", None),
                ("rc_pag_local", candidate_lookup[selected["rc_pag_local"]]),
                ("rc_pag_history", candidate_lookup[selected["rc_pag_history"]]),
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

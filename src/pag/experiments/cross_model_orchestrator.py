from __future__ import annotations

import gc
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pag.experiments.budget import BudgetGuard
from pag.experiments.config import inclusive_range
from pag.experiments.cross_model_config import CrossModelConfig
from pag.experiments.cross_model_runtime import (
    CrossModelRuntime,
    UnifiedCrossModelRuntime,
    derive_budget_stats,
    records_to_trace_sequences,
)
from pag.experiments.datasets import (
    ExperimentSample,
    complement_math500,
    materialize_fresh_gsm8k,
)
from pag.experiments.orchestrator import ControlledStop, environment_metadata
from pag.experiments.records import RecordStore
from pag.experiments.residual import ResidualEstimator
from pag.experiments.runtime import sha256_file


def run_missing_records(
    *,
    store: RecordStore,
    runtime: CrossModelRuntime,
    stage: str,
    samples: Sequence[ExperimentSample],
    methods: Sequence[str],
    policy: dict[str, Any] | None = None,
    before_run: Callable[[], None] | None = None,
) -> None:
    for sample in samples:
        baseline = (
            store.read(stage, "adablock", sample.sample_id)
            if store.is_complete(stage, "adablock", sample.sample_id)
            else None
        )
        for method in methods:
            if store.is_complete(stage, method, sample.sample_id):
                if method == "adablock":
                    baseline = store.read(stage, method, sample.sample_id)
                continue
            if method != "adablock" and baseline is None:
                raise ValueError(f"{stage}/{method} requires paired AdaBlock output")
            if before_run is not None:
                before_run()
            payload = runtime.run(
                sample,
                method,
                baseline=baseline,
                policy=policy,
            )
            store.write(stage, method, sample.sample_id, payload)
            if method == "adablock":
                baseline = store.read(stage, method, sample.sample_id)


def select_joint_policy(
    records: dict[str, dict[str, list[dict[str, object]]]],
    *,
    max_correct_loss: int,
) -> dict[str, object]:
    models = tuple(sorted(records))
    if not models:
        raise ValueError("selection requires model records")
    candidates = set(records[models[0]]) - {"adablock"}
    if any((set(records[model]) - {"adablock"}) != candidates for model in models):
        raise ValueError("selection candidates differ across models")
    rows: list[dict[str, object]] = []
    for candidate in sorted(candidates):
        losses: dict[str, int] = {}
        normalized_nfe: list[float] = []
        for model in models:
            baseline = records[model]["adablock"]
            proposed = records[model][candidate]
            if {row["sample_id"] for row in baseline} != {row["sample_id"] for row in proposed}:
                raise ValueError(f"incomplete calibration coverage for {model}/{candidate}")
            baseline_correct = sum(bool(row["grade"]["is_correct"]) for row in baseline)
            candidate_correct = sum(bool(row["grade"]["is_correct"]) for row in proposed)
            losses[model] = baseline_correct - candidate_correct
            normalized_nfe.append(
                float(np.mean([float(row["total_nfe"]) for row in proposed]))
                / float(np.mean([float(row["total_nfe"]) for row in baseline]))
            )
        eligible = all(loss <= max_correct_loss for loss in losses.values())
        rows.append(
            {
                "method": candidate,
                "correct_loss": losses,
                "joint_nfe_ratio": float(np.mean(normalized_nfe)),
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    fallback = not eligible_rows
    selected = (
        min(rows, key=lambda row: (sum(row["correct_loss"].values()), row["joint_nfe_ratio"]))
        if fallback
        else min(eligible_rows, key=lambda row: (row["joint_nfe_ratio"], row["method"]))
    )
    return {"selected": selected["method"], "fallback": fallback, "candidates": rows}


def load_cross_model_samples(
    config: CrossModelConfig,
) -> tuple[
    tuple[ExperimentSample, ...], tuple[ExperimentSample, ...], tuple[ExperimentSample, ...]
]:
    from datasets import load_dataset

    gsm = load_dataset(
        config.gsm8k.path,
        config.gsm8k.config,
        revision=config.gsm8k.revision,
    )
    fresh = materialize_fresh_gsm8k(
        gsm["train"],
        calibration=inclusive_range(config.gsm8k.calibration_indices),
        test=inclusive_range(config.gsm8k.test_indices),
    )
    math_rows = load_dataset(
        config.math500.path,
        revision=config.math500.revision,
        split="test",
    )
    manifest_path = Path(config.math500.prior_selection_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = {str(value) for value in manifest["math500"]}
    math = complement_math500(math_rows, excluded_ids=excluded)
    if len(math) != config.math500.expected_complement:
        raise ValueError(
            f"expected {config.math500.expected_complement} MATH-500 records, got {len(math)}"
        )
    return fresh.calibration, fresh.test, math


def _candidate_name(quantile: float, correction: int) -> str:
    return f"residual_q{round(quantile * 100):02d}_c{correction}"


def _candidate_parameters(name: str) -> tuple[float, int]:
    try:
        quantile_text, correction_text = name.removeprefix("residual_q").split("_c", 1)
        return int(quantile_text) / 100, int(correction_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid residual candidate: {name}") from exc


class CrossModelOrchestrator:
    def __init__(
        self,
        *,
        config: CrossModelConfig,
        run_dir: str | Path,
        device: str,
        llada_trace_path: str | Path,
        budget_usd: float | None = None,
        gpu_rate: float | None = None,
        runtime_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = device
        self.llada_trace_path = Path(llada_trace_path)
        self.store = RecordStore(
            self.run_dir,
            {"config_hash": config.config_hash, "protocol": "cross_model_residual_pag_v1"},
        )
        self.guard = BudgetGuard(
            budget_usd=config.budget_usd if budget_usd is None else budget_usd,
            hourly_rate=config.gpu_rate if gpu_rate is None else gpu_rate,
            reserve_fraction=config.reserve_fraction,
        )
        self.runtime_factory = runtime_factory or (
            lambda model: UnifiedCrossModelRuntime(
                model_name=model,
                config=config,
                device=device,
                llada_trace_path=self.llada_trace_path,
            )
        )
        self.seconds_per_run: float | None = None

    def _manifest(self, stage: str, status: str, **extra: object) -> None:
        self.store.write_manifest(
            {
                "stage": stage,
                "status": status,
                "config_hash": self.config.config_hash,
                "elapsed_seconds": self.guard.elapsed_seconds,
                "estimated_spend_usd": self.guard.estimated_spend_usd,
                **extra,
            }
        )

    def _admit(self, *, stage: str, remaining_runs: int) -> None:
        if self.seconds_per_run is None or remaining_runs < 1:
            return
        decision = self.guard.can_start(
            stage=stage,
            projected_seconds=self.seconds_per_run * remaining_runs * 1.25,
        )
        if not decision.allowed:
            self._manifest(stage, "controlled_stop", reason=decision.reason)
            raise ControlledStop(
                f"stopped before {stage}: projected spend ${decision.estimated_spend_usd:.2f}"
            )

    def _calibrate_model(
        self,
        model: str,
        samples: Sequence[ExperimentSample],
    ) -> None:
        stage = f"calibration/{model}"
        runtime = self.runtime_factory(model)
        started = self.guard.elapsed_seconds
        run_missing_records(
            store=self.store,
            runtime=runtime,
            stage=stage,
            samples=samples,
            methods=("adablock",),
        )
        baseline = self.store.records(stage, "adablock")
        if self.seconds_per_run is None and baseline:
            elapsed = self.guard.elapsed_seconds - started
            self.seconds_per_run = elapsed / len(baseline) if elapsed > 0 else 1.0
        stats = derive_budget_stats(baseline)
        sequences = records_to_trace_sequences(baseline)
        estimator = ResidualEstimator.fit(
            sequences,
            stats,
            seed=self.config.seed,
            n_estimators=self.config.policy.n_estimators,
            window_size=self.config.policy.window_size,
        )
        estimator_path = self.run_dir / "estimators" / f"{model}.joblib"
        estimator.save(estimator_path)
        runtime.configure(stats=stats, estimator=estimator)
        candidate_count = len(self.config.policy.quantiles) * len(
            self.config.policy.max_abs_corrections
        )
        self._admit(
            stage=stage,
            remaining_runs=len(samples) * (1 + candidate_count),
        )
        run_missing_records(
            store=self.store,
            runtime=runtime,
            stage=stage,
            samples=samples,
            methods=("adablock", "size_lookup"),
        )
        for quantile in self.config.policy.quantiles:
            for correction in self.config.policy.max_abs_corrections:
                method = _candidate_name(quantile, correction)
                runtime.configure(
                    stats=stats,
                    estimator=estimator,
                    quantile=quantile,
                    max_abs_correction=correction,
                )
                run_missing_records(
                    store=self.store,
                    runtime=runtime,
                    stage=stage,
                    samples=samples,
                    methods=("adablock", method),
                )
        self.store.write_named(
            f"estimators/{model}.json",
            {
                "path": str(estimator_path),
                "sha256": sha256_file(estimator_path),
                "stats": asdict(stats),
            },
        )
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _freeze_policy(self) -> dict[str, Any]:
        candidate_names = [
            _candidate_name(quantile, correction)
            for quantile in self.config.policy.quantiles
            for correction in self.config.policy.max_abs_corrections
        ]
        records = {
            model: {
                "adablock": self.store.records(f"calibration/{model}", "adablock"),
                **{
                    candidate: self.store.records(f"calibration/{model}", candidate)
                    for candidate in candidate_names
                },
            }
            for model in self.config.models
        }
        selection = select_joint_policy(records, max_correct_loss=2)
        selected = str(selection["selected"])
        quantile, correction = _candidate_parameters(selected)
        frozen = {
            **selection,
            "quantile": quantile,
            "max_abs_correction": correction,
            "config_hash": self.config.config_hash,
            "estimators": {
                model: json.loads(
                    (self.run_dir / "estimators" / f"{model}.json").read_text(encoding="utf-8")
                )
                for model in self.config.models
            },
        }
        self.store.write_named("frozen_policy.json", frozen)
        return frozen

    def _run_confirmatory(
        self,
        frozen: dict[str, Any],
        gsm_test: Sequence[ExperimentSample],
        math500: Sequence[ExperimentSample],
    ) -> None:
        for model in self.config.models:
            runtime = self.runtime_factory(model)
            metadata = frozen["estimators"][model]
            raw_stats = metadata["stats"]
            from pag.experiments.residual import TraceBudgetStats

            stats = TraceBudgetStats(
                content_median=int(raw_stats["content_median"]),
                delimiter_median=int(raw_stats["delimiter_median"]),
                by_size={int(key): int(value) for key, value in raw_stats["by_size"].items()},
            )
            estimator = ResidualEstimator.load(metadata["path"])
            runtime.configure(
                stats=stats,
                estimator=estimator,
                quantile=float(frozen["quantile"]),
                max_abs_correction=int(frozen["max_abs_correction"]),
            )
            for stage_name, samples in (("test_gsm8k", gsm_test), ("test_math500", math500)):
                stage = f"{stage_name}/{model}"
                remaining = len(samples) * len(self.config.confirmatory_methods)
                self._admit(stage=stage, remaining_runs=remaining)
                run_missing_records(
                    store=self.store,
                    runtime=runtime,
                    stage=stage,
                    samples=samples,
                    methods=self.config.confirmatory_methods,
                )
            del runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self) -> Path:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        if not self.llada_trace_path.is_file():
            raise FileNotFoundError(f"LLaDA trace not found: {self.llada_trace_path}")
        self.store.write_named("environment.json", environment_metadata(self.device))
        calibration, gsm_test, math500 = load_cross_model_samples(self.config)
        self.store.write_named(
            "samples.json",
            {
                "calibration": [sample.sample_id for sample in calibration],
                "gsm8k_test": [sample.sample_id for sample in gsm_test],
                "math500": [sample.sample_id for sample in math500],
            },
        )
        for model in self.config.models:
            self._manifest(f"calibration/{model}", "running")
            self._calibrate_model(model, calibration)
            self._manifest(f"calibration/{model}", "complete")
        frozen = self._freeze_policy()
        self._manifest("freeze_policy", "complete", selected=frozen["selected"])
        self._run_confirmatory(frozen, gsm_test, math500)
        self._manifest("report", "pending")
        return self.run_dir

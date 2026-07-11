from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import signal
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pag.experiments.budget import BudgetGuard
from pag.experiments.config import ExperimentConfig
from pag.experiments.datasets import ExperimentSample, GSM8KSplits, load_datasets
from pag.experiments.grading import grade_math500
from pag.experiments.records import RecordStore
from pag.experiments.report import write_report
from pag.experiments.runtime import ExperimentRuntime, sha256_file


@dataclass(frozen=True, slots=True)
class PreflightResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    environment: dict[str, object]


class ControlledStop(RuntimeError):
    """Raised after safely recording a budget or signal stop."""


@dataclass(frozen=True, slots=True)
class StageWork:
    remaining_runs: int
    observed_seconds_per_run: float | None


def inspect_stage_work(
    store: RecordStore,
    stage: str,
    samples: Sequence[ExperimentSample],
    methods: Sequence[str],
) -> StageWork:
    remaining = 0
    observed: list[float] = []
    for method in methods:
        for record in store.records(stage, method):
            elapsed = record.get("elapsed_sec")
            if isinstance(elapsed, (int, float)) and elapsed > 0:
                observed.append(float(elapsed))
        for sample in samples:
            if not store.is_complete(stage, method, sample.sample_id):
                remaining += 1
    return StageWork(
        remaining_runs=remaining,
        observed_seconds_per_run=float(np.median(observed)) if observed else None,
    )


def select_history_free(
    records: dict[str, list[dict[str, Any]]], *, max_correct_loss: int
) -> dict[str, object]:
    baseline_correct = sum(bool(row["grade"]["is_correct"]) for row in records["adablock"])
    candidates: list[dict[str, object]] = []
    for method in ("constant_budget", "size_lookup"):
        rows = records[method]
        correct = sum(bool(row["grade"]["is_correct"]) for row in rows)
        candidates.append(
            {
                "method": method,
                "correct": correct,
                "correct_loss": baseline_correct - correct,
                "mean_nfe": float(np.mean([float(row["total_nfe"]) for row in rows])),
                "eligible": baseline_correct - correct <= max_correct_loss,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    fallback = not eligible
    selected = (
        next(row for row in candidates if row["method"] == "constant_budget")
        if fallback
        else min(
            eligible,
            key=lambda row: (
                float(row["mean_nfe"]),
                -int(row["correct"]),
                str(row["method"]),
            ),
        )
    )
    return {"selected": selected["method"], "fallback": fallback, "candidates": candidates}


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_metadata(device: str) -> dict[str, object]:
    packages = {}
    for name in ("torch", "transformers", "datasets", "math-verify", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    metadata: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
        "device": device,
        "torch_cuda": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        metadata["gpu"] = {"name": properties.name, "total_memory": int(properties.total_memory)}
    return metadata


def run_preflight(
    *,
    config: ExperimentConfig,
    predictor_ckpt: Path,
    trace_path: Path,
    output_root: Path,
    device: str,
) -> PreflightResult:
    del config
    errors: list[str] = []
    warnings: list[str] = []
    if not predictor_ckpt.is_file():
        errors.append(f"predictor checkpoint not found: {predictor_ckpt}")
    else:
        try:
            checkpoint = torch.load(predictor_ckpt, map_location="cpu", weights_only=False)
            expected_fields = [
                "block_size",
                "nfe",
                "mean_top1_confidence",
                "min_top1_confidence",
                "digit_fraction",
                "delimiter_fraction",
            ]
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("input_fields") != expected_fields
            ):
                errors.append("predictor checkpoint has an incompatible six-feature schema")
        except Exception as exc:
            errors.append(f"predictor checkpoint could not be loaded: {type(exc).__name__}: {exc}")
    if not trace_path.is_file():
        errors.append(f"training trace not found: {trace_path}")
    else:
        try:
            with trace_path.open(encoding="utf-8") as file_obj:
                trace_rows = [json.loads(line) for line in file_obj if line.strip()]
            trace_ids = [str(row.get("sample_id")) for row in trace_rows]
            if (
                len(trace_ids) != 5000
                or trace_ids[0] != "gsm8k-train-0000"
                or trace_ids[-1] != "gsm8k-train-4999"
            ):
                errors.append("training trace must contain exactly gsm8k-train-0000 through 4999")
        except Exception as exc:
            errors.append(f"training trace could not be validated: {type(exc).__name__}: {exc}")
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < 10 * 1024**3:
        errors.append("artifact filesystem has less than 10 GiB free")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            errors.append("CUDA device requested but torch.cuda.is_available() is false")
        elif int(torch.cuda.get_device_properties(0).total_memory) < 40 * 1024**3:
            errors.append("strategy 1 requires a GPU with at least 40 GiB memory")
    else:
        warnings.append("CPU mode is suitable only for dry runs and mocked tests")
    fixture = grade_math500("Final answer: \\boxed{1/2}", "\\frac{2}{4}")
    if not fixture.is_correct:
        errors.append(f"Math-Verify fixture failed: {fixture.error}")
    return PreflightResult(not errors, tuple(errors), tuple(warnings), environment_metadata(device))


class StrategyOneOrchestrator:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        model_path: str,
        predictor_ckpt: Path,
        trace_path: Path,
        device: str,
        output_root: Path,
        run_id: str,
        budget_usd: float,
        gpu_rate: float,
        dtype: str | None = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.predictor_ckpt = predictor_ckpt
        self.trace_path = trace_path
        self.device = device
        self.dtype = dtype
        self.run_dir = output_root / run_id
        self.store = RecordStore(
            self.run_dir,
            {
                "config_hash": config.config_hash,
                "model_path": model_path,
                "checkpoint_sha256": sha256_file(predictor_ckpt),
            },
        )
        self.guard = BudgetGuard(
            budget_usd=budget_usd,
            hourly_rate=gpu_rate,
            reserve_fraction=config.budget.reserve_fraction,
        )
        self.stop_requested = False
        self.runtime: ExperimentRuntime | None = None
        self.seconds_per_run: float | None = None
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        def request_stop(signum, frame):
            del signum, frame
            self.stop_requested = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    def _manifest(self, stage: str, status: str, **extra: object) -> None:
        self.store.write_manifest(
            {
                "run_id": self.run_dir.name,
                "stage": stage,
                "status": status,
                "identity": self.store.identity,
                "resolved_config": self.config.raw,
                "elapsed_seconds": self.guard.elapsed_seconds,
                "estimated_spend_usd": self.guard.estimated_spend_usd,
                **extra,
            }
        )

    def _admit(
        self,
        stage: str,
        remaining_runs: int,
        *,
        observed_seconds_per_run: float | None = None,
    ) -> None:
        seconds_per_run = observed_seconds_per_run or self.seconds_per_run
        if seconds_per_run is None or remaining_runs == 0:
            return
        decision = self.guard.can_start(
            stage=stage,
            projected_seconds=seconds_per_run * remaining_runs * 1.25,
        )
        if not decision.allowed:
            self._manifest(stage, "controlled_stop", reason=decision.reason)
            raise ControlledStop(
                f"stopped before {stage}: projected spend ${decision.estimated_spend_usd:.2f}"
            )

    def _run_record(
        self,
        stage: str,
        method: str,
        sample: ExperimentSample,
        baseline: dict[str, Any] | None,
        *,
        record_id: str | None = None,
        measure_memory: bool = False,
    ) -> dict[str, Any]:
        if self.runtime is None:
            raise RuntimeError("runtime is not loaded")
        key = record_id or sample.sample_id
        if self.store.is_complete(stage, method, key):
            return self.store.read(stage, method, key)
        if self.stop_requested:
            raise ControlledStop("signal received")
        seed = None
        if baseline is not None:
            seed = (int(baseline["block_history"][0]), int(baseline["nfe_history"][0]))
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                result = self.runtime.run(
                    sample, method, baseline_seed=seed, measure_memory=measure_memory
                ).to_dict()
                if record_id is not None:
                    result["source_sample_id"] = result["sample_id"]
                    result["sample_id"] = record_id
                self.store.write(stage, method, key, result)
                return self.store.read(stage, method, key)
            except Exception as exc:
                last_error = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt == 0:
                    continue
        raise RuntimeError(f"{stage}/{method}/{key} failed twice") from last_error

    def _run_methods(
        self,
        stage: str,
        samples: Sequence[ExperimentSample],
        methods: Sequence[str],
    ) -> None:
        work = inspect_stage_work(self.store, stage, samples, methods)
        self._admit(
            stage,
            work.remaining_runs,
            observed_seconds_per_run=work.observed_seconds_per_run,
        )
        if work.remaining_runs == 0:
            self._manifest(
                stage, "complete", resumed=True, skipped_runs=len(samples) * len(methods)
            )
            print(f"[{stage}] already complete; all records verified", flush=True)
            return
        self._manifest(stage, "running")
        for index, sample in enumerate(samples, start=1):
            baseline = self._run_record(stage, "adablock", sample, None)
            for method in methods:
                if method != "adablock":
                    self._run_record(stage, method, sample, baseline)
            print(
                f"[{stage}] {index}/{len(samples)} {sample.sample_id} "
                f"elapsed={self.guard.elapsed_seconds / 3600:.2f}h "
                f"cost=${self.guard.estimated_spend_usd:.2f}",
                flush=True,
            )
        self._manifest(stage, "complete")

    def _timing_samples(self, gsm: GSM8KSplits) -> list[ExperimentSample]:
        baseline = self.store.records("gsm8k_test", "adablock")
        by_id = {sample.sample_id: sample for sample in gsm.full_test}
        ordered = sorted(baseline, key=lambda row: (float(row["total_nfe"]), row["sample_id"]))
        indices = np.linspace(0, len(ordered) - 1, self.config.timing.prompts, dtype=int)
        return [by_id[str(ordered[index]["sample_id"])] for index in indices]

    def _run_timing(self, gsm: GSM8KSplits) -> None:
        if self.runtime is None:
            raise RuntimeError("runtime is not loaded")
        samples = self._timing_samples(gsm)
        self._admit("timing", len(samples) * self.config.timing.repetitions * 2)
        self._manifest("timing", "running")
        for sample in samples[: self.config.timing.warmups]:
            baseline = self.runtime.run(sample, "adablock")
            self.runtime.run(sample, "pag", baseline_seed=baseline.first_block_seed)
        for repetition in range(self.config.timing.repetitions):
            for index, sample in enumerate(samples):
                order = (
                    ("adablock", "pag") if (index + repetition) % 2 == 0 else ("pag", "adablock")
                )
                baseline = self.store.read("gsm8k_test", "adablock", sample.sample_id)
                for method in order:
                    record_id = f"{sample.sample_id}::rep{repetition}"
                    row = self._run_record(
                        "timing",
                        method,
                        sample,
                        None if method == "adablock" else baseline,
                        record_id=record_id,
                        measure_memory=True,
                    )
                    if method == "adablock":
                        baseline = row
        self._manifest("timing", "complete")

    def run(self) -> Path:
        preflight = run_preflight(
            config=self.config,
            predictor_ckpt=self.predictor_ckpt,
            trace_path=self.trace_path,
            output_root=self.run_dir,
            device=self.device,
        )
        self.store.write_named("preflight/result.json", asdict(preflight))
        if not preflight.ok:
            self._manifest("preflight", "failed", errors=list(preflight.errors))
            raise RuntimeError("preflight failed: " + "; ".join(preflight.errors))
        self.store.write_named("environment.json", preflight.environment)
        gsm, math500 = load_datasets(self.config)
        self.store.write_named(
            "selected_samples.json",
            {
                "development": list(gsm.development_ids),
                "confirmatory": [sample.sample_id for sample in gsm.confirmatory],
                "math500": [sample.sample_id for sample in math500],
            },
        )
        self.runtime = ExperimentRuntime(
            config=self.config,
            model_path=self.model_path,
            predictor_ckpt=self.predictor_ckpt,
            trace_path=self.trace_path,
            device=self.device,
            dtype=self.dtype,
        )
        smoke_started = self.guard.elapsed_seconds
        for sample in gsm.development[:2]:
            baseline = self.runtime.run(sample, "adablock")
            self.runtime.run(sample, "pag", baseline_seed=baseline.first_block_seed)
        self.seconds_per_run = (self.guard.elapsed_seconds - smoke_started) / 4
        self._run_methods("development", gsm.development, self.config.methods.development)
        development = {
            method: self.store.records("development", method)
            for method in self.config.methods.development
        }
        selection = select_history_free(
            development, max_correct_loss=self.config.promotion.max_correct_loss
        )
        self.store.write_named("selection.json", selection)
        final_methods = tuple(
            dict.fromkeys((*self.config.methods.final_required, str(selection["selected"])))
        )
        self._run_methods("gsm8k_test", gsm.full_test, final_methods)
        self._run_methods("math500", math500, self.config.methods.math500)
        self._run_timing(gsm)
        method_map = {
            "development": self.config.methods.development,
            "gsm8k_test": final_methods,
            "math500": self.config.methods.math500,
            "timing": ("adablock", "pag"),
        }
        stages = {
            stage: {method: self.store.records(stage, method) for method in methods}
            for stage, methods in method_map.items()
        }
        write_report(
            self.run_dir / "report",
            stages=stages,
            bootstrap_samples=self.config.statistics.bootstrap_samples,
            seed=self.config.seed,
            confirmatory_ids={sample.sample_id for sample in gsm.confirmatory},
        )
        self._manifest("report", "complete")
        return self.run_dir


def protocol_summary(
    config: ExperimentConfig, budget_usd: float, gpu_rate: float
) -> dict[str, object]:
    return {
        "stages": [
            "preflight",
            "development",
            "promotion",
            "gsm8k_test",
            "math500",
            "timing",
            "report",
        ],
        "development_methods": list(config.methods.development),
        "final_required": list(config.methods.final_required),
        "math500_samples": config.math500.sample_size,
        "confirmatory_gsm8k_samples": 919,
        "full_gsm8k_samples": 1319,
        "budget_usd": budget_usd,
        "usable_budget_usd": budget_usd * (1 - config.budget.reserve_fraction),
        "gpu_rate": gpu_rate,
    }

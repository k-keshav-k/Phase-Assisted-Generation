from __future__ import annotations

import importlib
import statistics
import sys
import time
import types
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import torch

from pag.experiments.config import load_experiment_config
from pag.experiments.cross_model_config import CrossModelConfig
from pag.experiments.datasets import ExperimentSample
from pag.experiments.grading import grade_gsm8k, grade_math500
from pag.experiments.residual import (
    ResidualBudgetScheduler,
    ResidualEstimator,
    SizeLookupScheduler,
    TraceBudgetStats,
)
from pag.experiments.runtime import ExperimentRuntime

REPO_ROOT = Path(__file__).resolve().parents[3]
DREAM_DIR = REPO_ROOT / "AdaBlock-dLLM" / "dream"


class CrossModelRuntime(Protocol):
    def run(
        self,
        sample: ExperimentSample,
        method: str,
        *,
        baseline: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def records_to_trace_sequences(
    records: Sequence[dict[str, Any]],
) -> list[list[dict[str, float]]]:
    sequences: list[list[dict[str, float]]] = []
    for record in records:
        blocks = list(record.get("block_history", ()))
        nfes = list(record.get("nfe_history", ()))
        schedules = list(record.get("schedule_history", ()))
        if len(blocks) != len(nfes):
            raise ValueError(f"block/NFE length mismatch for {record.get('sample_id')}")
        if schedules and len(schedules) != len(blocks):
            raise ValueError(f"schedule length mismatch for {record.get('sample_id')}")
        sequence: list[dict[str, float]] = []
        for index, (block_size, nfe) in enumerate(zip(blocks, nfes, strict=True)):
            schedule = schedules[index] if schedules else {}
            sequence.append(
                {
                    "block_size": float(block_size),
                    "nfe": float(nfe),
                    "mean_top1_confidence": float(schedule.get("mean_top1_confidence", 1.0)),
                    "min_top1_confidence": float(schedule.get("min_top1_confidence", 1.0)),
                    "digit_fraction": float(schedule.get("digit_fraction", 0.0)),
                    "delimiter_fraction": float(
                        schedule.get(
                            "delimiter_fraction",
                            1.0 if int(block_size) == 1 else 0.0,
                        )
                    ),
                }
            )
        if sequence:
            sequences.append(sequence)
    if not sequences:
        raise ValueError("records contain no block traces")
    return sequences


def derive_budget_stats(records: Sequence[dict[str, Any]]) -> TraceBudgetStats:
    sequences = records_to_trace_sequences(records)
    content: list[int] = []
    delimiters: list[int] = []
    by_size: dict[int, list[int]] = {}
    for sequence in sequences:
        for row in sequence:
            block_size = max(1, int(row["block_size"]))
            nfe = max(1, int(row["nfe"]))
            is_delimiter = block_size == 1 or row["delimiter_fraction"] >= 0.5
            (delimiters if is_delimiter else content).append(nfe)
            by_size.setdefault(block_size, []).append(nfe)
    if not content:
        raise ValueError("calibration traces contain no content blocks")
    if not delimiters:
        delimiters = [1]
    return TraceBudgetStats(
        content_median=max(1, round(statistics.median(content))),
        delimiter_median=max(1, round(statistics.median(delimiters))),
        by_size={
            block_size: max(1, round(statistics.median(values)))
            for block_size, values in by_size.items()
        },
    )


class UnifiedCrossModelRuntime:
    def __init__(
        self,
        *,
        model_name: str,
        config: CrossModelConfig,
        device: str,
        llada_trace_path: str | Path,
    ) -> None:
        if model_name not in config.models:
            raise ValueError(f"unknown model: {model_name}")
        self.model_name = model_name
        self.config = config
        self.device = device
        self.stats: TraceBudgetStats | None = None
        self.estimator: ResidualEstimator | None = None
        self.quantile = 0.25
        self.max_abs_correction = 2
        if model_name == "llada":
            strategy = load_experiment_config(
                REPO_ROOT / "configs" / "experiments" / "neurips_strategy1.yaml"
            )
            self.backend = ExperimentRuntime(
                config=strategy,
                model_path=config.models[model_name],
                predictor_ckpt=REPO_ROOT / "unused-residual-checkpoint.pt",
                trace_path=llada_trace_path,
                device=device,
            )
        else:
            self.backend = self._load_dream_backend()

    def configure(
        self,
        *,
        stats: TraceBudgetStats,
        estimator: ResidualEstimator | None = None,
        quantile: float = 0.25,
        max_abs_correction: int = 2,
    ) -> None:
        self.stats = stats
        self.estimator = estimator
        self.quantile = float(quantile)
        self.max_abs_correction = int(max_abs_correction)
        if self.model_name == "llada" and estimator is not None:
            self.backend.configure_residual_policy(
                stats=stats,
                estimator=estimator,
                quantile=quantile,
                max_abs_correction=max_abs_correction,
            )
        elif self.model_name == "llada":
            variants = importlib.import_module("scheduler_variants")
            self.backend.trace_stats = variants.TraceBudgetStats(
                content_median=stats.content_median,
                delimiter_median=stats.delimiter_median,
                by_size=dict(stats.by_size),
            )

    def _load_dream_backend(self):
        if str(DREAM_DIR) not in sys.path:
            sys.path.insert(0, str(DREAM_DIR))
        dream_class = importlib.import_module("eval_dream_adablock").Dream
        backend = dream_class(
            pretrained=self.config.models["dream"],
            device=self.device,
            max_new_tokens=256,
            diffusion_steps=64,
            temperature=0.0,
            alg="confidence_threshold",
            threshold=0.9,
            use_cache=True,
            dual_cache=True,
            block_length=32,
            delimiter_threshold=0.3,
        )
        digit_ids: list[int] = []
        for token_id in range(int(getattr(backend.tokenizer, "vocab_size", 0))):
            text = backend.tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if text and all(character.isdigit() for character in text):
                digit_ids.append(token_id)
        backend.model.pag_digit_ids = torch.tensor(
            digit_ids,
            dtype=torch.long,
            device=backend.device,
        )
        backend.model.pag_delimiter_ids = torch.tensor(
            (
                198,
                271,
                280,
                319,
                340,
                382,
                401,
                532,
                624,
                630,
                692,
                698,
                921,
                1248,
                1837,
                1939,
                2219,
                2533,
                3276,
                3876,
                4894,
                5267,
                14750,
                68327,
            ),
            dtype=torch.long,
            device=backend.device,
        )
        return backend

    def _dream_baseline(self, sample: ExperimentSample):
        mixin = importlib.import_module("model.generation_utils_adablock").DreamGenerationMixin
        self.backend.model.diffusion_generate = types.MethodType(
            mixin.diffusion_generate,
            self.backend.model,
        )
        self.backend.model._sample = types.MethodType(
            mixin._sample_adablock_cache,
            self.backend.model,
        )
        self.backend.model._compute_block_length = types.MethodType(
            mixin._compute_block_length,
            self.backend.model,
        )
        responses, nfe_history, block_history = self.backend._generate_batch([sample.prompt])
        schedules = [
            {
                "block_index": index,
                "applied_block_size": int(block_size),
                "actual_nfe_used": int(nfe),
            }
            for index, (block_size, nfe) in enumerate(zip(block_history, nfe_history, strict=True))
        ]
        return responses[0], nfe_history, block_history, schedules, 0.0

    def _dream_controlled(
        self,
        sample: ExperimentSample,
        scheduler: SizeLookupScheduler | ResidualBudgetScheduler,
    ):
        mixin = importlib.import_module("model.generation_utils_pag").DreamGenerationMixin
        adablock_mixin = importlib.import_module(
            "model.generation_utils_adablock"
        ).DreamGenerationMixin
        model = self.backend.model
        model.diffusion_generate = types.MethodType(mixin.diffusion_generate, model)
        model._sample = types.MethodType(mixin._sample_pag_cache, model)
        model._compute_block_length = types.MethodType(
            adablock_mixin._compute_block_length,
            model,
        )
        model.pag_scheduler = scheduler
        prompt = sample.prompt
        if self.backend.add_bos_token:
            prompt = self.backend.tokenizer.bos_token + prompt
        prompt_ids = self.backend.tokenizer(
            [prompt], return_tensors="pt", padding=True, padding_side="left"
        ).input_ids.to(self.backend.device)
        attention_mask = prompt_ids.ne(self.backend.tokenizer.pad_token_id)
        result = model.diffusion_generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.backend.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.backend.diffusion_steps,
            temperature=self.backend.temperature,
            top_p=self.backend.top_p,
            top_k=self.backend.top_k,
            alg=self.backend.alg,
            alg_temp=self.backend.alg_temp,
            threshold=self.backend.threshold,
            dual_cache=True,
            block_length=self.backend.block_length,
            max_block_length=self.backend.max_new_tokens,
            max_refinement_steps=self.backend.diffusion_steps,
            delimiter_threshold=self.backend.delimiter_threshold,
        )
        generated = result.sequences[0][prompt_ids.shape[1] :]
        text = self.backend.tokenizer.decode(generated.tolist(), skip_special_tokens=True).split(
            self.backend.tokenizer.eos_token
        )[0]
        return (
            text,
            result.nfe_history,
            result.block_history,
            result.schedule_history,
            float(scheduler.scheduler_predict_time_sec),
        )

    def run(
        self,
        sample: ExperimentSample,
        method: str,
        *,
        baseline: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del policy
        runtime_method = "residual_pag" if method.startswith("residual_") else method
        if self.model_name == "llada":
            seed = None
            if baseline is not None:
                seed = (int(baseline["block_history"][0]), int(baseline["nfe_history"][0]))
            payload = self.backend.run(sample, runtime_method, baseline_seed=seed).to_dict()
            payload["method"] = method
            payload["model"] = self.model_name
            return payload
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        if runtime_method == "adablock":
            text, nfes, blocks, schedules, predict_time = self._dream_baseline(sample)
        else:
            if baseline is None or self.stats is None:
                raise ValueError(f"{method} requires baseline and calibration stats")
            seed_nfe = int(baseline["nfe_history"][0])
            if runtime_method == "size_lookup":
                scheduler = SizeLookupScheduler(seed_budget=seed_nfe, stats=self.stats)
            elif runtime_method == "residual_pag" and self.estimator is not None:
                scheduler = ResidualBudgetScheduler(
                    seed_budget=seed_nfe,
                    stats=self.stats,
                    estimator=self.estimator,
                    quantile=self.quantile,
                    max_abs_correction=self.max_abs_correction,
                )
            else:
                raise ValueError(f"unsupported Dream method: {method}")
            text, nfes, blocks, schedules, predict_time = self._dream_controlled(sample, scheduler)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        grade = (
            grade_gsm8k(text, sample.gold_answer)
            if sample.dataset == "gsm8k"
            else grade_math500(text, sample.gold_answer)
        )
        return {
            "model": self.model_name,
            "method": method,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "generated_text": text,
            "grade": asdict(grade),
            "total_nfe": int(sum(nfes)),
            "nfe_history": [int(value) for value in nfes],
            "block_history": [int(value) for value in blocks],
            "schedule_history": schedules,
            "elapsed_sec": elapsed,
            "scheduler_predict_time_sec": predict_time,
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else 0
            ),
            "peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if self.device.startswith("cuda") else 0
            ),
        }

from __future__ import annotations

import hashlib
import importlib
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pag.experiments.config import ExperimentConfig
from pag.experiments.datasets import ExperimentSample
from pag.experiments.grading import grade_gsm8k, grade_math500
from pag.experiments.residual import (
    ResidualBudgetScheduler,
    ResidualEstimator,
    TraceBudgetStats,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    method: str
    sample_id: str
    dataset: str
    generated_text: str
    grade: dict[str, object]
    total_nfe: int
    nfe_history: list[int]
    block_history: list[int]
    schedule_history: list[dict[str, object]]
    elapsed_sec: float
    scheduler_predict_time_sec: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    seed_block_size: int
    seed_nfe: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def first_block_seed(self) -> tuple[int, int]:
        if not self.block_history or not self.nfe_history:
            raise ValueError("generation returned no first-block seed")
        return self.block_history[0], self.nfe_history[0]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentRuntime:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        model_path: str,
        predictor_ckpt: str | Path,
        trace_path: str | Path,
        device: str,
        dtype: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.predictor_ckpt = Path(predictor_ckpt)
        self.trace_path = Path(trace_path)
        self.device = device
        if model is None or tokenizer is None:
            loader = importlib.import_module("run_pag_dummy_api")._load_model_and_tokenizer
            model, tokenizer = loader(model_path, device, dtype, disable_torch_compile=True)
        self.model = model
        self.tokenizer = tokenizer
        variants = importlib.import_module("scheduler_variants")
        self.trace_sequences = variants.load_trace_sequences(self.trace_path)
        self.trace_stats = variants.derive_trace_budget_stats(self.trace_sequences)
        self._rf_scheduler = None
        self._residual_estimator: ResidualEstimator | None = None
        self._residual_stats: TraceBudgetStats | None = None
        self._residual_quantile = 0.25
        self._residual_max_abs_correction = 2
        self.digit_ids_tensor, self.delimiter_ids_tensor = self._token_type_tensors()

    def configure_residual_policy(
        self,
        *,
        stats: TraceBudgetStats,
        estimator: ResidualEstimator,
        quantile: float,
        max_abs_correction: int,
    ) -> None:
        self._residual_stats = stats
        self._residual_estimator = estimator
        self._residual_quantile = float(quantile)
        self._residual_max_abs_correction = int(max_abs_correction)
        variants = importlib.import_module("scheduler_variants")
        self.trace_stats = variants.TraceBudgetStats(
            content_median=stats.content_median,
            delimiter_median=stats.delimiter_median,
            by_size=dict(stats.by_size),
        )

    def _token_type_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        digit_ids: set[int] = set()
        delimiter_ids = set(self.config.decoding.delimiter_ids)
        for token_id in range(int(getattr(self.tokenizer, "vocab_size", 0))):
            text = self.tokenizer.decode([token_id]).strip()
            if text and all(character.isdigit() for character in text):
                digit_ids.add(token_id)
            if text in {"\n", "<|endoftext|>", "<|eot_id|>"}:
                delimiter_ids.add(token_id)
        return (
            torch.tensor(sorted(digit_ids), dtype=torch.long, device=self.device),
            torch.tensor(sorted(delimiter_ids), dtype=torch.long, device=self.device),
        )

    def _prompt_ids(self, sample: ExperimentSample) -> torch.Tensor:
        prompt = sample.prompt
        if "instruct" in self.model_path.lower():
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return torch.tensor(
            self.tokenizer(prompt)["input_ids"], dtype=torch.long, device=self.device
        ).unsqueeze(0)

    def _scheduler(self, method: str, *, seed: tuple[int, int]):
        variants = importlib.import_module("scheduler_variants")
        seed_size, seed_nfe = seed
        if method == "gates_only":
            return variants.GatesOnlyScheduler()
        if method == "constant_budget":
            return variants.ConstantBudgetScheduler(
                seed_budget=seed_nfe,
                content_budget=self.trace_stats.content_median,
                delimiter_budget=self.trace_stats.delimiter_median,
            )
        if method == "size_lookup":
            return variants.SizeLookupBudgetScheduler(seed_budget=seed_nfe, stats=self.trace_stats)
        if method == "previous_nfe":
            return variants.PreviousNFEScheduler(seed_budget=seed_nfe)
        if method == "random_forest":
            if self._rf_scheduler is None:
                self._rf_scheduler = variants.RandomForestBudgetScheduler.fit(
                    self.trace_sequences, seed_budget=seed_nfe, seed=self.config.seed
                )
            self._rf_scheduler.seed_budget = seed_nfe
            self._rf_scheduler.reset()
            return self._rf_scheduler
        if method == "residual_pag":
            if self._residual_stats is None or self._residual_estimator is None:
                raise ValueError("residual_pag requires a configured estimator")
            return ResidualBudgetScheduler(
                seed_budget=seed_nfe,
                stats=self._residual_stats,
                estimator=self._residual_estimator,
                quantile=self._residual_quantile,
                max_abs_correction=self._residual_max_abs_correction,
            )
        if method in {"pag", "pag_hard_cap"}:
            scheduler_class = importlib.import_module("pag_predictor").PAGTupleScheduler
            return scheduler_class(
                predictor_ckpt=self.predictor_ckpt,
                seed_block_length=seed_size,
                seed_refinement_steps=seed_nfe,
                predictor_device="cpu",
                min_refinement_steps=1,
                min_block_length=1,
                refinement_step_offset=0,
            )
        raise ValueError(f"unsupported controlled method: {method}")

    def _synchronize(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def run(
        self,
        sample: ExperimentSample,
        method: str,
        *,
        baseline_seed: tuple[int, int] | None = None,
        measure_memory: bool = False,
    ) -> GenerationRecord:
        input_ids = self._prompt_ids(sample)
        if measure_memory and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        self._synchronize()
        started = time.perf_counter()
        scheduler_predict_time = 0.0
        if method == "adablock":
            generator = importlib.import_module("generate_adablock").generate_adablock_dual_cache
            output_ids, nfe_history, block_history = generator(
                self.model,
                input_ids,
                steps=self.config.decoding.steps,
                gen_length=self.config.decoding.gen_length,
                init_block_length=32,
                temperature=self.config.decoding.temperature,
                remasking="low_confidence",
                mask_id=126336,
                threshold=self.config.decoding.threshold,
                delimiter_ids=list(self.config.decoding.delimiter_ids),
                delimiter_threshold=self.config.decoding.delimiter_threshold,
            )
            schedule_history = [
                {
                    "block_index": index,
                    "applied_block_size": block_size,
                    "actual_nfe_used": nfe,
                    "exit_reason": "adablock_complete",
                }
                for index, (block_size, nfe) in enumerate(
                    zip(block_history, nfe_history, strict=True)
                )
            ]
        else:
            if baseline_seed is None:
                raise ValueError(f"{method} requires the paired AdaBlock first-block seed")
            scheduler = self._scheduler(method, seed=baseline_seed)
            generator = importlib.import_module("generate_pag").generate_pag_dual_cache
            output_ids, nfe_history, block_history, schedule_history = generator(
                self.model,
                input_ids,
                scheduler,
                steps=self.config.decoding.steps,
                gen_length=self.config.decoding.gen_length,
                temperature=self.config.decoding.temperature,
                remasking="low_confidence",
                mask_id=126336,
                threshold=self.config.decoding.threshold,
                max_block_length=self.config.decoding.gen_length,
                max_refinement_steps=self.config.decoding.steps,
                digit_ids_tensor=self.digit_ids_tensor,
                delimiter_ids_tensor=self.delimiter_ids_tensor,
                delimiter_ids=list(self.config.decoding.delimiter_ids),
                delimiter_threshold=self.config.decoding.delimiter_threshold,
                tau_commit=self.config.decoding.tau_commit,
                tau_stable_steps=self.config.decoding.tau_stable_steps,
                default_block_length=32,
                enforcement_mode="hard_cap" if method == "pag_hard_cap" else "soft_gate",
            )
            scheduler_predict_time = float(getattr(scheduler, "scheduler_predict_time_sec", 0.0))
        self._synchronize()
        elapsed = time.perf_counter() - started
        generated_text = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        grade_result = (
            grade_gsm8k(generated_text, sample.gold_answer)
            if sample.dataset == "gsm8k"
            else grade_math500(generated_text, sample.gold_answer)
        )
        peak_allocated = (
            int(torch.cuda.max_memory_allocated())
            if measure_memory and self.device.startswith("cuda")
            else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved())
            if measure_memory and self.device.startswith("cuda")
            else 0
        )
        seed_size, seed_nfe = baseline_seed or (block_history[0], nfe_history[0])
        return GenerationRecord(
            method=method,
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            generated_text=generated_text,
            grade=asdict(grade_result),
            total_nfe=sum(int(value) for value in nfe_history),
            nfe_history=[int(value) for value in nfe_history],
            block_history=[int(value) for value in block_history],
            schedule_history=schedule_history,
            elapsed_sec=elapsed,
            scheduler_predict_time_sec=scheduler_predict_time,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            seed_block_size=int(seed_size),
            seed_nfe=int(seed_nfe),
        )

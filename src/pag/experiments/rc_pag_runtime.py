from __future__ import annotations

import gc
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from pag.experiments.grading import GradeResult, grade_gsm8k, grade_math500
from pag.experiments.rc_pag_config import PolicyCandidateSpec, RCPAGConfig
from pag.experiments.rc_pag_features import (
    RealizedBlock,
    StepObservation,
    extract_features,
)
from pag.experiments.rc_pag_orchestrator import SampleRef
from pag.experiments.rc_pag_policy import (
    CalibratedRiskEstimator,
    NormalizedNFEReductionEstimator,
    RemainingNFEEstimator,
    RiskEstimator,
    RiskStoppingPolicy,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
DREAM_DIR = REPO_ROOT / "AdaBlock-dLLM" / "dream"
MASK_IDS = {"llada": 126336, "dream": 151666}


def _ensure_llada_config_compatibility(config: Any) -> Any:
    """Bridge fields added by the bundled AdaBlock LLaDA model fork."""
    if not hasattr(config, "train_max_sequence_length"):
        if not hasattr(config, "max_sequence_length"):
            raise ValueError("LLaDA config has no maximum sequence length")
        config.train_max_sequence_length = int(config.max_sequence_length)
    return config


def _import_llada_model_class_without_compile() -> Any:
    """Import LLaDA with eager SDPA so Triton is not a runtime requirement."""
    original_compile = getattr(torch, "compile", None)
    if original_compile is None:
        return importlib.import_module("model.modeling_llada").LLaDAModelLM

    def _identity_torch_compile(fn=None, *args, **kwargs):
        del args, kwargs
        if fn is None:
            return _identity_torch_compile
        return fn

    torch.compile = _identity_torch_compile
    try:
        return importlib.import_module("model.modeling_llada").LLaDAModelLM
    finally:
        torch.compile = original_compile


def _observation(payload: Mapping[str, Any]) -> StepObservation:
    return StepObservation.from_arrays(
        step_index=int(payload["step_index"]),
        block_size=int(payload["block_size"]),
        masked=payload["masked"],
        top1_probs=payload["top1_probs"],
        top2_probs=payload["top2_probs"],
        entropies=payload["entropies"],
        token_ids=payload["token_ids"],
        temporal_js=payload.get("temporal_js"),
        digit_ids=payload.get("digit_ids", ()),
        delimiter_ids=payload.get("delimiter_ids", ()),
    )


def training_examples_from_schedules(
    schedules: Sequence[Mapping[str, Any]],
    *,
    history_window: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    history: list[RealizedBlock] = []
    for block in schedules:
        previous = None
        final_tokens = tuple(int(value) for value in block.get("final_tokens", ()))
        for raw_step in block.get("risk_steps", ()):
            observation = _observation(raw_step["observation"])
            proposed = tuple(int(value) for value in raw_step["proposed_tokens"])
            if not final_tokens or len(proposed) != len(final_tokens):
                raise ValueError("training trace is missing same-block final tokens")
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
        raise ValueError("generation produced no RC-PAG training observations")
    return examples


def prompt_loss_from_schedules(schedules: Sequence[Mapping[str, Any]]) -> int:
    losses = [
        int(loss)
        for block in schedules
        for loss in block.get("shadow_losses", ())
        if loss is not None
    ]
    if not losses:
        if any(block.get("risk_steps") for block in schedules):
            return 0
        raise ValueError("calibration generation produced no policy decisions")
    if any(loss not in (0, 1) for loss in losses):
        raise ValueError("on-policy shadow labels must be binary")
    return int(any(losses))


@dataclass(frozen=True, slots=True)
class _PhaseTuple:
    block_size: int
    refinement_steps: int


@dataclass(frozen=True, slots=True)
class _Schedule:
    predicted_tuple: _PhaseTuple
    applied_block_size: int
    budgeted_refinement_steps: int


class _BudgetScheduler:
    def __init__(
        self,
        *,
        budget: int,
        override_block_size: int | None = None,
        size_scaled: bool = False,
    ) -> None:
        self.budget = int(budget)
        self.override_block_size = override_block_size
        self.size_scaled = bool(size_scaled)
        self.scheduler_predict_time_sec = 0.0
        self.reset()

    def reset(self) -> None:
        self.history = []

    def next_schedule(
        self,
        *,
        block_size: int,
        remaining_tokens: int,
        max_block_length: int,
        max_refinement_steps: int,
    ) -> _Schedule:
        size = min(
            remaining_tokens,
            max_block_length,
            self.override_block_size or max(1, int(block_size)),
        )
        budget = (
            max(2, min(max_refinement_steps, round(2 * math.sqrt(size))))
            if self.size_scaled
            else min(max_refinement_steps, max(1, self.budget))
        )
        return _Schedule(_PhaseTuple(size, budget), size, budget)

    def record_realized(self, block_size: int, nfe: int, *metrics: float) -> None:
        self.history.append((block_size, nfe, *metrics))


class _TracePolicy:
    def reset_prompt(self) -> None:
        self.history = []

    def start_block(self) -> None:
        pass

    def observe(self, observation: StepObservation):
        del observation
        return SimpleNamespace(
            should_stop=False,
            risk_score=1.0,
            safe_streak=0,
            reason="collect_full_trajectory",
        )

    def record_realized(self, block: RealizedBlock) -> None:
        self.history.append(block)


class _PilotStopPolicy(_TracePolicy):
    """Force one early decision so the pilot exercises the shadow/cache path."""

    def observe(self, observation: StepObservation):
        should_stop = observation.step_index >= 2
        return SimpleNamespace(
            should_stop=should_stop,
            risk_score=0.0 if should_stop else 1.0,
            safe_streak=1 if should_stop else 0,
            reason="pilot_shadow_stop" if should_stop else "continue",
        )


class _HeuristicPolicy(_TracePolicy):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.reset_prompt()

    def start_block(self) -> None:
        self.previous: StepObservation | None = None
        self.safe_streak = 0

    def observe(self, observation: StepObservation):
        masked = [index for index, value in enumerate(observation.masked) if value]
        confidences = [observation.top1_probs[index] for index in masked]
        entropies = [observation.entropies[index] for index in masked]
        min_confidence = min(confidences, default=1.0)
        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        temporal_js = [observation.temporal_js[index] for index in masked]
        max_temporal_js = max(temporal_js, default=0.0)
        churn = 1.0
        if self.previous is not None:
            churn = float(
                np.mean(np.asarray(observation.token_ids) != np.asarray(self.previous.token_ids))
            )
        progress = 1.0 - sum(observation.masked) / observation.block_size
        if self.kind == "fast_dllm_style":
            safe = min_confidence >= 0.90
        elif self.kind == "sched_style":
            safe = progress >= 0.75 and mean_entropy <= 1.0
        elif self.kind == "entropy_sum":
            safe = sum(entropies) <= max(0.5, 0.25 * observation.block_size)
        elif self.kind == "entropy_sum_gate":
            safe = sum(entropies) <= max(0.5, 0.25 * observation.block_size)
        elif self.kind == "confidence_gate":
            safe = min_confidence >= 0.95
        elif self.kind == "stability_gate":
            safe = churn == 0.0
        elif self.kind == "mutual_stability_gate":
            safe = min_confidence >= 0.90 and churn == 0.0 and mean_entropy <= 1.0
        elif self.kind == "stability_weighted_style":
            safe = progress >= 0.50 and min_confidence >= 0.80 and max_temporal_js <= 0.05
        elif self.kind == "token_convergence_style":
            safe = (
                progress >= 0.50
                and min_confidence >= 0.75
                and churn == 0.0
                and max_temporal_js <= 0.02
            )
        elif self.kind == "pag":
            safe = min_confidence >= 0.88 and churn <= 0.10 and observation.step_index >= 2
        elif self.kind == "residual_pag":
            history_nfe = self.history[-1].nfe if self.history else 8
            safe = observation.step_index >= max(2, history_nfe - 1) and min_confidence >= 0.85
        else:
            raise ValueError(f"unsupported heuristic policy: {self.kind}")
        self.safe_streak = self.safe_streak + 1 if safe else 0
        self.previous = observation
        needs_patience = self.kind in {
            "stability_gate",
            "mutual_stability_gate",
            "stability_weighted_style",
            "token_convergence_style",
        }
        stop = self.safe_streak >= (2 if needs_patience else 1)
        return SimpleNamespace(
            should_stop=stop,
            risk_score=float(max(0.0, min(1.0, 1 - min_confidence))),
            safe_streak=self.safe_streak,
            reason=f"{self.kind}_stop" if stop else "continue",
        )


def _activate_model_package(directory: Path) -> None:
    for name in tuple(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    for value in (str(LLADA_DIR), str(DREAM_DIR)):
        while value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(directory))


def _extract_code(text: str) -> str:
    fences = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fences[-1] if fences else text).strip()


def _grade_code(text: str, sample: Mapping[str, Any]) -> GradeResult:
    code = _extract_code(text)
    tests = str(sample.get("tests", ""))
    setup = str(sample.get("setup", ""))
    entry_point = str(sample.get("entry_point", ""))
    if not code or not tests:
        return GradeResult(False, None, "tests", "missing generated code or tests")
    check = f"\ncheck({entry_point})\n" if entry_point and "def check(" in tests else "\n"
    source = f"{setup}\n{code}\n{tests}{check}"
    try:
        with tempfile.TemporaryDirectory(prefix="rc_pag_code_") as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(source, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
                env={"PATH": os.environ.get("PATH", "")},
            )
        return GradeResult(
            completed.returncode == 0,
            code,
            "tests",
            None if completed.returncode == 0 else completed.stderr[-1000:],
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GradeResult(False, code, "tests", f"{type(exc).__name__}: {exc}")


class UnifiedRCPAGRuntime:
    is_mock = False

    def __init__(
        self,
        *,
        config: RCPAGConfig,
        model: str,
        device: str,
        run_dir: str | Path,
    ) -> None:
        if model not in config.models:
            raise ValueError(f"unknown RC-PAG model: {model}")
        self.config = config
        self.model_name = model
        self.device = device
        self.run_dir = Path(run_dir)
        self.model = None
        self.tokenizer = None
        self._pools: dict[str, Any] = {}
        self.digit_ids: torch.Tensor | None = None
        self.delimiter_ids: torch.Tensor | None = None

    def preflight(self, *, model: str, spec, device: str) -> Mapping[str, Any]:
        errors = []
        if model != self.model_name:
            errors.append("runtime/model mismatch")
        if not device.startswith("cuda"):
            errors.append("real RC-PAG execution requires CUDA")
        elif not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
        elif torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3:
            errors.append("RC-PAG requires at least 40 GiB GPU memory")
        try:
            from huggingface_hub import model_info

            info = model_info(spec.repository, revision=spec.revision)
            if info.sha != spec.revision:
                errors.append("resolved model revision differs from the frozen SHA")
        except Exception as exc:
            errors.append(f"model revision could not be resolved: {type(exc).__name__}: {exc}")
        return {
            "ok": not errors,
            "mock": False,
            "repository": spec.repository,
            "revision": spec.revision,
            "errors": errors,
        }

    def _load_pool(self, name: str):
        if name in self._pools:
            return self._pools[name]
        from datasets import concatenate_datasets, load_dataset

        spec = self.config.datasets[name]
        parts = [
            load_dataset(
                spec.path,
                dataset_config,
                split=spec.split,
                revision=spec.revision,
            )
            for dataset_config in spec.configs
        ]
        rows = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        if name.endswith("_train"):
            rows = rows.shuffle(seed=self.config.seed)
        self._pools[name] = rows
        return rows

    def _sample(self, ref: SampleRef) -> dict[str, Any]:
        row = dict(self._load_pool(ref.pool)[ref.index])
        if ref.pool.startswith("gsm8k"):
            answer = str(row["answer"])
            gold = answer.rsplit("####", 1)[-1].strip()
            prompt = (
                f"{row['question']}\n\nSolve the problem step by step. "
                "End with a line formatted exactly as Final answer: <number>."
            )
            kind = "gsm8k"
        elif ref.pool.startswith("math"):
            prompt = (
                f"{row['problem']}\n\nSolve the problem step by step and put the final answer "
                "inside \\boxed{}."
            )
            gold = str(row.get("answer", row.get("solution", "")))
            kind = "math"
        elif ref.pool.startswith("mbpp"):
            description = row.get("text", row.get("prompt", ""))
            examples = "\n".join(str(value) for value in row.get("test_list", ()))
            examples_prompt = (
                f"\nThe function must satisfy these examples:\n{examples}"
                if self.config.protocol_version in {"v2", "v3", "v4", "v5", "v6"} and examples
                else ""
            )
            prompt = f"Write a correct Python solution for this task:\n{description}"
            prompt += f"{examples_prompt}\nReturn code only."
            gold = "tests"
            kind = "code"
            row["tests"] = "\n".join(row.get("test_list", ()))
            row["setup"] = str(row.get("test_setup_code", ""))
        else:
            prompt = str(row["prompt"])
            gold = "tests"
            kind = "code"
            row["tests"] = str(row["test"])
            row["entry_point"] = str(row["entry_point"])
        return {"prompt": prompt, "gold": gold, "kind": kind, "row": row}

    def _ensure_backend(self) -> None:
        if self.model is not None:
            return
        spec = self.config.models[self.model_name]
        if self.model_name == "llada":
            _activate_model_package(LLADA_DIR)
            from transformers import AutoConfig, AutoTokenizer

            model_class = _import_llada_model_class_without_compile()
            model_config = AutoConfig.from_pretrained(
                spec.repository, revision=spec.revision, trust_remote_code=True
            )
            _ensure_llada_config_compatibility(model_config)
            model_config.flash_attention = True
            self.model = (
                model_class.from_pretrained(
                    spec.repository,
                    revision=spec.revision,
                    config=model_config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
                .to(self.device)
                .eval()
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                spec.repository, revision=spec.revision, trust_remote_code=True
            )
            delimiter = [198]
        else:
            _activate_model_package(DREAM_DIR)
            from transformers import AutoTokenizer

            model_class = importlib.import_module("model.modeling_dream").DreamModel
            self.model = (
                model_class.from_pretrained(
                    spec.repository,
                    revision=spec.revision,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
                .to(self.device)
                .eval()
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                spec.repository, revision=spec.revision, trust_remote_code=True
            )
            delimiter = [198, 271, 280, 624, 151645]
        digit_ids = set()
        for digit in "0123456789":
            digit_ids.update(self.tokenizer.encode(digit, add_special_tokens=False))
            digit_ids.update(self.tokenizer.encode(f" {digit}", add_special_tokens=False))
        self.digit_ids = torch.tensor(sorted(digit_ids), dtype=torch.long, device=self.device)
        self.delimiter_ids = torch.tensor(delimiter, dtype=torch.long, device=self.device)

    def _prompt_ids(self, prompt: str) -> torch.Tensor:
        assert self.tokenizer is not None
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer(rendered, return_tensors="pt").input_ids.to(self.device)

    def _risk_policy(
        self,
        candidate: PolicyCandidateSpec,
        estimator_paths: Mapping[str, str],
    ) -> RiskStoppingPolicy:
        uses_budgeted_heads = (
            self.config.protocol_version == "v6" and candidate.variant == "rc_pag_budgeted"
        )
        uses_advantage_heads = (
            self.config.protocol_version == "v5" and candidate.variant == "rc_pag_advantage"
        )
        if uses_budgeted_heads:
            risk_key = f"{self.model_name}_rc_pag_budgeted_risk"
            benefit_key = f"{self.model_name}_remaining_nfe"
            missing = [key for key in (risk_key, benefit_key) if key not in estimator_paths]
            if missing:
                raise ValueError(f"missing fitted v6 estimators: {', '.join(missing)}")
            estimator = CalibratedRiskEstimator.load(estimator_paths[risk_key])
            benefit = RemainingNFEEstimator.load(estimator_paths[benefit_key])
            required_features = {"local.temporal_js_mean", "local.temporal_js_max"}
            if not required_features.issubset(estimator.names) or not required_features.issubset(
                benefit.names
            ):
                raise ValueError("v6 estimator is missing the temporal-JS feature schema")
            if (
                estimator.include_history
                or benefit.include_history
                or estimator.kind != "hist_gradient_boosting"
            ):
                raise ValueError("v6 requires frozen local histogram-boosting estimators")
            if tuple(estimator.names) != tuple(benefit.names):
                raise ValueError("v6 risk and benefit estimator feature schemas differ")
        elif uses_advantage_heads:
            harm_key = f"{self.model_name}_rc_pag_advantage_harm"
            gain_key = f"{self.model_name}_rc_pag_advantage_gain"
            missing = [key for key in (harm_key, gain_key) if key not in estimator_paths]
            if missing:
                raise ValueError(f"missing fitted v5 advantage estimators: {', '.join(missing)}")
            estimator = RiskEstimator.load(estimator_paths[harm_key])
            benefit = NormalizedNFEReductionEstimator.load(estimator_paths[gain_key])
            required_features = {"local.temporal_js_mean", "local.temporal_js_max"}
            if not required_features.issubset(estimator.names) or not required_features.issubset(
                benefit.names
            ):
                raise ValueError("v5 advantage estimator is missing the temporal-JS feature schema")
            if (
                estimator.include_history
                or benefit.include_history
                or estimator.kind != "hist_gradient_boosting"
            ):
                raise ValueError("v5 requires frozen local histogram-boosting advantage heads")
            if tuple(estimator.names) != tuple(benefit.names):
                raise ValueError("v5 harm and gain estimator feature schemas differ")
        else:
            key = f"{self.model_name}_{candidate.variant}"
            if key not in estimator_paths:
                raise ValueError(f"missing fitted estimator for {key}")
            estimator = RiskEstimator.load(estimator_paths[key])
            benefit = None
        if self.config.protocol_version == "v4" or (
            self.config.protocol_version == "v5" and candidate.variant == "rc_pag_local"
        ):
            required_features = {"local.temporal_js_mean", "local.temporal_js_max"}
            if not required_features.issubset(estimator.names):
                raise ValueError("v4 estimator is missing the frozen temporal-JS feature schema")
            if estimator.include_history or estimator.kind != "hist_gradient_boosting":
                raise ValueError("v4 requires the frozen local histogram-boosting estimator")
        if (
            not uses_advantage_heads
            and not uses_budgeted_heads
            and candidate.min_predicted_nfe_savings > 0
        ):
            benefit_key = f"{self.model_name}_remaining_nfe"
            if benefit_key not in estimator_paths:
                raise ValueError(f"missing fitted benefit estimator for {benefit_key}")
            benefit = RemainingNFEEstimator.load(estimator_paths[benefit_key])
        return RiskStoppingPolicy(
            estimator,
            threshold=candidate.threshold,
            min_steps=candidate.min_steps,
            patience=candidate.patience,
            include_history=candidate.variant == "rc_pag_history",
            history_window=self.config.history_window,
            max_remaining_fraction=candidate.max_remaining_fraction,
            benefit_scorer=benefit,
            min_predicted_nfe_savings=candidate.min_predicted_nfe_savings,
            max_temporal_js=candidate.max_temporal_js,
            require_exact_agreement=candidate.require_exact_agreement,
            total_risk_budget=candidate.total_risk_budget,
            max_prompt_stops=candidate.max_prompt_stops,
        )

    def _method_components(
        self,
        method: str,
        candidate: PolicyCandidateSpec | None,
        estimator_paths: Mapping[str, str],
    ):
        max_steps = self.config.decoding.max_refinement_steps
        scheduler = _BudgetScheduler(budget=max_steps)
        policy = None
        enforcement = "soft_gate"
        provenance = method
        if candidate is not None:
            policy = self._risk_policy(candidate, estimator_paths)
            provenance = candidate.variant
        elif method == "full_budget" or method == "full_budget_shadow":
            policy = _TracePolicy() if method == "full_budget_shadow" else None
        elif method == "fixed":
            scheduler = _BudgetScheduler(budget=max_steps, override_block_size=32)
        elif method == "constant_budget":
            scheduler = _BudgetScheduler(budget=8)
            enforcement = "hard_cap"
        elif method == "size_lookup" or method == "best_nonlearned":
            scheduler = _BudgetScheduler(budget=8, size_scaled=True)
            enforcement = "hard_cap"
        elif method == "oracle":
            policy = _TracePolicy()
        elif method == "pilot_shadow":
            policy = _PilotStopPolicy()
            provenance = "pilot_forced_stop_shadow_smoke"
        elif method in {
            "fast_dllm",
            "sched",
            "entropy_sum",
            "confidence_gate",
            "stability_gate",
            "pag",
            "residual_pag",
            "entropy_sum_gate",
            "mutual_stability_gate",
            "stability_weighted_style",
            "token_convergence_style",
        }:
            style_name = {"fast_dllm": "fast_dllm_style", "sched": "sched_style"}.get(
                method, method
            )
            policy = _HeuristicPolicy(style_name)
            provenance = style_name
        elif method != "adablock":
            raise ValueError(f"unsupported RC-PAG method: {method}")
        return scheduler, policy, enforcement, provenance

    def _generate_llada(
        self,
        input_ids: torch.Tensor,
        *,
        method: str,
        scheduler,
        policy,
        enforcement: str,
        shadow: bool,
    ):
        modern_protocol = self.config.protocol_version in {"v2", "v3", "v4", "v5", "v6"}
        if method == "adablock" or modern_protocol:
            module = importlib.import_module("generate_adablock")
            result = module.generate_adablock_dual_cache(
                self.model,
                input_ids,
                steps=self.config.decoding.max_refinement_steps,
                gen_length=self.config.decoding.gen_length,
                init_block_length=32,
                temperature=0.0,
                mask_id=MASK_IDS["llada"],
                threshold=self.config.decoding.transfer_threshold,
                delimiter_ids=self.delimiter_ids.tolist(),
                delimiter_threshold=self.config.decoding.delimiter_threshold,
                risk_policy=policy if modern_protocol else None,
                digit_ids_tensor=self.digit_ids,
                delimiter_ids_tensor=self.delimiter_ids,
                return_schedule_history=modern_protocol,
            )
            if modern_protocol:
                return result
            output, nfes, blocks = result
            schedules = [
                {
                    "applied_block_size": block,
                    "actual_nfe_used": nfe,
                    "exit_reason": "adablock",
                }
                for block, nfe in zip(blocks, nfes, strict=True)
            ]
            return output, nfes, blocks, schedules
        module = importlib.import_module("generate_pag")
        callback = (
            module.make_llada_shadow_callback(
                self.model,
                mode="dual_cache",
                mask_id=MASK_IDS["llada"],
                threshold=self.config.decoding.transfer_threshold,
                max_steps=self.config.decoding.max_refinement_steps,
            )
            if shadow
            else None
        )
        return module.generate_pag_dual_cache(
            self.model,
            input_ids,
            scheduler,
            steps=self.config.decoding.max_refinement_steps,
            gen_length=self.config.decoding.gen_length,
            temperature=0.0,
            mask_id=MASK_IDS["llada"],
            threshold=self.config.decoding.transfer_threshold,
            max_block_length=self.config.decoding.max_block_length,
            max_refinement_steps=self.config.decoding.max_refinement_steps,
            digit_ids_tensor=self.digit_ids,
            delimiter_ids_tensor=self.delimiter_ids,
            delimiter_ids=self.delimiter_ids.tolist(),
            delimiter_threshold=self.config.decoding.delimiter_threshold,
            enforcement_mode=enforcement,
            risk_policy=policy,
            shadow_callback=callback,
            shadow_all_steps=False,
        )

    def _generate_dream(
        self,
        input_ids: torch.Tensor,
        *,
        method: str,
        scheduler,
        policy,
        enforcement: str,
        shadow: bool,
    ):
        del enforcement
        adablock = importlib.import_module("model.generation_utils_adablock")
        pag = importlib.import_module("model.generation_utils_pag")
        use_exact_adablock_loop = self.config.protocol_version in {
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
        }
        generation_module = adablock if method == "adablock" or use_exact_adablock_loop else pag
        self.model.diffusion_generate = types.MethodType(
            generation_module.DreamGenerationMixin.diffusion_generate,
            self.model,
        )
        self.model._sample = types.MethodType(
            (
                adablock.DreamGenerationMixin._sample_adablock_cache
                if method == "adablock" or use_exact_adablock_loop
                else pag.DreamGenerationMixin._sample_pag_cache
            ),
            self.model,
        )
        self.model._compute_block_length = types.MethodType(
            adablock.DreamGenerationMixin._compute_block_length,
            self.model,
        )
        self.model.pag_scheduler = scheduler
        self.model.pag_risk_policy = policy
        self.model.pag_digit_ids = self.digit_ids
        self.model.pag_delimiter_ids = self.delimiter_ids
        self.model.pag_shadow_callback = "auto" if shadow else None
        attention = input_ids.ne(self.tokenizer.pad_token_id)
        result = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention,
            max_new_tokens=self.config.decoding.gen_length,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.config.decoding.max_refinement_steps,
            temperature=0.0,
            alg="confidence_threshold",
            threshold=self.config.decoding.transfer_threshold,
            dual_cache=True,
            block_length=32,
            max_block_length=self.config.decoding.max_block_length,
            max_refinement_steps=self.config.decoding.max_refinement_steps,
            delimiter_threshold=self.config.decoding.delimiter_threshold,
        )
        schedules = getattr(result, "schedule_history", None) or [
            {
                "applied_block_size": block,
                "actual_nfe_used": nfe,
                "exit_reason": "adablock",
            }
            for block, nfe in zip(result.block_history, result.nfe_history, strict=True)
        ]
        return result.sequences, result.nfe_history, result.block_history, schedules

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
        if model != self.model_name:
            raise ValueError("runtime/model mismatch")
        self._ensure_backend()
        data = self._sample(sample)
        input_ids = self._prompt_ids(data["prompt"])
        resolved_method = method
        if method == "best_nonlearned":
            summary_path = self.run_dir / "screening_summary.json"
            if summary_path.is_file():
                selected = json.loads(summary_path.read_text())["best_nonlearned"]
                resolved_method = (
                    str(selected[self.model_name]) if isinstance(selected, dict) else str(selected)
                )
        scheduler, policy, enforcement, provenance = self._method_components(
            resolved_method,
            candidate,
            estimator_paths or {},
        )
        shadow = (
            self.config.protocol_version == "v1" and stage == "calibrate" and candidate is not None
        ) or method == "pilot_shadow"
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        generated, nfes, blocks, schedules = (
            self._generate_llada(
                input_ids,
                method=resolved_method,
                scheduler=scheduler,
                policy=policy,
                enforcement=enforcement,
                shadow=shadow,
            )
            if self.model_name == "llada"
            else self._generate_dream(
                input_ids,
                method=resolved_method,
                scheduler=scheduler,
                policy=policy,
                enforcement=enforcement,
                shadow=shadow,
            )
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated_ids = generated[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        grade = (
            grade_gsm8k(text, data["gold"])
            if data["kind"] == "gsm8k"
            else grade_math500(text, data["gold"])
            if data["kind"] == "math"
            else _grade_code(text, data["row"])
        )
        payload: dict[str, Any] = {
            "generated_text": text,
            "generated_ids": [int(value) for value in generated_ids.tolist()],
            "grade": asdict(grade),
            "is_correct": grade.is_correct,
            "total_nfe": int(sum(int(value) for value in nfes)),
            "nfe_history": [int(value) for value in nfes],
            "block_history": [int(value) for value in blocks],
            "schedule_history": schedules,
            "elapsed_sec": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "implementation": provenance,
            "mock": False,
        }
        if stage == "collect":
            payload["training_examples"] = training_examples_from_schedules(
                schedules,
                history_window=self.config.history_window,
            )
        if shadow:
            payload["shadow_losses"] = [prompt_loss_from_schedules(schedules)]
        if resolved_method == "oracle":
            oracle_nfe = sum(
                min(
                    (
                        int(step["step_index"])
                        for step in block.get("risk_steps", ())
                        if tuple(step["proposed_tokens"]) == tuple(block.get("final_tokens", ()))
                    ),
                    default=int(block["actual_nfe_used"]),
                )
                for block in schedules
            )
            payload["oracle_counterfactual_nfe"] = oracle_nfe
            payload["total_nfe"] = oracle_nfe
        if stage == "pilot":
            payload["artifact_bytes"] = len(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return payload

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self._pools.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

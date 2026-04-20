from __future__ import annotations

import hashlib
import os
import random
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_DEFAULT_DELIMITER_TEXTS = ("\n", ".", ",", ":", ";", "(", ")", "!", "?")
_DELIMITER_NAME_OVERRIDES = {
    "\n": "newline",
    ".": "period",
    ",": "comma",
    ":": "colon",
    ";": "semicolon",
    "(": "left_paren",
    ")": "right_paren",
    "!": "exclamation",
    "?": "question",
}


@dataclass(slots=True)
class DreamGenerationConfig:
    model_name: str
    max_new_tokens: int = 256
    steps: int = 256
    temperature: float | None = 0.2
    top_p: float | None = 0.95
    top_k: int | None = None
    alg: str = "entropy"
    alg_temp: float | None = 0.0
    device: str | None = None
    torch_dtype: str = "auto"
    trace_profile: str = "entropy_det"
    seed: int = 0
    delimiter_texts: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_DELIMITER_TEXTS)


@dataclass(frozen=True, slots=True)
class _DelimiterFeatureSpec:
    text: str
    feature_key: str
    token_id: int


class DreamTraceCollector:
    """Collect raw Dream step dumps via the official diffusion_generate API.

    Assumptions:
    - Dream is loaded through Hugging Face `AutoModel(..., trust_remote_code=True)`.
    - `generation_tokens_hook_func(step, x, logits)` is invoked once per denoising step.
    - `x` contains the current token canvas, and `logits` contains per-position logits for the
      same step. This follows the official Dream examples and README.
    """

    def __init__(self, config: DreamGenerationConfig) -> None:
        self._config = config
        self._torch, self._auto_model, self._auto_tokenizer = _import_dream_runtime()

        self._device = config.device or _select_device(self._torch)
        self._dtype = _resolve_torch_dtype(self._torch, config.torch_dtype, self._device)

        self._model = self._auto_model.from_pretrained(
            config.model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )
        self._tokenizer = self._auto_tokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        self._model = self._model.to(self._device).eval()
        self._mask_token_id = getattr(self._tokenizer, "mask_token_id", None)
        self._mask_token_text = getattr(self._tokenizer, "mask_token", None) or "<|mask|>"
        self._delimiter_features = _resolve_delimiter_features(
            self._tokenizer,
            config.delimiter_texts,
        )

    def collect(self, prompt_record: dict[str, Any]) -> dict[str, Any]:
        prompt_seed = _prompt_seed(self._config.seed, self._config.trace_profile, prompt_record)
        _seed_generation(self._torch, prompt_seed)
        prompt = str(prompt_record["prompt"])
        inputs = _build_inputs(self._tokenizer, prompt)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        prompt_length = int(input_ids.shape[1])

        recorder = _StepRecorder(
            torch_module=self._torch,
            tokenizer=self._tokenizer,
            prompt_length=prompt_length,
            max_new_tokens=self._config.max_new_tokens,
            mask_token_id=self._mask_token_id,
            mask_token_text=self._mask_token_text,
            delimiter_features=self._delimiter_features,
        )

        def generation_tokens_hook_func(step, x, logits):
            recorder.capture(step, x, logits)
            return x

        generation_kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "max_new_tokens": self._config.max_new_tokens,
            "output_history": False,
            "return_dict_in_generate": True,
            "steps": self._config.steps,
            "alg": self._config.alg,
            "generation_tokens_hook_func": generation_tokens_hook_func,
        }
        if self._config.temperature is not None and self._config.temperature > 0:
            generation_kwargs["temperature"] = self._config.temperature
        if self._config.top_p is not None:
            generation_kwargs["top_p"] = self._config.top_p
        if self._config.top_k is not None:
            generation_kwargs["top_k"] = self._config.top_k
        if self._config.alg_temp is not None:
            generation_kwargs["alg_temp"] = self._config.alg_temp

        with self._torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"`do_sample` is set to `False`\. However, `temperature` is set to `0\.0`"
                ),
                category=UserWarning,
                module=r"transformers\.generation\.configuration_utils",
            )
            output = self._model.diffusion_generate(input_ids, **generation_kwargs)

        final_sequence = output.sequences[0].detach().to("cpu")
        generated_ids = _truncate_generated_ids(
            final_sequence,
            prompt_length=prompt_length,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if not generated_ids:
            msg = (
                "Dream returned no non-special generated tokens. "
                f"Prompt sample_id={prompt_record.get('sample_id', 'unknown')!r}."
            )
            raise ValueError(msg)
        return recorder.to_raw_step_dump(
            prompt_record=prompt_record,
            config=self._config,
            generated_ids=generated_ids,
            device=self._device,
            prompt_seed=prompt_seed,
        )


class _StepRecorder:
    def __init__(
        self,
        *,
        torch_module,
        tokenizer,
        prompt_length: int,
        max_new_tokens: int,
        mask_token_id: int | None,
        mask_token_text: str,
        delimiter_features: tuple[_DelimiterFeatureSpec, ...],
    ) -> None:
        self._torch = torch_module
        self._tokenizer = tokenizer
        self._prompt_length = prompt_length
        self._max_new_tokens = max_new_tokens
        self._mask_token_id = mask_token_id
        self._mask_token_text = mask_token_text
        self._delimiter_features = delimiter_features
        self._snapshots: list[_StepSnapshot] = []

    def capture(self, step: int | None, x, logits) -> None:
        step_index = _normalize_hook_step(step)
        if step_index is None:
            return

        token_canvas = _normalize_token_canvas(x)
        generated_ids = token_canvas[
            :,
            self._prompt_length : self._prompt_length + self._max_new_tokens,
        ]
        if int(generated_ids.shape[0]) != 1:
            msg = (
                "Dream trace collection currently expects a single prompt per hook call. "
                f"Got batch size {int(generated_ids.shape[0])}."
            )
            raise ValueError(msg)
        generated_ids_row = generated_ids[0].detach().to("cpu", dtype=self._torch.long)

        selected_logits: list[float | None]
        selected_probs: list[float | None]
        top2_probs: list[float | None]
        entropies: list[float | None]
        delimiter_probabilities: dict[str, list[float | None]]

        if logits is None:
            selected_logits = [None] * int(generated_ids_row.shape[0])
            selected_probs = [None] * int(generated_ids_row.shape[0])
            top2_probs = [None] * int(generated_ids_row.shape[0])
            entropies = [None] * int(generated_ids_row.shape[0])
            delimiter_probabilities = {
                feature.feature_key: [None] * int(generated_ids_row.shape[0])
                for feature in self._delimiter_features
            }
        else:
            step_logits = _slice_generation_logits(
                logits=logits,
                prompt_length=self._prompt_length,
                generated_length=int(generated_ids_row.shape[0]),
            )
            if int(step_logits.shape[0]) != 1:
                msg = (
                    "Dream trace collection currently expects logits for a single prompt. "
                    f"Got batch size {int(step_logits.shape[0])}."
                )
                raise ValueError(msg)
            # Softmax/logsumexp in float32 avoids many exact 1.0 probabilities from low-precision
            # Dream logits, especially when the model itself is running in bf16 on GPU.
            step_logits_row = step_logits[0].to(dtype=self._torch.float32)
            token_ids_on_device = generated_ids_row.to(
                device=step_logits_row.device,
                dtype=self._torch.long,
            ).unsqueeze(-1)
            (
                selected_logits,
                selected_probs,
                top2_probs,
                entropies,
            ) = _selected_token_stats(
                self._torch,
                step_logits_row,
                token_ids_on_device,
            )
            delimiter_probabilities = _delimiter_probabilities(
                step_logits_row,
                self._delimiter_features,
            )

        self._snapshots.append(
            _StepSnapshot(
                step_index=step_index,
                token_ids=generated_ids_row.tolist(),
                selected_logits=selected_logits,
                selected_probs=selected_probs,
                top2_probs=top2_probs,
                entropies=entropies,
                delimiter_probabilities=delimiter_probabilities,
            )
        )

    def to_raw_step_dump(
        self,
        *,
        prompt_record: dict[str, Any],
        config: DreamGenerationConfig,
        generated_ids: list[int],
        device: str,
        prompt_seed: int,
    ) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        final_length = len(generated_ids)
        previous_token_ids: list[int] | None = None
        for snapshot in self._snapshots:
            current_token_ids = [int(token_id) for token_id in snapshot.token_ids[:final_length]]
            changed_flags = [
                False
                if previous_token_ids is None
                else current_token_ids[token_index] != previous_token_ids[token_index]
                for token_index in range(final_length)
            ]
            tokens: list[dict[str, Any]] = []
            delimiter_confidences: list[float | None] = []
            for token_index in range(final_length):
                token_id = current_token_ids[token_index]
                delimiter_values = [
                    snapshot.delimiter_probabilities[feature.feature_key][token_index]
                    for feature in self._delimiter_features
                ]
                resolved_delimiter_values = [
                    value for value in delimiter_values if value is not None
                ]
                delimiter_prob_max = (
                    max(resolved_delimiter_values) if resolved_delimiter_values else None
                )
                delimiter_confidences.append(delimiter_prob_max)
                is_mask = (
                    self._mask_token_id is not None and token_id == self._mask_token_id
                )
                extras = {
                    "entropy": _or_zero(snapshot.entropies[token_index]),
                    "is_mask": 1.0 if is_mask else 0.0,
                    "changed_from_prev_step": 1.0 if changed_flags[token_index] else 0.0,
                    "delimiter_prob_max": _or_zero(delimiter_prob_max),
                }
                for feature in self._delimiter_features:
                    extras[feature.feature_key] = _or_zero(
                        snapshot.delimiter_probabilities[feature.feature_key][token_index]
                    )
                tokens.append(
                    {
                        "token_index": token_index,
                        "token_id": token_id,
                        "token_text": _decode_single_token(self._tokenizer, token_id),
                        "top1_prob": _maybe_round(snapshot.selected_probs[token_index]),
                        "selected_logit": _maybe_round(snapshot.selected_logits[token_index]),
                        "top2_prob": _maybe_round(snapshot.top2_probs[token_index]),
                        "extras": {
                            key: _maybe_round(value) for key, value in extras.items()
                        },
                    }
                )
            mask_positions = [
                index
                for index, token_id in enumerate(current_token_ids)
                if self._mask_token_id is not None and token_id == self._mask_token_id
            ]
            valid_delimiter_confidences = [
                (index, confidence)
                for index, confidence in enumerate(delimiter_confidences)
                if confidence is not None
            ]
            if valid_delimiter_confidences:
                best_delimiter_index, max_delimiter_confidence = max(
                    valid_delimiter_confidences,
                    key=lambda item: item[1],
                )
            else:
                best_delimiter_index, max_delimiter_confidence = None, None
            steps.append(
                {
                    "step_index": snapshot.step_index,
                    "summary": {
                        "mask_count": len(mask_positions),
                        "changed_count": sum(changed_flags),
                        "active_start": mask_positions[0] if mask_positions else None,
                        "active_end": (mask_positions[-1] + 1) if mask_positions else None,
                        "active_count": len(mask_positions),
                        "best_delimiter_index": best_delimiter_index,
                        "max_delimiter_confidence": _maybe_round(max_delimiter_confidence),
                    },
                    "tokens": tokens,
                }
            )
            previous_token_ids = current_token_ids

        final_text = self._tokenizer.decode(
            generated_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        sample_id = str(prompt_record.get("sample_id", "dream-trace"))
        return {
            "trace_id": sample_id,
            "backend": "dream",
            "prompt": str(prompt_record["prompt"]),
            "model_name": config.model_name,
            "final_text": final_text,
            "tags": list(prompt_record.get("tags", [])),
            "created_at": datetime.now(UTC).isoformat(),
            "decoding_metadata": {
                "run_id": os.environ.get("SLURM_JOB_ID", "local"),
                "max_new_tokens": config.max_new_tokens,
                "steps": config.steps,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "alg": config.alg,
                "alg_temp": config.alg_temp,
                "trace_profile": config.trace_profile,
                "seed": config.seed,
                "prompt_seed": prompt_seed,
                "device": device,
                "mask_token_id": self._mask_token_id,
                "mask_token_text": self._mask_token_text,
                "delimiter_features": [
                    {
                        "text": feature.text,
                        "feature_key": feature.feature_key,
                        "token_id": feature.token_id,
                    }
                    for feature in self._delimiter_features
                ],
            },
            "steps": steps,
        }


@dataclass(slots=True)
class _StepSnapshot:
    step_index: int
    token_ids: list[int]
    selected_logits: list[float | None]
    selected_probs: list[float | None]
    top2_probs: list[float | None]
    entropies: list[float | None]
    delimiter_probabilities: dict[str, list[float | None]]


def _build_inputs(tokenizer, prompt: str) -> dict[str, Any]:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
    encoded = tokenizer(prompt, return_tensors="pt")
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
    }


def _decode_single_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def _resolve_delimiter_features(
    tokenizer,
    delimiter_texts: tuple[str, ...],
) -> tuple[_DelimiterFeatureSpec, ...]:
    resolved: list[_DelimiterFeatureSpec] = []
    seen_token_ids: set[int] = set()
    for delimiter_text in delimiter_texts:
        token_ids = tokenizer.encode(delimiter_text, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = int(token_ids[0])
        if token_id in seen_token_ids:
            continue
        seen_token_ids.add(token_id)
        resolved.append(
            _DelimiterFeatureSpec(
                text=delimiter_text,
                feature_key=f"delimiter_prob_{_delimiter_feature_name(delimiter_text)}",
                token_id=token_id,
            )
        )
    return tuple(resolved)


def _import_dream_runtime():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        msg = (
            "Dream trace collection requires PyTorch and Transformers. "
            "Install versions compatible with Dream, such as torch==2.5.1 and "
            "transformers==4.46.2."
        )
        raise ImportError(msg) from error
    return torch, AutoModel, AutoTokenizer


def _maybe_round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 8)


def _normalize_hook_step(step: object) -> int | None:
    if step is None:
        return None
    return int(step)


def _delimiter_probabilities(
    step_logits_row,
    delimiter_features: tuple[_DelimiterFeatureSpec, ...],
) -> dict[str, list[float | None]]:
    if not delimiter_features:
        return {}
    probabilities = step_logits_row.log_softmax(dim=-1).exp()
    return {
        feature.feature_key: probabilities[:, feature.token_id]
        .detach()
        .to("cpu", dtype=probabilities.dtype)
        .tolist()
        for feature in delimiter_features
    }


def _selected_token_stats(torch_module, step_logits_row, token_ids_on_device):
    gathered_logits = step_logits_row.gather(dim=-1, index=token_ids_on_device).squeeze(-1)
    log_probs = step_logits_row.log_softmax(dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=token_ids_on_device).squeeze(-1)
    selected_probs = selected_log_probs.exp()
    probs = log_probs.exp()
    entropies = -(probs * log_probs).sum(dim=-1)

    topk = min(2, int(step_logits_row.shape[-1]))
    top_log_probs = log_probs.topk(k=topk, dim=-1).values
    runner_up = (
        top_log_probs[..., 1].exp()
        if topk == 2
        else torch_module.full_like(selected_probs, float("nan"))
    )

    selected_logits_cpu = gathered_logits.detach().to("cpu", dtype=torch_module.float32).tolist()
    selected_probs_cpu = selected_probs.detach().to("cpu", dtype=torch_module.float32).tolist()
    top2_cpu = runner_up.detach().to("cpu", dtype=torch_module.float32).tolist()
    entropy_cpu = entropies.detach().to("cpu", dtype=torch_module.float32).tolist()
    top2_probs = [None if value != value else value for value in top2_cpu]
    entropies_list = [None if value != value else value for value in entropy_cpu]
    return selected_logits_cpu, selected_probs_cpu, top2_probs, entropies_list


def _normalize_token_canvas(x):
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim != 2:
        msg = f"Unexpected Dream token canvas rank: expected 1 or 2 dims, got {x.ndim}."
        raise ValueError(msg)
    return x


def _resolve_torch_dtype(torch_module, dtype_name: str, device: str):
    if dtype_name == "auto":
        if device == "cuda":
            return torch_module.bfloat16
        if device == "mps":
            return torch_module.float16
        return torch_module.float32
    dtype = getattr(torch_module, dtype_name, None)
    if dtype is None:
        msg = f"Unsupported torch dtype {dtype_name!r}"
        raise ValueError(msg)
    return dtype


def _select_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _slice_generation_logits(*, logits, prompt_length: int, generated_length: int):
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3:
        msg = f"Unexpected Dream logits rank: expected 2 or 3 dims, got {logits.ndim}."
        raise ValueError(msg)
    if logits.shape[1] == prompt_length + generated_length:
        return logits[:, prompt_length:, :]
    if logits.shape[1] == generated_length:
        return logits
    msg = (
        "Unexpected Dream logits shape during trace collection. "
        f"Expected sequence dimension {generated_length} or {prompt_length + generated_length}, "
        f"got {int(logits.shape[1])}."
    )
    raise ValueError(msg)


def _truncate_generated_ids(
    sequence,
    *,
    prompt_length: int,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> list[int]:
    generated = sequence[prompt_length:].tolist()
    truncated: list[int] = []
    for token_id in generated:
        if eos_token_id is not None and token_id == eos_token_id:
            break
        if pad_token_id is not None and token_id == pad_token_id:
            break
        truncated.append(int(token_id))
    return truncated


def _seed_generation(torch_module, seed: int) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _prompt_seed(
    base_seed: int,
    trace_profile: str,
    prompt_record: dict[str, Any],
) -> int:
    seed_material = (
        f"{base_seed}:{trace_profile}:"
        f"{prompt_record.get('sample_id', prompt_record.get('prompt', 'dream-trace'))}"
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _delimiter_feature_name(delimiter_text: str) -> str:
    return _DELIMITER_NAME_OVERRIDES.get(
        delimiter_text,
        delimiter_text.encode("unicode_escape").decode("ascii").replace("\\", "_"),
    )


def _or_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)

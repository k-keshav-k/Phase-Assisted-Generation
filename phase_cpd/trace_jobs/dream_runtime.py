from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


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

    def collect(self, prompt_record: dict[str, Any]) -> dict[str, Any]:
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
        )


class _StepRecorder:
    def __init__(self, *, torch_module, tokenizer, prompt_length: int, max_new_tokens: int) -> None:
        self._torch = torch_module
        self._tokenizer = tokenizer
        self._prompt_length = prompt_length
        self._max_new_tokens = max_new_tokens
        self._snapshots: list[_StepSnapshot] = []

    def capture(self, step: int | None, x, logits) -> None:
        step_index = _normalize_hook_step(step)
        if step_index is None:
            return

        token_canvas = _normalize_token_canvas(x)
        generated_ids = token_canvas[:, self._prompt_length : self._prompt_length + self._max_new_tokens]
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

        if logits is None:
            selected_logits = [None] * int(generated_ids_row.shape[0])
            selected_probs = [None] * int(generated_ids_row.shape[0])
            top2_probs = [None] * int(generated_ids_row.shape[0])
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
            step_logits_row = step_logits[0]
            token_ids_on_device = generated_ids_row.to(step_logits_row.device).unsqueeze(-1)
            gathered_logits = step_logits_row.gather(dim=-1, index=token_ids_on_device).squeeze(-1)
            log_partition = step_logits_row.logsumexp(dim=-1)
            probs = (gathered_logits - log_partition).exp()
            topk = min(2, int(step_logits_row.shape[-1]))
            top_logits = step_logits_row.topk(k=topk, dim=-1).values
            runner_up = (
                (top_logits[..., 1] - log_partition).exp()
                if topk == 2
                else self._torch.full_like(probs, float("nan"))
            )

            selected_logits = gathered_logits.detach().to("cpu", dtype=self._torch.float32).tolist()
            selected_probs = probs.detach().to("cpu", dtype=self._torch.float32).tolist()
            top2_cpu = runner_up.detach().to("cpu", dtype=self._torch.float32).tolist()
            top2_probs = [None if value != value else value for value in top2_cpu]

        self._snapshots.append(
            _StepSnapshot(
                step_index=step_index,
                token_ids=generated_ids_row.tolist(),
                selected_logits=selected_logits,
                selected_probs=selected_probs,
                top2_probs=top2_probs,
            )
        )

    def to_raw_step_dump(
        self,
        *,
        prompt_record: dict[str, Any],
        config: DreamGenerationConfig,
        generated_ids: list[int],
        device: str,
    ) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        final_length = len(generated_ids)
        for snapshot in self._snapshots:
            tokens: list[dict[str, Any]] = []
            for token_index in range(final_length):
                token_id = int(snapshot.token_ids[token_index])
                tokens.append(
                    {
                        "token_index": token_index,
                        "token_id": token_id,
                        "token_text": _decode_single_token(self._tokenizer, token_id),
                        "top1_prob": _maybe_round(snapshot.selected_probs[token_index]),
                        "selected_logit": _maybe_round(snapshot.selected_logits[token_index]),
                        "top2_prob": _maybe_round(snapshot.top2_probs[token_index]),
                    }
                )
            steps.append({"step_index": snapshot.step_index, "tokens": tokens})

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
                "device": device,
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

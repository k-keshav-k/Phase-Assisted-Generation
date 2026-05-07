from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Predictor = importlib.import_module("phase_predict.predict").Predictor
PhaseTuple = importlib.import_module("phase_predict.schema").PhaseTuple
ExtendedPhaseTuple = importlib.import_module("phase_predict.schema").ExtendedPhaseTuple

# Populated by run_pag_vs_adablock_eval.py or eval_dream_pag.py before generation.
DIGIT_IDS_TENSOR: torch.Tensor | None = None
DELIM_IDS_TENSOR: torch.Tensor | None = None

DEFAULT_PREDICTOR_CKPT = ROOT / "output" / "phase_predict_model_checkpoint.pt"
DEFAULT_LOG_FILE = ROOT / "logs" / "llada_pag_inference.jsonl"
BlockTuple = PhaseTuple


@dataclass(slots=True)
class ScheduledBlock:
    predicted_tuple: BlockTuple
    applied_block_size: int
    budgeted_refinement_steps: int


@dataclass(slots=True)
class PromptRecord:
    prompt: str
    prompt_id: str | None = None
    category: str | None = None
    tags: list[str] | None = None
    notes: str | None = None


@dataclass(slots=True)
class EffectiveSeed:
    block_length: int
    refinement_steps: int
    source: str
    context_stabilizing_steps: int | None = None
    context_mean_confidence: float = 1.0
    context_min_confidence: float = 1.0
    context_digit_fraction: float = 0.0
    context_delimiter_fraction: float = 0.0


def _tuple_to_dict(value: BlockTuple) -> dict[str, int]:
    return {
        "block_size": int(value.block_size),
        "refinement_steps": int(value.refinement_steps),
    }


def _extended_tuple_to_dict(et: ExtendedPhaseTuple) -> dict[str, float]:
    return dict(et.values)


def _normalize_tuple(block_size: int, refinement_steps: int) -> BlockTuple:
    return BlockTuple(
        block_size=max(1, int(block_size)),
        refinement_steps=max(1, int(refinement_steps)),
    )


def _normalize_stabilizing_tuple(block_size: int, stabilizing_steps: int) -> BlockTuple:
    return BlockTuple(
        block_size=max(1, int(block_size)),
        refinement_steps=max(0, int(stabilizing_steps)),
    )


def _max_stabilizing_step(
    predictions_by_step: list[object],
    final_tokens: object,
) -> int:
    """Return max per-token stabilizing step over one decoded block.

    Step indices match trace files: first refinement/model pass is step 0,
    and the final possible step is ``nfe - 1``.
    """
    import torch

    if not predictions_by_step:
        return 0

    stacked = torch.stack([step.detach().cpu().reshape(-1) for step in predictions_by_step])
    final = final_tokens.detach().cpu().reshape(-1)
    if final.numel() == 0:
        return 0

    max_stable_step = 0
    for token_index in range(final.numel()):
        final_id = final[token_index]
        stable_step = int(stacked.shape[0] - 1)
        for step_index in range(stacked.shape[0]):
            if torch.all(stacked[step_index:, token_index] == final_id):
                stable_step = int(step_index)
                break
        max_stable_step = max(max_stable_step, stable_step)
    return max_stable_step


def parse_tuple_schedule(raw: str | None) -> list[BlockTuple]:
    if raw is None or raw.strip() == "":
        return []

    schedule: list[BlockTuple] = []
    for entry in raw.split(","):
        block_size_str, refinement_steps_str = entry.strip().split(":")
        schedule.append(
            BlockTuple(
                block_size=int(block_size_str),
                refinement_steps=int(refinement_steps_str),
            )
        )
    return schedule


def _coerce_prompt_record(value: object, *, index: int) -> PromptRecord:
    if isinstance(value, str):
        return PromptRecord(prompt=value, prompt_id=f"prompt_{index:03d}")
    if not isinstance(value, dict):
        msg = f"Prompt entry {index} must be a string or JSON object"
        raise TypeError(msg)
    prompt = value.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        msg = f"Prompt entry {index} is missing a non-empty 'prompt' field"
        raise ValueError(msg)
    raw_tags = value.get("tags", [])
    if raw_tags is None:
        tags: list[str] | None = None
    elif isinstance(raw_tags, list):
        tags = [str(item) for item in raw_tags]
    else:
        tags = [str(raw_tags)]
    return PromptRecord(
        prompt=prompt,
        prompt_id=str(value.get("id", f"prompt_{index:03d}")),
        category=str(value["category"]) if value.get("category") is not None else None,
        tags=tags,
        notes=str(value["notes"]) if value.get("notes") is not None else None,
    )


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    prompt_path = Path(path)
    records: list[PromptRecord] = []
    if prompt_path.suffix.lower() == ".json":
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        entries = payload.get("prompts", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            msg = "JSON prompt files must contain a list or an object with a 'prompts' list"
            raise ValueError(msg)
        return [_coerce_prompt_record(item, index=index) for index, item in enumerate(entries)]

    with prompt_path.open(encoding="utf-8") as file_obj:
        for index, line in enumerate(file_obj):
            line = line.strip()
            if not line:
                continue
            records.append(_coerce_prompt_record(json.loads(line), index=index))
    return records


def _prompt_record_from_args(args: argparse.Namespace) -> PromptRecord:
    return PromptRecord(
        prompt=args.prompt,
        prompt_id=args.prompt_id,
        category=args.prompt_category,
        tags=[tag.strip() for tag in args.prompt_tags.split(",") if tag.strip()]
        if args.prompt_tags
        else None,
    )


class DummyTupleAPI:
    """A stand-in for a remote tuple prediction service."""

    def __init__(
        self,
        *,
        scripted_tuples: list[BlockTuple],
        fallback_block_size: int,
        fallback_refinement_steps: int,
        verbose: bool = False,
    ) -> None:
        self.scripted_tuples = list(scripted_tuples)
        self.fallback_tuple = _normalize_tuple(
            fallback_block_size,
            fallback_refinement_steps,
        )
        self.verbose = verbose
        self.reset()

    def reset(self) -> None:
        self._next_index = 0
        self.requests: list[dict[str, object]] = []
        self.responses: list[dict[str, object]] = []

    def predict_tuple(
        self,
        *,
        prompt_text: str,
        block_index: int,
        history: list[BlockTuple],
        remaining_tokens: int,
    ) -> BlockTuple:
        request = {
            "prompt": prompt_text,
            "block_index": int(block_index),
            "remaining_tokens": int(remaining_tokens),
            "history": [_tuple_to_dict(item) for item in history],
        }
        self.requests.append(request)

        if self._next_index < len(self.scripted_tuples):
            predicted = self.scripted_tuples[self._next_index]
            source = "scripted"
            self._next_index += 1
        else:
            predicted = BlockTuple(
                block_size=min(self.fallback_tuple.block_size, int(remaining_tokens)),
                refinement_steps=self.fallback_tuple.refinement_steps,
            )
            source = "fallback"

        response = {
            "block_size": int(predicted.block_size),
            "refinement_steps": int(predicted.refinement_steps),
            "source": source,
        }
        self.responses.append(response)

        if self.verbose:
            print("[dummy-api] request:")
            print(json.dumps(request, indent=2))
            print("[dummy-api] response:")
            print(json.dumps(response, indent=2))

        return predicted


class DummyAPIScheduler:
    """Use the explicit seed for block 0, then query the dummy API per block."""

    def __init__(
        self,
        *,
        prompt_text: str,
        seed_block_length: int,
        seed_refinement_steps: int,
        api: DummyTupleAPI,
    ) -> None:
        self.prompt_text = prompt_text
        self.seed_tuple = _normalize_tuple(
            seed_block_length,
            seed_refinement_steps,
        )
        self.api = api
        self.reset()

    def reset(self) -> None:
        self._history: list[BlockTuple] = []
        self._block_index = 0
        self.api.reset()
        self.prediction_trace: list[dict[str, object]] = []
        self.scheduler_predict_time_sec = 0.0

    def next_schedule(
        self,
        *,
        remaining_tokens: int,
        max_block_length: int,
        max_refinement_steps: int,
    ) -> ScheduledBlock:
        if remaining_tokens < 1:
            msg = "remaining_tokens must be positive"
            raise ValueError(msg)

        if self._block_index == 0:
            predicted_tuple = self.seed_tuple
            source = "seed"
            context: list[BlockTuple] = []
            predict_time_sec = 0.0
        else:
            predict_start = time.perf_counter()
            predicted_tuple = self.api.predict_tuple(
                prompt_text=self.prompt_text,
                block_index=self._block_index,
                history=list(self._history),
                remaining_tokens=remaining_tokens,
            )
            predict_time_sec = time.perf_counter() - predict_start
            self.scheduler_predict_time_sec += predict_time_sec
            source = "dummy_api"
            context = list(self._history)

        block_index = self._block_index
        self._block_index += 1
        applied_block_size = max(
            1,
            min(
                int(predicted_tuple.block_size),
                min(int(max_block_length), int(remaining_tokens)),
            ),
        )
        budgeted_refinement_steps = max(
            1,
            min(int(predicted_tuple.refinement_steps), int(max_refinement_steps)),
        )
        self.prediction_trace.append(
            {
                "block_index": int(block_index),
                "source": source,
                "predicted_tuple": _tuple_to_dict(predicted_tuple),
                "context": [_tuple_to_dict(item) for item in context],
                "remaining_tokens": int(remaining_tokens),
                "applied_block_size": int(applied_block_size),
                "budgeted_refinement_steps": int(budgeted_refinement_steps),
                "predict_time_sec": float(predict_time_sec),
            }
        )
        return ScheduledBlock(
            predicted_tuple=predicted_tuple,
            applied_block_size=applied_block_size,
            budgeted_refinement_steps=budgeted_refinement_steps,
        )

    def record_realized(self, applied_block_size: int, actual_nfe_used: int) -> None:
        realized_tuple = _normalize_tuple(applied_block_size, actual_nfe_used)
        self._history.append(realized_tuple)
        self.prediction_trace[-1]["realized_tuple"] = _tuple_to_dict(realized_tuple)

    @property
    def history(self) -> list[BlockTuple]:
        return list(self._history)


class CheckpointTupleScheduler:
    """Use the explicit seed for block 0, then query a phase_predict checkpoint."""

    def __init__(
        self,
        *,
        prompt_text: str,
        predictor_ckpt: str | Path,
        seed_block_length: int,
        seed_refinement_steps: int,
        predictor_device: str = "cpu",
        predictor: Predictor | None = None,
        context_seed_block_length: int | None = None,
        context_seed_stabilizing_steps: int | None = None,
        min_refinement_steps: int = 1,
        seed: EffectiveSeed | None = None,
    ) -> None:
        self.prompt_text = prompt_text
        self.seed_tuple = _normalize_tuple(
            seed_block_length,
            seed_refinement_steps,
        )

        if seed is not None:
            self.context_seed_tuple = ExtendedPhaseTuple(values={
                "block_size": seed.block_length,
                "nfe": seed.refinement_steps,
                "mean_top1_confidence": seed.context_mean_confidence,
                "min_top1_confidence": seed.context_min_confidence,
                "digit_fraction": seed.context_digit_fraction,
                "delimiter_fraction": seed.context_delimiter_fraction,
            })
        else:
            self.context_seed_tuple = _normalize_stabilizing_tuple(
                seed_block_length
                if context_seed_block_length is None
                else context_seed_block_length,
                seed_refinement_steps - 1
                if context_seed_stabilizing_steps is None
                else context_seed_stabilizing_steps,
            )
        self.min_refinement_steps = max(1, int(min_refinement_steps))

        if predictor is None:
            predictor = Predictor.from_checkpoint(
                str(_resolve_predictor_ckpt(predictor_ckpt)),
                device=_resolve_torch_device(predictor_device),
            )
        self.predictor = predictor
        self.reset()

    def reset(self) -> None:
        self._history: list[ExtendedPhaseTuple] = []
        self._block_index = 0
        self.prediction_trace: list[dict[str, object]] = []
        self.scheduler_predict_time_sec = 0.0

    def _padded_context(self) -> list[ExtendedPhaseTuple]:
        window_size = int(self.predictor.config.window_size)
        history = self._history[-window_size:]
        pad_count = max(0, window_size - len(history))
        return ([self.context_seed_tuple] * pad_count) + history

    def next_schedule(
        self,
        *,
        remaining_tokens: int,
        max_block_length: int,
        max_refinement_steps: int,
    ) -> ScheduledBlock:
        if remaining_tokens < 1:
            msg = "remaining_tokens must be positive"
            raise ValueError(msg)

        is_seed_block = self._block_index == 0
        if is_seed_block:
            predicted_tuple = self.seed_tuple
            result = SimpleNamespace(
                raw_output=None,
                metadata={"source": "seed", "window_size_used": 0},
            )
            context: list[ExtendedPhaseTuple] = []
            predict_time_sec = 0.0
        else:
            context = self._padded_context()
            predict_start = time.perf_counter()
            result = self.predictor.predict(context)
            predict_time_sec = time.perf_counter() - predict_start
            self.scheduler_predict_time_sec += predict_time_sec
            raw_predicted_tuple = result.predicted_tuple
            predicted_tuple = _normalize_tuple(
                raw_predicted_tuple.block_size,
                int(raw_predicted_tuple.refinement_steps),
            )
            result.metadata = {
                **dict(result.metadata),
                "source": "checkpoint",
                "stabilizing_step_offset": 1,
            }

        block_index = self._block_index
        self._block_index += 1

        # Detect death spiral: 3+ consecutive blocks with size <= 4
        recent = self._history[-3:]
        in_spiral = len(recent) >= 3 and all(
            hasattr(h, "values") and h.values.get("block_size", 0) <= 4
            for h in recent
        )
        if in_spiral and int(predicted_tuple.block_size) <= 4:
            applied_block_size = min(int(max_block_length), int(remaining_tokens))
        else:
            applied_block_size = max(
                1,
                min(
                    int(predicted_tuple.block_size),
                    min(int(max_block_length), int(remaining_tokens)),
                ),
            )
        budget_floor = (
            1
            if is_seed_block
            else min(self.min_refinement_steps, int(max_refinement_steps))
        )
        budgeted_refinement_steps = min(
            int(max_refinement_steps),
            max(budget_floor, int(predicted_tuple.refinement_steps)),
        )
        self.prediction_trace.append(
            {
                "block_index": int(block_index),
                "source": str(result.metadata.get("source", "checkpoint")),
                "predicted_tuple": _tuple_to_dict(predicted_tuple),
                "context": [_extended_tuple_to_dict(item) for item in context],
                "remaining_tokens": int(remaining_tokens),
                "applied_block_size": int(applied_block_size),
                "budgeted_refinement_steps": int(budgeted_refinement_steps),
                "raw_output": result.raw_output,
                "metadata": dict(result.metadata),
                "predict_time_sec": float(predict_time_sec),
                "context_seed_tuple": _extended_tuple_to_dict(self.context_seed_tuple),
            }
        )
        return ScheduledBlock(
            predicted_tuple=predicted_tuple,
            applied_block_size=applied_block_size,
            budgeted_refinement_steps=budgeted_refinement_steps,
        )

    def record_realized(
        self,
        applied_block_size: int,
        actual_nfe_used: int,
        mean_confidence: float = 1.0,
        min_confidence: float = 1.0,
        digit_fraction: float = 0.0,
        delimiter_fraction: float = 0.0,
    ) -> None:
        decode_tuple = _normalize_tuple(applied_block_size, actual_nfe_used)
        realized_et = ExtendedPhaseTuple(values={
            "block_size": max(1, int(applied_block_size)),
            "nfe": max(3, max(0, int(actual_nfe_used))) if int(applied_block_size) > 1 else max(0, int(actual_nfe_used)),
            "mean_top1_confidence": float(mean_confidence),
            "min_top1_confidence": float(min_confidence),
            "digit_fraction": float(digit_fraction),
            "delimiter_fraction": float(delimiter_fraction),
        })
        self._history.append(realized_et)
        self.prediction_trace[-1]["realized_tuple"] = realized_et.values
        self.prediction_trace[-1]["realized_decode_tuple"] = _tuple_to_dict(decode_tuple)

    @property
    def history(self) -> list[ExtendedPhaseTuple]:
        return list(self._history)


def _resolve_predictor_ckpt(path: str | Path | None) -> Path:
    candidate = DEFAULT_PREDICTOR_CKPT if path is None else Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _resolve_torch_device(device_name: str):
    import torch

    return torch.device(device_name)


def _resolve_torch_dtype(dtype_name: str | None):
    import torch

    if dtype_name is None:
        return None
    return getattr(torch, dtype_name)


def _maybe_disable_torch_compile(disable_torch_compile: bool) -> None:
    import torch

    if not disable_torch_compile or not hasattr(torch, "compile"):
        return
    if getattr(torch.compile, "__name__", "") == "_identity_torch_compile":
        return

    def _identity_torch_compile(fn=None, *args, **kwargs):
        del args, kwargs
        if fn is None:
            return _identity_torch_compile
        return fn

    torch.compile = _identity_torch_compile


def _load_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype_name: str | None,
    *,
    disable_torch_compile: bool,
):
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        msg = f"Requested device '{device}', but CUDA is not available"
        raise RuntimeError(msg)

    _maybe_disable_torch_compile(disable_torch_compile)

    dtype = _resolve_torch_dtype(dtype_name)
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    from model.modeling_llada import LLaDAModelLM

    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        config=config,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def _build_prompt(tokenizer, model_path: str, prompt: str) -> str:
    if "instruct" in model_path.lower():
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def _build_block_visualization(
    *,
    tokenizer,
    input_ids,
    output_ids,
    schedule_history: list[dict[str, object]],
    prediction_trace: list[dict[str, object]],
) -> list[dict[str, object]]:
    prompt_length = int(input_ids.shape[1])
    generated_ids = output_ids[0][prompt_length:].tolist()
    blocks: list[dict[str, object]] = []

    for index, schedule in enumerate(schedule_history):
        rel_start = int(schedule["block_start"]) - prompt_length
        rel_end = int(schedule["block_end"]) - prompt_length
        block_token_ids = generated_ids[rel_start:rel_end]
        block_text = tokenizer.decode(block_token_ids, skip_special_tokens=True)
        text_so_far = tokenizer.decode(generated_ids[:rel_end], skip_special_tokens=True)
        blocks.append(
            {
                "block_index": int(schedule["block_index"]),
                "predicted_tuple": dict(schedule["predicted_tuple"]),
                "applied_block_size": int(schedule["applied_block_size"]),
                "budgeted_refinement_steps": int(
                    schedule["budgeted_refinement_steps"]
                ),
                "actual_nfe_used": int(schedule["actual_nfe_used"]),
                "generated_span": {
                    "start": int(rel_start),
                    "end": int(rel_end),
                },
                "block_token_ids": block_token_ids,
                "block_text": block_text,
                "text_so_far": text_so_far,
                "predictor_trace": prediction_trace[index]
                if index < len(prediction_trace)
                else None,
            }
        )
    return blocks


def _adablock_first_seed(
    *,
    args: argparse.Namespace,
    model,
    prompt,
) -> EffectiveSeed:
    import torch
    from generate_adablock import compute_block_length, get_transfer_index
    from generate_pag import add_gumbel_noise

    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    assert args.threshold is not None, "threshold must be set"

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + args.gen_length),
        args.mask_id,
        dtype=torch.long,
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    with torch.no_grad():
        output = model(x)
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=args.temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe = 1

        block_length = compute_block_length(
            logits,
            predicted_tokens,
            prompt,
            args.gen_length,
            0,
            args.adablock_init_block_length,
            delimiter_ids=args.delimiter_ids,
            delimiter_threshold=args.delimiter_threshold,
        )
        block_start = prompt.shape[1]
        block_end = block_start + int(block_length)
        predictions_by_step = [predicted_tokens[0, block_start:block_end].detach().cpu()]

        mask_index = x == args.mask_id
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(
            logits,
            predicted_tokens,
            args.remasking,
            mask_index,
            x,
            None,
            args.threshold,
        )
        x[transfer_index] = x0[transfer_index]

        while (x[:, block_start:block_end] == args.mask_id).sum() != 0:
            mask_index = x == args.mask_id
            mask_index[:, block_end:] = 0
            block_output = model(x)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(
                block_logits,
                temperature=args.temperature,
            )
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            predictions_by_step.append(
                block_predicted_tokens[0, block_start:block_end].detach().cpu()
            )
            x0, transfer_index = get_transfer_index(
                block_logits,
                block_predicted_tokens,
                args.remasking,
                mask_index,
                x,
                None,
                args.threshold,
            )
            x[transfer_index] = x0[transfer_index]

        stabilizing_steps = _max_stabilizing_step(
            predictions_by_step,
            x[0, block_start:block_end],
        )

        # Compute confidence from final forward pass logits
        probs = torch.softmax(block_logits[:, block_start:block_end, :], dim=-1)
        max_probs = probs.max(dim=-1).values
        mean_conf = max_probs.mean().item()
        min_conf = max_probs.min().item()

        # Token-type fractions from decoded token IDs
        final_tokens = x[0, block_start:block_end]
        digit_ids = DIGIT_IDS_TENSOR
        delim_ids = DELIM_IDS_TENSOR
        if digit_ids is not None:
            digit_frac = torch.isin(final_tokens, digit_ids.to(x.device)).float().mean().item()
        else:
            digit_frac = 0.0
        if delim_ids is not None:
            delim_frac = torch.isin(final_tokens, delim_ids.to(x.device)).float().mean().item()
        else:
            delim_frac = 0.0

    return EffectiveSeed(
        block_length=max(1, int(block_length)),
        refinement_steps=max(1, int(nfe)),
        source="adablock_first_block",
        context_stabilizing_steps=max(0, int(stabilizing_steps)),
        context_mean_confidence=float(mean_conf),
        context_min_confidence=float(min_conf),
        context_digit_fraction=float(digit_frac),
        context_delimiter_fraction=float(delim_frac),
    )


def _effective_seed(
    *,
    args: argparse.Namespace,
    model,
    input_ids,
) -> EffectiveSeed:
    if not args.seed_from_adablock_first_block:
        return EffectiveSeed(
            block_length=max(1, int(args.seed_block_length)),
            refinement_steps=max(1, int(args.seed_refinement_steps)),
            source="explicit",
            context_stabilizing_steps=max(0, int(args.seed_refinement_steps) - 1),
        )
    return _adablock_first_seed(args=args, model=model, prompt=input_ids)


def _make_scheduler(
    args: argparse.Namespace,
    prompt_text: str,
    *,
    seed: EffectiveSeed,
):
    scripted_tuples = parse_tuple_schedule(args.dummy_tuples)
    if scripted_tuples:
        dummy_api = DummyTupleAPI(
            scripted_tuples=scripted_tuples,
            fallback_block_size=(
                args.fallback_block_length
                if args.fallback_block_length is not None
                else seed.block_length
            ),
            fallback_refinement_steps=(
                args.fallback_refinement_steps
                if args.fallback_refinement_steps is not None
                else seed.refinement_steps
            ),
            verbose=not args.quiet_api,
        )
        return DummyAPIScheduler(
            prompt_text=prompt_text,
            seed_block_length=seed.block_length,
            seed_refinement_steps=seed.refinement_steps,
            api=dummy_api,
        )

    predictor_ckpt = _resolve_predictor_ckpt(args.predictor_ckpt)
    if not predictor_ckpt.exists():
        msg = (
            "Predictor checkpoint not found at "
            f"{predictor_ckpt}. Pass --predictor-ckpt or use --dummy-tuples."
        )
        raise FileNotFoundError(msg)
    context_seed_block_length = (
        args.context_seed_block_length
        if args.context_seed_block_length is not None
        else args.seed_block_length
    )
    context_seed_stabilizing_steps = (
        args.context_seed_stabilizing_steps
        if args.context_seed_stabilizing_steps is not None
        else seed.context_stabilizing_steps
        if seed.context_stabilizing_steps is not None
        else max(0, int(args.seed_refinement_steps) - 1)
    )
    return CheckpointTupleScheduler(
        prompt_text=prompt_text,
        predictor_ckpt=predictor_ckpt,
        seed_block_length=seed.block_length,
        seed_refinement_steps=seed.refinement_steps,
        predictor_device=args.predictor_device,
        context_seed_block_length=context_seed_block_length,
        context_seed_stabilizing_steps=context_seed_stabilizing_steps,
        min_refinement_steps=args.min_refinement_steps,
        seed=seed,
    )


def _run_one_prompt(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    prompt_record: PromptRecord,
    run_id: str,
) -> dict[str, object]:
    import torch
    from generate_pag import generate_pag, generate_pag_dual_cache, generate_pag_prefix_cache

    user_input = _build_prompt(tokenizer, args.model_path, prompt_record.prompt)
    input_ids = torch.tensor(
        tokenizer(user_input)["input_ids"],
        device=args.device,
    ).unsqueeze(0)
    seed = _effective_seed(args=args, model=model, input_ids=input_ids)
    if seed.source == "adablock_first_block":
        print(
            "  AdaBlock seed: "
            f"block_length={seed.block_length} refinement_steps={seed.refinement_steps}"
        )
    scheduler = _make_scheduler(args, prompt_record.prompt, seed=seed)

    if args.use_cache and args.dual_cache:
        generator = generate_pag_dual_cache
    elif args.use_cache:
        generator = generate_pag_prefix_cache
    else:
        generator = generate_pag

    output_ids, nfe_history, block_history, schedule_history = generator(
        model,
        input_ids,
        scheduler,
        steps=args.steps,
        gen_length=args.gen_length,
        temperature=args.temperature,
        remasking=args.remasking,
        mask_id=args.mask_id,
        threshold=args.threshold,
        max_block_length=args.max_block_length,
        max_refinement_steps=args.max_refinement_steps,
    )

    generated_text = tokenizer.decode(
        output_ids[0][input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    block_visualization = _build_block_visualization(
        tokenizer=tokenizer,
        input_ids=input_ids,
        output_ids=output_ids,
        schedule_history=schedule_history,
        prediction_trace=scheduler.prediction_trace,
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "prompt_id": prompt_record.prompt_id,
        "prompt_category": prompt_record.category,
        "prompt_tags": prompt_record.tags or [],
        "prompt_notes": prompt_record.notes,
        "prompt": prompt_record.prompt,
        "wrapped_prompt": user_input,
        "generated_text": generated_text,
        "nfe_history": nfe_history,
        "block_history": block_history,
        "schedule_history": schedule_history,
        "scheduler_trace": scheduler.prediction_trace,
        "block_visualization": block_visualization,
        "realized_history": [_tuple_to_dict(item) for item in scheduler.history],
        "predictor_checkpoint": str(_resolve_predictor_ckpt(args.predictor_ckpt)),
        "used_dummy_schedule": bool(parse_tuple_schedule(args.dummy_tuples)),
        "config": {
            "model_path": args.model_path,
            "gen_length": args.gen_length,
            "steps": args.steps,
            "threshold": args.threshold,
            "requested_seed_block_length": args.seed_block_length,
            "requested_seed_refinement_steps": args.seed_refinement_steps,
            "effective_seed_block_length": seed.block_length,
            "effective_seed_refinement_steps": seed.refinement_steps,
            "effective_context_seed_stabilizing_steps": seed.context_stabilizing_steps,
            "seed_source": seed.source,
            "seed_from_adablock_first_block": args.seed_from_adablock_first_block,
            "adablock_init_block_length": args.adablock_init_block_length,
            "delimiter_ids": args.delimiter_ids,
            "delimiter_threshold": args.delimiter_threshold,
            "predictor_device": args.predictor_device,
            "device": args.device,
            "dtype": args.dtype,
            "use_cache": args.use_cache,
            "dual_cache": args.dual_cache,
            "disable_torch_compile": args.disable_torch_compile,
            "max_block_length": args.max_block_length,
            "max_refinement_steps": args.max_refinement_steps,
            "min_refinement_steps": args.min_refinement_steps,
            "context_seed_block_length": args.context_seed_block_length,
            "context_seed_stabilizing_steps": args.context_seed_stabilizing_steps,
        },
        "summary": {
            "num_blocks": len(block_history),
            "total_nfe": sum(nfe_history),
            "avg_block_size": (sum(block_history) / len(block_history))
            if block_history
            else 0,
            "avg_refinement_steps": (sum(nfe_history) / len(nfe_history))
            if nfe_history
            else 0,
        },
    }


def _resolve_log_file(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def write_log_record(path: str | Path | None, record: dict[str, object]) -> None:
    log_path = _resolve_log_file(path)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_records_for_args(args: argparse.Namespace) -> list[PromptRecord]:
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = ROOT / prompt_path
        records = load_prompt_records(prompt_path)
    else:
        if not args.prompt:
            msg = "Either --prompt or --prompt-file is required"
            raise ValueError(msg)
        records = [_prompt_record_from_args(args)]
    if args.max_prompts is not None:
        records = records[: args.max_prompts]
    return records


def run_prompts(args: argparse.Namespace) -> list[dict[str, object]]:
    model, tokenizer = _load_model_and_tokenizer(
        args.model_path,
        args.device,
        args.dtype,
        disable_torch_compile=args.disable_torch_compile,
    )
    records = _load_records_for_args(args)
    if not records:
        msg = "No prompts were found to run"
        raise ValueError(msg)
    run_id = args.run_id or uuid4().hex
    results = []
    for index, prompt_record in enumerate(records, start=1):
        print(
            f"[{index}/{len(records)}] Running prompt "
            f"{prompt_record.prompt_id or index}: {prompt_record.prompt[:90]}"
        )
        result = _run_one_prompt(
            args=args,
            model=model,
            tokenizer=tokenizer,
            prompt_record=prompt_record,
            run_id=run_id,
        )
        results.append(result)
        write_log_record(args.log_file, result)
        if args.log_file is not None:
            print(f"  wrote log record: {_resolve_log_file(args.log_file)}")
    return results


def run_prompt(args: argparse.Namespace) -> dict[str, object]:
    return run_prompts(args)[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one LLaDA PAG generation using a phase_predict checkpoint by "
            "default, or a dummy tuple schedule when --dummy-tuples is set."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--prompt-category", default=None)
    parser.add_argument(
        "--prompt-tags",
        default=None,
        help="Comma-separated tags for a single --prompt run.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=(
            "JSONL or JSON prompt suite. JSONL entries may contain id, category, "
            "tags, notes, and prompt."
        ),
    )
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE.relative_to(ROOT)),
        help="Append one structured JSON record per prompt to this JSONL file.",
    )
    parser.add_argument(
        "--no-log",
        dest="log_file",
        action="store_const",
        const=None,
        help="Disable JSONL logging.",
    )
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed-block-length", type=int, default=32)
    parser.add_argument("--seed-refinement-steps", type=int, default=4)
    parser.add_argument(
        "--seed-from-adablock-first-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run an AdaBlock-style first-block probe and use its realized "
            "(block_size, nfe) as the PAG seed tuple. Pass "
            "--no-seed-from-adablock-first-block to use explicit seed args."
        ),
    )
    parser.add_argument(
        "--adablock-init-block-length",
        type=int,
        default=32,
        help="Default AdaBlock block length used by --seed-from-adablock-first-block.",
    )
    parser.add_argument(
        "--delimiter-threshold",
        type=float,
        default=0.3,
        help="AdaBlock delimiter confidence threshold used for the seed probe.",
    )
    parser.add_argument(
        "--delimiter-ids",
        type=lambda raw: [int(item.strip()) for item in raw.split(",") if item.strip()],
        default=[198],
        help="Comma-separated delimiter token ids used for the AdaBlock seed probe.",
    )
    parser.add_argument(
        "--predictor-ckpt",
        default=None,
        help=(
            "Path to the phase_predict checkpoint. Defaults to "
            "output/phase_predict_model_checkpoint.pt relative to the repo root."
        ),
    )
    parser.add_argument(
        "--predictor-device",
        default="cpu",
        help="Device used for tuple prediction, for example cpu or cuda.",
    )
    parser.add_argument(
        "--dummy-tuples",
        default="",
        help=(
            "Optional comma-separated tuples used after block 0, for example "
            "'16:3,8:2,8:1'. If provided, the real predictor checkpoint is bypassed."
        ),
    )
    parser.add_argument("--fallback-block-length", type=int, default=None)
    parser.add_argument("--fallback-refinement-steps", type=int, default=None)
    parser.add_argument("--max-block-length", type=int, default=None)
    parser.add_argument("--max-refinement-steps", type=int, default=None)
    parser.add_argument(
        "--min-refinement-steps",
        type=int,
        default=3,
        help="Minimum total PAG refinement budget for checkpoint-predicted blocks.",
    )
    parser.add_argument(
        "--context-seed-block-length",
        type=int,
        default=None,
        help=(
            "Block size used for predictor context left-padding. Defaults to "
            "--seed-block-length, not the AdaBlock-derived decode seed."
        ),
    )
    parser.add_argument(
        "--context-seed-stabilizing-steps",
        type=int,
        default=None,
        help=(
            "Stabilizing-step value used for predictor context left-padding. "
            "Defaults to --seed-refinement-steps - 1."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        choices=["low_confidence", "random"],
        default="low_confidence",
    )
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--dual-cache", action="store_true")
    parser.add_argument("--quiet-api", action="store_true")
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        default=True,
        help=(
            "Disable the torch.compile decorator used by the LLaDA attention "
            "wrapper. Keep this enabled unless Triton is installed and working."
        ),
    )
    parser.add_argument(
        "--enable-torch-compile",
        dest="disable_torch_compile",
        action="store_false",
        help="Re-enable torch.compile for LLaDA if your environment has Triton.",
    )
    return parser


def _print_result(result: dict[str, object]) -> None:
    print("=" * 20)
    print(f"Prompt: {result['prompt']}")
    if result.get("prompt_id"):
        print(f"Prompt ID: {result['prompt_id']}")
    if result.get("prompt_category"):
        print(f"Category: {result['prompt_category']}")
    print(f"Predictor checkpoint: {result['predictor_checkpoint']}")
    print(f"Dummy schedule mode: {result['used_dummy_schedule']}")
    config = result.get("config", {})
    print(
        "Seed tuple: "
        f"source={config.get('seed_source')} "
        f"effective=({config.get('effective_seed_block_length')}, "
        f"{config.get('effective_seed_refinement_steps')})"
    )
    print()
    print("Generated text:")
    print(result["generated_text"])
    print()
    print(f"NFE history: {result['nfe_history']}")
    print(f"Block history: {result['block_history']}")
    print()
    print("Block-by-block trace:")
    for block in result["block_visualization"]:
        predicted = block["predicted_tuple"]
        print(
            f"[Block {block['block_index']}] "
            f"predicted=({predicted['block_size']}, {predicted['refinement_steps']}) "
            f"applied={block['applied_block_size']} "
            f"budget={block['budgeted_refinement_steps']} "
            f"actual_nfe={block['actual_nfe_used']} "
            f"span={block['generated_span']['start']}:{block['generated_span']['end']}"
        )
        predictor_trace = block["predictor_trace"]
        if predictor_trace is not None:
            print(f"  source: {predictor_trace['source']}")
            if predictor_trace.get("context"):
                print(f"  context: {json.dumps(predictor_trace['context'])}")
            if predictor_trace.get("raw_output") is not None:
                raw_output = [round(float(value), 3) for value in predictor_trace["raw_output"]]
                print(f"  raw_output: {raw_output}")
        print(f"  block_text: {json.dumps(block['block_text'])}")
        print(f"  token_ids: {block['block_token_ids']}")
        print(f"  text_so_far: {json.dumps(block['text_so_far'])}")
    print("=" * 20)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.dual_cache and not args.use_cache:
        parser.error("--dual-cache requires --use-cache")

    results = run_prompts(args)
    for result in results:
        _print_result(result)
    if args.log_file is not None:
        print(f"Structured log written to: {_resolve_log_file(args.log_file)}")


if __name__ == "__main__":
    main()

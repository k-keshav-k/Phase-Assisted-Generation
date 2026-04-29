from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Predictor = importlib.import_module("phase_predict.predict").Predictor
PhaseTuple = importlib.import_module("phase_predict.schema").PhaseTuple

DEFAULT_PREDICTOR_CKPT = ROOT / "output" / "phase_predict_model_checkpoint.pt"
BlockTuple = PhaseTuple


@dataclass(slots=True)
class ScheduledBlock:
    predicted_tuple: BlockTuple
    applied_block_size: int
    budgeted_refinement_steps: int


def _tuple_to_dict(value: BlockTuple) -> dict[str, int]:
    return {
        "block_size": int(value.block_size),
        "refinement_steps": int(value.refinement_steps),
    }


def _normalize_tuple(block_size: int, refinement_steps: int) -> BlockTuple:
    return BlockTuple(
        block_size=max(1, int(block_size)),
        refinement_steps=max(1, int(refinement_steps)),
    )


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
        else:
            predicted_tuple = self.api.predict_tuple(
                prompt_text=self.prompt_text,
                block_index=self._block_index,
                history=list(self._history),
                remaining_tokens=remaining_tokens,
            )
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
    ) -> None:
        self.prompt_text = prompt_text
        self.seed_tuple = _normalize_tuple(
            seed_block_length,
            seed_refinement_steps,
        )

        if predictor is None:
            predictor = Predictor.from_checkpoint(
                str(_resolve_predictor_ckpt(predictor_ckpt)),
                device=_resolve_torch_device(predictor_device),
            )
        self.predictor = predictor
        self.reset()

    def reset(self) -> None:
        self._history: list[BlockTuple] = []
        self._block_index = 0
        self.prediction_trace: list[dict[str, object]] = []

    def _padded_context(self) -> list[BlockTuple]:
        window_size = int(self.predictor.config.window_size)
        history = self._history[-window_size:]
        pad_count = max(0, window_size - len(history))
        return ([self.seed_tuple] * pad_count) + history

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
            result = SimpleNamespace(
                raw_output=None,
                metadata={"source": "seed", "window_size_used": 0},
            )
            context: list[BlockTuple] = []
        else:
            context = self._padded_context()
            result = self.predictor.predict(context)
            predicted_tuple = _normalize_tuple(
                result.predicted_tuple.block_size,
                result.predicted_tuple.refinement_steps,
            )
            result.metadata = {
                **dict(result.metadata),
                "source": "checkpoint",
            }

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
                "source": str(result.metadata.get("source", "checkpoint")),
                "predicted_tuple": _tuple_to_dict(predicted_tuple),
                "context": [_tuple_to_dict(item) for item in context],
                "remaining_tokens": int(remaining_tokens),
                "applied_block_size": int(applied_block_size),
                "budgeted_refinement_steps": int(budgeted_refinement_steps),
                "raw_output": result.raw_output,
                "metadata": dict(result.metadata),
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


def _make_scheduler(args: argparse.Namespace):
    scripted_tuples = parse_tuple_schedule(args.dummy_tuples)
    if scripted_tuples:
        dummy_api = DummyTupleAPI(
            scripted_tuples=scripted_tuples,
            fallback_block_size=(
                args.fallback_block_length
                if args.fallback_block_length is not None
                else args.seed_block_length
            ),
            fallback_refinement_steps=(
                args.fallback_refinement_steps
                if args.fallback_refinement_steps is not None
                else args.seed_refinement_steps
            ),
            verbose=not args.quiet_api,
        )
        return DummyAPIScheduler(
            prompt_text=args.prompt,
            seed_block_length=args.seed_block_length,
            seed_refinement_steps=args.seed_refinement_steps,
            api=dummy_api,
        )

    predictor_ckpt = _resolve_predictor_ckpt(args.predictor_ckpt)
    if not predictor_ckpt.exists():
        msg = (
            "Predictor checkpoint not found at "
            f"{predictor_ckpt}. Pass --predictor-ckpt or use --dummy-tuples."
        )
        raise FileNotFoundError(msg)
    return CheckpointTupleScheduler(
        prompt_text=args.prompt,
        predictor_ckpt=predictor_ckpt,
        seed_block_length=args.seed_block_length,
        seed_refinement_steps=args.seed_refinement_steps,
        predictor_device=args.predictor_device,
    )


def run_prompt(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from generate_pag import (
        generate_pag,
        generate_pag_dual_cache,
        generate_pag_prefix_cache,
    )

    model, tokenizer = _load_model_and_tokenizer(
        args.model_path,
        args.device,
        args.dtype,
        disable_torch_compile=args.disable_torch_compile,
    )
    user_input = _build_prompt(tokenizer, args.model_path, args.prompt)
    input_ids = torch.tensor(
        tokenizer(user_input)["input_ids"],
        device=args.device,
    ).unsqueeze(0)
    scheduler = _make_scheduler(args)

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
        "prompt": args.prompt,
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
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one LLaDA PAG generation using a phase_predict checkpoint by "
            "default, or a dummy tuple schedule when --dummy-tuples is set."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed-block-length", type=int, default=32)
    parser.add_argument("--seed-refinement-steps", type=int, default=4)
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


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.dual_cache and not args.use_cache:
        parser.error("--dual-cache requires --use-cache")

    result = run_prompt(args)
    print("=" * 20)
    print(f"Prompt: {result['prompt']}")
    print(f"Predictor checkpoint: {result['predictor_checkpoint']}")
    print(f"Dummy schedule mode: {result['used_dummy_schedule']}")
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


if __name__ == "__main__":
    main()

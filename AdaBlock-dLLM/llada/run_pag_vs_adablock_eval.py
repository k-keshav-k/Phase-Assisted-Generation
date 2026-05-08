from __future__ import annotations

import argparse
import copy
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch
from run_pag_dummy_api import (
    ROOT,
    EffectiveSeed,
    PromptRecord,
    _build_block_visualization,
    _build_prompt,
    _load_model_and_tokenizer,
    _make_scheduler,
    _resolve_log_file,
    _resolve_predictor_ckpt,
    write_log_record,
)

DEFAULT_PROMPT_FILE = ROOT / "AdaBlock-dLLM" / "llada" / "quick_eval_prompts.jsonl"
DEFAULT_LOG_FILE = ROOT / "logs" / "llada_pag_vs_adablock_eval.jsonl"


@dataclass(slots=True)
class EvalPromptRecord(PromptRecord):
    expected_contains: list[str] | None = None
    expected_answers: list[str] | None = None


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def load_eval_prompts(path: str | Path) -> list[EvalPromptRecord]:
    prompt_path = _resolve_path(path)
    records: list[EvalPromptRecord] = []
    with prompt_path.open(encoding="utf-8") as file_obj:
        for index, line in enumerate(file_obj):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            expected = payload.get("expected_contains") or []
            if isinstance(expected, str):
                expected = [expected]
            expected_answers = (
                payload.get("expected_answers")
                or payload.get("accepted_answers")
                or payload.get("answer")
                or []
            )
            if isinstance(expected_answers, str):
                expected_answers = [expected_answers]
            records.append(
                EvalPromptRecord(
                    prompt=payload["prompt"],
                    prompt_id=str(payload.get("id", f"prompt_{index:03d}")),
                    category=payload.get("category"),
                    tags=[str(tag) for tag in payload.get("tags", [])],
                    notes=payload.get("notes"),
                    expected_contains=[str(item) for item in expected],
                    expected_answers=[str(item) for item in expected_answers],
                )
            )
    return records


def _normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("$", "")
    value = value.replace(",", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _substring_score(text: str, expected_contains: list[str] | None) -> dict[str, object]:
    expected = expected_contains or []
    normalized = _normalize_text(text)
    matched = [item for item in expected if _normalize_text(item) in normalized]
    return {
        "expected_contains": expected,
        "matched": matched,
        "missing": [item for item in expected if item not in matched],
        "score": (len(matched) / len(expected)) if expected else None,
    }


def _answer_present(text: str, answer: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_answer = _normalize_text(answer)
    if not normalized_answer:
        return False
    if re.search(r"\d", normalized_answer):
        # Avoid counting "72" as correct when the model emitted "772".
        pattern = rf"(?<!\d){re.escape(normalized_answer)}(?!\d)"
        return re.search(pattern, normalized_text) is not None
    return normalized_answer in normalized_text


def _answer_score(text: str, expected_answers: list[str] | None) -> dict[str, object]:
    expected = expected_answers or []
    matched = [answer for answer in expected if _answer_present(text, answer)]
    return {
        "expected_answers": expected,
        "matched": matched,
        "missing": [answer for answer in expected if answer not in matched],
        "is_correct": bool(matched) if expected else None,
        "score": 1.0 if matched else 0.0 if expected else None,
    }


def _decode_generation(tokenizer, output_ids: torch.Tensor, input_ids: torch.Tensor) -> str:
    return tokenizer.decode(
        output_ids[0][input_ids.shape[1] :],
        skip_special_tokens=True,
    )


def _synchronize_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_history_block_visualization(
    *,
    tokenizer,
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    block_history: list[int],
    nfe_history: list[int],
) -> list[dict[str, object]]:
    prompt_length = int(input_ids.shape[1])
    generated_ids = output_ids[0][prompt_length:].tolist()
    blocks: list[dict[str, object]] = []
    cursor = 0

    for index, block_size in enumerate(block_history):
        rel_start = cursor
        rel_end = min(cursor + int(block_size), len(generated_ids))
        block_token_ids = generated_ids[rel_start:rel_end]
        actual_nfe = int(nfe_history[index]) if index < len(nfe_history) else None
        block_text = tokenizer.decode(block_token_ids, skip_special_tokens=True)
        text_so_far = tokenizer.decode(generated_ids[:rel_end], skip_special_tokens=True)
        blocks.append(
            {
                "block_index": index,
                "predicted_tuple": {
                    "block_size": int(block_size),
                    "refinement_steps": actual_nfe,
                },
                "applied_block_size": int(block_size),
                "budgeted_refinement_steps": actual_nfe,
                "actual_nfe_used": actual_nfe,
                "generated_span": {
                    "start": int(rel_start),
                    "end": int(rel_end),
                },
                "block_token_ids": block_token_ids,
                "block_text": block_text,
                "text_so_far": text_so_far,
                "predictor_trace": None,
            }
        )
        cursor = rel_end
        if cursor >= len(generated_ids):
            break

    return blocks


def _summarize_method(
    *,
    method: str,
    generated_text: str,
    nfe_history: list[int],
    block_history: list[int],
    elapsed_sec: float,
    expected_contains: list[str] | None,
    expected_answers: list[str] | None,
    block_visualization: list[dict[str, object]] | None = None,
    scheduler_predict_time_sec: float = 0.0,
) -> dict[str, object]:
    scheduler_predict_time_sec = max(0.0, float(scheduler_predict_time_sec))
    total_elapsed_sec = max(0.0, float(elapsed_sec))
    llada_decode_time_sec = max(0.0, total_elapsed_sec - scheduler_predict_time_sec)
    return {
        "method": method,
        "generated_text": generated_text,
        "nfe_history": nfe_history,
        "block_history": block_history,
        "block_visualization": block_visualization or [],
        "metrics": {
            "elapsed_sec": total_elapsed_sec,
            "total_elapsed_sec": total_elapsed_sec,
            "scheduler_predict_time_sec": scheduler_predict_time_sec,
            "llada_decode_time_sec": llada_decode_time_sec,
            "total_nfe": sum(nfe_history),
            "num_blocks": len(block_history),
            "avg_block_size": (sum(block_history) / len(block_history))
            if block_history
            else 0,
            "avg_nfe_per_block": (sum(nfe_history) / len(nfe_history))
            if nfe_history
            else 0,
            "decoded_chars": len(generated_text),
            "substring_check": _substring_score(generated_text, expected_contains),
            "answer_check": _answer_score(generated_text, expected_answers),
        },
    }


def _run_pag(args: argparse.Namespace, model, tokenizer, record: EvalPromptRecord):
    from generate_pag import generate_pag, generate_pag_dual_cache, generate_pag_prefix_cache
    from run_pag_dummy_api import DIGIT_IDS_TENSOR, DELIM_IDS_TENSOR

    user_input = _build_prompt(tokenizer, args.model_path, record.prompt)
    input_ids = torch.tensor(tokenizer(user_input)["input_ids"], device=args.device).unsqueeze(0)

    # Precompute digit and delimiter token ID sets for this tokenizer
    digit_ids = set()
    for tid in range(tokenizer.vocab_size):
        text = tokenizer.decode([tid]).strip()
        if text and all(c.isdigit() for c in text):
            digit_ids.add(tid)
    delimiter_ids = set(args.delimiter_ids or [198])
    for tid in range(tokenizer.vocab_size):
        text = tokenizer.decode([tid]).strip()
        if text in {"\n", "<|endoftext|>", "<|eot_id|>"}:
            delimiter_ids.add(tid)
    # Assign to module-level globals for the probe function
    import run_pag_dummy_api as rpda
    rpda.DIGIT_IDS_TENSOR = torch.tensor(list(digit_ids), dtype=torch.long)
    rpda.DELIM_IDS_TENSOR = torch.tensor(list(delimiter_ids), dtype=torch.long)
    digit_cache = rpda.DIGIT_IDS_TENSOR.to(args.device)
    delim_cache = rpda.DELIM_IDS_TENSOR.to(args.device)
    scheduler = _make_scheduler(
        args,
        record.prompt,
        seed=EffectiveSeed(
            block_length=args.seed_block_length,
            refinement_steps=args.seed_refinement_steps,
            source="comparator_effective",
        ),
    )
    generator = (
        generate_pag_dual_cache
        if args.use_cache and args.dual_cache
        else generate_pag_prefix_cache
        if args.use_cache
        else generate_pag
    )

    _synchronize_if_cuda(args.device)
    start = time.perf_counter()
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
        digit_ids_tensor=digit_cache,
        delimiter_ids_tensor=delim_cache,
        delimiter_ids=args.delimiter_ids,
        delimiter_threshold=args.delimiter_threshold,
        tau_commit=args.tau_commit,
        tau_stable_steps=args.tau_stable_steps,
        default_block_length=args.adablock_init_block_length,
    )
    _synchronize_if_cuda(args.device)
    elapsed = time.perf_counter() - start
    scheduler_predict_time_sec = float(getattr(scheduler, "scheduler_predict_time_sec", 0.0))
    generated_text = _decode_generation(tokenizer, output_ids, input_ids)
    block_visualization = _build_block_visualization(
        tokenizer=tokenizer,
        input_ids=input_ids,
        output_ids=output_ids,
        schedule_history=schedule_history,
        prediction_trace=scheduler.prediction_trace,
    )
    return _summarize_method(
        method="pag",
        generated_text=generated_text,
        nfe_history=nfe_history,
        block_history=block_history,
        elapsed_sec=elapsed,
        expected_contains=record.expected_contains,
        expected_answers=record.expected_answers,
        block_visualization=block_visualization,
        scheduler_predict_time_sec=scheduler_predict_time_sec,
    )


def _run_adablock(args: argparse.Namespace, model, tokenizer, record: EvalPromptRecord):
    from generate_adablock import (
        generate_adablock,
        generate_adablock_dual_cache,
        generate_adablock_prefix_cache,
    )

    user_input = _build_prompt(tokenizer, args.model_path, record.prompt)
    input_ids = torch.tensor(tokenizer(user_input)["input_ids"], device=args.device).unsqueeze(0)
    generator = (
        generate_adablock_dual_cache
        if args.use_cache and args.dual_cache
        else generate_adablock_prefix_cache
        if args.use_cache
        else generate_adablock
    )

    _synchronize_if_cuda(args.device)
    start = time.perf_counter()
    output_ids, nfe_history, block_history = generator(
        model,
        input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        init_block_length=args.adablock_init_block_length,
        temperature=args.temperature,
        remasking=args.remasking,
        mask_id=args.mask_id,
        threshold=args.threshold,
        delimiter_ids=args.delimiter_ids,
        delimiter_threshold=args.delimiter_threshold,
    )
    _synchronize_if_cuda(args.device)
    elapsed = time.perf_counter() - start
    generated_text = _decode_generation(tokenizer, output_ids, input_ids)
    block_visualization = _build_history_block_visualization(
        tokenizer=tokenizer,
        input_ids=input_ids,
        output_ids=output_ids,
        block_history=block_history,
        nfe_history=nfe_history,
    )
    return _summarize_method(
        method="adablock",
        generated_text=generated_text,
        nfe_history=nfe_history,
        block_history=block_history,
        elapsed_sec=elapsed,
        expected_contains=record.expected_contains,
        expected_answers=record.expected_answers,
        block_visualization=block_visualization,
    )


def _comparison_delta(pag: dict[str, object], adablock: dict[str, object]) -> dict[str, object]:
    pag_metrics = pag["metrics"]
    adablock_metrics = adablock["metrics"]
    pag_total_elapsed = pag_metrics.get("total_elapsed_sec", pag_metrics["elapsed_sec"])
    adablock_total_elapsed = adablock_metrics.get(
        "total_elapsed_sec",
        adablock_metrics["elapsed_sec"],
    )
    pag_decode = pag_metrics.get("llada_decode_time_sec", pag_total_elapsed)
    adablock_decode = adablock_metrics.get("llada_decode_time_sec", adablock_total_elapsed)
    pag_predict = pag_metrics.get("scheduler_predict_time_sec", 0.0)
    adablock_predict = adablock_metrics.get("scheduler_predict_time_sec", 0.0)
    return {
        "nfe_delta_pag_minus_adablock": pag_metrics["total_nfe"] - adablock_metrics["total_nfe"],
        "nfe_ratio_pag_over_adablock": (
            pag_metrics["total_nfe"] / adablock_metrics["total_nfe"]
            if adablock_metrics["total_nfe"]
            else None
        ),
        "elapsed_delta_sec_pag_minus_adablock": (
            pag_metrics["elapsed_sec"] - adablock_metrics["elapsed_sec"]
        ),
        "total_elapsed_delta_sec_pag_minus_adablock": (
            pag_total_elapsed - adablock_total_elapsed
        ),
        "llada_decode_delta_sec_pag_minus_adablock": (
            pag_decode - adablock_decode
        ),
        "scheduler_predict_delta_sec_pag_minus_adablock": (
            pag_predict - adablock_predict
        ),
        "block_count_delta_pag_minus_adablock": (
            pag_metrics["num_blocks"] - adablock_metrics["num_blocks"]
        ),
        "substring_score_delta_pag_minus_adablock": (
            pag_metrics["substring_check"]["score"] - adablock_metrics["substring_check"]["score"]
            if pag_metrics["substring_check"]["score"] is not None
            and adablock_metrics["substring_check"]["score"] is not None
            else None
        ),
        "answer_score_delta_pag_minus_adablock": (
            pag_metrics["answer_check"]["score"] - adablock_metrics["answer_check"]["score"]
            if pag_metrics["answer_check"]["score"] is not None
            and adablock_metrics["answer_check"]["score"] is not None
            else None
        ),
    }


def _args_with_pag_seed(
    args: argparse.Namespace,
    *,
    seed: EffectiveSeed,
) -> argparse.Namespace:
    seeded_args = copy.copy(args)
    seeded_args.seed_block_length = max(1, int(seed.block_length))
    seeded_args.seed_refinement_steps = max(1, int(seed.refinement_steps))
    if seed.context_stabilizing_steps is not None:
        seeded_args.context_seed_stabilizing_steps = max(
            0,
            int(seed.context_stabilizing_steps),
        )
    return seeded_args


def _adablock_first_seed(adablock: dict[str, object]) -> tuple[int, int]:
    block_history = adablock.get("block_history") or []
    nfe_history = adablock.get("nfe_history") or []
    if not block_history or not nfe_history:
        msg = "AdaBlock did not return a first block/nfe seed"
        raise ValueError(msg)
    return int(block_history[0]), int(nfe_history[0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a quick qualitative/efficiency comparison of PAG vs AdaBlock."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE.relative_to(ROOT)))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE.relative_to(ROOT)))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed-block-length", type=int, default=32)
    parser.add_argument("--seed-refinement-steps", type=int, default=4)
    parser.add_argument("--predictor-ckpt", default=None)
    parser.add_argument("--predictor-device", default="cpu")
    parser.add_argument("--max-block-length", type=int, default=None)
    parser.add_argument("--max-refinement-steps", type=int, default=None)
    parser.add_argument("--min-refinement-steps", type=int, default=3)
    parser.add_argument("--min-block-length", type=int, default=4,
                        help="Minimum block size (default: 4). Set 1 for no floor.")
    parser.add_argument("--refinement-step-offset", type=int, default=1,
                        help="Offset added to predicted refinement steps (default: 1). Set 0 for no offset.")
    parser.add_argument("--tau-commit", type=float, default=0.80,
                        help="Min confidence for soft-cap exit (default: 0.80)")
    parser.add_argument("--tau-stable-steps", type=int, default=2,
                        help="Steps of stable predictions required for exit (default: 2)")
    parser.add_argument("--context-seed-block-length", type=int, default=None)
    parser.add_argument("--context-seed-stabilizing-steps", type=int, default=None)
    parser.add_argument("--adablock-init-block-length", type=int, default=32)
    parser.add_argument("--delimiter-threshold", type=float, default=0.3)
    parser.add_argument(
        "--delimiter-ids",
        type=lambda raw: [int(item.strip()) for item in raw.split(",") if item.strip()],
        default=[198],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        choices=["low_confidence", "random"],
        default="low_confidence",
    )
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--dual-cache", action="store_true")
    parser.add_argument("--dummy-tuples", default="")
    parser.add_argument("--fallback-block-length", type=int, default=None)
    parser.add_argument("--fallback-refinement-steps", type=int, default=None)
    parser.add_argument("--quiet-api", action="store_true")
    parser.add_argument(
        "--seed-from-adablock-first-block",
        "--match-adablock-initial-seed",
        dest="seed_from_adablock_first_block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Seed PAG block 0 from AdaBlock's realized first "
            "(block_size, nfe) for the same prompt. Pass "
            "--no-seed-from-adablock-first-block to use explicit seed args."
        ),
    )
    parser.add_argument("--disable-torch-compile", action="store_true", default=True)
    parser.add_argument(
        "--enable-torch-compile",
        dest="disable_torch_compile",
        action="store_false",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dual_cache and not args.use_cache:
        raise ValueError("--dual-cache requires --use-cache")

    records = load_eval_prompts(args.prompt_file)
    if args.max_prompts is not None:
        records = records[: args.max_prompts]
    if not records:
        raise ValueError("No evaluation prompts found")

    model, tokenizer = _load_model_and_tokenizer(
        args.model_path,
        args.device,
        args.dtype,
        disable_torch_compile=args.disable_torch_compile,
    )
    run_id = args.run_id or uuid4().hex
    log_file = _resolve_log_file(args.log_file)
    print(f"Writing comparison logs to: {log_file}")

    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record.prompt_id}: {record.prompt[:90]}")
        adablock = _run_adablock(args, model, tokenizer, record)
        pag_args = args
        seed_source = "explicit"
        context_seed_stabilizing_steps = None
        if args.seed_from_adablock_first_block:
            block_size, nfe = _adablock_first_seed(adablock)
            seed = EffectiveSeed(
                block_length=block_size,
                refinement_steps=nfe,
                source="adablock_first_block",
                context_stabilizing_steps=max(0, nfe - 1),
            )
            pag_args = _args_with_pag_seed(args, seed=seed)
            seed_source = "adablock_first_block"
            context_seed_stabilizing_steps = max(0, nfe - 1)
        pag = _run_pag(pag_args, model, tokenizer, record)
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "prompt_id": record.prompt_id,
            "prompt_category": record.category,
            "prompt_tags": record.tags or [],
            "prompt": record.prompt,
            "expected_contains": record.expected_contains or [],
            "expected_answers": record.expected_answers or [],
            "predictor_checkpoint": str(_resolve_predictor_ckpt(args.predictor_ckpt)),
            "config": {
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "steps": args.steps,
                "threshold": args.threshold,
                "requested_seed_block_length": args.seed_block_length,
                "requested_seed_refinement_steps": args.seed_refinement_steps,
                "effective_seed_block_length": pag_args.seed_block_length,
                "effective_seed_refinement_steps": pag_args.seed_refinement_steps,
                "effective_context_seed_stabilizing_steps": (
                    context_seed_stabilizing_steps
                ),
                "seed_source": seed_source,
                "seed_from_adablock_first_block": args.seed_from_adablock_first_block,
                "pag_seed_source": seed_source,
                "match_adablock_initial_seed": args.seed_from_adablock_first_block,
                "adablock_init_block_length": args.adablock_init_block_length,
                "delimiter_threshold": args.delimiter_threshold,
                "delimiter_ids": args.delimiter_ids,
                "use_cache": args.use_cache,
                "dual_cache": args.dual_cache,
                "max_block_length": args.max_block_length,
                "max_refinement_steps": args.max_refinement_steps,
                "min_refinement_steps": args.min_refinement_steps,
                "min_block_length": args.min_block_length,
                "refinement_step_offset": args.refinement_step_offset,
                "context_seed_block_length": args.context_seed_block_length,
                "context_seed_stabilizing_steps": args.context_seed_stabilizing_steps,
            },
            "pag": pag,
            "adablock": adablock,
            "delta": _comparison_delta(pag, adablock),
        }
        write_log_record(args.log_file, result)
        print(
            "  PAG total_nfe={pag_nfe} blocks={pag_blocks} answer={pag_answer} score={pag_score}; "
            "AdaBlock total_nfe={ada_nfe} blocks={ada_blocks} answer={ada_answer} "
            "score={ada_score}".format(
                pag_nfe=pag["metrics"]["total_nfe"],
                pag_blocks=pag["metrics"]["num_blocks"],
                pag_answer=pag["metrics"]["answer_check"]["is_correct"],
                pag_score=pag["metrics"]["substring_check"]["score"],
                ada_nfe=adablock["metrics"]["total_nfe"],
                ada_blocks=adablock["metrics"]["num_blocks"],
                ada_answer=adablock["metrics"]["answer_check"]["is_correct"],
                ada_score=adablock["metrics"]["substring_check"]["score"],
            )
        )


if __name__ == "__main__":
    main()

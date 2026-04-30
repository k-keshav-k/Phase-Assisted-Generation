from __future__ import annotations

import argparse
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
) -> dict[str, object]:
    return {
        "method": method,
        "generated_text": generated_text,
        "nfe_history": nfe_history,
        "block_history": block_history,
        "block_visualization": block_visualization or [],
        "metrics": {
            "elapsed_sec": elapsed_sec,
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

    user_input = _build_prompt(tokenizer, args.model_path, record.prompt)
    input_ids = torch.tensor(tokenizer(user_input)["input_ids"], device=args.device).unsqueeze(0)
    scheduler = _make_scheduler(args, record.prompt)
    generator = (
        generate_pag_dual_cache
        if args.use_cache and args.dual_cache
        else generate_pag_prefix_cache
        if args.use_cache
        else generate_pag
    )

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
    )
    elapsed = time.perf_counter() - start
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
    elapsed = time.perf_counter() - start
    return _summarize_method(
        method="adablock",
        generated_text=_decode_generation(tokenizer, output_ids, input_ids),
        nfe_history=nfe_history,
        block_history=block_history,
        elapsed_sec=elapsed,
        expected_contains=record.expected_contains,
        expected_answers=record.expected_answers,
    )


def _comparison_delta(pag: dict[str, object], adablock: dict[str, object]) -> dict[str, object]:
    pag_metrics = pag["metrics"]
    adablock_metrics = adablock["metrics"]
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
        pag = _run_pag(args, model, tokenizer, record)
        adablock = _run_adablock(args, model, tokenizer, record)
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
                "seed_block_length": args.seed_block_length,
                "seed_refinement_steps": args.seed_refinement_steps,
                "adablock_init_block_length": args.adablock_init_block_length,
                "delimiter_threshold": args.delimiter_threshold,
                "delimiter_ids": args.delimiter_ids,
                "use_cache": args.use_cache,
                "dual_cache": args.dual_cache,
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

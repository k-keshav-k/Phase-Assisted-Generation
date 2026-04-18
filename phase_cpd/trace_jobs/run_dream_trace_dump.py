from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if sys.version_info < (3, 11):  # noqa: UP036
    raise SystemExit("phase_cpd Dream trace collection requires Python 3.11+.")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase_cpd.trace_jobs.dream_local_adapter import collect_trace
from phase_cpd.trace_jobs.dream_runtime import DreamGenerationConfig


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DreamGenerationConfig(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    prompt_records = _load_prompt_records(prompts_path)
    if args.limit is not None:
        prompt_records = prompt_records[: args.limit]

    written_paths: list[Path] = []
    for prompt_record in prompt_records:
        payload = collect_trace(prompt_record, config)
        normalized = _normalize_payload(payload, prompt_record, config.model_name)
        target = output_dir / f"{normalized['trace_id']}.json"
        target.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        written_paths.append(target)

    for path in written_paths:
        print(path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate raw Dream step-dump JSON files for Phase CPD conversion."
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="JSONL prompt manifest. Each row should include at least sample_id and prompt.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where one raw Dream trace JSON file per prompt will be written.",
    )
    parser.add_argument(
        "--model-name",
        default="Dream-org/Dream-v0-Instruct-7B",
        help="Hugging Face model id or local path for Dream.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of prompts to process.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--device", help="Override Dream runtime device, e.g. cuda or cpu.")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype name, e.g. auto, bfloat16, float16, or float32.",
    )
    return parser


def _load_prompt_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def _normalize_payload(
    payload: dict[str, Any],
    prompt_record: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    sample_id = str(prompt_record.get("sample_id", "trace"))
    prompt = str(prompt_record["prompt"])
    normalized = dict(payload)
    normalized.setdefault("trace_id", sample_id)
    normalized.setdefault("prompt", prompt)
    normalized.setdefault("model_name", model_name)
    normalized.setdefault("tags", list(prompt_record.get("tags", [])))
    normalized.setdefault("decoding_metadata", {})

    steps = normalized.get("steps")
    if not isinstance(steps, list) or not steps:
        msg = (
            "Dream trace payload must contain a non-empty 'steps' list. "
            f"Got payload for sample_id={sample_id!r} without usable step observations."
        )
        raise ValueError(msg)

    return normalized


if __name__ == "__main__":
    raise SystemExit(main())

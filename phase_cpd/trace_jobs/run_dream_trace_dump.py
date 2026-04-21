from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info < (3, 11):  # noqa: UP036
    raise SystemExit("phase_cpd Dream trace collection requires Python 3.11+.")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase_cpd.trace_jobs.dream_local_adapter import clear_collector_cache, collect_trace
from phase_cpd.trace_jobs.dream_runtime import DreamGenerationConfig

_DEFAULT_TRAINING_TRACE_PROFILE = "entropy_stochastic"
_TRACE_PROFILES: dict[str, tuple[str, float | None]] = {
    "entropy_det": ("entropy", 0.0),
    "entropy_stochastic": ("entropy", 0.1),
    "origin_random": ("origin", None),
}
_DEFAULT_ALG_TEMP_BY_ALG: dict[str, float | None] = {
    "entropy": 0.1,
    "origin": None,
}
_PROMPT_METADATA_KEYS = (
    "expected_answer",
    "reference_answer",
    "gold_answer",
    "target",
    "answer",
    "exact_match",
    "task_correct",
    "is_correct",
    "correct",
)


@dataclass(frozen=True, slots=True)
class _ResolvedTraceProfile:
    name: str
    alg: str
    alg_temp: float | None


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = _load_prompt_records(prompts_path)
    if args.limit is not None:
        prompt_records = prompt_records[: args.limit]

    written_paths: list[Path] = []
    skipped_prompts: list[str] = []
    profiles = _resolve_trace_profiles(
        trace_profile=args.trace_profile,
        alg=args.alg,
        alg_temp=args.alg_temp,
    )
    steps = args.max_new_tokens if args.steps is None else args.steps
    for profile in profiles:
        config = DreamGenerationConfig(
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            steps=steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            alg=profile.alg,
            alg_temp=profile.alg_temp,
            device=args.device,
            torch_dtype=args.torch_dtype,
            trace_profile=profile.name,
            seed=args.seed,
        )
        try:
            for prompt_record in prompt_records:
                try:
                    payload = collect_trace(prompt_record, config)
                except ValueError as error:
                    if not _should_skip_prompt_error(error):
                        raise
                    sample_id = str(prompt_record.get("sample_id", "unknown"))
                    skipped_prompts.append(f"{sample_id}:{profile.name}")
                    print(
                        (
                            "Skipping prompt after empty Dream generation: "
                            f"sample_id={sample_id!r}, trace_profile={profile.name!r}. "
                            "Dream returned no non-special generated tokens."
                        ),
                        file=sys.stderr,
                    )
                    continue
                normalized = _normalize_payload(payload, prompt_record, config)
                target = output_dir / f"{normalized['trace_id']}.json"
                target.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
                written_paths.append(target)
        finally:
            clear_collector_cache()

    if not written_paths:
        msg = (
            "Dream trace collection did not produce any trace files. "
            f"Skipped prompts: {', '.join(skipped_prompts) if skipped_prompts else 'none'}"
        )
        raise ValueError(msg)

    for path in written_paths:
        print(path)
    if skipped_prompts:
        print(
            f"Skipped {len(skipped_prompts)} prompt/profile runs due to empty Dream generations.",
            file=sys.stderr,
        )
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
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=512,
        help=(
            "Minimum generated tokens before EOS/PAD may be selected. "
            "The default keeps deterministic Dream traces long enough for segmentation."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Dream denoising steps. Defaults to --max-new-tokens when omitted.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Generation temperature controlling token sampling randomness. "
            "The training default is 0.0; use --alg-temp for refinement-order randomness."
        ),
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--trace-profile",
        choices=[*sorted(_TRACE_PROFILES), "all"],
        help=(
            "Vanilla Dream remasking profile to collect. Defaults to "
            f"{_DEFAULT_TRAINING_TRACE_PROFILE}; use all for comparison/ablation runs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used to derive per-prompt deterministic seeds.",
    )
    parser.add_argument(
        "--alg",
        help="Backward-compatible override. Must match one of the supported trace profiles.",
    )
    parser.add_argument(
        "--alg-temp",
        type=float,
        help="Backward-compatible override. Must match one of the supported trace profiles.",
    )
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
    config: DreamGenerationConfig,
) -> dict[str, Any]:
    sample_id = str(prompt_record.get("sample_id", "trace"))
    prompt = str(prompt_record["prompt"])
    normalized = dict(payload)
    normalized["trace_id"] = _profile_trace_id(
        sample_id=sample_id,
        trace_profile=config.trace_profile,
        seed=config.seed,
    )
    normalized.setdefault("prompt", prompt)
    normalized.setdefault("model_name", config.model_name)
    normalized.setdefault("tags", list(prompt_record.get("tags", [])))
    metadata = dict(normalized.setdefault("decoding_metadata", {}))
    metadata.setdefault("trace_profile", config.trace_profile)
    metadata.setdefault("alg", config.alg)
    metadata.setdefault("alg_temp", config.alg_temp)
    metadata.setdefault("temperature", config.temperature)
    metadata.setdefault("min_new_tokens", config.min_new_tokens)
    metadata.setdefault("seed", config.seed)
    for key in _PROMPT_METADATA_KEYS:
        if key in prompt_record:
            metadata.setdefault(key, prompt_record[key])
    normalized["decoding_metadata"] = metadata

    steps = normalized.get("steps")
    if not isinstance(steps, list) or not steps:
        msg = (
            "Dream trace payload must contain a non-empty 'steps' list. "
            f"Got payload for sample_id={sample_id!r} without usable step observations."
        )
        raise ValueError(msg)

    return normalized


def _resolve_trace_profiles(
    *,
    trace_profile: str | None,
    alg: str | None,
    alg_temp: float | None,
) -> list[_ResolvedTraceProfile]:
    if trace_profile == "all":
        if alg is not None or alg_temp is not None:
            msg = "--alg/--alg-temp cannot be combined with --trace-profile=all"
            raise ValueError(msg)
        return [
            _ResolvedTraceProfile(name=name, alg=profile_alg, alg_temp=profile_alg_temp)
            for name, (profile_alg, profile_alg_temp) in _TRACE_PROFILES.items()
        ]

    if trace_profile is not None:
        profile_alg, profile_alg_temp = _TRACE_PROFILES[trace_profile]
        if alg is not None and alg != profile_alg:
            msg = f"--alg={alg!r} conflicts with --trace-profile={trace_profile!r}"
            raise ValueError(msg)
        if alg_temp is not None and alg_temp != profile_alg_temp:
            msg = f"--alg-temp={alg_temp!r} conflicts with --trace-profile={trace_profile!r}"
            raise ValueError(msg)
        return [
            _ResolvedTraceProfile(
                name=trace_profile,
                alg=profile_alg,
                alg_temp=profile_alg_temp,
            )
        ]

    inferred = _infer_trace_profile(alg=alg, alg_temp=alg_temp)
    profile_alg, profile_alg_temp = _TRACE_PROFILES[inferred]
    return [
        _ResolvedTraceProfile(
            name=inferred,
            alg=profile_alg,
            alg_temp=profile_alg_temp,
        )
    ]


def _infer_trace_profile(*, alg: str | None, alg_temp: float | None) -> str:
    if alg is None and alg_temp is None:
        return _DEFAULT_TRAINING_TRACE_PROFILE

    normalized_alg = "entropy" if alg is None else alg
    normalized_alg_temp = (
        _DEFAULT_ALG_TEMP_BY_ALG.get(normalized_alg)
        if alg_temp is None
        else alg_temp
    )
    for profile_name, (profile_alg, profile_alg_temp) in _TRACE_PROFILES.items():
        if normalized_alg != profile_alg:
            continue
        if normalized_alg_temp == profile_alg_temp:
            return profile_name
    msg = (
        "Unsupported --alg/--alg-temp combination. "
        "Use one of the built-in trace profiles instead."
    )
    raise ValueError(msg)


def _profile_trace_id(*, sample_id: str, trace_profile: str, seed: int) -> str:
    return f"{sample_id}__{trace_profile}__seed-{seed}"


def _should_skip_prompt_error(error: ValueError) -> bool:
    return "Dream returned no non-special generated tokens." in str(error)


if __name__ == "__main__":
    raise SystemExit(main())

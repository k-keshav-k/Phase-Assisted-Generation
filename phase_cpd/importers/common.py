from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase_cpd.io import trace_from_dict
from phase_cpd.schema import TokenStepObservation, TraceRecord, TraceToken


def load_step_dump_as_trace(
    source: str | Path,
    *,
    backend: str,
    default_model_name: str,
) -> TraceRecord:
    source_path = Path(source)
    payload = json.loads(source_path.read_text(encoding="utf-8"))

    if _looks_like_unified_trace(payload):
        trace = trace_from_dict(payload)
        if trace.source_path is None:
            trace.source_path = str(source_path)
        if not trace.backend:
            trace.backend = backend
        return trace

    steps = list(payload.get("steps", []))
    if not steps:
        msg = (
            f"{source_path} must contain either unified trace fields or a non-empty 'steps' list "
            "for stepwise raw trace import."
        )
        raise ValueError(msg)

    token_rows: dict[int, dict[str, Any]] = {}
    observations_by_token: dict[int, list[TokenStepObservation]] = defaultdict(list)

    for step in steps:
        step_index = int(step["step_index"])
        for inferred_index, token in enumerate(step.get("tokens", [])):
            token_index = int(token.get("token_index", inferred_index))
            # Real denoising traces can change the token content across steps. Keep the latest
            # token row so the final converted trace reflects the converged text, not step 0.
            token_rows[token_index] = dict(token)
            observations_by_token[token_index].append(
                TokenStepObservation(
                    step_index=step_index,
                    top1_prob=_maybe_float(token.get("top1_prob")),
                    selected_logit=_maybe_float(token.get("selected_logit")),
                    top2_prob=_maybe_float(token.get("top2_prob")),
                    extras={
                        str(key): float(value)
                        for key, value in dict(token.get("extras", {})).items()
                    },
                )
            )

    sorted_indices = sorted(token_rows)
    cursor = 0
    tokens: list[TraceToken] = []
    for token_index in sorted_indices:
        token_row = token_rows[token_index]
        token_text = str(token_row.get("token_text", token_row.get("token", "")))
        char_start = token_row.get("char_start")
        char_end = token_row.get("char_end")
        if char_start is None or char_end is None:
            char_start = cursor
            char_end = cursor + len(token_text)
        tokens.append(
            TraceToken(
                token_index=token_index,
                token_text=token_text,
                char_start=int(char_start),
                char_end=int(char_end),
                observations=sorted(
                    observations_by_token[token_index],
                    key=lambda observation: observation.step_index,
                ),
            )
        )
        cursor = int(char_end)

    final_text = payload.get("final_text") or "".join(token.token_text for token in tokens)
    return TraceRecord(
        trace_id=str(payload.get("trace_id", source_path.stem)),
        backend=str(payload.get("backend", backend)),
        model_name=str(payload.get("model_name", default_model_name)),
        prompt=str(payload["prompt"]),
        final_text=str(final_text),
        tokens=tokens,
        decoding_metadata=dict(payload.get("decoding_metadata", {})),
        tags=[str(tag) for tag in list(payload.get("tags", []))],
        source_path=str(source_path),
        created_at=_maybe_str(payload.get("created_at")),
    )


def _looks_like_unified_trace(payload: dict[str, Any]) -> bool:
    return (
        "trace_id" in payload
        and "tokens" in payload
        and all("observations" in token for token in payload.get("tokens", []))
    )


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)

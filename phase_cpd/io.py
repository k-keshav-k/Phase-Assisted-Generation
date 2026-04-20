from __future__ import annotations

import json
from pathlib import Path

from phase_cpd.schema import (
    FeatureSeries,
    SegmentSummary,
    TokenStepObservation,
    TraceRecord,
    TraceStepSummary,
    TraceToken,
)


def trace_to_dict(trace: TraceRecord) -> dict[str, object]:
    return {
        "trace_id": trace.trace_id,
        "backend": trace.backend,
        "model_name": trace.model_name,
        "prompt": trace.prompt,
        "final_text": trace.final_text,
        "tokens": [
            {
                "token_index": token.token_index,
                "token_text": token.token_text,
                "char_start": token.char_start,
                "char_end": token.char_end,
                "observations": [
                    {
                        "step_index": observation.step_index,
                        "token_id": observation.token_id,
                        "token_text": observation.token_text,
                        "top1_prob": observation.top1_prob,
                        "selected_logit": observation.selected_logit,
                        "top2_prob": observation.top2_prob,
                        "extras": observation.extras,
                    }
                    for observation in token.observations
                ],
            }
            for token in trace.tokens
        ],
        "step_summaries": [
            {
                "step_index": summary.step_index,
                "mask_count": summary.mask_count,
                "changed_count": summary.changed_count,
                "active_start": summary.active_start,
                "active_end": summary.active_end,
                "active_count": summary.active_count,
                "best_delimiter_index": summary.best_delimiter_index,
                "max_delimiter_confidence": summary.max_delimiter_confidence,
            }
            for summary in trace.step_summaries
        ],
        "decoding_metadata": trace.decoding_metadata,
        "tags": trace.tags,
        "source_path": trace.source_path,
        "created_at": trace.created_at,
    }


def trace_from_dict(payload: dict[str, object]) -> TraceRecord:
    return TraceRecord(
        trace_id=str(payload["trace_id"]),
        backend=str(payload["backend"]),
        model_name=str(payload["model_name"]),
        prompt=str(payload["prompt"]),
        final_text=str(payload["final_text"]),
        tokens=[
            TraceToken(
                token_index=int(token["token_index"]),
                token_text=str(token["token_text"]),
                char_start=int(token["char_start"]),
                char_end=int(token["char_end"]),
                observations=[
                    TokenStepObservation(
                        step_index=int(observation["step_index"]),
                        token_id=_maybe_int(observation.get("token_id")),
                        token_text=_maybe_str(observation.get("token_text")),
                        top1_prob=_maybe_float(observation.get("top1_prob")),
                        selected_logit=_maybe_float(observation.get("selected_logit")),
                        top2_prob=_maybe_float(observation.get("top2_prob")),
                        extras={
                            str(key): float(value)
                            for key, value in dict(observation.get("extras", {})).items()
                        },
                    )
                    for observation in list(token.get("observations", []))
                ],
            )
            for token in list(payload.get("tokens", []))
        ],
        step_summaries=[
            TraceStepSummary(
                step_index=int(summary["step_index"]),
                mask_count=int(summary.get("mask_count", 0)),
                changed_count=int(summary.get("changed_count", 0)),
                active_start=_maybe_int(summary.get("active_start")),
                active_end=_maybe_int(summary.get("active_end")),
                active_count=int(summary.get("active_count", 0)),
                best_delimiter_index=_maybe_int(summary.get("best_delimiter_index")),
                max_delimiter_confidence=_maybe_float(summary.get("max_delimiter_confidence")),
            )
            for summary in list(payload.get("step_summaries", []))
        ],
        decoding_metadata=dict(payload.get("decoding_metadata", {})),
        tags=[str(tag) for tag in list(payload.get("tags", []))],
        source_path=_maybe_str(payload.get("source_path")),
        created_at=_maybe_str(payload.get("created_at")),
    )


def save_trace(path: str | Path, trace: TraceRecord) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace_to_dict(trace), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_trace(path: str | Path) -> TraceRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return trace_from_dict(payload)


def feature_series_to_dict(series: FeatureSeries) -> dict[str, object]:
    return {
        "feature_name": series.feature_name,
        "token_indices": series.token_indices,
        "values": series.values,
        "metadata": series.metadata,
    }


def segment_summary_to_dict(summary: SegmentSummary) -> dict[str, object]:
    return {
        "start_token": summary.start_token,
        "end_token": summary.end_token,
        "length": summary.length,
        "text": summary.text,
        "mean": summary.mean,
        "std": summary.std,
        "minimum": summary.minimum,
        "maximum": summary.maximum,
    }


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)

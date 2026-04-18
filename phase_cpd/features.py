from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from phase_cpd.importers.common import load_step_dump_as_trace
from phase_cpd.schema import FeatureSeries, TraceRecord


class FeatureExtractor(Protocol):
    name: str

    def is_available(self, trace: TraceRecord) -> bool:
        ...

    def extract(self, trace: TraceRecord) -> FeatureSeries:
        ...


class StabilizingTop1ProbExtractor:
    name = "stabilizing_prob"

    def is_available(self, trace: TraceRecord) -> bool:
        try:
            identity_trace = _trace_with_identity_history(trace)
        except (FileNotFoundError, ValueError):
            return False
        return all(
            _stabilizing_observation(token).top1_prob is not None
            for token in identity_trace.tokens
        )

    def extract(self, trace: TraceRecord) -> FeatureSeries:
        identity_trace = _trace_with_identity_history(trace)
        token_indices: list[int] = []
        values: list[float] = []
        for token in identity_trace.tokens:
            observation = _stabilizing_observation(token)
            if observation.top1_prob is None:
                msg = f"Token {token.token_index} is missing top1_prob at its stabilization step"
                raise ValueError(msg)
            token_indices.append(token.token_index)
            values.append(observation.top1_prob)
        return FeatureSeries(
            feature_name=self.name,
            token_indices=token_indices,
            values=values,
            metadata={"reduction": "first_stable_step"},
        )

def _stabilizing_observation(token) -> object:
    if not token.observations:
        msg = f"Token {token.token_index} has no observations"
        raise ValueError(msg)

    observations = sorted(token.observations, key=lambda item: item.step_index)
    identities = [
        _observation_identity(observation, token.token_text)
        for observation in observations
    ]
    if any(identity is None for identity in identities):
        msg = (
            "Stabilization-based features require per-step token identity history. "
            f"Token {token.token_index} is missing token_id/token_text observations."
        )
        raise ValueError(msg)

    final_identity = identities[-1]
    for index, identity in enumerate(identities):
        if identity != final_identity:
            continue
        if all(later_identity == final_identity for later_identity in identities[index:]):
            return observations[index]

    msg = f"Could not determine a stabilization step for token {token.token_index}"
    raise ValueError(msg)


def _observation_identity(observation, final_token_text: str) -> tuple[str, object] | None:
    if observation.token_id is not None:
        return ("token_id", observation.token_id)
    if observation.token_text is not None:
        return ("token_text", observation.token_text)
    if final_token_text:
        return None
    return None


def _trace_with_identity_history(trace: TraceRecord) -> TraceRecord:
    if _trace_has_identity_history(trace):
        return trace

    if trace.source_path is None:
        msg = (
            "This trace does not contain per-step token identities and has no source_path to "
            "reload the raw Dream step dump."
        )
        raise FileNotFoundError(msg)

    source_path = Path(trace.source_path)
    if not source_path.exists():
        msg = (
            "This trace does not contain per-step token identities and its source_path is not "
            f"available: {source_path}"
        )
        raise FileNotFoundError(msg)

    return _load_identity_trace_from_source(str(source_path), trace.backend, trace.model_name)


def _trace_has_identity_history(trace: TraceRecord) -> bool:
    for token in trace.tokens:
        if not token.observations:
            return False
        if not any(
            observation.token_id is not None or observation.token_text is not None
            for observation in token.observations
        ):
            return False
    return True


@lru_cache(maxsize=64)
def _load_identity_trace_from_source(
    source_path: str,
    backend: str,
    model_name: str,
) -> TraceRecord:
    return load_step_dump_as_trace(
        source_path,
        backend=backend,
        default_model_name=model_name,
    )

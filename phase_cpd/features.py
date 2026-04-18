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


class Top1ProbExtractor:
    name = "top1_prob"

    def is_available(self, trace: TraceRecord) -> bool:
        return all(_final_observation_has_top1(token.observations) for token in trace.tokens)

    def extract(self, trace: TraceRecord) -> FeatureSeries:
        token_indices: list[int] = []
        values: list[float] = []
        for token in trace.tokens:
            if not token.observations:
                msg = f"Token {token.token_index} has no observations"
                raise ValueError(msg)
            observation = max(token.observations, key=lambda item: item.step_index)
            if observation.top1_prob is None:
                msg = f"Token {token.token_index} is missing top1_prob at the final refinement step"
                raise ValueError(msg)
            token_indices.append(token.token_index)
            values.append(observation.top1_prob)
        return FeatureSeries(
            feature_name=self.name,
            token_indices=token_indices,
            values=values,
            metadata={"reduction": "final_step"},
        )


class MeanTop1ProbExtractor:
    name = "top1_prob_mean"

    def is_available(self, trace: TraceRecord) -> bool:
        return all(_has_any_top1(token.observations) for token in trace.tokens)

    def extract(self, trace: TraceRecord) -> FeatureSeries:
        token_indices: list[int] = []
        values: list[float] = []
        for token in trace.tokens:
            top1_values = [
                observation.top1_prob
                for observation in token.observations
                if observation.top1_prob is not None
            ]
            if not top1_values:
                msg = f"Token {token.token_index} has no top1_prob observations"
                raise ValueError(msg)
            token_indices.append(token.token_index)
            values.append(sum(top1_values) / len(top1_values))
        return FeatureSeries(
            feature_name=self.name,
            token_indices=token_indices,
            values=values,
            metadata={"reduction": "mean_over_steps"},
        )


class StabilizingTop1ProbExtractor:
    name = "top1_prob_stabilize"

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


FEATURE_EXTRACTORS: dict[str, FeatureExtractor] = {
    StabilizingTop1ProbExtractor.name: StabilizingTop1ProbExtractor(),
    MeanTop1ProbExtractor.name: MeanTop1ProbExtractor(),
    Top1ProbExtractor.name: Top1ProbExtractor(),
}


def get_feature_extractor(name: str) -> FeatureExtractor:
    try:
        return FEATURE_EXTRACTORS[name]
    except KeyError as error:
        available = ", ".join(sorted(FEATURE_EXTRACTORS))
        msg = f"Unknown feature extractor '{name}'. Available: {available}"
        raise KeyError(msg) from error


def _final_observation_has_top1(observations: list) -> bool:
    if not observations:
        return False
    return max(observations, key=lambda item: item.step_index).top1_prob is not None


def _has_any_top1(observations: list) -> bool:
    return any(observation.top1_prob is not None for observation in observations)


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

from __future__ import annotations

from typing import Protocol

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


FEATURE_EXTRACTORS: dict[str, FeatureExtractor] = {
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


from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from statistics import pvariance
from typing import Any

from phase_cpd.cpd import CPDParameters, PeltDetector, get_detector
from phase_cpd.features import (
    StabilizingEntropyExtractor,
    StabilizingRefinementStepExtractor,
    get_feature_extractor,
)
from phase_cpd.schema import TokenStepObservation, TraceRecord, TraceStepSummary, TraceToken
from phase_cpd.segments import segment_ranges

_FALLBACK_MASK_TEXTS = {"<|mask|>", "<mask>", "[MASK]"}
_TASK_CORRECTNESS_KEYS = ("task_correct", "is_correct", "correct", "correctness")
_EXACT_MATCH_KEYS = ("exact_match", "em")
_EXPECTED_ANSWER_KEYS = (
    "expected_answer",
    "reference_answer",
    "gold_answer",
    "target",
    "answer",
)


@dataclass(slots=True)
class SchedulerDatasetConfig:
    feature_name: str = StabilizingEntropyExtractor.name
    detector_name: str = PeltDetector.name
    kernel: str = "rbf"
    cpd_params: CPDParameters = field(default_factory=CPDParameters)


def build_scheduler_rows(
    trace: TraceRecord,
    *,
    config: SchedulerDatasetConfig | None = None,
) -> list[dict[str, Any]]:
    if not trace.tokens:
        return []

    resolved_config = SchedulerDatasetConfig() if config is None else config
    feature_series = get_feature_extractor(resolved_config.feature_name).extract(trace)
    detector = get_detector(
        resolved_config.detector_name,
        kernel=resolved_config.kernel,
    )
    breakpoints = detector.detect(feature_series.values, resolved_config.cpd_params)
    oracle_segments = segment_ranges(len(trace.tokens), breakpoints)
    stable_steps = _stable_steps_by_token(trace)
    step_indices = _trace_step_indices(trace)
    if not step_indices:
        return []

    summaries_by_step = {summary.step_index: summary for summary in trace.step_summaries}
    profile = _trace_profile(trace)
    metadata = dict(trace.decoding_metadata)

    rows: list[dict[str, Any]] = []
    activation_step = step_indices[0]
    for segment_index, (start_token, end_token) in enumerate(oracle_segments):
        block_max_stable = max(stable_steps[start_token:end_token])
        for step_index in step_indices:
            if step_index < activation_step or step_index >= block_max_stable:
                continue
            summary = summaries_by_step.get(step_index, TraceStepSummary(step_index=step_index))
            block_observations = [
                _observation_at_or_before(trace.tokens[token_index], step_index)
                for token_index in range(start_token, end_token)
            ]
            frontier_observation = block_observations[0]
            rows.append(
                {
                    "trace_id": trace.trace_id,
                    "trace_profile": profile,
                    "prompt": trace.prompt,
                    "step_index": step_index,
                    "frontier": start_token,
                    "oracle_segment_index": segment_index,
                    "oracle_block_start": start_token,
                    "oracle_block_end": end_token,
                    "oracle_block_size": end_token - start_token,
                    "oracle_max_refinement_steps": block_max_stable - step_index,
                    "alg": metadata.get("alg"),
                    "alg_temp": metadata.get("alg_temp"),
                    "seed": metadata.get("seed"),
                    "feature_name": resolved_config.feature_name,
                    "detector_name": resolved_config.detector_name,
                    "mask_count": summary.mask_count,
                    "changed_count": summary.changed_count,
                    "active_start": summary.active_start,
                    "active_end": summary.active_end,
                    "active_count": summary.active_count,
                    "best_delimiter_index": summary.best_delimiter_index,
                    "max_delimiter_confidence": summary.max_delimiter_confidence,
                    "frontier_entropy": _extra(frontier_observation, "entropy"),
                    "frontier_top1_prob": frontier_observation.top1_prob,
                    "frontier_margin": _margin(frontier_observation),
                    "frontier_is_mask": _extra(frontier_observation, "is_mask"),
                    "frontier_changed_from_prev_step": _extra(
                        frontier_observation,
                        "changed_from_prev_step",
                    ),
                    "frontier_delimiter_prob_max": _extra(
                        frontier_observation,
                        "delimiter_prob_max",
                    ),
                    "block_mean_entropy": _mean(
                        _extra(observation, "entropy") for observation in block_observations
                    ),
                    "block_mean_top1_prob": _mean(
                        observation.top1_prob for observation in block_observations
                    ),
                    "block_mean_margin": _mean(
                        _margin(observation) for observation in block_observations
                    ),
                    "block_mask_fraction": _mean(
                        _extra(observation, "is_mask") for observation in block_observations
                    ),
                    "block_changed_fraction": _mean(
                        _extra(observation, "changed_from_prev_step")
                        for observation in block_observations
                    ),
                    "block_max_delimiter_prob": _max(
                        _extra(observation, "delimiter_prob_max")
                        for observation in block_observations
                    ),
                }
            )
        activation_step = max(activation_step, block_max_stable)
    return rows


def build_profile_report(
    traces: list[TraceRecord],
    *,
    config: SchedulerDatasetConfig | None = None,
) -> list[dict[str, Any]]:
    resolved_config = SchedulerDatasetConfig() if config is None else config
    rows_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tokens_by_profile: dict[str, list[TraceToken]] = defaultdict(list)
    traces_by_profile: dict[str, int] = defaultdict(int)
    exact_matches_by_profile: dict[str, list[bool]] = defaultdict(list)
    task_correctness_by_profile: dict[str, list[bool]] = defaultdict(list)
    stable_steps_by_profile: dict[str, list[int]] = defaultdict(list)
    token_index_r2_by_profile: dict[str, list[float]] = defaultdict(list)

    for trace in traces:
        profile = _trace_profile(trace)
        stable_steps = _stable_steps_by_token(trace)
        exact_match = _trace_exact_match(trace)
        task_correct = _trace_task_correct(trace)
        traces_by_profile[profile] += 1
        tokens_by_profile[profile].extend(trace.tokens)
        stable_steps_by_profile[profile].extend(stable_steps)
        token_index_r2_by_profile[profile].append(_token_index_stabilization_r2(stable_steps))
        if exact_match is not None:
            exact_matches_by_profile[profile].append(exact_match)
        if task_correct is not None:
            task_correctness_by_profile[profile].append(task_correct)
        rows_by_profile[profile].extend(build_scheduler_rows(trace, config=resolved_config))

    summaries: list[dict[str, Any]] = []
    for profile in sorted(traces_by_profile):
        tokens = tokens_by_profile[profile]
        rows = rows_by_profile[profile]
        exact_matches = exact_matches_by_profile[profile]
        task_correctness = task_correctness_by_profile[profile]
        stable_steps = stable_steps_by_profile[profile]
        direct_mask_to_final = sum(1 for token in tokens if _is_direct_mask_to_final(token))
        rewrite_counts = [_rewrite_count(token) for token in tokens]
        monotonicity_scores = [
            _stabilization_monotonicity(trace)
            for trace in traces
            if _trace_profile(trace) == profile
        ]
        summaries.append(
            {
                "trace_profile": profile,
                "trace_count": traces_by_profile[profile],
                "token_count": len(tokens),
                "row_count": len(rows),
                "task_correct_available_count": len(task_correctness),
                "task_correct_rate": _mean(task_correctness),
                "exact_match_available_count": len(exact_matches),
                "exact_match_rate": _mean(exact_matches),
                "direct_mask_to_final_fraction": (
                    direct_mask_to_final / len(tokens) if tokens else 0.0
                ),
                "mean_token_rewrite_count": _mean(rewrite_counts),
                "stabilization_monotonicity": _mean(monotonicity_scores),
                "token_index_stabilization_r2": _mean(token_index_r2_by_profile[profile]),
                "stabilization_step_min": _min(stable_steps),
                "stabilization_step_mean": _mean(stable_steps),
                "stabilization_step_std": _std(stable_steps),
                "stabilization_step_max": _max(stable_steps),
                "oracle_block_size_variance": _variance(
                    row["oracle_block_size"] for row in rows
                ),
                "oracle_max_refinement_steps_variance": _variance(
                    row["oracle_max_refinement_steps"] for row in rows
                ),
            }
        )
    return summaries


def _stable_steps_by_token(trace: TraceRecord) -> list[int]:
    return [
        int(value)
        for value in StabilizingRefinementStepExtractor().extract(trace).values
    ]


def _trace_step_indices(trace: TraceRecord) -> list[int]:
    step_indices = {summary.step_index for summary in trace.step_summaries}
    for token in trace.tokens:
        for observation in token.observations:
            step_indices.add(observation.step_index)
    return sorted(step_indices)


def _trace_profile(trace: TraceRecord) -> str:
    return str(trace.decoding_metadata.get("trace_profile", "unknown"))


def _observation_at_or_before(token: TraceToken, step_index: int) -> TokenStepObservation:
    observations = sorted(token.observations, key=lambda observation: observation.step_index)
    latest = observations[0]
    for observation in observations:
        if observation.step_index > step_index:
            break
        latest = observation
    return latest


def _extra(observation: TokenStepObservation, key: str) -> float | None:
    value = observation.extras.get(key)
    if value is None:
        return None
    return float(value)


def _margin(observation: TokenStepObservation) -> float | None:
    if observation.top1_prob is None or observation.top2_prob is None:
        return None
    return float(observation.top1_prob - observation.top2_prob)


def _mean(values) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _max(values) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _min(values) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return min(filtered)


def _std(values) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    if len(filtered) < 2:
        return 0.0
    return sqrt(float(pvariance(filtered)))


def _variance(values) -> float:
    filtered = [float(value) for value in values if value is not None]
    if len(filtered) < 2:
        return 0.0
    return float(pvariance(filtered))


def _stabilization_monotonicity(trace: TraceRecord) -> float:
    stable_steps = _stable_steps_by_token(trace)
    if len(stable_steps) < 2:
        return 1.0
    nondecreasing_pairs = sum(
        1
        for index in range(len(stable_steps) - 1)
        if stable_steps[index + 1] >= stable_steps[index]
    )
    return nondecreasing_pairs / (len(stable_steps) - 1)


def _token_index_stabilization_r2(stable_steps: list[int]) -> float:
    if len(stable_steps) < 2:
        return 0.0

    xs = [float(index) for index in range(len(stable_steps))]
    ys = [float(step) for step in stable_steps]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    total = sum((value - y_mean) ** 2 for value in ys)
    if x_var == 0.0 or total == 0.0:
        return 0.0

    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / x_var
    intercept = y_mean - (slope * x_mean)
    residual = sum(
        (y - (intercept + (slope * x))) ** 2 for x, y in zip(xs, ys, strict=True)
    )
    return max(0.0, min(1.0, 1.0 - (residual / total)))


def _trace_task_correct(trace: TraceRecord) -> bool | None:
    metadata = trace.decoding_metadata
    for key in (*_TASK_CORRECTNESS_KEYS, *_EXACT_MATCH_KEYS):
        if key in metadata:
            value = _maybe_bool(metadata[key])
            if value is not None:
                return value
    return _trace_exact_match(trace)


def _trace_exact_match(trace: TraceRecord) -> bool | None:
    metadata = trace.decoding_metadata
    for key in _EXACT_MATCH_KEYS:
        if key in metadata:
            value = _maybe_bool(metadata[key])
            if value is not None:
                return value
    for key in _EXPECTED_ANSWER_KEYS:
        if key in metadata:
            return _normalize_answer(trace.final_text) == _normalize_answer(metadata[key])
    return None


def _maybe_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _normalize_answer(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _is_direct_mask_to_final(token: TraceToken) -> bool:
    runs = _identity_runs(token)
    if len(runs) != 2:
        return False
    final_identity = runs[-1][0]
    return runs[0][1] and runs[1][0] == final_identity


def _rewrite_count(token: TraceToken) -> int:
    runs = _identity_runs(token)
    return max(0, len(runs) - 1)


def _identity_runs(token: TraceToken) -> list[tuple[object, bool]]:
    runs: list[tuple[object, bool]] = []
    previous_identity: object | None = None
    for observation in sorted(token.observations, key=lambda item: item.step_index):
        identity = _identity_key(token, observation)
        if runs and identity == previous_identity:
            continue
        runs.append((identity, _is_mask_observation(observation)))
        previous_identity = identity
    return runs


def _identity_key(token: TraceToken, observation: TokenStepObservation) -> object:
    if observation.token_id is not None:
        return ("token_id", observation.token_id)
    if observation.token_text is not None:
        return ("token_text", observation.token_text)
    return ("fallback_text", token.token_text)


def _is_mask_observation(observation: TokenStepObservation) -> bool:
    is_mask_extra = _extra(observation, "is_mask")
    if is_mask_extra is not None:
        return is_mask_extra >= 0.5
    if observation.token_text is not None and observation.token_text in _FALLBACK_MASK_TEXTS:
        return True
    return False

from __future__ import annotations

from phase_cpd.catalog import default_trace_dir, load_trace_by_id
from phase_cpd.features import (
    MeanTop1ProbExtractor,
    StabilizingTop1ProbExtractor,
    Top1ProbExtractor,
)
from phase_cpd.schema import TokenStepObservation, TraceRecord, TraceToken


def test_top1_prob_uses_final_refinement_step() -> None:
    trace = load_trace_by_id("prompt-001", default_trace_dir())
    feature_series = Top1ProbExtractor().extract(trace)

    assert feature_series.feature_name == "top1_prob"
    assert len(feature_series.values) == len(trace.tokens)
    assert all(0.0 <= value <= 1.0 for value in feature_series.values)


def test_top1_prob_mean_uses_all_refinement_steps() -> None:
    trace = load_trace_by_id("prompt-001", default_trace_dir())
    feature_series = MeanTop1ProbExtractor().extract(trace)

    assert feature_series.feature_name == "top1_prob_mean"
    assert feature_series.metadata["reduction"] == "mean_over_steps"
    assert len(feature_series.values) == len(trace.tokens)
    assert feature_series.values[0] < Top1ProbExtractor().extract(trace).values[0]


def test_stabilizing_top1_prob_uses_first_stable_observation() -> None:
    trace = TraceRecord(
        trace_id="stabilize-test",
        backend="dream",
        model_name="dream-test",
        prompt="Prompt",
        final_text="AB",
        tokens=[
            TraceToken(
                token_index=0,
                token_text="A",
                char_start=0,
                char_end=1,
                observations=[
                    TokenStepObservation(step_index=0, token_id=10, token_text="X", top1_prob=0.11),
                    TokenStepObservation(step_index=1, token_id=20, token_text="A", top1_prob=0.42),
                    TokenStepObservation(step_index=2, token_id=20, token_text="A", top1_prob=0.87),
                ],
            ),
            TraceToken(
                token_index=1,
                token_text="B",
                char_start=1,
                char_end=2,
                observations=[
                    TokenStepObservation(step_index=0, token_id=30, token_text="Y", top1_prob=0.08),
                    TokenStepObservation(step_index=1, token_id=31, token_text="Z", top1_prob=0.16),
                    TokenStepObservation(step_index=2, token_id=40, token_text="B", top1_prob=0.65),
                    TokenStepObservation(step_index=3, token_id=40, token_text="B", top1_prob=0.92),
                ],
            ),
        ],
    )

    feature_series = StabilizingTop1ProbExtractor().extract(trace)

    assert feature_series.feature_name == "top1_prob_stabilize"
    assert feature_series.values == [0.42, 0.65]
    assert feature_series.metadata["reduction"] == "first_stable_step"

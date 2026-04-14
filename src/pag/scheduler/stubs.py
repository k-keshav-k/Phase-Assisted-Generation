from __future__ import annotations

from pag.contracts.artifacts import AdaptiveRunArtifacts, BaselineRunArtifacts, PhaseArtifacts
from pag.contracts.enums import PhaseLabel, StageName
from pag.contracts.schemas import (
    DecodingResult,
    RunConfig,
    RunSummary,
    ScheduleDecision,
    SchedulePlan,
)


def mock_scheduler_runner(
    run_config: RunConfig,
    baseline_artifacts: BaselineRunArtifacts,
    phase_artifacts: PhaseArtifacts,
) -> AdaptiveRunArtifacts:
    """Default deterministic adaptive-decoding stub.

    TODO(team-scheduler): Replace the mock plan/decision generation and adaptive decode logic
    with a real scheduler while preserving returned artifact shapes.
    """

    prediction_by_sample = {
        prediction.sample_id: prediction for prediction in phase_artifacts.predictions
    }

    schedule_decisions: list[ScheduleDecision] = []
    schedule_plans: list[SchedulePlan] = []
    adaptive_results: list[DecodingResult] = []

    for completion in baseline_artifacts.completions:
        prediction = prediction_by_sample[completion.sample_id]
        decisions: list[ScheduleDecision] = []
        for step_index, span in enumerate(prediction.spans):
            is_easy = span.label == PhaseLabel.EASY.value
            decisions.append(
                ScheduleDecision(
                    sample_id=completion.sample_id,
                    step_index=step_index,
                    chunk_size=_choose_chunk_size(run_config, is_easy=is_easy),
                    refinement_steps=_choose_refinement_steps(run_config, is_easy=is_easy),
                    reason=f"phase:{span.label}",
                    phase_label=span.label,
                    metadata={"source_span": [span.start_token, span.end_token]},
                )
            )
        schedule_decisions.extend(decisions)
        schedule_plans.append(
            SchedulePlan(
                sample_id=completion.sample_id,
                planner_name=run_config.scheduler.name,
                decisions=decisions,
                metadata={"implementation": "mock_scheduler_runner"},
            )
        )

        adaptive_tokens = [*completion.tokens, "adaptive"]
        adaptive_results.append(
            DecodingResult(
                sample_id=completion.sample_id,
                request_id=completion.request_id,
                completion=" ".join(adaptive_tokens),
                tokens=adaptive_tokens,
                token_ids=list(range(len(adaptive_tokens))),
                metadata={"source": "mock-adaptive-decoder"},
            )
        )

    comparison_metrics = {
        "throughput_proxy": round(1.0 + (0.1 * len(schedule_decisions)), 3),
        "quality_proxy": round(0.8 + (0.05 * len(adaptive_results)), 3),
    }
    summary = RunSummary(
        run_id=run_config.run_id,
        stage=StageName.SCHEDULER.value,
        num_samples=len(adaptive_results),
        metrics={
            "num_schedule_decisions": float(len(schedule_decisions)),
            "num_schedule_plans": float(len(schedule_plans)),
            **comparison_metrics,
        },
        metadata={"implementation": "mock_scheduler_runner"},
    )
    return AdaptiveRunArtifacts(
        run_config=run_config,
        schedule_decisions=schedule_decisions,
        schedule_plans=schedule_plans,
        adaptive_results=adaptive_results,
        comparison_metrics=comparison_metrics,
        summary=summary,
    )


def _choose_chunk_size(run_config: RunConfig, *, is_easy: bool) -> int:
    if is_easy:
        return int(
            run_config.scheduler.parameters.get(
                "easy_chunk_size",
                run_config.decoding.chunk_size,
            )
        )
    return int(run_config.scheduler.parameters.get("hard_chunk_size", 1))


def _choose_refinement_steps(run_config: RunConfig, *, is_easy: bool) -> int:
    if is_easy:
        return int(
            run_config.scheduler.parameters.get(
                "easy_refinement_steps",
                run_config.decoding.refinement_steps,
            )
        )
    return int(
        run_config.scheduler.parameters.get(
            "hard_refinement_steps",
            run_config.decoding.refinement_steps,
        )
    )

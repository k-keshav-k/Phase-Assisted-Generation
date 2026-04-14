from __future__ import annotations

import pytest

from pag.contracts.schemas import (
    DecodingResult,
    EvaluationRecord,
    GenerationRequest,
    GenerationTrace,
    PhasePrediction,
    PhaseSpan,
    PredictorDatasetItem,
    RunConfig,
    RunSummary,
    ScheduleDecision,
    SchedulePlan,
    TokenSignal,
    TraceStep,
)
from pag.utils.ids import build_request_id


def test_shared_schema_construction(run_config: RunConfig, sample_records: list) -> None:
    sample = sample_records[0]
    request_id = build_request_id(run_config.run_id, sample.sample_id)
    request = GenerationRequest(
        run_id=run_config.run_id,
        request_id=request_id,
        sample=sample,
        model=run_config.model,
        decoding=run_config.decoding,
        requested_artifacts=["trace", "completion"],
    )
    signal = TokenSignal(
        sample_id=sample.sample_id,
        token_index=0,
        token_text="baseline",
        step_index=0,
        values={"entropy": 0.42},
    )
    trace = GenerationTrace(
        sample_id=sample.sample_id,
        request_id=request_id,
        steps=[
            TraceStep(
                step_index=0,
                chunk_size=2,
                refinement_steps=1,
                emitted_tokens=["baseline"],
            )
        ],
        final_tokens=["baseline", "completion"],
    )
    phase_span = PhaseSpan(sample_id=sample.sample_id, start_token=0, end_token=1, label="easy")
    prediction = PhasePrediction(
        sample_id=sample.sample_id,
        predictor_name=run_config.predictor.name,
        spans=[phase_span],
    )
    dataset_item = PredictorDatasetItem(
        sample_id=sample.sample_id,
        token_index=0,
        features={"entropy": 0.42},
        label="easy",
    )
    decision = ScheduleDecision(
        sample_id=sample.sample_id,
        step_index=0,
        chunk_size=4,
        refinement_steps=1,
        reason="phase:easy",
        phase_label="easy",
    )
    plan = SchedulePlan(
        sample_id=sample.sample_id,
        planner_name=run_config.scheduler.name,
        decisions=[decision],
    )
    result = DecodingResult(
        sample_id=sample.sample_id,
        request_id=request_id,
        completion="baseline completion",
        tokens=["baseline", "completion"],
        token_ids=[0, 1],
    )
    record = EvaluationRecord(
        sample_id=sample.sample_id,
        baseline_completion="baseline completion",
        adaptive_completion="baseline completion adaptive",
        metrics={"quality_proxy": 0.9},
    )
    summary = RunSummary(
        run_id=run_config.run_id,
        stage="baseline",
        num_samples=len(sample_records),
    )

    assert request.request_id == request_id
    assert signal.values["entropy"] == pytest.approx(0.42)
    assert trace.final_tokens[-1] == "completion"
    assert prediction.spans[0].label == "easy"
    assert dataset_item.label == "easy"
    assert plan.decisions[0].chunk_size == 4
    assert result.tokens == ["baseline", "completion"]
    assert record.metrics["quality_proxy"] == pytest.approx(0.9)
    assert summary.num_samples == len(sample_records)


def test_phase_span_rejects_invalid_token_range() -> None:
    with pytest.raises(ValueError):
        PhaseSpan(sample_id="sample-001", start_token=2, end_token=1, label="hard")


def test_run_config_requires_dataset_path(run_config: RunConfig) -> None:
    with pytest.raises(ValueError):
        RunConfig(
            run_id=run_config.run_id,
            output_root=run_config.output_root,
            dataset_path="",
            model=run_config.model,
            decoding=run_config.decoding,
            predictor=run_config.predictor,
            scheduler=run_config.scheduler,
            evaluation=run_config.evaluation,
        )

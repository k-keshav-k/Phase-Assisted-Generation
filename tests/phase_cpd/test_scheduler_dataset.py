from __future__ import annotations

import json
import sys
from pathlib import Path

from phase_cpd.export_scheduler_dataset import main as export_scheduler_dataset_main
from phase_cpd.io import save_trace
from phase_cpd.report_trace_profiles import main as report_trace_profiles_main
from phase_cpd.scheduler_dataset import (
    SchedulerDatasetConfig,
    build_profile_report,
    build_scheduler_rows,
)
from phase_cpd.schema import TokenStepObservation, TraceRecord, TraceStepSummary, TraceToken


def _make_trace(
    *,
    trace_id: str,
    trace_profile: str,
    stable_steps: list[int],
    stabilizing_entropies: list[float],
    expected_answer: str | None = None,
    task_correct: bool | None = None,
) -> TraceRecord:
    final_tokens = [chr(ord("A") + index) for index in range(len(stable_steps))]
    tokens: list[TraceToken] = []
    cursor = 0
    max_step = max(stable_steps)
    for token_index, (final_token, stable_step, entropy) in enumerate(
        zip(final_tokens, stable_steps, stabilizing_entropies, strict=True)
    ):
        observations: list[TokenStepObservation] = []
        for step_index in range(max_step + 1):
            is_stable = step_index >= stable_step
            token_text = final_token if is_stable else "<|mask|>"
            token_id = token_index + 10 if is_stable else 99
            top1_prob = 0.9 if is_stable else 0.1
            top2_prob = 0.05 if is_stable else 0.08
            observations.append(
                TokenStepObservation(
                    step_index=step_index,
                    token_id=token_id,
                    token_text=token_text,
                    top1_prob=top1_prob,
                    top2_prob=top2_prob,
                    extras={
                        "entropy": entropy if step_index == stable_step else (entropy + 1.0),
                        "is_mask": 0.0 if is_stable else 1.0,
                        "changed_from_prev_step": 1.0 if step_index == stable_step else 0.0,
                        "delimiter_prob_max": 0.1 * (token_index + 1),
                    },
                )
            )
        tokens.append(
            TraceToken(
                token_index=token_index,
                token_text=final_token,
                char_start=cursor,
                char_end=cursor + len(final_token),
                observations=observations,
            )
        )
        cursor += len(final_token)

    step_summaries: list[TraceStepSummary] = []
    for step_index in range(max_step + 1):
        mask_positions = [
            token_index
            for token_index, stable_step in enumerate(stable_steps)
            if step_index < stable_step
        ]
        step_summaries.append(
            TraceStepSummary(
                step_index=step_index,
                mask_count=len(mask_positions),
                changed_count=sum(1 for stable_step in stable_steps if stable_step == step_index),
                active_start=mask_positions[0] if mask_positions else None,
                active_end=(mask_positions[-1] + 1) if mask_positions else None,
                active_count=len(mask_positions),
                best_delimiter_index=(mask_positions[-1] if mask_positions else None),
                max_delimiter_confidence=0.25 + (0.05 * step_index),
            )
        )

    decoding_metadata = {
        "trace_profile": trace_profile,
        "alg": "entropy" if trace_profile != "origin_random" else "origin",
        "alg_temp": _alg_temp_for_profile(trace_profile),
        "seed": 0,
        "mask_token_id": 99,
        "mask_token_text": "<|mask|>",
    }
    if expected_answer is not None:
        decoding_metadata["expected_answer"] = expected_answer
    if task_correct is not None:
        decoding_metadata["task_correct"] = task_correct

    return TraceRecord(
        trace_id=trace_id,
        backend="dream",
        model_name="dream-test",
        prompt="Explain scheduler traces.",
        final_text="".join(final_tokens),
        tokens=tokens,
        step_summaries=step_summaries,
        decoding_metadata=decoding_metadata,
    )


def _alg_temp_for_profile(trace_profile: str) -> float | None:
    if trace_profile == "entropy_det":
        return 0.0
    if trace_profile == "entropy_stochastic":
        return 0.1
    return None


def test_build_scheduler_rows_advances_frontier_by_oracle_segment() -> None:
    trace = _make_trace(
        trace_id="trace-a",
        trace_profile="entropy_det",
        stable_steps=[1, 2, 5, 6],
        stabilizing_entropies=[0.0, 0.1, 5.0, 5.1],
    )
    config = SchedulerDatasetConfig()
    config.cpd_params.penalty = 0.01

    rows = build_scheduler_rows(trace, config=config)

    assert [row["step_index"] for row in rows] == [0, 1, 2, 3, 4, 5]
    assert [row["frontier"] for row in rows] == [0, 0, 2, 2, 2, 2]
    assert [row["oracle_block_end"] for row in rows] == [2, 2, 4, 4, 4, 4]
    assert [row["oracle_max_refinement_steps"] for row in rows] == [2, 1, 4, 3, 2, 1]
    assert rows[0]["mask_count"] == 4
    assert rows[2]["frontier_entropy"] == 6.0


def test_build_profile_report_groups_metrics_by_profile(monkeypatch) -> None:
    traces = [
        _make_trace(
            trace_id="trace-b",
            trace_profile="entropy_det",
            stable_steps=[1, 2, 3, 4],
            stabilizing_entropies=[0.0, 0.1, 4.0, 4.1],
            expected_answer="ABCD",
            task_correct=True,
        ),
        _make_trace(
            trace_id="trace-c",
            trace_profile="origin_random",
            stable_steps=[1, 2, 3, 4],
            stabilizing_entropies=[0.0, 0.1, 4.0, 4.1],
            expected_answer="wrong",
            task_correct=False,
        ),
    ]

    def _rows_for_trace(trace, *, config=None):
        del config
        if trace.decoding_metadata["trace_profile"] == "entropy_det":
            return [
                {"oracle_block_size": 2, "oracle_max_refinement_steps": 1},
                {"oracle_block_size": 4, "oracle_max_refinement_steps": 3},
            ]
        return [
            {"oracle_block_size": 3, "oracle_max_refinement_steps": 2},
            {"oracle_block_size": 3, "oracle_max_refinement_steps": 2},
        ]

    monkeypatch.setattr("phase_cpd.scheduler_dataset.build_scheduler_rows", _rows_for_trace)

    report = build_profile_report(traces)

    assert report == [
        {
            "trace_profile": "entropy_det",
            "trace_count": 1,
            "token_count": 4,
            "row_count": 2,
            "task_correct_available_count": 1,
            "task_correct_rate": 1.0,
            "exact_match_available_count": 1,
            "exact_match_rate": 1.0,
            "direct_mask_to_final_fraction": 1.0,
            "mean_token_rewrite_count": 1.0,
            "stabilization_monotonicity": 1.0,
            "token_index_stabilization_r2": 1.0,
            "stabilization_step_min": 1.0,
            "stabilization_step_mean": 2.5,
            "stabilization_step_std": 1.118033988749895,
            "stabilization_step_max": 4.0,
            "oracle_block_size_variance": 1.0,
            "oracle_max_refinement_steps_variance": 1.0,
        },
        {
            "trace_profile": "origin_random",
            "trace_count": 1,
            "token_count": 4,
            "row_count": 2,
            "task_correct_available_count": 1,
            "task_correct_rate": 0.0,
            "exact_match_available_count": 1,
            "exact_match_rate": 0.0,
            "direct_mask_to_final_fraction": 1.0,
            "mean_token_rewrite_count": 1.0,
            "stabilization_monotonicity": 1.0,
            "token_index_stabilization_r2": 1.0,
            "stabilization_step_min": 1.0,
            "stabilization_step_mean": 2.5,
            "stabilization_step_std": 1.118033988749895,
            "stabilization_step_max": 4.0,
            "oracle_block_size_variance": 0.0,
            "oracle_max_refinement_steps_variance": 0.0,
        },
    ]


def test_export_scheduler_dataset_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    trace = _make_trace(
        trace_id="trace-export",
        trace_profile="entropy_det",
        stable_steps=[1, 2, 5, 6],
        stabilizing_entropies=[0.0, 0.1, 5.0, 5.1],
    )
    trace_path = tmp_path / "trace.json"
    output_path = tmp_path / "rows.jsonl"
    save_trace(trace_path, trace)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_scheduler_dataset.py",
            "--source",
            str(trace_path),
            "--output",
            str(output_path),
            "--penalty",
            "0.01",
        ],
    )

    assert export_scheduler_dataset_main() == 0
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[0]["trace_profile"] == "entropy_det"


def test_report_trace_profiles_writes_json(tmp_path: Path, monkeypatch) -> None:
    trace = _make_trace(
        trace_id="trace-report",
        trace_profile="entropy_det",
        stable_steps=[1, 2, 5, 6],
        stabilizing_entropies=[0.0, 0.1, 5.0, 5.1],
    )
    trace_path = tmp_path / "trace.json"
    output_path = tmp_path / "report.json"
    save_trace(trace_path, trace)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_trace_profiles.py",
            "--source",
            str(trace_path),
            "--output",
            str(output_path),
            "--penalty",
            "0.01",
        ],
    )

    assert report_trace_profiles_main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["trace_profile"] == "entropy_det"

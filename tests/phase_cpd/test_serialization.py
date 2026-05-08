from __future__ import annotations

from phase_cpd.catalog import default_trace_dir, load_trace_by_id
from phase_cpd.io import load_trace, save_trace
from phase_cpd.schema import TokenStepObservation, TraceRecord, TraceStepSummary, TraceToken


def test_trace_json_round_trip(tmp_path) -> None:
    trace = load_trace_by_id("prompt-001", default_trace_dir())
    output_path = tmp_path / "trace.json"

    save_trace(output_path, trace)
    restored = load_trace(output_path)

    assert restored.trace_id == trace.trace_id
    assert restored.final_text == trace.final_text
    assert (
        restored.tokens[0].observations[-1].top1_prob == trace.tokens[0].observations[-1].top1_prob
    )
    assert restored.tokens[0].observations[-1].extras == trace.tokens[0].observations[-1].extras


def test_trace_json_round_trip_preserves_step_summaries(tmp_path) -> None:
    trace = TraceRecord(
        trace_id="scheduler-trace",
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
                observations=[TokenStepObservation(step_index=0, token_text="A")],
            ),
            TraceToken(
                token_index=1,
                token_text="B",
                char_start=1,
                char_end=2,
                observations=[TokenStepObservation(step_index=0, token_text="B")],
            ),
        ],
        step_summaries=[
            TraceStepSummary(
                step_index=0,
                mask_count=1,
                changed_count=0,
                active_start=0,
                active_end=1,
                active_count=1,
                best_delimiter_index=1,
                max_delimiter_confidence=0.8,
            )
        ],
        decoding_metadata={"trace_profile": "entropy_det"},
    )
    output_path = tmp_path / "trace.json"

    save_trace(output_path, trace)
    restored = load_trace(output_path)

    assert restored.step_summaries[0].mask_count == 1
    assert restored.step_summaries[0].best_delimiter_index == 1

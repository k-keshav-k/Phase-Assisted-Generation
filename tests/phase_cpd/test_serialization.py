from __future__ import annotations

from phase_cpd.catalog import default_trace_dir, load_trace_by_id
from phase_cpd.io import load_trace, save_trace


def test_trace_json_round_trip(tmp_path) -> None:
    trace = load_trace_by_id("mock-adaptive-001", default_trace_dir())
    output_path = tmp_path / "trace.json"

    save_trace(output_path, trace)
    restored = load_trace(output_path)

    assert restored.trace_id == trace.trace_id
    assert restored.final_text == trace.final_text
    assert restored.tokens[0].observations[-1].top1_prob == 0.91


from __future__ import annotations

from pathlib import Path

from phase_cpd.catalog import (
    default_trace_dir,
    filter_catalog_entries,
    list_catalog_entries,
    load_trace_by_id,
    trace_dir_signature,
)
from phase_cpd.io import save_trace
from phase_cpd.schema import TraceRecord, TraceToken
from tests.phase_cpd.trace_fixtures import make_stabilized_trace


def test_catalog_lists_trace_files(tmp_path: Path) -> None:
    trace = make_stabilized_trace(trace_id="prompt-001__entropy_stochastic__seed-0")
    save_trace(tmp_path / "trace.json", trace)
    entries = list_catalog_entries(tmp_path)

    assert len(entries) >= 1
    assert trace.trace_id in {entry.trace_id for entry in entries}
    assert all(Path(entry.path).exists() for entry in entries)


def test_load_trace_by_id_returns_expected_trace(tmp_path: Path) -> None:
    expected = make_stabilized_trace(trace_id="prompt-001__entropy_stochastic__seed-0")
    save_trace(tmp_path / "trace.json", expected)

    trace = load_trace_by_id(expected.trace_id, tmp_path)

    assert trace.backend == "dream"
    assert trace.decoding_metadata["run_id"]


def test_catalog_default_scope_points_to_real_traces_dir() -> None:
    trace_dir = default_trace_dir()

    assert trace_dir.name == "traces_real"


def test_catalog_extracts_profile_and_seed_metadata(tmp_path: Path) -> None:
    trace = TraceRecord(
        trace_id="prompt-001__entropy_det__seed-0",
        backend="dream",
        model_name="dream-test",
        prompt="Explain phases.",
        final_text="AB",
        tokens=[
            TraceToken(token_index=0, token_text="A", char_start=0, char_end=1),
            TraceToken(token_index=1, token_text="B", char_start=1, char_end=2),
        ],
        decoding_metadata={
            "run_id": "run-1",
            "trace_profile": "entropy_det",
            "alg": "entropy",
            "alg_temp": 0.0,
            "seed": 0,
        },
    )
    save_trace(tmp_path / "trace.json", trace)

    entries = list_catalog_entries(tmp_path)

    assert len(entries) == 1
    assert entries[0].trace_profile == "entropy_det"
    assert entries[0].profile_label == "entropy (alg_temp=0.0)"
    assert entries[0].seed == 0
    assert "entropy (alg_temp=0.0)" in entries[0].label
    assert "seed=0" in entries[0].label


def test_catalog_filters_by_profile_and_seed(tmp_path: Path) -> None:
    entropy_trace = TraceRecord(
        trace_id="entropy-trace",
        backend="dream",
        model_name="dream-test",
        prompt="Entropy trace",
        final_text="A",
        tokens=[TraceToken(token_index=0, token_text="A", char_start=0, char_end=1)],
        decoding_metadata={"alg": "entropy", "alg_temp": 0.0, "seed": 0},
    )
    origin_trace = TraceRecord(
        trace_id="origin-trace",
        backend="dream",
        model_name="dream-test",
        prompt="Origin trace",
        final_text="B",
        tokens=[TraceToken(token_index=0, token_text="B", char_start=0, char_end=1)],
        decoding_metadata={"alg": "origin", "seed": 3},
    )
    save_trace(tmp_path / "entropy.json", entropy_trace)
    save_trace(tmp_path / "origin.json", origin_trace)

    entries = list_catalog_entries(tmp_path)
    filtered = filter_catalog_entries(
        entries,
        profile_label="origin",
        seed=3,
    )

    assert [entry.trace_id for entry in filtered] == ["origin-trace"]


def test_trace_dir_signature_changes_when_trace_file_changes(tmp_path: Path) -> None:
    trace = TraceRecord(
        trace_id="signature-trace",
        backend="dream",
        model_name="dream-test",
        prompt="Prompt",
        final_text="A",
        tokens=[TraceToken(token_index=0, token_text="A", char_start=0, char_end=1)],
    )
    target = tmp_path / "trace.json"
    save_trace(target, trace)

    first_signature = trace_dir_signature(tmp_path)
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second_signature = trace_dir_signature(tmp_path)

    assert first_signature != second_signature

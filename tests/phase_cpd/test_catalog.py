from __future__ import annotations

from pathlib import Path

from phase_cpd.catalog import default_trace_dir, list_catalog_entries, load_trace_by_id


def test_catalog_lists_checked_in_mock_traces() -> None:
    trace_dir = default_trace_dir()
    entries = list_catalog_entries(trace_dir)

    assert len(entries) >= 2
    assert sorted(entry.trace_id for entry in entries) == [
        "mock-adaptive-001",
        "mock-throughput-002",
    ]
    assert all(Path(entry.path).exists() for entry in entries)


def test_load_trace_by_id_returns_expected_trace() -> None:
    trace = load_trace_by_id("mock-adaptive-001", default_trace_dir())

    assert trace.backend == "mock"
    assert trace.decoding_metadata["run_id"] == "mock-run-001"


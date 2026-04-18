from __future__ import annotations

from pathlib import Path

from phase_cpd.catalog import default_trace_dir, list_catalog_entries, load_trace_by_id


def test_catalog_lists_checked_in_real_traces() -> None:
    trace_dir = default_trace_dir()
    entries = list_catalog_entries(trace_dir)

    assert len(entries) >= 1
    assert "prompt-001" in {entry.trace_id for entry in entries}
    assert all(Path(entry.path).exists() for entry in entries)


def test_load_trace_by_id_returns_expected_trace() -> None:
    trace = load_trace_by_id("prompt-001", default_trace_dir())

    assert trace.backend == "dream"
    assert trace.decoding_metadata["run_id"]


def test_catalog_default_scope_uses_real_traces_only() -> None:
    entries = list_catalog_entries()

    assert len(entries) >= 2
    assert all(entry.path.parent.name == "traces_real" for entry in entries)

from __future__ import annotations

from phase_cpd.trace_jobs.dream_runtime import _normalize_hook_step


def test_normalize_hook_step_skips_none() -> None:
    assert _normalize_hook_step(None) is None


def test_normalize_hook_step_accepts_numeric_values() -> None:
    assert _normalize_hook_step(3) == 3
    assert _normalize_hook_step("4") == 4

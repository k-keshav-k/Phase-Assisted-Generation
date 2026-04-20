"""Unit tests for phase_predict.data_utils."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from phase_predict.data_utils import (
    _stabilizing_step,
    extract_per_segment,
    extract_per_token,
    tuples_from_trace,
)
from phase_predict.schema import PhaseTuple


# ---------------------------------------------------------------------------
# Minimal stubs that mimic phase_cpd schema objects without importing it
# ---------------------------------------------------------------------------


@dataclass
class _Obs:
    step_index: int
    token_id: int | None = None
    token_text: str | None = None


@dataclass
class _Token:
    token_index: int
    token_text: str = ""
    char_start: int = 0
    char_end: int = 1
    observations: list = field(default_factory=list)


@dataclass
class _Trace:
    trace_id: str = "stub"
    tokens: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tests for _stabilizing_step
# ---------------------------------------------------------------------------


class TestStabilizingStep:
    def test_stable_from_first_step(self) -> None:
        obs = [_Obs(0, token_id=42), _Obs(1, token_id=42), _Obs(2, token_id=42)]
        assert _stabilizing_step(obs) == 0

    def test_stabilises_at_step_2(self) -> None:
        obs = [_Obs(0, token_id=1), _Obs(1, token_id=2), _Obs(2, token_id=42), _Obs(3, token_id=42)]
        assert _stabilizing_step(obs) == 2

    def test_no_observations_returns_zero(self) -> None:
        assert _stabilizing_step([]) == 0

    def test_no_identity_returns_zero(self) -> None:
        obs = [_Obs(0), _Obs(1)]  # no token_id or token_text
        assert _stabilizing_step(obs) == 0

    def test_identity_via_token_text(self) -> None:
        obs = [
            _Obs(0, token_text="hello"),
            _Obs(1, token_text="world"),
            _Obs(2, token_text="world"),
        ]
        assert _stabilizing_step(obs) == 1


# ---------------------------------------------------------------------------
# Tests for extract_per_token
# ---------------------------------------------------------------------------


class TestExtractPerToken:
    def test_returns_one_tuple_per_token(self) -> None:
        tokens = [
            _Token(0, observations=[_Obs(0, token_id=1), _Obs(1, token_id=1)]),
            _Token(1, observations=[_Obs(0, token_id=2), _Obs(1, token_id=3), _Obs(2, token_id=3)]),
        ]
        trace = _Trace(tokens=tokens)
        result = extract_per_token(trace)
        assert len(result) == 2

    def test_block_size_always_one(self) -> None:
        tokens = [_Token(i, observations=[_Obs(0, token_id=i)]) for i in range(5)]
        trace = _Trace(tokens=tokens)
        result = extract_per_token(trace)
        assert all(t.block_size == 1 for t in result)

    def test_refinement_steps_equals_obs_count(self) -> None:
        tokens = [
            _Token(0, observations=[_Obs(0, token_id=1), _Obs(1, token_id=1), _Obs(2, token_id=1)]),
        ]
        trace = _Trace(tokens=tokens)
        result = extract_per_token(trace)
        assert result[0].refinement_steps == 3

    def test_empty_trace(self) -> None:
        trace = _Trace(tokens=[])
        assert extract_per_token(trace) == []


# ---------------------------------------------------------------------------
# Tests for extract_per_segment
# ---------------------------------------------------------------------------


class TestExtractPerSegment:
    def _make_trace(self, n: int) -> _Trace:
        tokens = [
            _Token(i, observations=[_Obs(0, token_id=i), _Obs(1, token_id=i)])
            for i in range(n)
        ]
        return _Trace(tokens=tokens)

    def test_no_breakpoints_returns_one_segment(self) -> None:
        trace = self._make_trace(10)
        result = extract_per_segment(trace, [])
        assert len(result) == 1
        assert result[0].block_size == 10

    def test_two_segments(self) -> None:
        trace = self._make_trace(10)
        result = extract_per_segment(trace, [5])
        assert len(result) == 2
        assert result[0].block_size == 5
        assert result[1].block_size == 5

    def test_segment_block_size_sum_equals_total(self) -> None:
        trace = self._make_trace(12)
        result = extract_per_segment(trace, [4, 8])
        assert sum(t.block_size for t in result) == 12

    def test_all_results_are_phase_tuples(self) -> None:
        trace = self._make_trace(10)
        result = extract_per_segment(trace, [3, 7])
        assert all(isinstance(t, PhaseTuple) for t in result)

    def test_empty_trace(self) -> None:
        trace = _Trace(tokens=[])
        assert extract_per_segment(trace, []) == []


# ---------------------------------------------------------------------------
# Tests for tuples_from_trace
# ---------------------------------------------------------------------------


class TestTuplesFromTrace:
    def test_without_breakpoints_delegates_to_per_token(self) -> None:
        tokens = [_Token(i, observations=[_Obs(0, token_id=i)]) for i in range(5)]
        trace = _Trace(tokens=tokens)
        result = tuples_from_trace(trace)
        assert all(t.block_size == 1 for t in result)

    def test_with_breakpoints_delegates_to_per_segment(self) -> None:
        tokens = [_Token(i, observations=[_Obs(0, token_id=i)]) for i in range(10)]
        trace = _Trace(tokens=tokens)
        result = tuples_from_trace(trace, breakpoints=[5])
        assert len(result) == 2

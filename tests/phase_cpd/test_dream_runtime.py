from __future__ import annotations

import pytest

from phase_cpd.trace_jobs.dream_runtime import _normalize_hook_step, _selected_token_stats


class _TorchStub:
    float32 = None

    @staticmethod
    def full_like(tensor, fill_value):
        import torch

        return torch.full_like(tensor, fill_value)


def test_normalize_hook_step_skips_none() -> None:
    assert _normalize_hook_step(None) is None


def test_normalize_hook_step_accepts_numeric_values() -> None:
    assert _normalize_hook_step(3) == 3
    assert _normalize_hook_step("4") == 4


def test_selected_token_stats_returns_entropy_and_top2() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    token_ids = torch.tensor([[0]], dtype=torch.long)

    selected_logits, selected_probs, top2_probs, entropies = _selected_token_stats(
        _TorchStub(),
        logits,
        token_ids,
    )

    assert selected_logits[0] == 2.0
    assert 0.0 < selected_probs[0] < 1.0
    assert 0.0 < top2_probs[0] < selected_probs[0]
    assert entropies[0] > 0.0

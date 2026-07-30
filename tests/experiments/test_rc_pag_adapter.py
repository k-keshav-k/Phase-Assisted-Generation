from __future__ import annotations

from dataclasses import astuple
from types import SimpleNamespace

import torch

from pag.experiments.rc_pag_adapter import (
    ShadowRequest,
    clone_tensor_tree,
    continue_shadow_refinement,
    observation_from_tensors,
    observe_policy_step,
)


class _StopPolicy:
    def __init__(self, should_stop: bool) -> None:
        self.should_stop = should_stop
        self.observations = []

    def observe(self, observation):
        self.observations.append(observation)
        return SimpleNamespace(
            should_stop=self.should_stop,
            risk_score=0.01,
            safe_streak=2,
            reason="risk_certified_candidate" if self.should_stop else "continue",
        )


def _inputs():
    logits = torch.tensor(
        [[[0.0, 2.0, 1.0, -1.0], [3.0, 0.0, 1.0, 2.0]]],
        dtype=torch.float32,
    )
    tokens = torch.tensor([[99, 2]], dtype=torch.long)
    return logits, tokens


def test_tensor_adapter_emits_compact_finite_online_observation():
    logits, tokens = _inputs()

    observation = observation_from_tensors(
        logits=logits,
        current_tokens=tokens,
        mask_token_id=99,
        step_index=2,
        digit_ids=torch.tensor([1, 2]),
        delimiter_ids=torch.tensor([0]),
    )

    assert observation.block_size == 2
    assert observation.masked == (True, False)
    assert observation.token_ids == (1, 0)
    assert all(0 <= value <= 1 for value in observation.top1_probs)
    assert all(value >= 0 for value in observation.entropies)
    assert not any(isinstance(value, torch.Tensor) for value in astuple(observation))


def test_policy_shadow_request_clones_state_and_labels_disagreement():
    logits, tokens = _inputs()
    full_tokens = torch.tensor([[4, 5, 99, 2, 99]], dtype=torch.long)
    cache = [(torch.ones(1, 1, 2, 3), torch.zeros(1, 1, 2, 3))]
    seen: list[ShadowRequest] = []

    def shadow(request: ShadowRequest):
        seen.append(request)
        request.tokens[0, 0] = -1
        return torch.tensor([[3, 2]])

    result = observe_policy_step(
        _StopPolicy(True),
        logits=logits,
        current_tokens=tokens,
        mask_token_id=99,
        step_index=2,
        full_tokens=full_tokens,
        block_start=2,
        block_end=4,
        cache=cache,
        shadow_callback=shadow,
    )

    assert result is not None
    assert result.decision.should_stop
    assert result.shadow_loss == 1
    assert full_tokens[0, 0].item() == 4
    assert seen[0].block_start == 2
    assert seen[0].block_end == 4
    assert seen[0].proposed_tokens.tolist() == [[1, 2]]
    assert seen[0].cache is not cache


def test_clone_tensor_tree_preserves_structure_without_aliasing():
    source = {"cache": (torch.tensor([1.0]), [torch.tensor([2.0])])}
    cloned = clone_tensor_tree(source)

    cloned["cache"][0][0] = 9
    assert source["cache"][0].item() == 1


def test_shadow_refinement_continues_identical_state_to_hard_ceiling():
    request = ShadowRequest(
        tokens=torch.tensor([[7, 99, 99, 8]]),
        proposed_tokens=torch.tensor([[1, 2]]),
        block_start=1,
        block_end=3,
        step_index=1,
        cache=None,
    )
    calls = []

    def forward(tokens, cache, block_start, block_end):
        del cache
        calls.append(tokens.clone())
        logits = torch.zeros((1, block_end - block_start, 4))
        logits[0, 0, 1] = 5
        logits[0, 1, 3] = 5
        return logits, None

    result = continue_shadow_refinement(
        request,
        forward=forward,
        mask_token_id=99,
        threshold=0.99,
        max_steps=2,
    )

    assert result.tokens.tolist() == [[1, 3]]
    assert result.additional_nfe == 1
    assert calls[0].tolist() == [[7, 99, 99, 8]]

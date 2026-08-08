from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pag.experiments.rc_pag_features import StepObservation
from pag.experiments.rc_pag_speculation import (
    RiskAdaptiveSpeculationPolicy,
    build_linear_draft,
    repeat_tensor_tree,
    serialize_speculation_result,
    verify_draft,
)


@dataclass
class _Scorer:
    risk: float

    def predict_risk(self, features):
        assert "local.remaining_count" in features
        return self.risk


def _observation() -> StepObservation:
    return StepObservation.from_arrays(
        step_index=3,
        block_size=4,
        masked=[False, True, True, True],
        top1_probs=[1.0, 0.95, 0.8, 0.6],
        top2_probs=[0.0, 0.03, 0.1, 0.2],
        entropies=[0.0, 0.2, 0.5, 0.9],
        token_ids=[10, 11, 12, 13],
        temporal_js=[0.0, 0.01, 0.02, 0.03],
    )


@pytest.mark.parametrize(
    ("risk", "expected_depth"),
    [(0.05, 4), (0.20, 2), (0.80, 0)],
)
def test_risk_controls_capacity_not_token_acceptance(risk: float, expected_depth: int) -> None:
    policy = RiskAdaptiveSpeculationPolicy(
        _Scorer(risk),
        max_depth=4,
        medium_depth=2,
        deep_risk_threshold=0.10,
        medium_risk_threshold=0.30,
        draft_width_multiplier=1.0,
    )

    plan = policy.choose(_observation(), last_transfer_count=2)

    assert plan.risk_score == risk
    assert plan.depth == expected_depth
    assert plan.draft_width == 2


def test_policy_validates_nested_thresholds_and_depths() -> None:
    with pytest.raises(ValueError, match="deep risk threshold"):
        RiskAdaptiveSpeculationPolicy(
            _Scorer(0.1),
            max_depth=4,
            medium_depth=2,
            deep_risk_threshold=0.4,
            medium_risk_threshold=0.3,
            draft_width_multiplier=1.0,
        )
    with pytest.raises(ValueError, match="medium depth"):
        RiskAdaptiveSpeculationPolicy(
            _Scorer(0.1),
            max_depth=2,
            medium_depth=3,
            deep_risk_threshold=0.1,
            medium_risk_threshold=0.3,
            draft_width_multiplier=1.0,
        )


def test_linear_draft_only_reveals_ranked_masked_positions() -> None:
    root = torch.tensor([[5, 99, 99, 8, 99]])
    proposed = torch.tensor([[5, 6, 7, 8, 9]])

    nodes = build_linear_draft(
        root,
        proposed_tokens=proposed,
        mask_token_id=99,
        ranked_positions=(4, 2, 1),
        depth=3,
        draft_width=1,
    )

    assert [node.tolist() for node in nodes] == [
        [[5, 99, 99, 8, 99]],
        [[5, 99, 99, 8, 9]],
        [[5, 99, 7, 8, 9]],
        [[5, 6, 7, 8, 9]],
    ]
    assert all(node[0, 0].item() == 5 and node[0, 3].item() == 8 for node in nodes)


def test_verifier_accepts_exact_prefix_then_uses_verified_rejection() -> None:
    nodes = (
        torch.tensor([[0, -1, -1]]),
        torch.tensor([[0, 1, -1]]),
        torch.tensor([[0, 1, 2]]),
    )
    verified_successors = torch.tensor(
        [
            [[0, 1, -1]],
            [[0, 1, 9]],
            [[0, 1, 2]],
        ]
    )

    def transition(state: torch.Tensor, verified: torch.Tensor) -> torch.Tensor:
        del state
        return verified.clone()

    result = verify_draft(nodes, verified_successors, transition, mask_token_id=-1)

    assert result.tokens.tolist() == [[0, 1, 9]]
    assert result.accepted_draft_edges == 1
    assert result.verified_transitions == 2
    assert result.evaluated_nodes == 3
    assert result.rejection_depth == 2
    assert result.sequence_safe is True


def test_verifier_uses_leaf_logits_for_an_additional_exact_transition() -> None:
    nodes = (
        torch.tensor([[0, -1, -1, -1]]),
        torch.tensor([[0, 1, -1, -1]]),
        torch.tensor([[0, 1, 2, -1]]),
    )
    verified_successors = torch.tensor(
        [
            [[0, 1, -1, -1]],
            [[0, 1, 2, -1]],
            [[0, 1, 2, 3]],
        ]
    )

    result = verify_draft(
        nodes,
        verified_successors,
        lambda _state, successor: successor.clone(),
        mask_token_id=-1,
    )

    assert result.tokens.tolist() == [[0, 1, 2, 3]]
    assert result.accepted_draft_edges == 2
    assert result.verified_transitions == 3
    assert result.rejection_depth is None
    assert serialize_speculation_result(result)["nfe_saved"] == 2


def test_verifier_does_not_count_a_terminal_draft_leaf_as_a_transition() -> None:
    nodes = (
        torch.tensor([[0, -1, -1]]),
        torch.tensor([[0, 1, -1]]),
        torch.tensor([[0, 1, 2]]),
    )
    outputs = torch.tensor(
        [
            [[0, 1, -1]],
            [[0, 1, 2]],
            [[0, 1, 2]],
        ]
    )

    result = verify_draft(
        nodes,
        outputs,
        lambda _state, successor: successor.clone(),
        mask_token_id=-1,
    )

    assert result.tokens.tolist() == [[0, 1, 2]]
    assert result.accepted_draft_edges == 2
    assert result.verified_transitions == 2
    assert result.nfe_saved == 1


def test_verifier_rejects_a_transition_that_rewrites_committed_tokens() -> None:
    nodes = (torch.tensor([[4, -1]]),)
    outputs = torch.tensor([[[7, 3]]])

    with pytest.raises(ValueError, match="rewrote an unmasked token"):
        verify_draft(
            nodes,
            outputs,
            lambda _state, successor: successor.clone(),
            mask_token_id=-1,
        )


def test_repeat_tensor_tree_expands_only_singleton_batch_leaves() -> None:
    cache = ((torch.tensor([[[1.0], [2.0]]]), torch.tensor([[[3.0]]])),)

    repeated = repeat_tensor_tree(cache, 3)

    assert repeated[0][0].shape == (3, 2, 1)
    assert repeated[0][1].shape == (3, 1, 1)
    assert torch.equal(repeated[0][0][2], cache[0][0][0])

from __future__ import annotations

import math

import pytest
import torch

from pag.experiments.rc_pag_equivalence import (
    EquivalenceCostArtifact,
    EquivalenceCostPolicy,
    EquivalenceEnvelope,
    decision_margins,
    fit_equivalence_artifact,
    guard_transition,
    state_digest,
    verify_guarded_draft,
)
from pag.experiments.rc_pag_features import StepObservation

MASK = 9


def _logits(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


def _observation(*, confidence: float = 0.95) -> StepObservation:
    return StepObservation(
        step_index=2,
        block_size=4,
        masked=(True, True, False, False),
        top1_probs=(confidence, confidence - 0.01, 1.0, 1.0),
        top2_probs=(0.02, 0.03, 0.0, 0.0),
        entropies=(0.1, 0.2, 0.0, 0.0),
        token_ids=(1, 2, 3, 4),
        temporal_js=(0.0, 0.0, 0.0, 0.0),
        digit_ids=frozenset(),
        delimiter_ids=frozenset(),
    )


def test_decision_margins_cover_token_threshold_and_forced_rank() -> None:
    logits = _logits(
        [
            [8.0, 1.0, 0.0],
            [7.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    state = torch.tensor([[MASK, MASK, 3]])

    margins = decision_margins(logits, state, mask_token_id=MASK, threshold=0.9)

    assert margins.token_margin == pytest.approx(6.0)
    assert margins.threshold_margin > 0.09
    assert margins.forced_rank_margin > 0.0


def test_guard_requires_every_margin_to_clear_the_envelope() -> None:
    logits = _logits([[2.197, 0.0, -10.0], [8.0, 0.0, 0.0]])
    state = torch.tensor([[MASK, MASK]])
    envelope = EquivalenceEnvelope(logit_epsilon=0.1, probability_epsilon=0.01)

    margins = decision_margins(logits, state, mask_token_id=MASK, threshold=0.9)
    decision = guard_transition(margins, envelope)

    assert margins.token_margin > 2 * envelope.logit_epsilon
    assert not decision.passed
    assert decision.reason in {"threshold_margin", "forced_rank_margin"}


def test_guard_boundary_is_strict() -> None:
    envelope = EquivalenceEnvelope(logit_epsilon=0.1, probability_epsilon=0.01)
    from pag.experiments.rc_pag_equivalence import DecisionMargins

    decision = guard_transition(
        DecisionMargins(token_margin=0.2, threshold_margin=0.02, forced_rank_margin=0.03),
        envelope,
    )

    assert not decision.passed
    assert decision.reason == "token_margin"


def test_audit_verifier_returns_canonical_root_successor() -> None:
    root = torch.tensor([[MASK, MASK]])
    child = torch.tensor([[1, MASK]])
    batched = torch.tensor(
        [
            [[5.0, 4.0], [5.0, 4.0]],
            [[5.0, 4.0], [5.0, 4.0]],
        ]
    )
    canonical = torch.tensor([[[4.0, 5.0], [5.0, 4.0]]])

    def transition(state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        successor = state.clone()
        first_mask = int(torch.nonzero(state[0] == MASK, as_tuple=False)[0].item())
        successor[0, first_mask] = int(logits[first_mask].argmax().item())
        return successor

    result = verify_guarded_draft(
        (root, child),
        batched,
        transition,
        guard=lambda state, logits: guard_transition(
            decision_margins(logits, state, mask_token_id=MASK, threshold=0.9),
            EquivalenceEnvelope(0.0, 0.0),
        ),
        mask_token_id=MASK,
        canonical_root_output=canonical[0],
    )

    assert torch.equal(result.tokens, torch.tensor([[1, MASK]]))
    assert result.reference_checked
    assert result.successor_equal_when_checked is False
    assert result.canonical_fallback_rows == 1
    assert result.reference_equivalent_transitions == 1


def test_production_verifier_falls_back_only_for_unsafe_root() -> None:
    root = torch.tensor([[MASK, MASK]])
    child = torch.tensor([[0, MASK]])
    outputs = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 1.0]],
            [[4.0, 1.0], [3.0, 1.0]],
        ]
    )
    fallback_calls = 0

    def transition(state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        successor = state.clone()
        first_mask = int(torch.nonzero(state[0] == MASK, as_tuple=False)[0].item())
        successor[0, first_mask] = int(logits[first_mask].argmax().item())
        return successor

    def canonical_root() -> torch.Tensor:
        nonlocal fallback_calls
        fallback_calls += 1
        return torch.tensor([[4.0, 1.0], [3.0, 1.0]])

    result = verify_guarded_draft(
        (root, child),
        outputs,
        transition,
        guard=lambda state, logits: guard_transition(
            decision_margins(logits, state, mask_token_id=MASK, threshold=0.9),
            EquivalenceEnvelope(0.1, 0.01),
        ),
        mask_token_id=MASK,
        canonical_root=canonical_root,
    )

    assert fallback_calls == 1
    assert result.canonical_fallback_rows == 1
    assert result.reason == "canonical_root_fallback"
    assert torch.equal(result.tokens, torch.tensor([[0, MASK]]))


def test_fit_artifact_inflates_envelope_and_policy_prefers_depth_one() -> None:
    key = "r2|t1|m3|q3|b1"
    events = []
    for depth, batch_ms in ((1, 1.0), (2, 1.4)):
        for _ in range(8):
            events.append(
                {
                    "batch_size": depth + 1,
                    "depth": depth,
                    "activation_key": key,
                    "max_logit_delta": 0.08,
                    "max_probability_delta": 0.004,
                    "full_acceptance": True,
                    "batched_latency_ms": batch_ms,
                    "canonical_latency_ms": 0.8,
                }
            )

    payload = fit_equivalence_artifact(
        events,
        fingerprint={"gpu": "A100"},
        safety_inflation=1.25,
        minimum_bin_count=4,
        minimum_acceptance_lcb=0.5,
    )
    artifact = EquivalenceCostArtifact.from_dict(payload)
    policy = EquivalenceCostPolicy.production(artifact, threshold=0.9)
    policy.activation_key = lambda observation, last_transfer_count: key  # type: ignore[method-assign]

    plan = policy.choose(_observation(), last_transfer_count=1)

    assert artifact.envelopes[2].logit_epsilon == pytest.approx(0.1)
    assert artifact.envelopes[2].probability_epsilon == pytest.approx(0.005)
    assert plan.depth == 1
    assert plan.reason == "certified_depth_1"


def test_sparse_or_unseen_cost_bin_falls_back_to_adablock() -> None:
    payload = fit_equivalence_artifact(
        [
            {
                "batch_size": 2,
                "depth": 1,
                "activation_key": "known",
                "max_logit_delta": 0.01,
                "max_probability_delta": 0.001,
                "full_acceptance": True,
                "batched_latency_ms": 1.0,
                "canonical_latency_ms": 1.0,
            }
        ],
        fingerprint={"gpu": "A100"},
        minimum_bin_count=4,
        minimum_acceptance_lcb=0.5,
    )
    policy = EquivalenceCostPolicy.production(
        EquivalenceCostArtifact.from_dict(payload), threshold=0.9
    )

    plan = policy.choose(_observation(), last_transfer_count=1)

    assert plan.depth == 0
    assert plan.reason == "uncertified_cost_bin"


def test_artifact_rejects_empty_or_nonfinite_events() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fit_equivalence_artifact([], fingerprint={"gpu": "A100"})
    event = {
        "batch_size": 2,
        "depth": 1,
        "activation_key": "x",
        "max_logit_delta": math.inf,
        "max_probability_delta": 0.0,
        "full_acceptance": True,
        "batched_latency_ms": 1.0,
        "canonical_latency_ms": 1.0,
    }
    with pytest.raises(ValueError, match="finite"):
        fit_equivalence_artifact([event], fingerprint={"gpu": "A100"})


def test_state_digest_is_shape_and_position_sensitive() -> None:
    state = torch.tensor([[1, 2, 3]])

    assert state_digest(state, block_start=4) == state_digest(state.clone(), block_start=4)
    assert state_digest(state, block_start=4) != state_digest(state, block_start=5)
    assert state_digest(state, block_start=4) != state_digest(state[:, :2], block_start=4)

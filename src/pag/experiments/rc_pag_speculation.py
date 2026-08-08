from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from pag.experiments.rc_pag_features import (
    RealizedBlock,
    StepObservation,
    extract_features,
    feature_names,
)


class SpeculationRiskScorer(Protocol):
    def predict_risk(self, features: Mapping[str, float]) -> float: ...


@dataclass(frozen=True, slots=True)
class SpeculationPlan:
    risk_score: float
    depth: int
    draft_width: int
    reason: str

    @property
    def node_budget(self) -> int:
        return self.depth + 1


@dataclass(frozen=True, slots=True)
class SpeculationResult:
    tokens: torch.Tensor
    accepted_draft_edges: int
    verified_transitions: int
    evaluated_nodes: int
    rejection_depth: int | None
    sequence_safe: bool = True

    @property
    def nfe_saved(self) -> int:
        return max(0, self.verified_transitions - 1)


class RiskAdaptiveSpeculationPolicy:
    """Allocate verifier capacity without granting token-acceptance authority."""

    def __init__(
        self,
        scorer: SpeculationRiskScorer,
        *,
        max_depth: int,
        medium_depth: int,
        deep_risk_threshold: float,
        medium_risk_threshold: float,
        draft_width_multiplier: float,
        include_history: bool = False,
        history_window: int = 4,
    ) -> None:
        if max_depth < 1:
            raise ValueError("maximum speculation depth must be positive")
        if not 1 <= medium_depth <= max_depth:
            raise ValueError("medium depth must be between one and maximum depth")
        if not 0.0 <= deep_risk_threshold <= 1.0:
            raise ValueError("deep risk threshold must be in [0, 1]")
        if not deep_risk_threshold <= medium_risk_threshold <= 1.0:
            raise ValueError("deep risk threshold must not exceed the medium risk threshold")
        if not math.isfinite(draft_width_multiplier) or draft_width_multiplier <= 0.0:
            raise ValueError("draft width multiplier must be finite and positive")
        if history_window < 1:
            raise ValueError("history window must be positive")
        self.scorer = scorer
        self.max_depth = int(max_depth)
        self.medium_depth = int(medium_depth)
        self.deep_risk_threshold = float(deep_risk_threshold)
        self.medium_risk_threshold = float(medium_risk_threshold)
        self.draft_width_multiplier = float(draft_width_multiplier)
        self.include_history = bool(include_history)
        self.history_window = int(history_window)
        self.reset_prompt()

    @property
    def history(self) -> tuple[RealizedBlock, ...]:
        return tuple(self._history)

    def reset_prompt(self) -> None:
        self._history: list[RealizedBlock] = []
        self.start_block()

    def start_block(self) -> None:
        self._previous: StepObservation | None = None

    def record_realized(self, block: RealizedBlock) -> None:
        self._history.append(block)

    def choose(
        self,
        observation: StepObservation,
        *,
        last_transfer_count: int,
    ) -> SpeculationPlan:
        if last_transfer_count < 1:
            raise ValueError("last transfer count must be positive")
        all_features = extract_features(
            observation,
            previous=self._previous,
            history=self._history if self.include_history else (),
            history_window=self.history_window,
        )
        selected = {
            name: all_features[name] for name in feature_names(include_history=self.include_history)
        }
        risk = float(self.scorer.predict_risk(selected))
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError("speculation risk scorer must return a probability in [0, 1]")
        if risk <= self.deep_risk_threshold:
            depth = self.max_depth
            reason = "deep_verified_speculation"
        elif risk <= self.medium_risk_threshold:
            depth = self.medium_depth
            reason = "medium_verified_speculation"
        else:
            depth = 0
            reason = "single_verified_transition"
        width = max(1, round(last_transfer_count * self.draft_width_multiplier))
        self._previous = observation
        return SpeculationPlan(
            risk_score=risk,
            depth=depth,
            draft_width=int(width),
            reason=reason,
        )


def _validate_state_transition(
    parent: torch.Tensor,
    child: torch.Tensor,
    *,
    mask_token_id: int,
) -> None:
    if parent.ndim != 2 or parent.shape[0] != 1 or child.shape != parent.shape:
        raise ValueError("speculative states must have matching [1, block_size] shapes")
    committed = parent != int(mask_token_id)
    if bool(torch.any(child[committed] != parent[committed])):
        raise ValueError("verified transition rewrote an unmasked token")


def build_linear_draft(
    root: torch.Tensor,
    *,
    proposed_tokens: torch.Tensor,
    mask_token_id: int,
    ranked_positions: Sequence[int],
    depth: int,
    draft_width: int,
) -> tuple[torch.Tensor, ...]:
    if root.ndim != 2 or root.shape[0] != 1:
        raise ValueError("draft root must have shape [1, block_size]")
    if proposed_tokens.shape != root.shape:
        raise ValueError("proposed tokens must match the draft root")
    if depth < 0 or draft_width < 1:
        raise ValueError("draft depth must be non-negative and width must be positive")
    positions = tuple(int(position) for position in ranked_positions)
    if len(set(positions)) != len(positions):
        raise ValueError("ranked draft positions must be unique")
    block_size = root.shape[1]
    for position in positions:
        if not 0 <= position < block_size:
            raise ValueError("ranked draft position is outside the active block")
        if int(root[0, position].item()) != int(mask_token_id):
            raise ValueError("ranked draft positions must be masked in the root")
        if int(proposed_tokens[0, position].item()) == int(mask_token_id):
            raise ValueError("a draft proposal cannot reveal another mask token")

    nodes = [root.detach().clone()]
    cursor = 0
    for _ in range(depth):
        if cursor >= len(positions):
            break
        child = nodes[-1].detach().clone()
        selected = positions[cursor : cursor + draft_width]
        for position in selected:
            child[0, position] = proposed_tokens[0, position]
        _validate_state_transition(nodes[-1], child, mask_token_id=mask_token_id)
        nodes.append(child)
        cursor += len(selected)
    return tuple(nodes)


def verify_draft(
    nodes: Sequence[torch.Tensor],
    verified_outputs: Sequence[Any] | torch.Tensor,
    transition: Callable[[torch.Tensor, Any], torch.Tensor],
    *,
    mask_token_id: int,
) -> SpeculationResult:
    if not nodes:
        raise ValueError("a verification draft must contain a root node")
    if len(verified_outputs) != len(nodes):
        raise ValueError("each speculative node needs one verified model output")
    frozen_nodes = tuple(node.detach().clone() for node in nodes)
    for parent, child in zip(frozen_nodes, frozen_nodes[1:], strict=False):
        _validate_state_transition(parent, child, mask_token_id=mask_token_id)

    accepted = 0
    for index, node in enumerate(frozen_nodes):
        if not bool(torch.any(node == int(mask_token_id))):
            return SpeculationResult(
                tokens=node.detach().clone(),
                accepted_draft_edges=accepted,
                verified_transitions=accepted,
                evaluated_nodes=len(frozen_nodes),
                rejection_depth=None,
            )
        successor = transition(node.detach().clone(), verified_outputs[index])
        _validate_state_transition(node, successor, mask_token_id=mask_token_id)
        if index + 1 < len(frozen_nodes) and torch.equal(successor, frozen_nodes[index + 1]):
            accepted += 1
            continue
        rejection_depth = index + 1 if index + 1 < len(frozen_nodes) else None
        return SpeculationResult(
            tokens=successor.detach().clone(),
            accepted_draft_edges=accepted,
            verified_transitions=accepted + 1,
            evaluated_nodes=len(frozen_nodes),
            rejection_depth=rejection_depth,
        )
    raise AssertionError("verification loop must return from a finite non-empty node sequence")


def repeat_tensor_tree(value: Any, repeats: int) -> Any:
    if repeats < 1:
        raise ValueError("cache repeat count must be positive")
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().clone()
        if value.shape[0] != 1:
            raise ValueError("cache tensor leaves must have singleton batch dimension")
        multiples = (repeats,) + (1,) * (value.ndim - 1)
        return value.detach().repeat(multiples)
    if isinstance(value, tuple):
        return tuple(repeat_tensor_tree(item, repeats) for item in value)
    if isinstance(value, list):
        return [repeat_tensor_tree(item, repeats) for item in value]
    if isinstance(value, Mapping):
        return {key: repeat_tensor_tree(item, repeats) for key, item in value.items()}
    return value


def serialize_speculation_plan(plan: SpeculationPlan) -> dict[str, Any]:
    return {
        "risk_score": plan.risk_score,
        "proposed_depth": plan.depth,
        "draft_width": plan.draft_width,
        "node_budget": plan.node_budget,
        "reason": plan.reason,
    }


def serialize_speculation_result(result: SpeculationResult) -> dict[str, Any]:
    return {
        "accepted_draft_edges": result.accepted_draft_edges,
        "verified_transitions": result.verified_transitions,
        "evaluated_nodes": result.evaluated_nodes,
        "rejection_depth": result.rejection_depth,
        "sequence_safe": result.sequence_safe,
        "nfe_saved": result.nfe_saved,
    }

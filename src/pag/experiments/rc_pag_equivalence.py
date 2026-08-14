from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from scipy.stats import beta

from pag.experiments.rc_pag_features import RealizedBlock, StepObservation
from pag.experiments.rc_pag_speculation import SpeculationPlan


@dataclass(frozen=True, slots=True)
class DecisionMargins:
    token_margin: float
    threshold_margin: float
    forced_rank_margin: float

    def __post_init__(self) -> None:
        values = (self.token_margin, self.threshold_margin, self.forced_rank_margin)
        if any(math.isnan(value) or value < 0.0 for value in values):
            raise ValueError("decision margins must be non-negative and not NaN")


@dataclass(frozen=True, slots=True)
class EquivalenceEnvelope:
    logit_epsilon: float
    probability_epsilon: float
    safety_inflation: float = 1.25

    def __post_init__(self) -> None:
        values = (self.logit_epsilon, self.probability_epsilon, self.safety_inflation)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("equivalence envelope values must be finite")
        if self.logit_epsilon < 0.0 or self.probability_epsilon < 0.0:
            raise ValueError("equivalence epsilons must be non-negative")
        if self.safety_inflation < 1.0:
            raise ValueError("equivalence safety inflation must be at least one")


@dataclass(frozen=True, slots=True)
class GuardDecision:
    passed: bool
    margins: DecisionMargins
    reason: str


@dataclass(frozen=True, slots=True)
class CostRule:
    activation_key: str
    depth: int
    count: int
    full_acceptance_rate: float
    acceptance_lower_bound: float
    latency_reduction_lower_bound: float
    enabled: bool

    def __post_init__(self) -> None:
        if not self.activation_key:
            raise ValueError("cost rule activation key is required")
        if self.depth not in {1, 2} or self.count < 1:
            raise ValueError("cost rules require depth one/two and a positive count")
        probabilities = (self.full_acceptance_rate, self.acceptance_lower_bound)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("cost rule acceptance values must be probabilities")
        if not math.isfinite(self.latency_reduction_lower_bound):
            raise ValueError("cost rule latency lower bound must be finite")


@dataclass(frozen=True, slots=True)
class EquivalenceCostArtifact:
    schema_version: int
    fingerprint: dict[str, Any]
    envelopes: dict[int, EquivalenceEnvelope]
    cost_rules: tuple[CostRule, ...]
    safety_inflation: float
    minimum_bin_count: int
    minimum_acceptance_lcb: float
    artifact_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("equivalence artifact schema_version must be one")
        if not self.fingerprint:
            raise ValueError("equivalence artifact fingerprint is required")
        if not self.envelopes or any(batch not in {2, 3} for batch in self.envelopes):
            raise ValueError("equivalence artifact requires batch-two/three envelopes")
        if self.minimum_bin_count < 1:
            raise ValueError("minimum cost-bin count must be positive")
        if not 0.0 <= self.minimum_acceptance_lcb <= 1.0:
            raise ValueError("minimum acceptance lower bound must be a probability")
        if not self.artifact_hash:
            raise ValueError("equivalence artifact hash is required")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EquivalenceCostArtifact:
        return cls(
            schema_version=int(payload["schema_version"]),
            fingerprint=dict(payload["fingerprint"]),
            envelopes={
                int(batch): EquivalenceEnvelope(**dict(values))
                for batch, values in dict(payload["envelopes"]).items()
            },
            cost_rules=tuple(CostRule(**dict(rule)) for rule in payload.get("cost_rules", ())),
            safety_inflation=float(payload["safety_inflation"]),
            minimum_bin_count=int(payload["minimum_bin_count"]),
            minimum_acceptance_lcb=float(payload["minimum_acceptance_lcb"]),
            artifact_hash=str(payload["artifact_hash"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "envelopes": {
                str(batch): asdict(envelope) for batch, envelope in sorted(self.envelopes.items())
            },
            "cost_rules": [asdict(rule) for rule in self.cost_rules],
            "safety_inflation": self.safety_inflation,
            "minimum_bin_count": self.minimum_bin_count,
            "minimum_acceptance_lcb": self.minimum_acceptance_lcb,
            "artifact_hash": self.artifact_hash,
        }


@dataclass(frozen=True, slots=True)
class GuardedSpeculationResult:
    tokens: torch.Tensor
    accepted_draft_edges: int
    reference_equivalent_transitions: int
    evaluated_nodes: int
    canonical_fallback_rows: int
    guard_passed: bool
    reference_checked: bool
    successor_equal_when_checked: bool | None
    transition_states: tuple[torch.Tensor, ...]
    reason: str
    margins: tuple[DecisionMargins, ...] = ()

    @property
    def serial_forward_calls(self) -> int:
        return 1 + self.canonical_fallback_rows

    @property
    def evaluated_rows(self) -> int:
        return self.evaluated_nodes + self.canonical_fallback_rows

    @property
    def nfe_saved(self) -> int:
        return max(0, self.reference_equivalent_transitions - self.serial_forward_calls)


def _normalize_logits(logits: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 2 or state.shape[0] != 1:
        raise ValueError("equivalence states must have shape [1, length]")
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("one state requires one logits row")
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[0] != state.shape[1]:
        raise ValueError("logits must have shape [length, vocabulary]")
    if logits.shape[1] < 2:
        raise ValueError("decision margins require at least two vocabulary logits")
    return logits.float()


def decision_margins(
    logits: torch.Tensor,
    state: torch.Tensor,
    *,
    mask_token_id: int,
    threshold: float,
) -> DecisionMargins:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("transfer threshold must be in [0, 1]")
    normalized = _normalize_logits(logits, state)
    masked = state[0] == int(mask_token_id)
    if not bool(masked.any()):
        return DecisionMargins(math.inf, math.inf, math.inf)
    selected = normalized[masked]
    top_two = torch.topk(selected, k=2, dim=-1).values
    token_margin = float(torch.min(top_two[:, 0] - top_two[:, 1]).item())
    probabilities = torch.softmax(selected, dim=-1)
    top_probability = torch.max(probabilities, dim=-1).values
    threshold_margin = float(torch.min(torch.abs(top_probability - float(threshold))).item())
    if top_probability.numel() == 1:
        forced_rank_margin = math.inf
    else:
        ranked = torch.topk(top_probability, k=2).values
        forced_rank_margin = float((ranked[0] - ranked[1]).item())
    return DecisionMargins(token_margin, threshold_margin, forced_rank_margin)


def guard_transition(
    margins: DecisionMargins,
    envelope: EquivalenceEnvelope,
) -> GuardDecision:
    conditions = (
        (margins.token_margin > 2.0 * envelope.logit_epsilon, "token_margin"),
        (margins.threshold_margin > envelope.probability_epsilon, "threshold_margin"),
        (margins.forced_rank_margin > 2.0 * envelope.probability_epsilon, "forced_rank_margin"),
    )
    for passed, reason in conditions:
        if not passed:
            return GuardDecision(False, margins, reason)
    return GuardDecision(True, margins, "certified")


def _validate_transition(parent: torch.Tensor, child: torch.Tensor, *, mask_token_id: int) -> None:
    if parent.ndim != 2 or parent.shape[0] != 1 or child.shape != parent.shape:
        raise ValueError("guarded speculative states must share shape [1, length]")
    committed = parent != int(mask_token_id)
    if bool(torch.any(parent[committed] != child[committed])):
        raise ValueError("guarded transition rewrote a committed token")


def verify_guarded_draft(
    nodes: Sequence[torch.Tensor],
    batched_outputs: Sequence[Any] | torch.Tensor,
    transition: Callable[[torch.Tensor, Any], torch.Tensor],
    guard: Callable[[torch.Tensor, Any], GuardDecision],
    *,
    mask_token_id: int,
    canonical_root: Callable[[], Any] | None = None,
    canonical_root_output: Any | None = None,
) -> GuardedSpeculationResult:
    if not nodes:
        raise ValueError("a guarded draft requires at least one node")
    if len(nodes) != len(batched_outputs):
        raise ValueError("each guarded draft node requires one model output")
    if canonical_root is not None and canonical_root_output is not None:
        raise ValueError("provide either a lazy or eager canonical root, not both")
    frozen = tuple(node.detach().clone() for node in nodes)
    for parent, child in zip(frozen, frozen[1:], strict=False):
        _validate_transition(parent, child, mask_token_id=mask_token_id)

    if canonical_root_output is not None:
        batched_successor = transition(frozen[0].clone(), batched_outputs[0])
        canonical_successor = transition(frozen[0].clone(), canonical_root_output)
        _validate_transition(frozen[0], canonical_successor, mask_token_id=mask_token_id)
        root_guard = guard(frozen[0], batched_outputs[0])
        return GuardedSpeculationResult(
            tokens=canonical_successor.detach().clone(),
            accepted_draft_edges=int(len(frozen) > 1 and torch.equal(batched_successor, frozen[1])),
            reference_equivalent_transitions=1,
            evaluated_nodes=len(frozen),
            canonical_fallback_rows=1,
            guard_passed=root_guard.passed,
            reference_checked=True,
            successor_equal_when_checked=torch.equal(batched_successor, canonical_successor),
            transition_states=(canonical_successor.detach().clone(),),
            reason="audit_reference",
            margins=(root_guard.margins,),
        )

    accepted = 0
    transitions: list[torch.Tensor] = []
    margins: list[DecisionMargins] = []
    for index, node in enumerate(frozen):
        if not bool(torch.any(node == int(mask_token_id))):
            return GuardedSpeculationResult(
                tokens=node.detach().clone(),
                accepted_draft_edges=accepted,
                reference_equivalent_transitions=len(transitions),
                evaluated_nodes=len(frozen),
                canonical_fallback_rows=0,
                guard_passed=True,
                reference_checked=False,
                successor_equal_when_checked=None,
                transition_states=tuple(transitions),
                reason="complete_draft_state",
                margins=tuple(margins),
            )
        decision = guard(node, batched_outputs[index])
        margins.append(decision.margins)
        if not decision.passed:
            if index == 0:
                if canonical_root is None:
                    raise ValueError("unsafe guarded root requires a canonical fallback")
                canonical_output = canonical_root()
                batched_successor = transition(node.clone(), batched_outputs[index])
                canonical_successor = transition(node.clone(), canonical_output)
                _validate_transition(node, canonical_successor, mask_token_id=mask_token_id)
                return GuardedSpeculationResult(
                    tokens=canonical_successor.detach().clone(),
                    accepted_draft_edges=0,
                    reference_equivalent_transitions=1,
                    evaluated_nodes=len(frozen),
                    canonical_fallback_rows=1,
                    guard_passed=False,
                    reference_checked=True,
                    successor_equal_when_checked=torch.equal(
                        batched_successor, canonical_successor
                    ),
                    transition_states=(canonical_successor.detach().clone(),),
                    reason="canonical_root_fallback",
                    margins=tuple(margins),
                )
            return GuardedSpeculationResult(
                tokens=node.detach().clone(),
                accepted_draft_edges=accepted,
                reference_equivalent_transitions=len(transitions),
                evaluated_nodes=len(frozen),
                canonical_fallback_rows=0,
                guard_passed=False,
                reference_checked=False,
                successor_equal_when_checked=None,
                transition_states=tuple(transitions),
                reason=f"unsafe_depth_{index}",
                margins=tuple(margins),
            )
        successor = transition(node.clone(), batched_outputs[index])
        _validate_transition(node, successor, mask_token_id=mask_token_id)
        transitions.append(successor.detach().clone())
        if index + 1 < len(frozen) and torch.equal(successor, frozen[index + 1]):
            accepted += 1
            continue
        return GuardedSpeculationResult(
            tokens=successor.detach().clone(),
            accepted_draft_edges=accepted,
            reference_equivalent_transitions=len(transitions),
            evaluated_nodes=len(frozen),
            canonical_fallback_rows=0,
            guard_passed=True,
            reference_checked=False,
            successor_equal_when_checked=None,
            transition_states=tuple(transitions),
            reason="draft_rejected" if index + 1 < len(frozen) else "guarded_leaf",
            margins=tuple(margins),
        )
    raise AssertionError("finite guarded verification must return")


def state_digest(state: torch.Tensor, *, block_start: int) -> str:
    if block_start < 0:
        raise ValueError("block_start must be non-negative")
    values = state.detach().to(device="cpu").contiguous()
    metadata = json.dumps(
        {
            "block_start": int(block_start),
            "dtype": str(values.dtype),
            "shape": list(values.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(metadata + b"\0" + values.numpy().tobytes()).hexdigest()


def _binomial_lower(successes: int, count: int, *, confidence: float = 0.95) -> float:
    if not 0 <= successes <= count or count < 1:
        raise ValueError("invalid binomial counts")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, count - successes + 1))


def _lower_empirical(values: Sequence[float], *, confidence: float = 0.95) -> float:
    if not values:
        raise ValueError("cannot summarize an empty latency bin")
    return float(np.quantile(np.asarray(values, dtype=np.float64), 1.0 - confidence))


def fit_equivalence_artifact(
    events: Sequence[Mapping[str, Any]],
    *,
    fingerprint: Mapping[str, Any],
    safety_inflation: float = 1.25,
    minimum_bin_count: int = 8,
    minimum_acceptance_lcb: float = 0.8,
) -> dict[str, Any]:
    if not events:
        raise ValueError("equivalence fitting requires non-empty audit events")
    if not fingerprint:
        raise ValueError("equivalence fitting requires an execution fingerprint")
    if not math.isfinite(safety_inflation) or safety_inflation < 1.0:
        raise ValueError("safety inflation must be finite and at least one")
    if minimum_bin_count < 1:
        raise ValueError("minimum bin count must be positive")
    if not 0.0 <= minimum_acceptance_lcb <= 1.0:
        raise ValueError("minimum acceptance lower bound must be in [0, 1]")

    normalized: list[dict[str, Any]] = []
    numeric_fields = (
        "max_logit_delta",
        "max_probability_delta",
        "batched_latency_ms",
        "canonical_latency_ms",
    )
    for event in events:
        row = dict(event)
        batch_size = int(row.get("batch_size", 0))
        depth = int(row.get("depth", -1))
        if batch_size not in {2, 3} or depth != batch_size - 1:
            raise ValueError("audit events require matching depth-one/two batch shapes")
        values = [float(row.get(field, math.nan)) for field in numeric_fields]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("audit event numeric values must be finite")
        if any(value < 0.0 for value in values):
            raise ValueError("audit event numeric values must be non-negative")
        if values[3] <= 0.0:
            raise ValueError("canonical audit latency must be positive")
        key = str(row.get("activation_key", ""))
        if not key:
            raise ValueError("audit event activation key is required")
        row.update(zip(numeric_fields, values, strict=True))
        row.update({"batch_size": batch_size, "depth": depth, "activation_key": key})
        normalized.append(row)

    envelopes: dict[int, EquivalenceEnvelope] = {}
    for batch_size in sorted({int(row["batch_size"]) for row in normalized}):
        selected = [row for row in normalized if int(row["batch_size"]) == batch_size]
        envelopes[batch_size] = EquivalenceEnvelope(
            logit_epsilon=max(float(row["max_logit_delta"]) for row in selected) * safety_inflation,
            probability_epsilon=max(float(row["max_probability_delta"]) for row in selected)
            * safety_inflation,
            safety_inflation=safety_inflation,
        )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[(str(row["activation_key"]), int(row["depth"]))].append(row)
    rules: list[CostRule] = []
    for (key, depth), rows in sorted(grouped.items()):
        count = len(rows)
        successes = sum(bool(row.get("full_acceptance", False)) for row in rows)
        acceptance_lcb = _binomial_lower(successes, count)
        latency_reductions = [
            1.0
            - float(row["batched_latency_ms"]) / (float(row["canonical_latency_ms"]) * (depth + 1))
            for row in rows
        ]
        latency_lcb = _lower_empirical(latency_reductions)
        enabled = (
            count >= minimum_bin_count
            and acceptance_lcb >= minimum_acceptance_lcb
            and latency_lcb > 0.0
        )
        rules.append(
            CostRule(
                activation_key=key,
                depth=depth,
                count=count,
                full_acceptance_rate=successes / count,
                acceptance_lower_bound=acceptance_lcb,
                latency_reduction_lower_bound=latency_lcb,
                enabled=enabled,
            )
        )

    core = {
        "schema_version": 1,
        "fingerprint": dict(fingerprint),
        "envelopes": {
            str(batch): asdict(envelope) for batch, envelope in sorted(envelopes.items())
        },
        "cost_rules": [asdict(rule) for rule in rules],
        "safety_inflation": safety_inflation,
        "minimum_bin_count": minimum_bin_count,
        "minimum_acceptance_lcb": minimum_acceptance_lcb,
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**core, "artifact_hash": hashlib.sha256(encoded).hexdigest()}


class EquivalenceCostPolicy:
    """Select shallow batches; numerical guards retain all token authority."""

    def __init__(
        self,
        *,
        mode: str,
        threshold: float,
        fixed_depth: int = 0,
        artifact: EquivalenceCostArtifact | None = None,
    ) -> None:
        if mode not in {"audit", "production"}:
            raise ValueError("equivalence policy mode must be audit or production")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("equivalence policy threshold must be in [0, 1]")
        if fixed_depth not in {0, 1, 2}:
            raise ValueError("equivalence audit depth must be zero, one, or two")
        if mode == "audit" and fixed_depth not in {1, 2}:
            raise ValueError("equivalence audit requires depth one or two")
        if mode == "production" and artifact is None:
            raise ValueError("production equivalence policy requires a fitted artifact")
        self.mode = mode
        self.threshold = float(threshold)
        self.fixed_depth = int(fixed_depth)
        self.artifact = artifact
        self._rules = {
            (rule.activation_key, rule.depth): rule
            for rule in (() if artifact is None else artifact.cost_rules)
        }
        self.reset_prompt()

    @classmethod
    def audit(cls, *, depth: int, threshold: float) -> EquivalenceCostPolicy:
        return cls(mode="audit", fixed_depth=depth, threshold=threshold)

    @classmethod
    def production(
        cls,
        artifact: EquivalenceCostArtifact,
        *,
        threshold: float,
    ) -> EquivalenceCostPolicy:
        return cls(mode="production", artifact=artifact, threshold=threshold)

    @property
    def audit_reference(self) -> bool:
        return self.mode == "audit"

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

    def activation_key(
        self,
        observation: StepObservation,
        last_transfer_count: int,
    ) -> str:
        masked = [index for index, is_masked in enumerate(observation.masked) if bool(is_masked)]
        remaining_fraction = len(masked) / observation.block_size
        remaining_bucket = min(4, max(1, math.ceil(4 * remaining_fraction)))
        transfer_bucket = 1 if last_transfer_count == 1 else 2 if last_transfer_count <= 3 else 3
        probability_margin = min(
            (observation.top1_probs[index] - observation.top2_probs[index] for index in masked),
            default=1.0,
        )
        threshold_margin = min(
            (abs(observation.top1_probs[index] - self.threshold) for index in masked),
            default=1.0,
        )
        margin_bucket = 1 if probability_margin < 0.05 else 2 if probability_margin < 0.2 else 3
        threshold_bucket = 1 if threshold_margin < 0.01 else 2 if threshold_margin < 0.05 else 3
        block_bucket = (
            1 if observation.block_size <= 16 else 2 if observation.block_size <= 32 else 3
        )
        return (
            f"r{remaining_bucket}|t{transfer_bucket}|m{margin_bucket}|"
            f"q{threshold_bucket}|b{block_bucket}"
        )

    def choose(
        self,
        observation: StepObservation,
        *,
        last_transfer_count: int,
    ) -> SpeculationPlan:
        if last_transfer_count < 1:
            raise ValueError("last transfer count must be positive")
        key = self.activation_key(observation, last_transfer_count)
        if self.audit_reference:
            depth = self.fixed_depth
            reason = f"audit_depth_{depth}"
        else:
            depth = 0
            reason = "uncertified_cost_bin"
            for candidate_depth in (1, 2):
                rule = self._rules.get((key, candidate_depth))
                if rule is not None and rule.enabled:
                    depth = candidate_depth
                    reason = f"certified_depth_{depth}"
                    break
        self._previous = observation
        return SpeculationPlan(
            risk_score=0.0,
            depth=depth,
            draft_width=max(1, int(last_transfer_count)),
            reason=reason,
        )

    def envelope_for(self, *, batch_size: int) -> EquivalenceEnvelope:
        if self.artifact is None or batch_size not in self.artifact.envelopes:
            raise ValueError(f"missing calibrated envelope for batch size {batch_size}")
        return self.artifact.envelopes[batch_size]

    def guard(
        self,
        state: torch.Tensor,
        logits: torch.Tensor,
        *,
        batch_size: int,
        mask_token_id: int,
    ) -> GuardDecision:
        margins = decision_margins(
            logits,
            state,
            mask_token_id=mask_token_id,
            threshold=self.threshold,
        )
        if self.audit_reference:
            return GuardDecision(False, margins, "audit_requires_reference")
        return guard_transition(margins, self.envelope_for(batch_size=batch_size))


def serialize_guarded_result(result: GuardedSpeculationResult) -> dict[str, Any]:
    return {
        "accepted_draft_edges": result.accepted_draft_edges,
        "reference_equivalent_transitions": result.reference_equivalent_transitions,
        "verified_transitions": result.reference_equivalent_transitions,
        "evaluated_nodes": result.evaluated_nodes,
        "canonical_fallback_rows": result.canonical_fallback_rows,
        "evaluated_rows": result.evaluated_rows,
        "serial_forward_calls": result.serial_forward_calls,
        "guard_passed": result.guard_passed,
        "reference_checked": result.reference_checked,
        "successor_equal_when_checked": result.successor_equal_when_checked,
        "reason": result.reason,
        "margins": [asdict(margin) for margin in result.margins],
        "nfe_saved": result.nfe_saved,
    }

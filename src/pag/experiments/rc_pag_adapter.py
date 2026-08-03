from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from pag.experiments.rc_pag_features import StepObservation


class OnlineRiskPolicy(Protocol):
    def observe(self, observation: StepObservation) -> Any: ...


@dataclass(frozen=True, slots=True)
class ShadowRequest:
    tokens: torch.Tensor
    proposed_tokens: torch.Tensor
    block_start: int
    block_end: int
    step_index: int
    cache: Any | None


@dataclass(frozen=True, slots=True)
class PolicyStep:
    observation: StepObservation
    decision: Any
    shadow_loss: int | None
    proposed_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ShadowContinuation:
    tokens: torch.Tensor
    additional_nfe: int


ShadowCallback = Callable[[ShadowRequest], torch.Tensor | Sequence[int]]


def clone_tensor_tree(value: Any) -> Any:
    """Clone tensor leaves while preserving common cache container types."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tensor_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {key: clone_tensor_tree(item) for key, item in value.items()}
    return value


@torch.no_grad()
def observation_from_tensors(
    *,
    logits: torch.Tensor,
    previous_logits: torch.Tensor | None = None,
    current_tokens: torch.Tensor,
    mask_token_id: int,
    step_index: int,
    digit_ids: torch.Tensor | Sequence[int] | None = None,
    delimiter_ids: torch.Tensor | Sequence[int] | None = None,
) -> StepObservation:
    """Reduce one active-block logit tensor to the shared compact CPU schema."""
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape [1, block_size, vocab>=2]")
    if current_tokens.ndim != 2 or current_tokens.shape != logits.shape[:2]:
        raise ValueError("current_tokens must have shape [1, block_size]")
    if step_index < 1:
        raise ValueError("step_index must be positive")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    top_values, top_ids = torch.topk(probabilities, k=2, dim=-1)
    entropies = -(probabilities * log_probs).sum(dim=-1)
    if previous_logits is None:
        temporal_js = torch.zeros_like(entropies)
    else:
        if previous_logits.shape != logits.shape:
            raise ValueError("previous logits must match current logits")
        temporal_js = torch.zeros_like(entropies)
        active_mask = current_tokens == int(mask_token_id)
        if bool(active_mask.any()):
            current_log_probs = log_probs[active_mask]
            current_probabilities = probabilities[active_mask]
            previous_log_probs = torch.log_softmax(previous_logits[active_mask].float(), dim=-1)
            previous_probabilities = previous_log_probs.exp()
            mixture_log_probs = torch.logaddexp(current_log_probs, previous_log_probs) - math.log(
                2.0
            )
            active_js = 0.5 * (
                (current_probabilities * (current_log_probs - mixture_log_probs)).sum(dim=-1)
                + (previous_probabilities * (previous_log_probs - mixture_log_probs)).sum(dim=-1)
            )
            temporal_js[active_mask] = (active_js / math.log(2.0)).clamp(0.0, 1.0)

    def frozen_ids(values: torch.Tensor | Sequence[int] | None) -> frozenset[int]:
        if values is None:
            return frozenset()
        if isinstance(values, torch.Tensor):
            return frozenset(int(value) for value in values.detach().cpu().reshape(-1).tolist())
        return frozenset(int(value) for value in values)

    observation = StepObservation.from_arrays(
        step_index=step_index,
        block_size=int(logits.shape[1]),
        masked=(current_tokens == int(mask_token_id)).detach().cpu().reshape(-1).tolist(),
        top1_probs=top_values[..., 0].detach().cpu().reshape(-1).tolist(),
        top2_probs=top_values[..., 1].detach().cpu().reshape(-1).tolist(),
        entropies=entropies.detach().cpu().reshape(-1).tolist(),
        token_ids=top_ids[..., 0].detach().cpu().reshape(-1).tolist(),
        temporal_js=temporal_js.detach().cpu().reshape(-1).tolist(),
        digit_ids=frozen_ids(digit_ids),
        delimiter_ids=frozen_ids(delimiter_ids),
    )
    if any(not math.isfinite(value) for value in observation.entropies):
        raise ValueError("adapter produced non-finite entropy")
    return observation


def shadow_disagreement(
    proposed_tokens: torch.Tensor,
    shadow_tokens: torch.Tensor | Sequence[int],
) -> int:
    proposed = proposed_tokens.detach().cpu().reshape(-1).to(torch.long)
    shadow = torch.as_tensor(shadow_tokens).detach().cpu().reshape(-1).to(torch.long)
    if proposed.shape != shadow.shape:
        raise ValueError("shadow output shape differs from the proposed active block")
    return int(torch.any(proposed != shadow).item())


@torch.no_grad()
def continue_shadow_refinement(
    request: ShadowRequest,
    *,
    forward: Callable[[torch.Tensor, Any | None, int, int], tuple[torch.Tensor, Any | None]],
    mask_token_id: int,
    threshold: float,
    max_steps: int,
) -> ShadowContinuation:
    """Continue a cloned active block without changing the policy trajectory."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("shadow threshold must be in [0, 1]")
    if max_steps < request.step_index:
        raise ValueError("shadow max_steps precedes the proposed stopping step")
    tokens = request.tokens.detach().clone()
    cache = clone_tensor_tree(request.cache)
    additional_nfe = 0
    step = int(request.step_index)
    while (
        bool((tokens[:, request.block_start : request.block_end] == int(mask_token_id)).any())
        and step < max_steps
    ):
        local_logits, cache = forward(
            tokens,
            cache,
            request.block_start,
            request.block_end,
        )
        active = tokens[:, request.block_start : request.block_end]
        if local_logits.shape[:2] != active.shape:
            raise ValueError("shadow forward returned logits for a different active block")
        probabilities = torch.softmax(local_logits.float(), dim=-1)
        confidences, predictions = probabilities.max(dim=-1)
        masked = active == int(mask_token_id)
        step += 1
        additional_nfe += 1
        transfer = masked & (confidences >= threshold)
        if bool(masked.any()) and not bool(transfer.any()):
            ranked = torch.where(masked, confidences, torch.full_like(confidences, -torch.inf))
            transfer.scatter_(1, ranked.argmax(dim=1, keepdim=True), True)
        if step >= max_steps:
            transfer = masked
        active[transfer] = predictions[transfer]
    if bool((tokens[:, request.block_start : request.block_end] == int(mask_token_id)).any()):
        raise RuntimeError("shadow continuation reached the ceiling with unresolved masks")
    return ShadowContinuation(
        tokens=tokens[:, request.block_start : request.block_end].detach().clone(),
        additional_nfe=additional_nfe,
    )


@torch.no_grad()
def observe_policy_step(
    policy: OnlineRiskPolicy | None,
    *,
    logits: torch.Tensor,
    previous_logits: torch.Tensor | None = None,
    current_tokens: torch.Tensor,
    mask_token_id: int,
    step_index: int,
    full_tokens: torch.Tensor,
    block_start: int,
    block_end: int,
    digit_ids: torch.Tensor | Sequence[int] | None = None,
    delimiter_ids: torch.Tensor | Sequence[int] | None = None,
    cache: Any | None = None,
    shadow_callback: ShadowCallback | None = None,
    shadow_all_steps: bool = False,
) -> PolicyStep | None:
    if policy is None:
        return None
    observation = observation_from_tensors(
        logits=logits,
        previous_logits=previous_logits,
        current_tokens=current_tokens,
        mask_token_id=mask_token_id,
        step_index=step_index,
        digit_ids=digit_ids,
        delimiter_ids=delimiter_ids,
    )
    decision = policy.observe(observation)
    predicted = torch.tensor(
        observation.token_ids,
        dtype=torch.long,
        device=full_tokens.device,
    ).reshape(1, -1)
    proposed = torch.where(
        current_tokens.to(full_tokens.device) == int(mask_token_id),
        predicted,
        current_tokens.to(full_tokens.device),
    )
    loss = None
    if shadow_callback is not None and (bool(decision.should_stop) or shadow_all_steps):
        request = ShadowRequest(
            tokens=full_tokens.detach().clone(),
            proposed_tokens=proposed.detach().clone(),
            block_start=int(block_start),
            block_end=int(block_end),
            step_index=int(step_index),
            cache=clone_tensor_tree(cache),
        )
        shadow_tokens = shadow_callback(request)
        loss = shadow_disagreement(request.proposed_tokens, shadow_tokens)
    return PolicyStep(
        observation=observation,
        decision=decision,
        shadow_loss=loss,
        proposed_tokens=tuple(int(value) for value in proposed.detach().cpu().reshape(-1)),
    )


def serialize_policy_step(step: PolicyStep) -> dict[str, Any]:
    return {
        "step_index": step.observation.step_index,
        "block_size": step.observation.block_size,
        "remaining_masks": sum(step.observation.masked),
        "risk_score": float(step.decision.risk_score),
        "safe_streak": int(step.decision.safe_streak),
        "should_stop": bool(step.decision.should_stop),
        "reason": str(step.decision.reason),
        "predicted_nfe_savings": float(getattr(step.decision, "predicted_nfe_savings", 0.0)),
        "temporal_js": float(getattr(step.decision, "temporal_js", 0.0)),
        "risk_spent": float(getattr(step.decision, "risk_spent", 0.0)),
        "prompt_stops": int(getattr(step.decision, "prompt_stops", 0)),
        "shadow_loss": step.shadow_loss,
        "proposed_tokens": list(step.proposed_tokens),
        "observation": {
            "step_index": step.observation.step_index,
            "block_size": step.observation.block_size,
            "masked": list(step.observation.masked),
            "top1_probs": list(step.observation.top1_probs),
            "top2_probs": list(step.observation.top2_probs),
            "entropies": list(step.observation.entropies),
            "token_ids": list(step.observation.token_ids),
            "temporal_js": list(step.observation.temporal_js),
            "digit_ids": sorted(step.observation.digit_ids),
            "delimiter_ids": sorted(step.observation.delimiter_ids),
        },
    }

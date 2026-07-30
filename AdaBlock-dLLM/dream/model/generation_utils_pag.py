from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from transformers.utils import ModelOutput

from model.generation_utils_adablock import (
    DreamGenerationConfig,
    sample_tokens,
)
from model.generation_utils_adablock import (
    DreamGenerationMixin as AdaBlockDreamGenerationMixin,
)
from pag.experiments.rc_pag_adapter import (
    continue_shadow_refinement,
    observation_from_tensors,
    observe_policy_step,
    serialize_policy_step,
)
from pag.experiments.rc_pag_features import RealizedBlock


def _rc_pag_observation(
    logits: torch.Tensor,
    current_tokens: torch.Tensor,
    *,
    mask_token_id: int,
    step_index: int,
    digit_ids: torch.Tensor | None = None,
    delimiter_ids: torch.Tensor | None = None,
):
    return observation_from_tensors(
        logits=logits,
        current_tokens=current_tokens,
        mask_token_id=mask_token_id,
        step_index=step_index,
        digit_ids=digit_ids,
        delimiter_ids=delimiter_ids,
    )


def _observe_rc_pag_step(
    model,
    *,
    local_logits: torch.Tensor,
    local_tokens: torch.Tensor,
    mask_token_id: int,
    step_index: int,
    full_tokens: torch.Tensor,
    block_start: int,
    block_end: int,
    cache,
    records: list[dict[str, object]],
) -> bool:
    policy = getattr(model, "pag_risk_policy", None)
    step = observe_policy_step(
        policy,
        logits=local_logits,
        current_tokens=local_tokens,
        mask_token_id=mask_token_id,
        step_index=step_index,
        full_tokens=full_tokens,
        block_start=block_start,
        block_end=block_end,
        digit_ids=getattr(model, "pag_digit_ids", None),
        delimiter_ids=getattr(model, "pag_delimiter_ids", None),
        cache=cache,
        shadow_callback=getattr(model, "pag_shadow_callback", None),
        shadow_all_steps=bool(getattr(model, "pag_shadow_all_steps", False)),
    )
    if step is None:
        return False
    records.append(serialize_policy_step(step))
    return bool(step.decision.should_stop)


def _record_rc_pag_realized(
    model,
    *,
    block_size: int,
    nfe: int,
    mean_confidence: float,
    min_confidence: float,
    digit_fraction: float,
    delimiter_fraction: float,
) -> None:
    policy = getattr(model, "pag_risk_policy", None)
    if policy is None:
        return
    policy.record_realized(
        RealizedBlock(
            block_size=int(block_size),
            nfe=int(nfe),
            mean_confidence=float(mean_confidence),
            min_confidence=float(min_confidence),
            digit_fraction=float(digit_fraction),
            delimiter_fraction=float(delimiter_fraction),
        )
    )


def make_dream_shadow_callback(
    model,
    *,
    mode: str,
    attention_mask,
    tok_idx,
    mask_token_id: int,
    threshold: float,
    max_steps: int,
):
    """Create an on-policy shadow continuation for uncached or dual-cache Dream."""
    if mode not in {"uncached", "dual_cache"}:
        raise ValueError("Dream shadow mode must be uncached or dual_cache")

    def callback(request):
        def forward(tokens, cache, block_start, block_end):
            if mode == "uncached":
                output = model(
                    tokens,
                    attention_mask if attention_mask != "full" else attention_mask,
                    tok_idx if tok_idx is not None else None,
                )
                logits = torch.cat([output.logits[:, :1], output.logits[:, :-1]], dim=1)
                return logits[:, block_start:block_end, :], None
            replace_position = torch.zeros_like(tokens, dtype=torch.bool)
            replace_position[:, block_start:block_end] = True
            current_attention = (
                attention_mask[:, :, :, block_start:]
                if attention_mask != "full"
                else attention_mask
            )
            output = model(
                tokens[:, block_start:block_end],
                current_attention,
                tok_idx[:, block_start:block_end] if tok_idx is not None else None,
                past_key_values=cache,
                use_cache=True,
                dual_cache=True,
                replace_position=replace_position,
            )
            logits = torch.cat([output.logits[:, :1], output.logits[:, :-1]], dim=1)
            return logits, cache

        return continue_shadow_refinement(
            request,
            forward=forward,
            mask_token_id=mask_token_id,
            threshold=threshold,
            max_steps=max_steps,
        ).tokens

    return callback


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: tuple[torch.FloatTensor] | None = None
    nfe_history: list | None = None
    block_history: list | None = None
    schedule_history: list | None = None


def _apply_confidence_threshold_sample(
    *,
    target_tokens: torch.Tensor,
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    mask_token_id: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    threshold: float,
    force_all: bool,
) -> None:
    if target_tokens.shape[0] != 1:
        msg = "PAG decoding currently supports batch size 1 only"
        raise AssertionError(msg)

    if mask_index.sum().item() == 0:
        return

    mask_logits = logits[mask_index]
    confidence, x0 = sample_tokens(
        mask_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    candidate_tokens = torch.full_like(target_tokens, mask_token_id)
    full_confidence = torch.full_like(
        target_tokens,
        -torch.inf,
        device=target_tokens.device,
        dtype=logits.dtype,
    )
    candidate_tokens[mask_index] = x0.clone()
    full_confidence[mask_index] = confidence

    if force_all:
        transfer_index = mask_index
    else:
        current_transfer_tokens = int(mask_index.sum().item())
        selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
        transfer_index = torch.zeros_like(
            target_tokens,
            device=target_tokens.device,
            dtype=torch.bool,
        )
        transfer_index[0, select_index[0]] = True
        for k in range(1, current_transfer_tokens):
            if selected_confidence[0, k] < threshold:
                transfer_index[0, select_index[0, k]] = False

    target_tokens[transfer_index] = candidate_tokens[transfer_index]


def _token_fraction(tokens: torch.Tensor, token_ids: torch.Tensor | None) -> float:
    if token_ids is None or token_ids.numel() == 0 or tokens.numel() == 0:
        return 0.0
    return float(torch.isin(tokens, token_ids.to(tokens.device)).float().mean().item())


def _realized_features(
    model,
    tokens: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[float, float, float, float]:
    top1_confidence = torch.softmax(logits.float(), dim=-1).amax(dim=-1)
    return (
        float(top1_confidence.mean().item()),
        float(top1_confidence.min().item()),
        _token_fraction(tokens, getattr(model, "pag_digit_ids", None)),
        _token_fraction(tokens, getattr(model, "pag_delimiter_ids", None)),
    )


class DreamGenerationMixin(AdaBlockDreamGenerationMixin):
    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: DreamGenerationConfig | None = None,
        **kwargs,
    ) -> DreamModelOutput | torch.LongTensor:
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None and generation_config.max_length is not None
        )
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        threshold = kwargs.get("threshold", 0.9)
        block_length = kwargs.get("block_length", 8)
        dual_cache = kwargs.get("dual_cache", False)
        max_block_length = kwargs.get("max_block_length", block_length)
        max_refinement_steps = kwargs.get("max_refinement_steps", generation_config.steps)
        delimiter_threshold = kwargs.get("delimiter_threshold", 0.3)

        return self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            threshold=threshold,
            block_length=block_length,
            dual_cache=dual_cache,
            max_block_length=max_block_length,
            max_refinement_steps=max_refinement_steps,
            delimiter_threshold=delimiter_threshold,
        )

    def _sample_pag(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None,
        generation_config: DreamGenerationConfig,
        threshold: float | None = 0.9,
        block_length: int | None = 32,
        dual_cache: bool = False,
        max_block_length: int | None = None,
        max_refinement_steps: int | None = None,
        delimiter_threshold: float = 0.3,
    ) -> DreamModelOutput | torch.LongTensor:
        del block_length, dual_cache

        if not hasattr(self, "pag_scheduler") or self.pag_scheduler is None:
            msg = "PAG decoding requires a pag_scheduler attached to the model"
            raise ValueError(msg)

        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg

        if alg != "confidence_threshold":
            raise NotImplementedError(alg)

        histories = [] if (return_dict_in_generate and output_history) else None

        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        max_block_length = gen_length if max_block_length is None else int(max_block_length)
        max_refinement_steps = (
            int(generation_config.steps)
            if max_refinement_steps is None
            else int(max_refinement_steps)
        )

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(
                attention_mask,
                (0, max_length - attention_mask.shape[1]),
                value=1.0,
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        prompt_length = input_ids.shape[1]
        generated_length = 0
        nfe_history: list[int] = []
        block_history: list[int] = []
        schedule_history: list[dict[str, object]] = []

        self.pag_scheduler.reset()
        if getattr(self, "pag_risk_policy", None) is not None:
            self.pag_risk_policy.reset_prompt()
            if getattr(self, "pag_shadow_callback", None) == "auto":
                self.pag_shadow_callback = make_dream_shadow_callback(
                    self,
                    mode="uncached",
                    attention_mask=attention_mask,
                    tok_idx=tok_idx,
                    mask_token_id=mask_token_id,
                    threshold=float(threshold),
                    max_steps=max_refinement_steps,
                )

        while generated_length < gen_length:
            remaining_tokens = gen_length - generated_length
            block_start = prompt_length + generated_length
            model_output = self(
                x,
                attention_mask if attention_mask != "full" else attention_mask,
                tok_idx if tok_idx is not None else None,
            )
            logits = model_output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            nfe = 1
            proposed_block_size = self._compute_block_length(
                logits,
                prompt_length,
                gen_length,
                generated_length,
                max_block_length,
                delimiter_threshold=delimiter_threshold,
            )
            schedule = self.pag_scheduler.next_schedule(
                block_size=proposed_block_size,
                remaining_tokens=remaining_tokens,
                max_block_length=max_block_length,
                max_refinement_steps=max_refinement_steps,
            )
            block_end = block_start + schedule.applied_block_size
            generated_length += schedule.applied_block_size
            risk_steps: list[dict[str, object]] = []
            if getattr(self, "pag_risk_policy", None) is not None:
                self.pag_risk_policy.start_block()

            local_logits = logits[:, block_start:block_end, :]
            mask_index = x[:, block_start:block_end] == mask_token_id
            risk_force_commit = _observe_rc_pag_step(
                self,
                local_logits=local_logits,
                local_tokens=x[:, block_start:block_end],
                mask_token_id=mask_token_id,
                step_index=nfe,
                full_tokens=x,
                block_start=block_start,
                block_end=block_end,
                cache=None,
                records=risk_steps,
            )
            _apply_confidence_threshold_sample(
                target_tokens=x[:, block_start:block_end],
                logits=local_logits,
                mask_index=mask_index,
                mask_token_id=mask_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                threshold=float(threshold),
                force_all=risk_force_commit or nfe >= schedule.budgeted_refinement_steps,
            )
            while True:
                if (x[:, block_start:block_end] == mask_token_id).sum() == 0:
                    break

                model_output = self(
                    x,
                    attention_mask if attention_mask != "full" else attention_mask,
                    tok_idx if tok_idx is not None else None,
                )
                logits = model_output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                nfe += 1

                local_logits = logits[:, block_start:block_end, :]
                mask_index = x[:, block_start:block_end] == mask_token_id
                risk_force_commit = _observe_rc_pag_step(
                    self,
                    local_logits=local_logits,
                    local_tokens=x[:, block_start:block_end],
                    mask_token_id=mask_token_id,
                    step_index=nfe,
                    full_tokens=x,
                    block_start=block_start,
                    block_end=block_end,
                    cache=None,
                    records=risk_steps,
                )
                _apply_confidence_threshold_sample(
                    target_tokens=x[:, block_start:block_end],
                    logits=local_logits,
                    mask_index=mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    threshold=float(threshold),
                    force_all=risk_force_commit or nfe >= schedule.budgeted_refinement_steps,
                )

                if nfe >= schedule.budgeted_refinement_steps:
                    break

            mean_conf, min_conf, digit_frac, delim_frac = _realized_features(
                self,
                x[:, block_start:block_end],
                local_logits,
            )
            self.pag_scheduler.record_realized(
                schedule.applied_block_size,
                nfe,
                mean_conf,
                min_conf,
                digit_frac,
                delim_frac,
            )
            _record_rc_pag_realized(
                self,
                block_size=schedule.applied_block_size,
                nfe=nfe,
                mean_confidence=mean_conf,
                min_confidence=min_conf,
                digit_fraction=digit_frac,
                delimiter_fraction=delim_frac,
            )
            nfe_history.append(nfe)
            block_history.append(schedule.applied_block_size)
            schedule_history.append(
                {
                    "block_index": len(schedule_history),
                    "predicted_tuple": {
                        "block_size": int(schedule.predicted_tuple.block_size),
                        "refinement_steps": int(schedule.predicted_tuple.refinement_steps),
                    },
                    "applied_block_size": int(schedule.applied_block_size),
                    "budgeted_refinement_steps": int(schedule.budgeted_refinement_steps),
                    "actual_nfe_used": int(nfe),
                    "mean_top1_confidence": mean_conf,
                    "min_top1_confidence": min_conf,
                    "digit_fraction": digit_frac,
                    "delimiter_fraction": delim_frac,
                    "block_start": int(block_start),
                    "block_end": int(block_end),
                }
            )
            if risk_steps:
                schedule_history[-1]["risk_steps"] = risk_steps
                schedule_history[-1]["final_tokens"] = (
                    x[:, block_start:block_end].detach().cpu().reshape(-1).tolist()
                )
                schedule_history[-1]["shadow_losses"] = [
                    row["shadow_loss"]
                    for row in risk_steps
                    if row.get("shadow_loss") is not None
                ]

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                nfe_history=nfe_history,
                block_history=block_history,
                schedule_history=schedule_history,
            )
        return x

    def _sample_pag_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None,
        generation_config: DreamGenerationConfig,
        threshold: float | None = 0.9,
        block_length: int | None = 32,
        dual_cache: bool = False,
        max_block_length: int | None = None,
        max_refinement_steps: int | None = None,
        delimiter_threshold: float = 0.3,
    ) -> DreamModelOutput | torch.LongTensor:
        del block_length

        if not dual_cache:
            msg = "Cached PAG decoding requires dual_cache=True"
            raise AssertionError(msg)
        if not hasattr(self, "pag_scheduler") or self.pag_scheduler is None:
            msg = "PAG decoding requires a pag_scheduler attached to the model"
            raise ValueError(msg)

        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg

        if alg != "confidence_threshold":
            raise NotImplementedError(alg)

        histories = [] if (return_dict_in_generate and output_history) else None

        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        max_block_length = gen_length if max_block_length is None else int(max_block_length)
        max_refinement_steps = (
            int(generation_config.steps)
            if max_refinement_steps is None
            else int(max_refinement_steps)
        )

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(
                attention_mask,
                (0, max_length - attention_mask.shape[1]),
                value=1.0,
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        prompt_length = input_ids.shape[1]
        generated_length = 0
        nfe_history: list[int] = []
        block_history: list[int] = []
        schedule_history: list[dict[str, object]] = []

        self.pag_scheduler.reset()
        if getattr(self, "pag_risk_policy", None) is not None:
            self.pag_risk_policy.reset_prompt()
            if getattr(self, "pag_shadow_callback", None) == "auto":
                self.pag_shadow_callback = make_dream_shadow_callback(
                    self,
                    mode="dual_cache",
                    attention_mask=attention_mask,
                    tok_idx=tok_idx,
                    mask_token_id=mask_token_id,
                    threshold=float(threshold),
                    max_steps=max_refinement_steps,
                )

        while generated_length < gen_length:
            remaining_tokens = gen_length - generated_length
            block_start = prompt_length + generated_length
            model_output = self(
                x,
                attention_mask if attention_mask != "full" else attention_mask,
                tok_idx if tok_idx is not None else None,
                use_cache=True,
            )
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            nfe = 1
            proposed_block_size = self._compute_block_length(
                logits,
                prompt_length,
                gen_length,
                generated_length,
                max_block_length,
                delimiter_threshold=delimiter_threshold,
            )
            schedule = self.pag_scheduler.next_schedule(
                block_size=proposed_block_size,
                remaining_tokens=remaining_tokens,
                max_block_length=max_block_length,
                max_refinement_steps=max_refinement_steps,
            )
            block_end = block_start + schedule.applied_block_size
            generated_length += schedule.applied_block_size
            risk_steps: list[dict[str, object]] = []
            if getattr(self, "pag_risk_policy", None) is not None:
                self.pag_risk_policy.start_block()

            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, block_start:block_end] = 1
            local_logits = logits[:, block_start:block_end, :]
            mask_index = x[:, block_start:block_end] == mask_token_id
            risk_force_commit = _observe_rc_pag_step(
                self,
                local_logits=local_logits,
                local_tokens=x[:, block_start:block_end],
                mask_token_id=mask_token_id,
                step_index=nfe,
                full_tokens=x,
                block_start=block_start,
                block_end=block_end,
                cache=past_key_values,
                records=risk_steps,
            )
            _apply_confidence_threshold_sample(
                target_tokens=x[:, block_start:block_end],
                logits=local_logits,
                mask_index=mask_index,
                mask_token_id=mask_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                threshold=float(threshold),
                force_all=risk_force_commit or nfe >= schedule.budgeted_refinement_steps,
            )

            while True:
                if (x[:, block_start:block_end] == mask_token_id).sum() == 0:
                    break

                current_attention_mask = (
                    attention_mask[:, :, :, block_start:]
                    if attention_mask != "full"
                    else attention_mask
                )
                model_output = self(
                    x[:, block_start:block_end],
                    current_attention_mask,
                    tok_idx[:, block_start:block_end] if tok_idx is not None else None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    dual_cache=True,
                    replace_position=replace_position,
                )
                local_logits = model_output.logits
                local_logits = torch.cat([local_logits[:, :1], local_logits[:, :-1]], dim=1)

                nfe += 1
                mask_index = x[:, block_start:block_end] == mask_token_id
                risk_force_commit = _observe_rc_pag_step(
                    self,
                    local_logits=local_logits,
                    local_tokens=x[:, block_start:block_end],
                    mask_token_id=mask_token_id,
                    step_index=nfe,
                    full_tokens=x,
                    block_start=block_start,
                    block_end=block_end,
                    cache=past_key_values,
                    records=risk_steps,
                )
                _apply_confidence_threshold_sample(
                    target_tokens=x[:, block_start:block_end],
                    logits=local_logits,
                    mask_index=mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    threshold=float(threshold),
                    force_all=risk_force_commit or nfe >= schedule.budgeted_refinement_steps,
                )

                if nfe >= schedule.budgeted_refinement_steps:
                    break

            mean_conf, min_conf, digit_frac, delim_frac = _realized_features(
                self,
                x[:, block_start:block_end],
                local_logits,
            )
            self.pag_scheduler.record_realized(
                schedule.applied_block_size,
                nfe,
                mean_conf,
                min_conf,
                digit_frac,
                delim_frac,
            )
            _record_rc_pag_realized(
                self,
                block_size=schedule.applied_block_size,
                nfe=nfe,
                mean_confidence=mean_conf,
                min_confidence=min_conf,
                digit_fraction=digit_frac,
                delimiter_fraction=delim_frac,
            )
            nfe_history.append(nfe)
            block_history.append(schedule.applied_block_size)
            schedule_history.append(
                {
                    "block_index": len(schedule_history),
                    "predicted_tuple": {
                        "block_size": int(schedule.predicted_tuple.block_size),
                        "refinement_steps": int(schedule.predicted_tuple.refinement_steps),
                    },
                    "applied_block_size": int(schedule.applied_block_size),
                    "budgeted_refinement_steps": int(schedule.budgeted_refinement_steps),
                    "actual_nfe_used": int(nfe),
                    "mean_top1_confidence": mean_conf,
                    "min_top1_confidence": min_conf,
                    "digit_fraction": digit_frac,
                    "delimiter_fraction": delim_frac,
                    "block_start": int(block_start),
                    "block_end": int(block_end),
                }
            )
            if risk_steps:
                schedule_history[-1]["risk_steps"] = risk_steps
                schedule_history[-1]["final_tokens"] = (
                    x[:, block_start:block_end].detach().cpu().reshape(-1).tolist()
                )
                schedule_history[-1]["shadow_losses"] = [
                    row["shadow_loss"]
                    for row in risk_steps
                    if row.get("shadow_loss") is not None
                ]

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                nfe_history=nfe_history,
                block_history=block_history,
                schedule_history=schedule_history,
            )
        return x

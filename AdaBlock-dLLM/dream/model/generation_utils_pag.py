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
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
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

        return self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            threshold=threshold,
            block_length=block_length,
            dual_cache=dual_cache,
            max_block_length=max_block_length,
            max_refinement_steps=max_refinement_steps,
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

        while generated_length < gen_length:
            remaining_tokens = gen_length - generated_length
            schedule = self.pag_scheduler.next_schedule(
                remaining_tokens=remaining_tokens,
                max_block_length=max_block_length,
                max_refinement_steps=max_refinement_steps,
            )
            block_start = prompt_length + generated_length
            block_end = block_start + schedule.applied_block_size
            generated_length += schedule.applied_block_size

            nfe = 0
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
                _apply_confidence_threshold_sample(
                    target_tokens=x[:, block_start:block_end],
                    logits=local_logits,
                    mask_index=mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    threshold=float(threshold),
                    force_all=nfe >= schedule.budgeted_refinement_steps,
                )

                if nfe >= schedule.budgeted_refinement_steps:
                    break

            self.pag_scheduler.record_realized(schedule.applied_block_size, nfe)
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
                    "block_start": int(block_start),
                    "block_end": int(block_end),
                }
            )

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

        while generated_length < gen_length:
            remaining_tokens = gen_length - generated_length
            schedule = self.pag_scheduler.next_schedule(
                remaining_tokens=remaining_tokens,
                max_block_length=max_block_length,
                max_refinement_steps=max_refinement_steps,
            )
            block_start = prompt_length + generated_length
            block_end = block_start + schedule.applied_block_size
            generated_length += schedule.applied_block_size

            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, block_start:block_end] = 1
            past_key_values = None
            nfe = 0

            while True:
                if (x[:, block_start:block_end] == mask_token_id).sum() == 0:
                    break

                if nfe == 0:
                    model_output = self(
                        x,
                        attention_mask if attention_mask != "full" else attention_mask,
                        tok_idx if tok_idx is not None else None,
                        use_cache=True,
                    )
                    past_key_values = model_output.past_key_values
                    logits = model_output.logits
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                    local_logits = logits[:, block_start:block_end, :]
                else:
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
                _apply_confidence_threshold_sample(
                    target_tokens=x[:, block_start:block_end],
                    logits=local_logits,
                    mask_index=mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    threshold=float(threshold),
                    force_all=nfe >= schedule.budgeted_refinement_steps,
                )

                if nfe >= schedule.budgeted_refinement_steps:
                    break

            self.pag_scheduler.record_realized(schedule.applied_block_size, nfe)
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
                    "block_start": int(block_start),
                    "block_end": int(block_end),
                }
            )

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                nfe_history=nfe_history,
                block_history=block_history,
                schedule_history=schedule_history,
            )
        return x

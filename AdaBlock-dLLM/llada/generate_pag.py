"""PAG generation with reactive block sizing and predictive residual budget.
Block boundaries are determined reactively from logits (AdaBlock-style).
Refinement budget is predicted by the classifier.
Budget enforcement is soft: exit early if tokens are few/confident/stable.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens,
    threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = predicted_tokens
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)
    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index
    if num_transfer_tokens is None:
        msg = "num_transfer_tokens must be a tensor when threshold is None."
        raise ValueError(msg)
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)
    _, idx = torch.sort(confidence, dim=1, descending=True)
    batch_size, seq_len = confidence.shape
    cols = torch.arange(seq_len, device=confidence.device).unsqueeze(0).expand(batch_size, seq_len)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(batch_size, seq_len)
    select_sorted = cols < k_expanded
    transfer_int = torch.zeros(batch_size, seq_len, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index


def _force_commit(
    predicted_tokens: torch.Tensor,
    mask_index: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = torch.where(mask_index, predicted_tokens, x)
    return x0, mask_index


def _record_schedule(
    schedule_history: list[dict[str, object]],
    *,
    schedule,
    nfe: int,
    block_start: int,
    block_end: int,
) -> None:
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


@torch.no_grad()
def generate_pag(
    model,
    prompt: torch.Tensor,
    scheduler,
    *,
    steps: int = 128,
    gen_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float | None = None,
    max_block_length: int | None = None,
    max_refinement_steps: int | None = None,
    digit_ids_tensor: torch.Tensor | None = None,
    delimiter_ids_tensor: torch.Tensor | None = None,
    delimiter_ids: list[int] | None = None,
    delimiter_threshold: float | None = None,
    tau_commit: float = 0.80,
    tau_stable_steps: int = 2,
    default_block_length: int = 32,
):
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    assert threshold is not None
    from generate_adablock import compute_block_length

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    scheduler.reset()
    max_block_length = gen_length if max_block_length is None else int(max_block_length)
    max_refinement_steps = steps if max_refinement_steps is None else int(max_refinement_steps)
    delimiter_ids = delimiter_ids or [198]

    generated_length = 0
    nfe_history: list[int] = []
    block_history: list[int] = []
    schedule_history: list[dict[str, object]] = []

    while generated_length < gen_length:
        output = model(x)
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)

        block_length = compute_block_length(
            logits, predicted_tokens, prompt, gen_length,
            generated_length, default_block_length=default_block_length,
            delimiter_ids=delimiter_ids,
            delimiter_threshold=delimiter_threshold or float("inf"),
        )
        block_length = min(int(block_length), max_block_length, gen_length - generated_length)

        schedule = scheduler.next_schedule(
            block_size=block_length,
            remaining_tokens=gen_length - generated_length,
            max_block_length=max_block_length,
            max_refinement_steps=max_refinement_steps,
        )
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        unmask_confs: list[torch.Tensor] = []
        mask_index = x == mask_id
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(
            logits, predicted_tokens, remasking, mask_index, x, None, threshold,
        )
        initial_probs = F.softmax(logits[:, block_start:block_end, :], dim=-1)
        initial_max_probs = initial_probs.max(dim=-1).values
        newly_unmasked = transfer_index[:, block_start:block_end]
        if newly_unmasked.any():
            unmask_confs.append(initial_max_probs[newly_unmasked])
        x[transfer_index] = x0[transfer_index]

        nfe = 0
        prev_predictions: list[torch.Tensor] = []

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            output = model(x)
            logits = output.logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
            nfe += 1
            mask_index = x == mask_id
            mask_index[:, block_end:] = 0
            block_probs = F.softmax(logits[:, block_start:block_end, :], dim=-1)
            block_max_probs = block_probs.max(dim=-1).values

            if nfe >= schedule.budgeted_refinement_steps:
                block_mask = x[:, block_start:block_end] == mask_id
                remaining_count = block_mask.sum().item()
                few_remaining = remaining_count <= max(1, math.ceil(0.10 * block_length))
                confident = False
                if remaining_count > 0:
                    remaining_logits = logits[:, block_start:block_end][block_mask]
                    remaining_confs = torch.softmax(remaining_logits, dim=-1).max(dim=-1).values
                    confident = remaining_confs.min().item() >= tau_commit
                else:
                    confident = True
                stable = False
                if remaining_count > 0 and len(prev_predictions) >= tau_stable_steps:
                    current_for_remaining = predicted_tokens[:, block_start:block_end][block_mask]
                    stable = all(
                        torch.all(current_for_remaining == p[block_mask])
                        for p in prev_predictions[-tau_stable_steps:]
                    )
                elif remaining_count == 0:
                    stable = True
                if few_remaining or confident or stable:
                    x0, transfer_index = _force_commit(predicted_tokens, mask_index, x)
                    x[transfer_index] = x0[transfer_index]
                    break
                if nfe >= max_refinement_steps:
                    x0, transfer_index = _force_commit(predicted_tokens, mask_index, x)
                    x[transfer_index] = x0[transfer_index]
                    break
                x0, transfer_index = get_transfer_index(
                    logits, predicted_tokens, remasking, mask_index, x, None, threshold,
                )
            else:
                x0, transfer_index = get_transfer_index(
                    logits, predicted_tokens, remasking, mask_index, x, None, threshold,
                )

            newly_unmasked = transfer_index[:, block_start:block_end]
            if newly_unmasked.any():
                unmask_confs.append(block_max_probs[newly_unmasked])
            x[transfer_index] = x0[transfer_index]
            prev_predictions.append(predicted_tokens[:, block_start:block_end].clone())
            prev_predictions = prev_predictions[-tau_stable_steps:]
            if nfe >= max_refinement_steps:
                break

        if unmask_confs:
            all_confs = torch.cat(unmask_confs)
            mean_conf = all_confs.mean().item()
            min_conf = all_confs.min().item()
        else:
            mean_conf = min_conf = 1.0
        block_tokens = x[0, block_start:block_end]
        digit_frac = torch.isin(block_tokens, digit_ids_tensor.to(x.device)).float().mean().item() if digit_ids_tensor is not None else 0.0
        delim_frac = torch.isin(block_tokens, delimiter_ids_tensor.to(x.device)).float().mean().item() if delimiter_ids_tensor is not None else 0.0
        scheduler.record_realized(block_length, nfe, mean_conf, min_conf, digit_frac, delim_frac)
        nfe_history.append(nfe)
        block_history.append(block_length)
        _record_schedule(schedule_history, schedule=schedule, nfe=nfe, block_start=block_start, block_end=block_end)

    return x, nfe_history, block_history, schedule_history


@torch.no_grad()
def generate_pag_prefix_cache(
    model,
    prompt: torch.Tensor,
    scheduler,
    *,
    steps: int = 128,
    gen_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float | None = None,
    max_block_length: int | None = None,
    max_refinement_steps: int | None = None,
    digit_ids_tensor: torch.Tensor | None = None,
    delimiter_ids_tensor: torch.Tensor | None = None,
    delimiter_ids: list[int] | None = None,
    delimiter_threshold: float | None = None,
    tau_commit: float = 0.80,
    tau_stable_steps: int = 2,
    default_block_length: int = 32,
):
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    assert threshold is not None
    from generate_adablock import compute_block_length

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    scheduler.reset()
    max_block_length = gen_length if max_block_length is None else int(max_block_length)
    max_refinement_steps = steps if max_refinement_steps is None else int(max_refinement_steps)
    delimiter_ids = delimiter_ids or [198]

    generated_length = 0
    nfe_history: list[int] = []
    block_history: list[int] = []
    schedule_history: list[dict[str, object]] = []

    while generated_length < gen_length:
        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)

        block_length = compute_block_length(
            logits, predicted_tokens, prompt, gen_length,
            generated_length, default_block_length=default_block_length,
            delimiter_ids=delimiter_ids,
            delimiter_threshold=delimiter_threshold or float("inf"),
        )
        block_length = min(int(block_length), max_block_length, gen_length - generated_length)

        schedule = scheduler.next_schedule(
            block_size=block_length,
            remaining_tokens=gen_length - generated_length,
            max_block_length=max_block_length,
            max_refinement_steps=max_refinement_steps,
        )
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        unmask_confs: list[torch.Tensor] = []
        mask_index = x == mask_id
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(
            logits, predicted_tokens, remasking, mask_index, x, None, threshold,
        )
        initial_probs = F.softmax(logits[:, block_start:block_end, :], dim=-1)
        initial_max_probs = initial_probs.max(dim=-1).values
        newly_unmasked = transfer_index[:, block_start:block_end]
        if newly_unmasked.any():
            unmask_confs.append(initial_max_probs[newly_unmasked])
        x[transfer_index] = x0[transfer_index]

        prefix_cache = []
        for cache_layer in full_cache:
            prefix_cache.append(())
            for cache_entry in cache_layer:
                prefix_cache[-1] += (cache_entry[:, :, :block_start],)

        nfe = 0
        prev_predictions: list[torch.Tensor] = []

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            if nfe >= max_refinement_steps:
                break

            mask_index = x[:, block_start:] == mask_id
            mask_index[:, block_length:] = 0
            block_output = model(x[:, block_start:], past_key_values=prefix_cache, use_cache=True)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1

            block_probs = F.softmax(block_logits, dim=-1)
            block_max_probs = block_probs.max(dim=-1).values

            if nfe >= schedule.budgeted_refinement_steps:
                block_mask = x[:, block_start:block_end] == mask_id
                remaining_count = block_mask.sum().item()
                few_remaining = remaining_count <= max(1, math.ceil(0.10 * block_length))
                confident = False
                if remaining_count > 0:
                    local_conf = block_max_probs[0, :block_length][block_mask[0, :block_length]]
                    confident = local_conf.min().item() >= tau_commit if local_conf.numel() > 0 else True
                else:
                    confident = True
                stable = False
                if remaining_count > 0 and len(prev_predictions) >= tau_stable_steps:
                    current_for_remaining = block_predicted_tokens[0, :block_length][block_mask[0, :block_length]]
                    stable = all(
                        torch.all(current_for_remaining == p[block_mask[0, :block_length]])
                        for p in prev_predictions[-tau_stable_steps:]
                    )
                elif remaining_count == 0:
                    stable = True
                if few_remaining or confident or stable:
                    x0, transfer_index = _force_commit(block_predicted_tokens, mask_index, x[:, block_start:])
                    x[:, block_start:][transfer_index] = x0[transfer_index]
                    break
                if nfe >= max_refinement_steps:
                    x0, transfer_index = _force_commit(block_predicted_tokens, mask_index, x[:, block_start:])
                    x[:, block_start:][transfer_index] = x0[transfer_index]
                    break
                x0, transfer_index = get_transfer_index(
                    block_logits, block_predicted_tokens, remasking, mask_index, x[:, block_start:], None, threshold,
                )
            else:
                x0, transfer_index = get_transfer_index(
                    block_logits, block_predicted_tokens, remasking, mask_index, x[:, block_start:], None, threshold,
                )

            newly_unmasked = transfer_index[:, :block_length]
            if newly_unmasked.any():
                unmask_confs.append(block_max_probs[0, :block_length][newly_unmasked[0]])
            x[:, block_start:][transfer_index] = x0[transfer_index]
            prev_predictions.append(block_predicted_tokens[0, :block_length].clone())
            prev_predictions = prev_predictions[-tau_stable_steps:]

        if unmask_confs:
            all_confs = torch.cat(unmask_confs)
            mean_conf = all_confs.mean().item()
            min_conf = all_confs.min().item()
        else:
            mean_conf = min_conf = 1.0
        block_tokens = x[0, block_start:block_end]
        digit_frac = torch.isin(block_tokens, digit_ids_tensor.to(x.device)).float().mean().item() if digit_ids_tensor is not None else 0.0
        delim_frac = torch.isin(block_tokens, delimiter_ids_tensor.to(x.device)).float().mean().item() if delimiter_ids_tensor is not None else 0.0
        scheduler.record_realized(block_length, nfe, mean_conf, min_conf, digit_frac, delim_frac)
        nfe_history.append(nfe)
        block_history.append(block_length)
        _record_schedule(schedule_history, schedule=schedule, nfe=nfe, block_start=block_start, block_end=block_end)

    return x, nfe_history, block_history, schedule_history
@torch.no_grad()
def generate_pag_dual_cache(
    model,
    prompt: torch.Tensor,
    scheduler,
    *,
    steps: int = 128,
    gen_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float | None = None,
    max_block_length: int | None = None,
    max_refinement_steps: int | None = None,
    digit_ids_tensor: torch.Tensor | None = None,
    delimiter_ids_tensor: torch.Tensor | None = None,
    delimiter_ids: list[int] | None = None,
    delimiter_threshold: float | None = None,
    tau_commit: float = 0.80,
    tau_stable_steps: int = 2,
    default_block_length: int = 32,
):
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    assert threshold is not None
    from generate_adablock import compute_block_length

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    scheduler.reset()
    max_block_length = gen_length if max_block_length is None else int(max_block_length)
    max_refinement_steps = steps if max_refinement_steps is None else int(max_refinement_steps)
    delimiter_ids = delimiter_ids or [198]

    generated_length = 0
    nfe_history: list[int] = []
    block_history: list[int] = []
    schedule_history: list[dict[str, object]] = []

    while generated_length < gen_length:
        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)

        block_length = compute_block_length(
            logits, predicted_tokens, prompt, gen_length,
            generated_length, default_block_length=default_block_length,
            delimiter_ids=delimiter_ids,
            delimiter_threshold=delimiter_threshold or float("inf"),
        )
        block_length = min(int(block_length), max_block_length, gen_length - generated_length)

        schedule = scheduler.next_schedule(
            block_size=block_length,
            remaining_tokens=gen_length - generated_length,
            max_block_length=max_block_length,
            max_refinement_steps=max_refinement_steps,
        )

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        # First unmask
        unmask_confs: list[torch.Tensor] = []
        mask_index = x == mask_id
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(
            logits, predicted_tokens, remasking, mask_index, x, None, threshold,
        )
        initial_probs = F.softmax(logits[:, block_start:block_end, :], dim=-1)
        initial_max_probs = initial_probs.max(dim=-1).values
        newly_unmasked = transfer_index[:, block_start:block_end]
        if newly_unmasked.any():
            unmask_confs.append(initial_max_probs[newly_unmasked])
        x[transfer_index] = x0[transfer_index]

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = 1

        # Refinement loop with dual cache
        nfe = 0
        prev_predictions: list[torch.Tensor] = []

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            if nfe >= max_refinement_steps:
                break

            mask_index = x[:, block_start:block_end] == mask_id
            block_output = model(
                x[:, block_start:block_end],
                past_key_values=full_cache,
                use_cache=True,
                replace_position=replace_position,
            )
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1

            block_probs = F.softmax(block_logits, dim=-1)
            block_max_probs = block_probs.max(dim=-1).values

            if nfe >= schedule.budgeted_refinement_steps:
                remaining_count = (x[:, block_start:block_end] == mask_id).sum().item()
                few_remaining = remaining_count <= max(1, math.ceil(0.10 * block_length))
                confident = False
                if remaining_count > 0:
                    local_conf = block_max_probs[0, mask_index[0]]
                    confident = local_conf.min().item() >= tau_commit if local_conf.numel() > 0 else True
                else:
                    confident = True
                stable = False
                if remaining_count > 0 and len(prev_predictions) >= tau_stable_steps:
                    current_for_remaining = block_predicted_tokens[0][mask_index[0]]
                    stable = all(
                        torch.all(current_for_remaining == p[mask_index[0]])
                        for p in prev_predictions[-tau_stable_steps:]
                    )
                elif remaining_count == 0:
                    stable = True
                if few_remaining or confident or stable:
                    x0, transfer_index = _force_commit(block_predicted_tokens, mask_index, x[:, block_start:block_end])
                    x[:, block_start:block_end][transfer_index] = x0[transfer_index]
                    break
                if nfe >= max_refinement_steps:
                    x0, transfer_index = _force_commit(block_predicted_tokens, mask_index, x[:, block_start:block_end])
                    x[:, block_start:block_end][transfer_index] = x0[transfer_index]
                    break
                x0, transfer_index = get_transfer_index(
                    block_logits, block_predicted_tokens, remasking, mask_index, x[:, block_start:block_end], None, threshold,
                )
            else:
                x0, transfer_index = get_transfer_index(
                    block_logits, block_predicted_tokens, remasking, mask_index, x[:, block_start:block_end], None, threshold,
                )

            newly_unmasked = transfer_index
            if newly_unmasked.any():
                unmask_confs.append(block_max_probs[newly_unmasked])
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
            prev_predictions.append(block_predicted_tokens[0].clone())
            prev_predictions = prev_predictions[-tau_stable_steps:]

        if unmask_confs:
            all_confs = torch.cat(unmask_confs)
            mean_conf = all_confs.mean().item()
            min_conf = all_confs.min().item()
        else:
            mean_conf = min_conf = 1.0
        block_tokens = x[0, block_start:block_end]
        digit_frac = torch.isin(block_tokens, digit_ids_tensor.to(x.device)).float().mean().item() if digit_ids_tensor is not None else 0.0
        delim_frac = torch.isin(block_tokens, delimiter_ids_tensor.to(x.device)).float().mean().item() if delimiter_ids_tensor is not None else 0.0
        scheduler.record_realized(block_length, nfe, mean_conf, min_conf, digit_frac, delim_frac)
        nfe_history.append(nfe)
        block_history.append(block_length)
        _record_schedule(schedule_history, schedule=schedule, nfe=nfe, block_start=block_start, block_end=block_end)

    return x, nfe_history, block_history, schedule_history

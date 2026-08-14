# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# Copyright 2025 Guanxi Lu, Imperial College London (modifications)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA
# Modified by Guanxi Lu, Imperial College London

import time

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def compute_block_length(
    logits,                
    predicted_tokens,      
    prompt,                
    gen_length,       
    generated_length,
    default_block_length,
    delimiter_ids=[198],  # default: newline token (11=comma, 13=period, 198=newline)
    delimiter_threshold=float('inf')
):
    """
    Compute adaptive block length based on delimiter confidence.
    Returns the position of the highest-confidence delimiter if above threshold,
    otherwise returns default_block_length.
    """
    prompt_length = prompt.shape[1]
    block_start = prompt_length + generated_length
    remaining_length = gen_length - generated_length
    
    # Create sampling window (25% of gen_length, capped by remaining)
    window_size = min(int(0.25 * gen_length), remaining_length)
    window_tokens = predicted_tokens[0, block_start:block_start + window_size]
    
    # Create mask for delimiter tokens
    delimiter_mask = torch.zeros_like(window_tokens, dtype=torch.bool)
    for token_id in delimiter_ids:
        delimiter_mask |= (window_tokens == token_id)

    # Fallback to default block length if no delimiter is found
    if not torch.any(delimiter_mask):
        return min(default_block_length, remaining_length)

    # Get positions of delimiters in the sequence
    delimiter_pos = block_start + torch.nonzero(delimiter_mask).squeeze(-1)
    
    # Compute confidence for each delimiter
    delimiter_logits = logits[0, delimiter_pos, predicted_tokens[0, delimiter_pos]]
    log_sum_exp = torch.logsumexp(logits[0, delimiter_pos, :], dim=-1)
    delimiter_confidences = torch.exp(delimiter_logits - log_sum_exp)

    # Find the delimiter with highest confidence
    max_confidence, best_idx = torch.max(delimiter_confidences, dim=0)
    max_confidence = max_confidence.item()
    best_delimiter_pos = delimiter_pos[best_idx].item()

    if max_confidence >= delimiter_threshold:
        block_length = best_delimiter_pos - block_start + 1
    else:
        block_length = min(default_block_length, remaining_length)
    return block_length

@ torch.no_grad()
def generate_adablock(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf')):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []  
    block_history = []
    while generated_length < gen_length: 
        nfe = 0

        output = model(x)
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        block_length = compute_block_length(logits, predicted_tokens, prompt, gen_length, generated_length, init_block_length, delimiter_ids=delimiter_ids, delimiter_threshold=delimiter_threshold)
        
        block_history.append(block_length)
        
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length
        
        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = 0
            block_output = model(x)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index, 
                                            x, None, threshold)
            x[transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history

@torch.no_grad()
def generate_adablock_prefix_cache(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf')):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []  
    block_history = []

    while generated_length < gen_length: 
        nfe = 0

        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        block_length = compute_block_length(logits, predicted_tokens, prompt, gen_length, generated_length, init_block_length, delimiter_ids=delimiter_ids, delimiter_threshold=delimiter_threshold)
        
        block_history.append(block_length)
        
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        # truncate cache to prefix only (before current block)
        prefix_cache = []
        for i in range(len(full_cache)):
            prefix_cache.append(())
            for j in range(len(full_cache[i])):
                prefix_cache[i] += (full_cache[i][j][:, :, :block_start],)

        # 2nd and later denoising steps in block
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, block_start:] == mask_id)
            mask_index[:, block_length:] = 0
            block_output = model(x[:, block_start:], past_key_values=prefix_cache, use_cache=True)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index, 
                                            x[:, block_start:], None, threshold)
            x[:, block_start:][transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history


@torch.no_grad()
def generate_adablock_dual_cache(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf'),
            risk_policy=None, speculation_policy=None, digit_ids_tensor=None,
            delimiter_ids_tensor=None, return_schedule_history=False,
            record_state_digests=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    if risk_policy is not None and speculation_policy is not None:
        raise ValueError("risk stopping and verified speculation are mutually exclusive")
    if speculation_policy is not None:
        from pag.experiments.rc_pag_equivalence import EquivalenceCostPolicy

        record_state_digests = record_state_digests or isinstance(
            speculation_policy, EquivalenceCostPolicy
        )
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []  
    block_history = []
    schedule_history = []
    if risk_policy is not None:
        risk_policy.reset_prompt()
    if speculation_policy is not None:
        speculation_policy.reset_prompt()
    
    while generated_length < gen_length: 
        nfe = 0

        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        block_length = compute_block_length(logits, predicted_tokens, prompt, gen_length, generated_length, init_block_length, delimiter_ids=delimiter_ids, delimiter_threshold=delimiter_threshold)
        
        block_history.append(block_length)
        
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length
        risk_steps = []
        speculation_steps = []
        state_digests = []
        if risk_policy is not None:
            risk_policy.start_block()
        if speculation_policy is not None:
            speculation_policy.start_block()
        
        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]
        if record_state_digests:
            from pag.experiments.rc_pag_equivalence import state_digest

            state_digests.append(
                state_digest(x[:, block_start:block_end], block_start=block_start)
            )
        last_transfer_count = int(transfer_index.sum().item())
        verified_transition_count = 1
        final_block_logits = logits[:, block_start:block_end]
        previous_policy_logits = None

        if risk_policy is not None:
            from pag.experiments.rc_pag_adapter import observe_policy_step, serialize_policy_step

            policy_step = observe_policy_step(
                risk_policy,
                logits=final_block_logits,
                current_tokens=x[:, block_start:block_end],
                mask_token_id=mask_id,
                step_index=nfe,
                full_tokens=x,
                block_start=block_start,
                block_end=block_end,
                digit_ids=digit_ids_tensor,
                delimiter_ids=delimiter_ids_tensor,
                cache=full_cache,
                previous_logits=previous_policy_logits,
            )
            risk_steps.append(serialize_policy_step(policy_step))
            previous_policy_logits = final_block_logits.detach()
            if policy_step.decision.should_stop:
                proposed = torch.tensor(
                    policy_step.proposed_tokens,
                    dtype=x.dtype,
                    device=x.device,
                ).reshape(1, -1)
                active = x[:, block_start:block_end]
                remaining = active == mask_id
                active[remaining] = proposed[remaining]

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = 1
        # 2nd and later denoising steps in block
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, block_start:block_end] == mask_id)
            if speculation_policy is not None:
                from pag.experiments.rc_pag_adapter import observation_from_tensors
                from pag.experiments.rc_pag_equivalence import (
                    EquivalenceCostPolicy,
                    serialize_guarded_result,
                    state_digest,
                    verify_guarded_draft,
                )
                from pag.experiments.rc_pag_speculation import (
                    build_linear_draft,
                    repeat_tensor_tree,
                    serialize_speculation_plan,
                    serialize_speculation_result,
                    verify_draft,
                )

                active = x[:, block_start:block_end]
                observation = observation_from_tensors(
                    logits=final_block_logits,
                    previous_logits=previous_policy_logits,
                    current_tokens=active,
                    mask_token_id=mask_id,
                    step_index=verified_transition_count,
                    digit_ids=digit_ids_tensor,
                    delimiter_ids=delimiter_ids_tensor,
                )
                plan = speculation_policy.choose(
                    observation,
                    last_transfer_count=last_transfer_count,
                )
                is_equivalence = isinstance(speculation_policy, EquivalenceCostPolicy)
                activation_key = (
                    speculation_policy.activation_key(observation, last_transfer_count)
                    if is_equivalence
                    else None
                )
                if is_equivalence and plan.depth == 0:
                    block_output = model(
                        x[:, block_start:block_end],
                        past_key_values=full_cache,
                        use_cache=True,
                        replace_position=replace_position,
                    )
                    block_logits = block_output.logits
                    final_block_logits = block_logits
                    block_predictions = torch.argmax(
                        add_gumbel_noise(block_logits, temperature=temperature),
                        dim=-1,
                    )
                    x0, transfer_index = get_transfer_index(
                        block_logits,
                        block_predictions,
                        remasking,
                        mask_index,
                        x[:, block_start:block_end],
                        None,
                        threshold,
                    )
                    x[:, block_start:block_end][transfer_index] = x0[transfer_index]
                    nfe += 1
                    verified_transition_count += 1
                    last_transfer_count = max(1, int(transfer_index.sum().item()))
                    previous_policy_logits = final_block_logits.detach()
                    if record_state_digests:
                        state_digests.append(
                            state_digest(
                                x[:, block_start:block_end],
                                block_start=block_start,
                            )
                        )
                    continue
                proposal = torch.argmax(final_block_logits, dim=-1)
                proposal = torch.where(mask_index, proposal, active)
                probabilities = torch.softmax(final_block_logits.float(), dim=-1)
                proposal_confidence = probabilities.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
                ranked_positions = torch.argsort(
                    torch.where(
                        mask_index,
                        proposal_confidence,
                        torch.full_like(proposal_confidence, -torch.inf),
                    ),
                    dim=-1,
                    descending=True,
                )[0]
                ranked_positions = [
                    int(position)
                    for position in ranked_positions.tolist()
                    if bool(mask_index[0, position])
                ]
                nodes = build_linear_draft(
                    active,
                    proposed_tokens=proposal,
                    mask_token_id=mask_id,
                    ranked_positions=ranked_positions,
                    depth=plan.depth,
                    draft_width=plan.draft_width,
                )
                node_count = len(nodes)
                batched_cache = repeat_tensor_tree(full_cache, node_count)
                batched_replace_position = replace_position.repeat(node_count, 1)
                if is_equivalence and speculation_policy.audit_reference and x.is_cuda:
                    torch.cuda.synchronize()
                batch_started = time.perf_counter()
                block_output = model(
                    torch.cat(nodes, dim=0),
                    past_key_values=batched_cache,
                    use_cache=True,
                    replace_position=batched_replace_position,
                )
                if is_equivalence and speculation_policy.audit_reference and x.is_cuda:
                    torch.cuda.synchronize()
                batched_latency_ms = (time.perf_counter() - batch_started) * 1000.0
                block_logits = block_output.logits
                nfe += 1
                root_masks = int(mask_index.sum().item())

                def verified_transition(state, verified_logits):
                    if verified_logits.ndim == 2:
                        verified_logits = verified_logits.unsqueeze(0)
                    verified_predictions = torch.argmax(
                        add_gumbel_noise(verified_logits, temperature=temperature),
                        dim=-1,
                    )
                    verified_mask = state == mask_id
                    verified_x0, verified_transfer = get_transfer_index(
                        verified_logits,
                        verified_predictions,
                        remasking,
                        verified_mask,
                        state,
                        None,
                        threshold,
                    )
                    successor = state.detach().clone()
                    successor[verified_transfer] = verified_x0[verified_transfer]
                    return successor

                audit_event = {}
                canonical_holder = {}
                if is_equivalence:
                    def canonical_root():
                        nonlocal nfe
                        if x.is_cuda:
                            torch.cuda.synchronize()
                        canonical_started = time.perf_counter()
                        canonical_output = model(
                            active.detach().clone(),
                            past_key_values=full_cache,
                            use_cache=True,
                            replace_position=replace_position,
                        )
                        if x.is_cuda:
                            torch.cuda.synchronize()
                        canonical_logits = canonical_output.logits[0]
                        canonical_holder["logits"] = canonical_logits
                        canonical_holder["latency_ms"] = (
                            time.perf_counter() - canonical_started
                        ) * 1000.0
                        nfe += 1
                        return canonical_logits

                    guard = lambda state, state_logits: speculation_policy.guard(
                        state,
                        state_logits,
                        batch_size=node_count,
                        mask_token_id=mask_id,
                    )
                    if speculation_policy.audit_reference:
                        canonical_logits = canonical_root()
                        result = verify_guarded_draft(
                            nodes,
                            block_logits,
                            verified_transition,
                            guard,
                            mask_token_id=mask_id,
                            canonical_root_output=canonical_logits,
                        )
                        batched_root = block_logits[0].float()
                        canonical_root_logits = canonical_logits.float()
                        full_acceptance = all(
                            torch.equal(
                                verified_transition(nodes[index], block_logits[index]),
                                nodes[index + 1],
                            )
                            for index in range(node_count - 1)
                        )
                        audit_event = {
                            "batch_size": node_count,
                            "depth": plan.depth,
                            "activation_key": activation_key,
                            "max_logit_delta": float(
                                torch.max(torch.abs(batched_root - canonical_root_logits)).item()
                            ),
                            "max_probability_delta": float(
                                torch.max(
                                    torch.abs(
                                        torch.softmax(batched_root, dim=-1)
                                        - torch.softmax(canonical_root_logits, dim=-1)
                                    )
                                ).item()
                            ),
                            "full_acceptance": bool(full_acceptance),
                            "batched_latency_ms": batched_latency_ms,
                            "canonical_latency_ms": canonical_holder["latency_ms"],
                        }
                    else:
                        result = verify_guarded_draft(
                            nodes,
                            block_logits,
                            verified_transition,
                            guard,
                            mask_token_id=mask_id,
                            canonical_root=canonical_root,
                        )
                else:
                    result = verify_draft(
                        nodes,
                        block_logits,
                        verified_transition,
                        mask_token_id=mask_id,
                    )
                x[:, block_start:block_end] = result.tokens
                used_node = min(result.accepted_draft_edges, node_count - 1)
                previous_policy_logits = final_block_logits.detach()
                if is_equivalence and canonical_holder:
                    final_block_logits = canonical_holder["logits"].unsqueeze(0)
                else:
                    final_block_logits = block_logits[used_node : used_node + 1]
                transition_count = (
                    result.reference_equivalent_transitions
                    if is_equivalence
                    else result.verified_transitions
                )
                verified_transition_count += transition_count
                last_transfer_count = max(
                    1,
                    root_masks - int((result.tokens == mask_id).sum().item()),
                )
                if is_equivalence and record_state_digests:
                    state_digests.extend(
                        state_digest(state, block_start=block_start)
                        for state in result.transition_states
                    )
                result_payload = (
                    serialize_guarded_result(result)
                    if is_equivalence
                    else serialize_speculation_result(result)
                )
                speculation_steps.append({
                    "step_index": int(verified_transition_count),
                    "remaining_masks_before": root_masks,
                    "remaining_masks_after": int((result.tokens == mask_id).sum().item()),
                    **serialize_speculation_plan(plan),
                    **result_payload,
                    **audit_event,
                })
                continue
            block_output = model(x[:, block_start:block_end], past_key_values=full_cache, use_cache=True, replace_position=replace_position)
            block_logits = block_output.logits
            final_block_logits = block_logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            force_commit = False
            if risk_policy is not None:
                from pag.experiments.rc_pag_adapter import observe_policy_step, serialize_policy_step

                policy_step = observe_policy_step(
                    risk_policy,
                    logits=block_logits,
                    current_tokens=x[:, block_start:block_end],
                    mask_token_id=mask_id,
                    step_index=nfe,
                    full_tokens=x,
                    block_start=block_start,
                    block_end=block_end,
                    digit_ids=digit_ids_tensor,
                    delimiter_ids=delimiter_ids_tensor,
                    cache=full_cache,
                    previous_logits=previous_policy_logits,
                )
                risk_steps.append(serialize_policy_step(policy_step))
                previous_policy_logits = block_logits.detach()
                force_commit = bool(policy_step.decision.should_stop)
                if force_commit:
                    proposed = torch.tensor(
                        policy_step.proposed_tokens,
                        dtype=x.dtype,
                        device=x.device,
                    ).reshape(1, -1)
                    active = x[:, block_start:block_end]
                    active[mask_index] = proposed[mask_index]
            if not force_commit:
                x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index,
                                                x[:, block_start:block_end], None, threshold)
                x[:, block_start:block_end][transfer_index] = x0[transfer_index]
            verified_transition_count += 1
            if record_state_digests:
                from pag.experiments.rc_pag_equivalence import state_digest

                state_digests.append(
                    state_digest(x[:, block_start:block_end], block_start=block_start)
                )
        nfe_history.append(nfe)

        if risk_policy is not None or speculation_policy is not None or return_schedule_history:
            from pag.experiments.rc_pag_features import RealizedBlock

            block_tokens = x[:, block_start:block_end]
            probabilities = torch.softmax(final_block_logits.float(), dim=-1)
            token_confidence = probabilities.gather(
                -1, block_tokens.unsqueeze(-1)
            ).squeeze(-1)
            digit_fraction = (
                torch.isin(block_tokens, digit_ids_tensor.to(x.device)).float().mean().item()
                if digit_ids_tensor is not None
                else 0.0
            )
            delimiter_fraction = (
                torch.isin(block_tokens, delimiter_ids_tensor.to(x.device)).float().mean().item()
                if delimiter_ids_tensor is not None
                else 0.0
            )
            if risk_policy is not None:
                risk_policy.record_realized(
                    RealizedBlock(
                        block_size=block_length,
                        nfe=nfe,
                        mean_confidence=token_confidence.mean().item(),
                        min_confidence=token_confidence.min().item(),
                        digit_fraction=digit_fraction,
                        delimiter_fraction=delimiter_fraction,
                    )
                )
            if speculation_policy is not None:
                speculation_policy.record_realized(
                    RealizedBlock(
                        block_size=block_length,
                        nfe=nfe,
                        mean_confidence=token_confidence.mean().item(),
                        min_confidence=token_confidence.min().item(),
                        digit_fraction=digit_fraction,
                        delimiter_fraction=delimiter_fraction,
                    )
                )
            schedule_history.append(
                {
                    "block_index": len(schedule_history),
                    "applied_block_size": int(block_length),
                    "budgeted_refinement_steps": int(steps),
                    "actual_nfe_used": int(nfe),
                    "mean_top1_confidence": token_confidence.mean().item(),
                    "min_top1_confidence": token_confidence.min().item(),
                    "digit_fraction": digit_fraction,
                    "delimiter_fraction": delimiter_fraction,
                    "block_start": int(block_start),
                    "block_end": int(block_end),
                    "risk_steps": risk_steps,
                    "speculation_steps": speculation_steps,
                    "verified_transition_count": int(verified_transition_count),
                    "speculative_nfe_saved": int(
                        sum(step["nfe_saved"] for step in speculation_steps)
                    ),
                    "verified_sequence_safe": all(
                        step.get("sequence_safe", step.get("guard_passed", False))
                        for step in speculation_steps
                    ),
                    "guarded_transition_evidence": all(
                        step.get("guard_passed", False)
                        or step.get("reference_checked", False)
                        for step in speculation_steps
                    ),
                    "state_digests": state_digests,
                    "model_time_sec": 0.0,
                    "final_tokens": block_tokens.detach().cpu().reshape(-1).tolist(),
                }
            )

    if return_schedule_history:
        return x, nfe_history, block_history, schedule_history
    return x, nfe_history, block_history

def get_transfer_index(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    x0 = predicted_tokens  # (B, L)

    # Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)  # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out_ids, nfe_history, block_history = generate_adablock(model, input_ids, steps=32, gen_length=256, init_block_length=16, temperature=0., remasking='low_confidence', threshold=0.9, delimiter_ids=[198], delimiter_threshold=0.3)
    
    print(tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print()
    print(f"NFE for each block: {nfe_history}")
    print(f"Total NFE: {sum(nfe_history)}")
    print(f"Average NFE per block: {sum(nfe_history) / len(nfe_history)}")
    print()
    print(f"Block length for each block: {block_history}")
    print(f"Number of blocks: {len(block_history)}")
    print(f"Average block length: {sum(block_history) / len(block_history)}")

if __name__ == '__main__':
    main()

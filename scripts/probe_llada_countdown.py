"""
LLaDA-8B baseline probe on the Countdown dataset.

Loads the first N samples from Jiayi-Pan/Countdown-Tasks-3to4, runs the full
T-step masked diffusion denoising loop on each, and saves per-token signals
to a JSONL file (one record per sample).

Collected signals per token:
  tau_commit          : denoising step at which token was unmasked
  tau_stable          : first step confidence crossed threshold
  max_refinement_step : last step token was still masked
  gap                 : tau_commit - tau_stable
  confidence/entropy  : at commit and stable steps
  prob_trajectory     : confidence at every step while masked

Usage:
    uv run python scripts/probe_llada_countdown.py
    uv run python scripts/probe_llada_countdown.py --n_samples 50 --steps 64
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

# Compatibility patch: LLaDA's custom model class doesn't implement
# all_tied_weights_keys expected by newer transformers. Safe to return {}
# because LLaDA has no tied weights.
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = property(lambda _: {})

MODEL_ID          = "GSAI-ML/LLaDA-8B-Base"
DATASET_ID        = "Jiayi-Pan/Countdown-Tasks-3to4"
N_SAMPLES         = 50
DENOISING_STEPS   = 64
TARGET_LEN        = 32   # tokens to generate per sample
CONFIDENCE_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int,   default=N_SAMPLES)
    parser.add_argument("--steps",      type=int,   default=DENOISING_STEPS)
    parser.add_argument("--target_len", type=int,   default=TARGET_LEN)
    parser.add_argument("--threshold",  type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--output",     type=str,   default="countdown_baseline_traces.jsonl")
    parser.add_argument("--split",      type=str,   default="train")
    return parser.parse_args()


def build_prompt(nums: list[int], target: int) -> str:
    nums_str = ", ".join(str(n) for n in nums)
    return (
        f"Using the numbers {nums_str}, find an arithmetic expression "
        f"that equals {target}. You may use +, -, *, / and each number at most once."
    )


def compute_entropy(probs: torch.Tensor) -> float:
    return float(-(probs * torch.log(probs + 1e-10)).sum().item())


def get_mask_token_id(tokenizer) -> int:
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    if mask_token_id is None or mask_token_id == tokenizer.unk_token_id:
        mask_token_id = 126336  # known LLaDA-8B mask token id
    return mask_token_id


def run_denoising_loop(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    target_len: int,
    T: int,
    threshold: float,
    mask_token_id: int,
    device: torch.device,
) -> dict:
    prompt_len = prompt_ids.shape[1]

    mask_ids = torch.full((1, target_len), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, mask_ids], dim=1)

    tau_commit      = [None] * target_len
    tau_stable      = [None] * target_len
    max_refine_step = [None] * target_len
    conf_at_commit  = [0.0]  * target_len
    conf_at_stable  = [0.0]  * target_len
    entr_at_commit  = [0.0]  * target_len
    entr_at_stable  = [0.0]  * target_len
    prob_trajectory = [[]    for _ in range(target_len)]
    committed       = [False] * target_len

    for step in range(T):
        masked_positions = [i for i in range(target_len) if not committed[i]]
        if not masked_positions:
            break

        with torch.no_grad():
            outputs = model(input_ids)
            logits  = outputs.logits[0, prompt_len:].float()

        probs      = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        top_ids    = probs.argmax(dim=-1)

        for i in masked_positions:
            conf = float(confidence[i].item())
            prob_trajectory[i].append(round(conf, 4))
            max_refine_step[i] = step
            if tau_stable[i] is None and conf >= threshold:
                tau_stable[i]    = step
                conf_at_stable[i] = conf
                entr_at_stable[i] = compute_entropy(probs[i])

        remaining_steps = T - step
        n_to_unmask = math.ceil(len(masked_positions) / remaining_steps)
        ranked    = sorted(masked_positions, key=lambda i: confidence[i].item(), reverse=True)
        to_unmask = ranked[:n_to_unmask]

        for i in to_unmask:
            tau_commit[i]    = step
            conf_at_commit[i] = float(confidence[i].item())
            entr_at_commit[i] = compute_entropy(probs[i])
            input_ids[0, prompt_len + i] = top_ids[i]
            committed[i] = True

    final_token_ids = input_ids[0, prompt_len:].tolist()
    final_tokens    = [tokenizer.decode([tid]).strip() for tid in final_token_ids]

    gap = [
        (tau_commit[i] - tau_stable[i])
        if (tau_commit[i] is not None and tau_stable[i] is not None)
        else None
        for i in range(target_len)
    ]

    return {
        "final_tokens":         final_tokens,
        "final_token_ids":      final_token_ids,
        "tau_commit":           tau_commit,
        "tau_stable":           tau_stable,
        "max_refinement_step":  max_refine_step,
        "gap":                  gap,
        "confidence_at_commit": conf_at_commit,
        "confidence_at_stable": conf_at_stable,
        "entropy_at_commit":    entr_at_commit,
        "entropy_at_stable":    entr_at_stable,
        "prob_trajectory":      prob_trajectory,
    }


def main() -> None:
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()
    print(f"Model on {device}\n")

    mask_token_id = get_mask_token_id(tokenizer)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading {DATASET_ID} (first {args.n_samples} samples) ...")
    ds      = load_dataset(DATASET_ID, split=args.split)
    samples = list(ds.select(range(args.n_samples)))
    print(f"Loaded {len(samples)} samples\n")

    # ── Run baseline loop ─────────────────────────────────────────────────────
    print(f"{'#':>4}  {'Target':>6}  {'Nums':<20}  {'Steps':>5}  {'Time':>6}")
    print("-" * 52)

    with open(args.output, "w") as out_f:
        for idx, sample in enumerate(samples):
            nums   = list(sample["nums"])
            target = int(sample["target"])
            prompt = build_prompt(nums, target)

            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            t0      = time.time()
            results = run_denoising_loop(
                model         = model,
                tokenizer     = tokenizer,
                prompt_ids    = prompt_ids,
                target_len    = args.target_len,
                T             = args.steps,
                threshold     = args.threshold,
                mask_token_id = mask_token_id,
                device        = device,
            )
            elapsed = time.time() - t0

            steps_used = max(
                (r for r in results["tau_commit"] if r is not None), default=0
            ) + 1

            nums_str = str(nums)
            print(f"{idx:>4}  {target:>6}  {nums_str:<20}  {steps_used:>5}  {elapsed:>5.1f}s")

            # Build per-token records
            token_records = []
            for i in range(args.target_len):
                token_records.append({
                    "position":             i,
                    "token":                results["final_tokens"][i],
                    "token_id":             results["final_token_ids"][i],
                    "tau_commit":           results["tau_commit"][i],
                    "tau_stable":           results["tau_stable"][i],
                    "max_refinement_step":  results["max_refinement_step"][i],
                    "gap":                  results["gap"][i],
                    "confidence_at_commit": round(results["confidence_at_commit"][i], 6),
                    "confidence_at_stable": round(results["confidence_at_stable"][i], 6),
                    "entropy_at_commit":    round(results["entropy_at_commit"][i], 6),
                    "entropy_at_stable":    round(results["entropy_at_stable"][i], 6),
                    "prob_trajectory":      results["prob_trajectory"][i],
                })

            record = {
                "sample_id":    f"countdown-{idx:04d}",
                "nums":         nums,
                "target":       target,
                "prompt":       prompt,
                "generated":    " ".join(results["final_tokens"]),
                "denoising_steps": args.steps,
                "target_len":   args.target_len,
                "threshold":    args.threshold,
                "token_records": token_records,
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()  # write immediately so progress is saved if job is killed

    print(f"\nDone. Traces saved to {args.output}")


if __name__ == "__main__":
    main()

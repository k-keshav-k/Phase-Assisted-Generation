"""
Full denoising loop probe script for LLaDA-8B.

Runs T denoising steps (default 64) and records per-token signals:
  - tau_commit          : step at which the token was unmasked
  - tau_stable          : step at which confidence first crossed threshold
  - max_refinement_step : last step the token was still masked
  - gap                 : tau_commit - tau_stable
  - confidence/entropy  : at both stable and commit steps
  - prob_trajectory     : confidence at every step while masked

Saves output to probe_llada_output.json.

Usage:
    uv run python scripts/probe_llada.py
    uv run python scripts/probe_llada.py --prompt "Your text" --target_len 32 --steps 64
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

# LLaDA's custom model class is missing all_tied_weights_keys which newer
# transformers expects. Patch it to return an empty dict — safe because
# LLaDA has no tied weights.
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = property(lambda _: {})

MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
PROMPT = "Explain why adaptive diffusion decoding can improve language generation quality."
TARGET_LEN = 20
DENOISING_STEPS = 64
CONFIDENCE_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--target_len", type=int, default=TARGET_LEN)
    parser.add_argument("--steps", type=int, default=DENOISING_STEPS)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--output", type=str, default="probe_llada_output.json")
    return parser.parse_args()


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
    """Run T denoising steps and collect per-token signals."""

    prompt_len = prompt_ids.shape[1]

    # Initialize: all target positions masked
    mask_ids = torch.full((1, target_len), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, mask_ids], dim=1)

    # Per-token tracking
    tau_commit = [None] * target_len
    tau_stable = [None] * target_len
    max_refine_step = [None] * target_len
    conf_at_commit = [0.0] * target_len
    conf_at_stable = [0.0] * target_len
    entr_at_commit = [0.0] * target_len
    entr_at_stable = [0.0] * target_len
    prob_trajectory = [[] for _ in range(target_len)]
    committed = [False] * target_len

    print(f"\nRunning {T} denoising steps ...")
    print(f"{'Step':>5}  {'Unmasked':>8}  {'Remaining':>9}  {'Avg conf (masked)':>18}")
    print("-" * 48)

    for step in range(T):
        masked_positions = [i for i in range(target_len) if not committed[i]]
        if not masked_positions:
            print(f"  All tokens committed at step {step}. Stopping early.")
            break

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, prompt_len:].float()  # (target_len, vocab_size)

        probs = torch.softmax(logits, dim=-1)  # (target_len, vocab_size)
        confidence = probs.max(dim=-1).values  # (target_len,)
        top_ids = probs.argmax(dim=-1)  # (target_len,)

        # Update trajectory and tau_stable for still-masked positions
        for i in masked_positions:
            conf = float(confidence[i].item())
            prob_trajectory[i].append(round(conf, 4))
            max_refine_step[i] = step

            if tau_stable[i] is None and conf >= threshold:
                tau_stable[i] = step
                conf_at_stable[i] = conf
                entr_at_stable[i] = compute_entropy(probs[i])

        # Decide how many to unmask this step:
        # ceil(remaining / remaining_steps) ensures we finish by step T
        remaining_steps = T - step
        n_to_unmask = math.ceil(len(masked_positions) / remaining_steps)

        # Pick the n most confident among still-masked
        ranked = sorted(masked_positions, key=lambda i: confidence[i].item(), reverse=True)
        to_unmask = ranked[:n_to_unmask]

        avg_conf = sum(confidence[i].item() for i in masked_positions) / len(masked_positions)
        print(f"{step:>5}  {n_to_unmask:>8}  {len(masked_positions):>9}  {avg_conf:>18.4f}")

        # Commit them
        for i in to_unmask:
            tau_commit[i] = step
            conf_at_commit[i] = float(confidence[i].item())
            entr_at_commit[i] = compute_entropy(probs[i])
            input_ids[0, prompt_len + i] = top_ids[i]
            committed[i] = True

    # Decode final tokens
    final_token_ids = input_ids[0, prompt_len:].tolist()
    final_tokens = [tokenizer.decode([tid]).strip() for tid in final_token_ids]

    return {
        "final_tokens": final_tokens,
        "final_token_ids": final_token_ids,
        "tau_commit": tau_commit,
        "tau_stable": tau_stable,
        "max_refinement_step": max_refine_step,
        "gap": [
            (tau_commit[i] - tau_stable[i])
            if (tau_commit[i] is not None and tau_stable[i] is not None)
            else None
            for i in range(target_len)
        ],
        "confidence_at_commit": conf_at_commit,
        "confidence_at_stable": conf_at_stable,
        "entropy_at_commit": entr_at_commit,
        "entropy_at_stable": entr_at_stable,
        "prob_trajectory": prob_trajectory,
    }


def main() -> None:
    args = parse_args()

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    mask_token_id = get_mask_token_id(tokenizer)
    print(f"Mask token id : {mask_token_id}")

    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"Prompt length : {prompt_ids.shape[1]} tokens")
    print(f"Target length : {args.target_len} masked positions")
    print(f"Denoising steps: {args.steps}")

    results = run_denoising_loop(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        target_len=args.target_len,
        T=args.steps,
        threshold=args.threshold,
        mask_token_id=mask_token_id,
        device=device,
    )

    # Print per-token summary table
    print(
        f"\n{'Pos':>4}  {'Token':<16}  {'tau_commit':>10}  {'tau_stable':>10}  {'gap':>5}  {'conf@commit':>11}  {'entr@commit':>11}"
    )
    print("-" * 80)

    token_records = []
    for i in range(args.target_len):
        tc = results["tau_commit"][i]
        ts = results["tau_stable"][i]
        gap = results["gap"][i]
        tok = results["final_tokens"][i]
        cc = results["confidence_at_commit"][i]
        ec = results["entropy_at_commit"][i]

        tc_str = str(tc) if tc is not None else "-"
        ts_str = str(ts) if ts is not None else "-"
        gap_str = str(gap) if gap is not None else "-"

        print(
            f"{i:>4}  {tok:<16}  {tc_str:>10}  {ts_str:>10}  {gap_str:>5}  {cc:>11.4f}  {ec:>11.4f}"
        )

        token_records.append(
            {
                "position": i,
                "token": tok,
                "token_id": results["final_token_ids"][i],
                "tau_commit": tc,
                "tau_stable": ts,
                "max_refinement_step": results["max_refinement_step"][i],
                "gap": gap,
                "confidence_at_commit": round(cc, 6),
                "confidence_at_stable": round(results["confidence_at_stable"][i], 6),
                "entropy_at_commit": round(ec, 6),
                "entropy_at_stable": round(results["entropy_at_stable"][i], 6),
                "prob_trajectory": results["prob_trajectory"][i],
            }
        )

    print("-" * 80)
    print(f"\nGenerated: {' '.join(results['final_tokens'])}")

    # Summary stats
    committed_taus = [r["tau_commit"] for r in token_records if r["tau_commit"] is not None]
    stable_taus = [r["tau_stable"] for r in token_records if r["tau_stable"] is not None]
    gaps = [r["gap"] for r in token_records if r["gap"] is not None]

    print(f"\nSummary")
    print(f"  Tokens committed          : {len(committed_taus)}/{args.target_len}")
    print(f"  Tokens that stabilized    : {len(stable_taus)}/{args.target_len}")
    print(
        f"  Mean tau_commit           : {sum(committed_taus) / len(committed_taus):.1f}"
        if committed_taus
        else "  Mean tau_commit : -"
    )
    print(
        f"  Mean tau_stable           : {sum(stable_taus) / len(stable_taus):.1f}"
        if stable_taus
        else "  Mean tau_stable : -"
    )
    print(
        f"  Mean gap                  : {sum(gaps) / len(gaps):.1f}" if gaps else "  Mean gap : -"
    )

    # Save
    output = {
        "model": MODEL_ID,
        "prompt": args.prompt,
        "target_len": args.target_len,
        "denoising_steps": args.steps,
        "confidence_threshold": args.threshold,
        "generated": " ".join(results["final_tokens"]),
        "token_records": token_records,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

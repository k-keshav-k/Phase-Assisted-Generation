"""
One-pass probe script for LLaDA-8B.

Runs a single forward pass on a masked sequence and prints per-position signals:
  - top predicted token
  - confidence  (prob of top token)
  - entropy     (uncertainty across full vocab)

Saves output to probe_llada_output.json in the working directory.

Usage:
    python scripts/probe_llada.py
    python scripts/probe_llada.py --prompt "Your prompt here" --target_len 20
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "GSAI-ML/LLaDA-8B-Base"

PROMPT = "Explain why adaptive diffusion decoding can improve language generation quality."
TARGET_LEN = 20  # number of masked positions (tokens to generate)
CONFIDENCE_THRESHOLD = 0.85  # for computing tau_stable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--target_len", type=int, default=TARGET_LEN)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--output", type=str, default="probe_llada_output.json")
    return parser.parse_args()


def compute_entropy(probs: torch.Tensor) -> float:
    """Shannon entropy of a probability distribution."""
    return float(-(probs * torch.log(probs + 1e-10)).sum().item())


def main() -> None:
    args = parse_args()

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}\n")

    # Build input: [prompt tokens] [MASK ... MASK]
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    mask_token_id = tokenizer.mask_token_id
    mask_ids = torch.full((1, args.target_len), mask_token_id, device=device)
    input_ids = torch.cat([prompt_ids, mask_ids], dim=1)

    prompt_len = prompt_ids.shape[1]
    print(f"Prompt length : {prompt_len} tokens")
    print(f"Masked target : {args.target_len} positions")
    print(f"Total input   : {input_ids.shape[1]} tokens\n")

    # Single forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Extract signals for masked positions only
    masked_logits = logits[0, prompt_len:].float()           # (target_len, vocab_size)
    probs = torch.softmax(masked_logits, dim=-1)             # (target_len, vocab_size)
    confidence = probs.max(dim=-1).values                    # (target_len,)
    top_token_ids = probs.argmax(dim=-1)                     # (target_len,)

    # Decode top tokens
    top_tokens = [tokenizer.decode([tid.item()]).strip() for tid in top_token_ids]

    # Print header
    print(f"{'Pos':>4}  {'Token':<18}  {'Confidence':>10}  {'Entropy':>10}  {'tau_stable':>10}")
    print("-" * 62)

    records = []
    for i in range(args.target_len):
        p = probs[i]
        conf = float(confidence[i].item())
        ent = compute_entropy(p)
        tau_stable = i if conf >= args.threshold else None  # step 0 = this single pass

        record = {
            "position": i,
            "top_token": top_tokens[i],
            "top_token_id": int(top_token_ids[i].item()),
            "confidence": round(conf, 6),
            "entropy": round(ent, 6),
            "tau_stable": tau_stable,
            "above_threshold": conf >= args.threshold,
        }
        records.append(record)

        tau_str = str(tau_stable) if tau_stable is not None else "-"
        print(
            f"{i:>4}  {top_tokens[i]:<18}  {conf:>10.4f}  {ent:>10.4f}  {tau_str:>10}"
        )

    # Summary
    confident_count = sum(1 for r in records if r["above_threshold"])
    mean_conf = sum(r["confidence"] for r in records) / len(records)
    mean_ent = sum(r["entropy"] for r in records) / len(records)

    print("-" * 62)
    print(f"\nSummary")
    print(f"  Positions above threshold ({args.threshold}): {confident_count}/{args.target_len}")
    print(f"  Mean confidence : {mean_conf:.4f}")
    print(f"  Mean entropy    : {mean_ent:.4f}")

    # Save
    output = {
        "model": MODEL_ID,
        "prompt": args.prompt,
        "target_len": args.target_len,
        "confidence_threshold": args.threshold,
        "single_pass": True,
        "token_records": records,
        "summary": {
            "mean_confidence": round(mean_conf, 6),
            "mean_entropy": round(mean_ent, 6),
            "confident_positions": confident_count,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

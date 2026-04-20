"""
LLaDA-8B baseline probe on the Countdown dataset.

Loads the first N samples from Jiayi-Pan/Countdown-Tasks-3to4, runs the full
T-step masked diffusion denoising loop on each, and saves traces to a JSONL
file in the unified TraceRecord format consumed by phase_cpd.

Output format (one JSON object per line) matches phase_cpd's TraceRecord schema
so that load_step_dump_as_trace() can import each record directly:

  trace_id            : "countdown-llada-NNNN"
  backend             : "llada"
  model_name          : MODEL_ID
  prompt              : raw task prompt
  final_text          : space-joined generated tokens
  decoding_metadata   : algorithm settings + per-token commit/gap summary
  tokens[]            : TraceToken list, each with observations[]

Per-step observations (TokenStepObservation) recorded while token is masked:
  step_index          : denoising step (0-indexed)
  token_id            : argmax token id predicted at this step
  token_text          : decoded text of predicted token
  top1_prob           : max softmax probability  (= confidence)
  top2_prob           : second-highest softmax probability
  extras["entropy"]   : Shannon entropy of full distribution

Per-token summary stored in decoding_metadata["token_summaries"]:
  tau_commit          : step at which token was committed (unmasked)
  tau_stable          : first step predicted identity matched final AND stayed
                        stable — uses phase_cpd stabilization definition
  max_refinement_step : last step token was still masked (== tau_commit)
  gap                 : tau_commit - tau_stable

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

MODEL_ID        = "GSAI-ML/LLaDA-8B-Base"
DATASET_ID      = "Jiayi-Pan/Countdown-Tasks-3to4"
N_SAMPLES       = 50
DENOISING_STEPS = 64
TARGET_LEN      = 32   # tokens to generate per sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int, default=N_SAMPLES)
    parser.add_argument("--steps",      type=int, default=DENOISING_STEPS)
    parser.add_argument("--target_len", type=int, default=TARGET_LEN)
    parser.add_argument("--output",     type=str, default="countdown_baseline_traces.jsonl")
    parser.add_argument("--split",      type=str, default="train")
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


def _first_stable_step(
    predicted_ids: list[int],
    final_token_id: int,
) -> int:
    """Return the first step index where the predicted identity matches the
    final committed token AND stays stable for all subsequent steps.

    This is the stabilization definition used by phase_cpd feature extractors
    (StabilizingEntropyExtractor etc.) — NOT the argmax-of-confidence definition.

    If the token was always predicted as something else until the commit step,
    returns tau_commit (the last entry index).
    """
    n = len(predicted_ids)
    for idx in range(n):
        if predicted_ids[idx] == final_token_id:
            # Check that it stays stable from here to the end
            if all(predicted_ids[j] == final_token_id for j in range(idx, n)):
                return idx
    # Never stabilized — return the commit step (last observation)
    return n - 1


def run_denoising_loop(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    target_len: int,
    T: int,
    mask_token_id: int,
    device: torch.device,
) -> dict:
    """Run T denoising steps and collect per-token per-step observations.

    Returns a dict with:
      final_token_ids  : list[int]  committed token ids
      final_tokens     : list[str]  decoded committed tokens
      tau_commit       : list[int]  step at which each token was committed
      tau_stable       : list[int]  first stable step (phase_cpd definition)
      gap              : list[int]  tau_commit - tau_stable
      observations     : list[list[dict]]  per-token step-level observations
                         each observation: {step_index, token_id, token_text,
                                           top1_prob, top2_prob, entropy}
    """
    prompt_len = prompt_ids.shape[1]

    mask_ids  = torch.full((1, target_len), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, mask_ids], dim=1)

    tau_commit      = [None] * target_len
    committed       = [False] * target_len

    # Per-step observation log: observations[i] = list of dicts, one per masked step
    observations: list[list[dict]] = [[] for _ in range(target_len)]

    for step in range(T):
        masked_positions = [i for i in range(target_len) if not committed[i]]
        if not masked_positions:
            break

        with torch.no_grad():
            outputs = model(input_ids)
            logits  = outputs.logits[0, prompt_len:].float()

        probs   = torch.softmax(logits, dim=-1)
        top2    = probs.topk(2, dim=-1)
        top_ids = top2.indices[:, 0]   # argmax token id per position
        top1_p  = top2.values[:, 0]    # max probability
        top2_p  = top2.values[:, 1]    # second-highest probability

        for i in masked_positions:
            tid       = int(top_ids[i].item())
            tok_text  = tokenizer.decode([tid]).strip()
            t1p       = round(float(top1_p[i].item()), 6)
            t2p       = round(float(top2_p[i].item()), 6)
            ent       = round(compute_entropy(probs[i]), 6)

            observations[i].append({
                "step_index": step,
                "token_id":   tid,
                "token_text": tok_text,
                "top1_prob":  t1p,
                "top2_prob":  t2p,
                "extras":     {"entropy": ent},
            })

        remaining_steps = T - step
        n_to_unmask = math.ceil(len(masked_positions) / remaining_steps)
        ranked      = sorted(masked_positions, key=lambda i: top1_p[i].item(), reverse=True)
        to_unmask   = ranked[:n_to_unmask]

        for i in to_unmask:
            tau_commit[i]              = step
            input_ids[0, prompt_len + i] = top_ids[i]
            committed[i]               = True

    final_token_ids = input_ids[0, prompt_len:].tolist()
    final_tokens    = [tokenizer.decode([tid]).strip() for tid in final_token_ids]

    # Compute tau_stable using phase_cpd stabilization definition:
    # first step where predicted token_id == final token_id AND stays stable.
    tau_stable = []
    for i in range(target_len):
        predicted_ids = [obs["token_id"] for obs in observations[i]]
        stable_idx    = _first_stable_step(predicted_ids, final_token_ids[i])
        # stable_idx is an index into observations[i]; convert to actual step number
        tau_stable.append(observations[i][stable_idx]["step_index"])

    gap = [tau_commit[i] - tau_stable[i] for i in range(target_len)]

    return {
        "final_token_ids": final_token_ids,
        "final_tokens":    final_tokens,
        "tau_commit":      tau_commit,
        "tau_stable":      tau_stable,
        "gap":             gap,
        "observations":    observations,
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
                mask_token_id = mask_token_id,
                device        = device,
            )
            elapsed = time.time() - t0

            steps_used = max(
                (r for r in results["tau_commit"] if r is not None), default=0
            ) + 1

            nums_str = str(nums)
            print(f"{idx:>4}  {target:>6}  {nums_str:<20}  {steps_used:>5}  {elapsed:>5.1f}s")

            final_text = " ".join(results["final_tokens"])

            # ── Build TraceToken list (phase_cpd TraceRecord schema) ──────────
            # char offsets are approximate (space-separated tokens)
            tokens = []
            cursor = 0
            for i in range(args.target_len):
                tok_text  = results["final_tokens"][i]
                char_start = cursor
                char_end   = cursor + len(tok_text)
                cursor     = char_end + 1  # +1 for the space separator

                tokens.append({
                    "token_index": i,
                    "token_text":  tok_text,
                    "char_start":  char_start,
                    "char_end":    char_end,
                    "observations": results["observations"][i],
                })

            # Per-token summary kept in decoding_metadata for downstream analysis
            token_summaries = []
            for i in range(args.target_len):
                token_summaries.append({
                    "position":            i,
                    "token":               results["final_tokens"][i],
                    "token_id":            results["final_token_ids"][i],
                    "tau_commit":          results["tau_commit"][i],
                    "tau_stable":          results["tau_stable"][i],
                    "max_refinement_step": results["tau_commit"][i],  # last masked step == commit
                    "gap":                 results["gap"][i],
                })

            # ── Unified TraceRecord output (phase_cpd compatible) ─────────────
            record = {
                "trace_id":    f"countdown-llada-{idx:04d}",
                "backend":     "llada",
                "model_name":  MODEL_ID,
                "prompt":      prompt,
                "final_text":  final_text,
                "tokens":      tokens,
                "decoding_metadata": {
                    "dataset":          DATASET_ID,
                    "nums":             nums,
                    "target":           target,
                    "denoising_steps":  args.steps,
                    "target_len":       args.target_len,
                    "algorithm":        "greedy_confidence",
                    "token_summaries":  token_summaries,
                },
                "tags": ["countdown", "arithmetic", "baseline"],
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"\nDone. Traces saved to {args.output}")


if __name__ == "__main__":
    main()

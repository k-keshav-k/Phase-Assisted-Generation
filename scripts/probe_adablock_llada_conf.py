"""Confidence-aware probe for AdaBlock LLaDA generation.

Extends probe_adablock_llada.py by tracking per-step logit argmax and softmax
probability of the final token at every diffusion step for each block position.

Definitions
-----------
  refinement_step  : step at which the token is actually unmasked in x
                     (x[pos] first contains final_token_id and stays)

  stabilizing_step : earliest step where argmax(logits[pos]) == final_token_id
                     AND the token is still masked in x (x[pos] == MASK_ID).
                     Defaults to refinement_step when the model's argmax only
                     matches the final token at or after unmask time.

  stabilizing_step <= refinement_step  always.

  gap = refinement_step - stabilizing_step
      = how many steps before unmask the model was already predicting
        the correct token. gap=0 means the model was uncertain until commit.

Per-token output:
  "stabilizing_step": int            (see above)
  "refinement_step":  int            step token was unmasked in x
  "p_final_per_step": [p0, p1, ...]  p(final_token) via softmax at each step
  "step_token_ids":   [id0, id1...]  x contents at each step (MASK_ID until unmasked)
  "step_logit_ids":   [id0, id1...]  argmax of logits at each step

Usage
-----
    python scripts/probe_adablock_llada_conf.py \
        --gsm8k \
        --gsm8k-split test \
        --output-dir traces/adablock \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

_ADABLOCK_LLADA = Path(__file__).resolve().parents[1] / "AdaBlock-dLLM" / "llada"
if str(_ADABLOCK_LLADA) not in sys.path:
    sys.path.insert(0, str(_ADABLOCK_LLADA))

from generate_adablock import add_gumbel_noise, compute_block_length, get_transfer_index  # noqa: E402
from model.modeling_llada import LLaDAModelLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MASK_ID = 126336


# ---------------------------------------------------------------------------
# Instrumented generation (confidence-aware)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_adablock_conf_traced(
    model,
    tokenizer,
    prompt: torch.Tensor,
    *,
    gen_length: int = 128,
    init_block_length: int = 16,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    threshold: float = 0.9,
    delimiter_ids: list[int] | None = None,
    delimiter_threshold: float = 0.3,
) -> dict[str, Any]:
    """generate_adablock recording per-step argmax token AND its softmax probability."""
    if delimiter_ids is None:
        delimiter_ids = [198]

    x = torch.full(
        (1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    generated_length = 0
    nfe_history: list[int] = []
    block_history: list[int] = []
    block_traces: list[dict[str, Any]] = []

    while generated_length < gen_length:
        nfe = 0

        # Pass 1 — pick block boundary
        output = model(x)
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1

        block_length = compute_block_length(
            logits,
            predicted_tokens,
            prompt,
            gen_length,
            generated_length,
            init_block_length,
            delimiter_ids=delimiter_ids,
            delimiter_threshold=delimiter_threshold,
        )
        block_history.append(block_length)

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        # First transfer
        mask_index = x == MASK_ID
        mask_index[:, block_end:] = False
        x0, transfer_index = get_transfer_index(
            logits, predicted_tokens, remasking, mask_index, x, None, threshold
        )
        x[transfer_index] = x0[transfer_index]

        # ── step 0 snapshots ────────────────────────────────────────────────
        # step_snapshots   : what is in x (MASK_ID until token is unmasked)
        # step_raw_logits  : raw block logits at each step (CPU, float32) — used
        #                    post-block to compute p(final_token) per step
        raw0 = logits[0, block_start:block_end].float().cpu()
        step_snapshots:  list[torch.Tensor] = [x[0, block_start:block_end].clone()]
        step_raw_logits: list[torch.Tensor] = [raw0.clone()]

        # Inner refinement loop
        while True:
            if (x[:, block_start:block_end] == MASK_ID).sum() == 0:
                break
            mask_index = x == MASK_ID
            mask_index[:, block_end:] = False
            block_output = model(x)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(
                block_logits, block_predicted_tokens, remasking, mask_index, x, None, threshold
            )
            x[transfer_index] = x0[transfer_index]
            step_snapshots.append(x[0, block_start:block_end].clone())
            step_raw_logits.append(block_logits[0, block_start:block_end].float().cpu().clone())

        nfe_history.append(nfe)

        # ── post-block: compute p(final_token) at every step ────────────────
        # all_logits: (nfe, block_len, vocab)
        # all_probs:  (nfe, block_len, vocab)  — softmax over vocab
        final_token_ids = x[0, block_start:block_end].tolist()
        all_logits = torch.stack(step_raw_logits, dim=0)          # (nfe, block_len, vocab)
        all_probs  = torch.softmax(all_logits, dim=-1)             # (nfe, block_len, vocab)
        final_ids_t = torch.tensor(final_token_ids, dtype=torch.long)  # (block_len,)
        # p_final[s, tok] = p(final_token_ids[tok]) at step s
        p_final_mat = all_probs[:, torch.arange(block_length), final_ids_t]  # (nfe, block_len)
        # argmax logits per step per token (for step_logit_ids)
        argmax_mat  = all_logits.argmax(dim=-1)                    # (nfe, block_len)

        token_traces: list[dict[str, Any]] = []

        for tok_idx in range(block_length):
            final_id    = final_token_ids[tok_idx]
            x_history   = [int(snap[tok_idx].item()) for snap in step_snapshots]
            logit_ids   = argmax_mat[:, tok_idx].tolist()
            p_final     = p_final_mat[:, tok_idx].tolist()

            # refinement_step: step token was actually unmasked in x
            refinement_step = len(x_history) - 1
            for s, tok_id in enumerate(x_history):
                if tok_id == final_id and all(h == final_id for h in x_history[s:]):
                    refinement_step = s
                    break

            # stabilizing_step: first step where argmax(logits) == final_token while
            # still masked AND argmax stays == final_token for all remaining masked
            # steps until unmask. Transient flips (correct at step s, wrong at s+1)
            # are ignored. Defaults to refinement_step when no such stable run exists.
            # Guarantees stabilizing_step <= refinement_step.
            stabilizing_step = refinement_step
            for s, (logit_id, tok_in_x) in enumerate(zip(logit_ids, x_history)):
                if tok_in_x == MASK_ID and logit_id == final_id:
                    if all(
                        x_h != MASK_ID or lid == final_id
                        for lid, x_h in zip(logit_ids[s:], x_history[s:])
                    ):
                        stabilizing_step = s
                        break

            token_traces.append({
                "token_index":      tok_idx,
                "token_id":         final_id,
                "token_text":       tokenizer.decode([final_id], clean_up_tokenization_spaces=False),
                "stabilizing_step": stabilizing_step,               # first step argmax==final while masked
                "refinement_step":  refinement_step,                # step token was unmasked in x
                "p_final_per_step": [round(p, 6) for p in p_final], # p(final_token) at each step
                "step_token_ids":   x_history,                      # x contents per step (MASK_ID until unmasked)
                "step_logit_ids":   [int(i) for i in logit_ids],    # argmax logit per step
            })

        block_traces.append({
            "block_index": len(block_traces),
            "block_size":  block_length,
            "nfe":         nfe,
            "tokens":      token_traces,
        })

        if eot_id is not None and any(t["token_id"] == eot_id for t in token_traces):
            break

    final_text = tokenizer.decode(
        x[0, prompt.shape[1]:].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return {
        "generated_text": final_text,
        "nfe_history":    nfe_history,
        "block_history":  block_history,
        "blocks":         block_traces,
        "total_nfe":      sum(nfe_history),
        "num_blocks":     len(block_history),
        "avg_block_size": sum(block_history) / len(block_history) if block_history else 0,
        "avg_nfe":        sum(nfe_history) / len(nfe_history) if nfe_history else 0,
    }


# ---------------------------------------------------------------------------
# I/O helpers (identical to probe_adablock_llada.py)
# ---------------------------------------------------------------------------

def _load_prompts_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _load_gsm8k(split: str = "test", limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        {
            "sample_id": f"gsm8k-{split}-{i:04d}",
            "prompt": ex["question"],
            "reference_answer": ex["answer"],
            "dataset": "gsm8k",
            "split": split,
        }
        for i, ex in enumerate(ds)
    ]


def _build_input(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return torch.tensor(tokenizer(text)["input_ids"]).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confidence-aware AdaBlock probe: records per-step softmax probs."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompts", type=Path)
    source.add_argument("--gsm8k", action="store_true")
    parser.add_argument("--gsm8k-split", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("traces/adablock"))
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--init-block-length", type=int, default=16)
    parser.add_argument("--delimiter-threshold", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output filename (overrides default). E.g. my_traces.jsonl")
    args = parser.parse_args(argv)

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print(f"Loading model {args.model} ...")
    model = LLaDAModelLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Model loaded.")

    if args.gsm8k:
        print(f"Loading GSM8K ({args.gsm8k_split}, limit={args.limit}) ...")
        prompts = _load_gsm8k(split=args.gsm8k_split, limit=args.limit)
        default_filename = f"gsm8k_{args.gsm8k_split}_conf_traces.jsonl"
    else:
        prompts_path = args.prompts or Path("phase_cpd/data/prompts/research_prompts.jsonl")
        prompts = _load_prompts_jsonl(prompts_path)
        if args.limit:
            prompts = prompts[: args.limit]
        default_filename = "adablock_llada_conf_traces.jsonl"

    out_filename = args.output_file if args.output_file else default_filename
    print(f"Loaded {len(prompts)} prompts.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / out_filename

    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["sample_id"])
        print(f"Resuming — {len(done_ids)} already done.")

    with out_path.open("a", encoding="utf-8") as f:
        for i, record in enumerate(prompts):
            sample_id = record.get("sample_id", f"sample-{i:04d}")
            if sample_id in done_ids:
                continue

            prompt_text = record["prompt"]
            print(f"  [{i+1}/{len(prompts)}] [{sample_id}] {prompt_text[:60]}...")

            input_ids = _build_input(tokenizer, prompt_text).to(device)

            result = generate_adablock_conf_traced(
                model,
                tokenizer,
                input_ids,
                gen_length=args.gen_length,
                init_block_length=args.init_block_length,
                temperature=args.temperature,
                threshold=args.threshold,
                delimiter_threshold=args.delimiter_threshold,
            )

            trace = {
                "sample_id":   sample_id,
                "prompt":      prompt_text,
                "model_name":  args.model,
                "dataset":     record.get("dataset", "custom"),
                "reference_answer": record.get("reference_answer"),
                "created_at":  datetime.now(timezone.utc).isoformat(),
                "decoding_config": {
                    "gen_length":          args.gen_length,
                    "init_block_length":   args.init_block_length,
                    "delimiter_threshold": args.delimiter_threshold,
                    "threshold":           args.threshold,
                    "temperature":         args.temperature,
                },
                **result,
            }
            f.write(json.dumps(trace) + "\n")
            f.flush()

            print(
                f"    → {result['num_blocks']} blocks | "
                f"avg block size {result['avg_block_size']:.1f} | "
                f"total NFE {result['total_nfe']} | "
                f"avg NFE/block {result['avg_nfe']:.2f}"
            )

    print(f"\nTraces written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Probe script for AdaBlock LLaDA generation.

Runs generate_adablock on a prompt file and records:
  - block_history: block size per block
  - nfe_history: model forward calls (refinement steps) per block
  - per-token stabilizing step within each block

Output: one JSONL file where each line is a full trace for one prompt.

Usage
-----
    python scripts/probe_adablock_llada.py \
        --prompts phase_cpd/data/prompts/research_prompts.jsonl \
        --output-dir traces/adablock \
        --model GSAI-ML/LLaDA-8B-Instruct \
        --gen-length 128 \
        --init-block-length 16 \
        --delimiter-threshold 0.3 \
        --threshold 0.9 \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Make AdaBlock llada helpers importable
_ADABLOCK_LLADA = Path(__file__).resolve().parents[1] / "AdaBlock-dLLM" / "llada"
if str(_ADABLOCK_LLADA) not in sys.path:
    sys.path.insert(0, str(_ADABLOCK_LLADA))

from generate_adablock import add_gumbel_noise, compute_block_length, get_transfer_index  # noqa: E402
from model.modeling_llada import LLaDAModelLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MASK_ID = 126336


# ---------------------------------------------------------------------------
# Instrumented generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_adablock_traced(
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
    """generate_adablock with per-token step history for stabilizing step extraction."""
    if delimiter_ids is None:
        delimiter_ids = [198]  # newline token

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

        # Full-sequence forward pass to pick block boundary
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

        # First transfer pass
        mask_index = x == MASK_ID
        mask_index[:, block_end:] = False
        x0, transfer_index = get_transfer_index(
            logits, predicted_tokens, remasking, mask_index, x, None, threshold
        )
        x[transfer_index] = x0[transfer_index]

        # Snapshot after first pass (step 0)
        step_snapshots: list[torch.Tensor] = [x[0, block_start:block_end].clone()]

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

        nfe_history.append(nfe)

        # Build per-token trace for this block
        final_token_ids = x[0, block_start:block_end].tolist()
        token_traces: list[dict[str, Any]] = []
        for tok_idx in range(block_length):
            history = [int(snap[tok_idx].item()) for snap in step_snapshots]
            final_id = final_token_ids[tok_idx]
            # First step at which the token matches final and never changes after
            stab_step = len(history) - 1
            for s, tok_id in enumerate(history):
                if tok_id == final_id and all(h == final_id for h in history[s:]):
                    stab_step = s
                    break
            token_traces.append({
                "token_index": tok_idx,
                "token_id": final_id,
                "token_text": tokenizer.decode(
                    [final_id], clean_up_tokenization_spaces=False
                ),
                "stabilizing_step": stab_step,
                "step_token_ids": history,
            })

        block_traces.append({
            "block_index": len(block_traces),
            "block_size": block_length,
            "nfe": nfe,
            "tokens": token_traces,
        })

        # stop early if eot was generated in this block
        if eot_id is not None and any(t["token_id"] == eot_id for t in token_traces):
            break

    final_text = tokenizer.decode(
        x[0, prompt.shape[1]:].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return {
        "generated_text": final_text,
        "nfe_history": nfe_history,
        "block_history": block_history,
        "blocks": block_traces,
        "total_nfe": sum(nfe_history),
        "num_blocks": len(block_history),
        "avg_block_size": sum(block_history) / len(block_history) if block_history else 0,
        "avg_nfe": sum(nfe_history) / len(nfe_history) if nfe_history else 0,
    }


# ---------------------------------------------------------------------------
# I/O helpers
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
    records = []
    for i, example in enumerate(ds):
        records.append({
            "sample_id": f"gsm8k-{split}-{i:04d}",
            "prompt": example["question"],
            "reference_answer": example["answer"],
            "dataset": "gsm8k",
            "split": split,
        })
    return records


def _build_input(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe AdaBlock LLaDA: record block_history, nfe_history, and stabilizing steps."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--prompts",
        type=Path,
        help="JSONL prompt file (default if neither --prompts nor --gsm8k given).",
    )
    source.add_argument(
        "--gsm8k",
        action="store_true",
        help="Load prompts from GSM8K (openai/gsm8k on HuggingFace).",
    )
    parser.add_argument(
        "--gsm8k-split", default="test", choices=["train", "test"],
        help="GSM8K split to use (default: test).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("traces/adablock"))
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--init-block-length", type=int, default=16)
    parser.add_argument("--delimiter-threshold", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
        out_filename = f"gsm8k_{args.gsm8k_split}_traces.jsonl"
    else:
        prompts_path = args.prompts or Path("phase_cpd/data/prompts/research_prompts.jsonl")
        prompts = _load_prompts_jsonl(prompts_path)
        if args.limit:
            prompts = prompts[: args.limit]
        out_filename = "adablock_llada_traces.jsonl"

    print(f"Loaded {len(prompts)} prompts.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / out_filename

    # resume: skip already-written sample_ids
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

            result = generate_adablock_traced(
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
                "sample_id": sample_id,
                "prompt": prompt_text,
                "model_name": args.model,
                "dataset": record.get("dataset", "custom"),
                "reference_answer": record.get("reference_answer"),
                "created_at": datetime.now(UTC).isoformat(),
                "decoding_config": {
                    "gen_length": args.gen_length,
                    "init_block_length": args.init_block_length,
                    "delimiter_threshold": args.delimiter_threshold,
                    "threshold": args.threshold,
                    "temperature": args.temperature,
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

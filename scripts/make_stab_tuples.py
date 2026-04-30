"""Convert AdaBlock trace JSONL → stabilizing-step phase_predict JSONL.

Same structure as make_phase_tuples.py but the per-block signal is the
stabilizing step (how many diffusion steps until each token locked in)
rather than NFE.

Each output line is one problem:
  {
    "sample_id": "gsm8k-train-0000",
    "dataset":   "gsm8k",
    "split":     "train",
    "tuples": [
      {
        "block_size":     16,
        "nfe":            9,
        "mean_stab_step": 1.2,   # mean step argmax==final while masked (across tokens)
        "max_stab_step":  4,     # latest token to reach stable argmax prediction
        "mean_ref_step":  3.8,   # mean step token was actually unmasked in x
        "max_ref_step":   8,     # latest token to be unmasked
        "mean_gap":       2.6,   # mean (refinement - stabilizing) per token
        "max_gap":        5,     # max gap in block
      },
      ...
    ]
  }

Definitions:
  stabilizing_step: first step argmax(logits)==final_token while token still masked,
                    AND argmax stays correct for all remaining masked steps (no transient flips).
                    stabilizing_step <= refinement_step always.
  refinement_step:  step the token was actually unmasked in x.
  gap:              refinement_step - stabilizing_step (how many steps between
                    model committing internally vs AdaBlock unmasking externally).

By default all blocks are kept. Pass --no-delimiters to drop delimiter blocks
(size=1, only special tokens) which have trivially zero stabilizing steps.

Usage
-----
    python scripts/make_stab_tuples.py \
        --traces traces/adablock/gsm8k_train_traces.jsonl \
        --output traces/adablock/stab_tuples_train.jsonl

    python scripts/make_stab_tuples.py \
        --traces traces/adablock/gsm8k_test_traces.jsonl \
        --output traces/adablock/stab_tuples_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECIAL_TOKENS = {"<|endoftext|>", "<|eot_id|>"}


def _is_content_block(block: dict) -> bool:
    return any(t["token_text"] not in SPECIAL_TOKENS for t in block["tokens"])


def _block_tuple(block: dict) -> dict:
    stab_steps = [t["stabilizing_step"] for t in block["tokens"]]
    ref_steps  = [t["refinement_step"]  for t in block["tokens"]]
    gaps = [r - s for r, s in zip(ref_steps, stab_steps)]
    n = len(block["tokens"])
    return {
        "block_size":      block["block_size"],
        "nfe":             block["nfe"],
        # stabilizing step: first step argmax==final while still masked
        "mean_stab_step":  round(sum(stab_steps) / n, 4) if n else 0.0,
        "max_stab_step":   max(stab_steps) if stab_steps else 0,
        # refinement step: step token was actually unmasked in x
        "mean_ref_step":   round(sum(ref_steps) / n, 4) if n else 0.0,
        "max_ref_step":    max(ref_steps) if ref_steps else 0,
        # gap = refinement - stabilizing: steps between model deciding and committing
        "mean_gap":        round(sum(gaps) / n, 4) if n else 0.0,
        "max_gap":         max(gaps) if gaps else 0,
    }


def convert(traces_path: Path, output_path: Path, keep_delimiters: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_problems = 0
    n_tuples = 0
    n_dropped = 0

    with traces_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            trace = json.loads(line)

            if keep_delimiters:
                blocks = trace["blocks"]
            else:
                blocks = [b for b in trace["blocks"] if _is_content_block(b)]
                n_dropped += len(trace["blocks"]) - len(blocks)

            if not blocks:
                continue

            record = {
                "sample_id": trace["sample_id"],
                "dataset":   trace.get("dataset", "unknown"),
                "split":     trace.get("decoding_config", {}).get("split", "unknown"),
                "tuples":    [_block_tuple(b) for b in blocks],
            }
            dst.write(json.dumps(record) + "\n")
            n_problems += 1
            n_tuples += len(blocks)

    print(f"Problems written : {n_problems}")
    print(f"Tuples written   : {n_tuples}")
    print(f"Delimiter blocks dropped: {n_dropped}")
    print(f"Output → {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Traces → stabilizing-step phase_predict JSONL")
    parser.add_argument("--traces", type=Path, required=True, help="Input traces JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output stab_tuples JSONL")
    parser.add_argument(
        "--no-delimiters",
        action="store_true",
        help="Drop delimiter blocks (size=1, all special tokens).",
    )
    args = parser.parse_args(argv)
    convert(args.traces, args.output, keep_delimiters=not args.no_delimiters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

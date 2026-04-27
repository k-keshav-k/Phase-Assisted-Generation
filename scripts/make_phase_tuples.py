"""Convert AdaBlock trace JSONL → phase_predict training JSONL.

Each output line is one problem:
  {
    "sample_id": "gsm8k-train-0000",
    "dataset":   "gsm8k",
    "split":     "train",
    "tuples":    [{"block_size": 16, "nfe": 4}, ...]   # content blocks only
  }

Delimiter blocks (size=1, nfe=1, only special tokens) are dropped — they carry
no predictive signal and make the NFE distribution bimodal.

Usage
-----
    python scripts/make_phase_tuples.py \
        --traces  traces/adablock/gsm8k_train_traces.jsonl \
        --output  traces/adablock/phase_tuples_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECIAL_TOKENS = {"<|endoftext|>", "<|eot_id|>"}


def _is_content_block(block: dict) -> bool:
    """True if the block has at least one non-special token."""
    return any(t["token_text"] not in SPECIAL_TOKENS for t in block["tokens"])


def convert(traces_path: Path, output_path: Path) -> None:
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

            content_blocks = [b for b in trace["blocks"] if _is_content_block(b)]
            dropped = len(trace["blocks"]) - len(content_blocks)
            n_dropped += dropped

            if not content_blocks:
                continue

            record = {
                "sample_id": trace["sample_id"],
                "dataset": trace.get("dataset", "unknown"),
                "split": trace.get("decoding_config", {}).get("split", "unknown"),
                "tuples": [
                    {"block_size": b["block_size"], "nfe": b["nfe"]}
                    for b in content_blocks
                ],
            }
            dst.write(json.dumps(record) + "\n")
            n_problems += 1
            n_tuples += len(content_blocks)

    print(f"Problems written : {n_problems}")
    print(f"Content tuples   : {n_tuples}")
    print(f"Delimiter blocks dropped: {n_dropped}")
    print(f"Output → {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Traces → phase_predict training JSONL")
    parser.add_argument("--traces", type=Path, required=True, help="Input traces JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output phase_tuples JSONL")
    args = parser.parse_args(argv)
    convert(args.traces, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

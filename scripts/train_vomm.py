"""Train and evaluate the Variable-Order Markov Model (VOMM) baseline.

This script mirrors the PhaseTransformer training entrypoint in spirit: it
loads `phase_tuples` JSONL sequences, fits a VOMM, evaluates on a held-out
validation set (or provided test JSONL), and saves the model checkpoint and
metrics for comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_predict.schema import PhaseTuple
from phase_predict.vomm import VariableOrderMarkovModel
from phase_predict.data_utils import tuple_sequences_from_phase_tuples_jsonl


def load_sequences(path: Path) -> List[List[PhaseTuple]]:
    jsonl_paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL files found in {path}")
    sequences = []
    for p in jsonl_paths:
        seqs = tuple_sequences_from_phase_tuples_jsonl(p)
        sequences.extend(seqs)
    return sequences


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate VOMM baseline.")
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("traces/phase_tuples_train.jsonl"),
        help="Path to training JSONL file or directory.",
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=None,
        help="Path to test JSONL file or directory (optional).",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=4,
        help="Maximum Markov order (number of previous tuples to consider).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/vomm.pt"),
        help="Path to save trained VOMM checkpoint.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("output/vomm_metrics.json"),
        help="Path to save evaluation metrics JSON.",
    )
    args = parser.parse_args(argv)

    if not args.train_jsonl.exists():
        parser.error(f"Training JSONL not found: {args.train_jsonl}")

    # Load data
    print(f"Loading training sequences from: {args.train_jsonl}")  # noqa: T201
    train_sequences = load_sequences(args.train_jsonl)

    if args.test_jsonl is not None and args.test_jsonl.exists():
        print(f"Loading test sequences from: {args.test_jsonl}")  # noqa: T201
        test_sequences = load_sequences(args.test_jsonl)
    else:
        # 80/20 split
        split_idx = max(1, int(len(train_sequences) * 0.8))
        split_idx = min(split_idx, len(train_sequences) - 1)
        test_sequences = train_sequences[split_idx:]
        train_sequences = train_sequences[:split_idx]
        print(f"Auto-split: {len(train_sequences)} train, {len(test_sequences)} val")  # noqa: T201

    if not train_sequences:
        print("ERROR: no train sequences available", file=sys.stderr)
        sys.exit(1)

    # Train VOMM
    print(f"\nFitting Variable-Order Markov Model (max_order={args.max_order})...")  # noqa: T201
    start = time.time()
    model = VariableOrderMarkovModel(max_order=args.max_order)
    model.fit(train_sequences)
    elapsed = time.time() - start
    print(f"Fitting complete ({elapsed:.1f}s)")  # noqa: T201

    # Evaluate
    val_mse = model.evaluate_mse(test_sequences)
    print(f"Validation MSE: {val_mse:.6f}")  # noqa: T201

    # Save model and metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))

    metrics = {
        "max_order": args.max_order,
        "train_sequences": len(train_sequences),
        "val_sequences": len(test_sequences),
        "val_mse": val_mse,
        "fit_time_sec": elapsed,
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved checkpoint: {args.output}")  # noqa: T201
    print(f"Saved metrics: {args.metrics_out}")  # noqa: T201


if __name__ == "__main__":
    main()

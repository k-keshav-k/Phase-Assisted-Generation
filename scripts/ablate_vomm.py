"""Run simple ablations for Variable-Order Markov Model (VOMM).

This script tries different `max_order` values, fits a VOMM on the training
sequences, evaluates on validation/test sequences, saves each checkpoint and
writes a CSV/JSON summary. The best model (lowest validation MSE) is reported
and its checkpoint path printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_predict.vomm import VariableOrderMarkovModel
from phase_predict.data_utils import tuple_sequences_from_phase_tuples_jsonl
from phase_predict.schema import PhaseTuple


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
    parser = argparse.ArgumentParser(description="Ablate VOMM max_order hyperparameter.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("traces/phase_tuples_train.jsonl"),
                        help="Training JSONL file or directory")
    parser.add_argument("--test-jsonl", type=Path, default=Path("traces/phase_tuples_test.jsonl"),
                        help="Test JSONL file or directory (optional)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/vomm_ablation"),
                        help="Directory to save checkpoints and results")
    parser.add_argument("--orders", type=int, nargs="+", default=[1,2,4,10,20,30],
                        help="List of max_order values to try")
    args = parser.parse_args(argv)

    if not args.train_jsonl.exists():
        parser.error(f"Training JSONL not found: {args.train_jsonl}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train sequences from {args.train_jsonl}")  # noqa: T201
    train_sequences = load_sequences(args.train_jsonl)

    if args.test_jsonl and args.test_jsonl.exists():
        print(f"Loading test sequences from {args.test_jsonl}")  # noqa: T201
        val_sequences = load_sequences(args.test_jsonl)
    else:
        split_idx = max(1, int(len(train_sequences) * 0.8))
        split_idx = min(split_idx, len(train_sequences) - 1)
        val_sequences = train_sequences[split_idx:]
        train_sequences = train_sequences[:split_idx]
        print(f"Auto-split: {len(train_sequences)} train, {len(val_sequences)} val")  # noqa: T201

    results = []
    best = None

    for order in args.orders:
        run_id = f"vomm_order{order}"
        print(f"\nRunning {run_id}")  # noqa: T201
        start = time.time()

        model = VariableOrderMarkovModel(max_order=order)
        model.fit(train_sequences)
        fit_time = time.time() - start

        val_mse = model.evaluate_mse(val_sequences)

        ckpt_path = args.output_dir / f"{run_id}.pt"
        model.save(str(ckpt_path))

        entry = {
            "run_id": run_id,
            "max_order": order,
            "val_mse": float(val_mse),
            "fit_time_sec": float(fit_time),
            "checkpoint": str(ckpt_path),
        }
        results.append(entry)

        print(f"  val_mse={val_mse:.6f}, time={fit_time:.1f}s, ckpt={ckpt_path}")  # noqa: T201

        if best is None or (not (entry["val_mse"] != entry["val_mse"]) and entry["val_mse"] < best["val_mse"]):
            best = entry

    # save CSV and JSON
    csv_path = args.output_dir / "vomm_ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["run_id", "max_order", "val_mse", "fit_time_sec", "checkpoint"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    json_path = args.output_dir / "vomm_ablation_results.json"
    with open(json_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"\nAblation finished. Results saved to {csv_path} and {json_path}")  # noqa: T201
    if best:
        print(f"Best: {best['run_id']} with val_mse={best['val_mse']:.6f}")  # noqa: T201
        print(best["checkpoint"])  # print path for scripting convenience


if __name__ == "__main__":
    main()

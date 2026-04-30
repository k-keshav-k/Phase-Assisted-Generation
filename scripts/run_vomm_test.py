"""Load a VOMM checkpoint and run test/evaluation and sample predictions.

Usage::

    python scripts/run_vomm_test.py --checkpoint output/vomm.pt

If `--test-jsonl` is not provided, the script will attempt to auto-split the
training file (same behaviour as `train_vomm.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

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
    parser = argparse.ArgumentParser(description="Load VOMM checkpoint and evaluate/test.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to VOMM checkpoint (.pt)")
    parser.add_argument("--sample", action="store_true", help="Run sample predictions")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    model = VariableOrderMarkovModel.load(str(args.checkpoint))
    print(f"Loaded VOMM checkpoint (max_order={model.max_order}): {args.checkpoint}")  # noqa: T201

    if args.sample:
        samples = [
            [PhaseTuple(15, 4), PhaseTuple(16, 7), PhaseTuple(16, 9), PhaseTuple(16, 11), PhaseTuple(14, 8), PhaseTuple(15, 3), PhaseTuple(8, 3), PhaseTuple(1, 1)],
            [PhaseTuple(26, 4), PhaseTuple(1, 1), PhaseTuple(14, 9), PhaseTuple(7, 4), PhaseTuple(8, 4), PhaseTuple(15, 3), PhaseTuple(8, 3)],
        ]
        for ctx in samples:
            pred = model.predict(ctx)
            print(*ctx, sep="\n")
            print(f"Predicted: {pred}\n")  # noqa: T201


if __name__ == "__main__":
    main()

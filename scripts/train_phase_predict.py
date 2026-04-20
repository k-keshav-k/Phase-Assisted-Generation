"""Train and evaluate the PhaseTransformer on phase_cpd trace data.

Usage
-----
Run from the repository root::

    python scripts/train_phase_predict.py

Optional flags::

    --trace-dir PATH      path to trace JSON directory (default: phase_cpd default)
    --window-size N       context window size (default: 8)
    --epochs N            max training epochs (default: 100)
    --lr LR               learning rate (default: 1e-3)
    --output PATH         where to save the trained model checkpoint (default: phase_predict.pt)
    --per-token           use one tuple per token instead of one per CPD segment

The script:
  1. Loads all traces from the trace directory.
  2. Extracts PhaseTuple sequences (one per CPD segment, or one per token).
  3. Concatenates sequences across all traces.
  4. Trains the PhaseTransformer with early stopping.
  5. Reports validation loss and saves the checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_cpd.catalog import default_trace_dir, list_catalog_entries
from phase_cpd.cpd import CPDParameters, PeltDetector
from phase_cpd.features import get_feature_extractor
from phase_cpd.io import load_trace

from phase_predict.data_utils import tuples_from_trace
from phase_predict.dataset import PhaseSequenceDataset
from phase_predict.model import PhaseTransformer
from phase_predict.predict import Predictor
from phase_predict.schema import ModelConfig, PhaseTuple, TrainConfig
from phase_predict.train import Trainer


def _extract_tuples_from_traces(
    trace_dir: Path,
    *,
    per_token: bool = False,
) -> list[PhaseTuple]:
    """Load all traces and extract PhaseTuple sequences."""
    entries = list_catalog_entries(trace_dir)
    all_tuples: list[PhaseTuple] = []

    detector = PeltDetector()
    cpd_params = CPDParameters(cost="l2", penalty=0.1, min_segment_length=2, smoothing_window=3)

    for entry in entries:
        try:
            trace = load_trace(entry.path)
            if per_token:
                tuples = tuples_from_trace(trace)
            else:
                # try to get CPD breakpoints; fall back to per-token on failure
                try:
                    extractor = get_feature_extractor("stabilizing_prob")
                    if not extractor.is_available(trace):
                        extractor = get_feature_extractor("stabilizing_entropy")
                    if not extractor.is_available(trace):
                        tuples = tuples_from_trace(trace)
                    else:
                        feature_series = extractor.extract(trace)
                        breakpoints = detector.detect(feature_series.values, cpd_params)
                        tuples = tuples_from_trace(trace, breakpoints=breakpoints)
                except Exception:
                    tuples = tuples_from_trace(trace)
            all_tuples.extend(tuples)
            print(f"  Loaded {entry.trace_id}: {len(tuples)} tuples")  # noqa: T201
        except Exception as exc:
            print(f"  Skipping {entry.trace_id}: {exc}", file=sys.stderr)  # noqa: T201

    return all_tuples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PhaseTransformer on phase_cpd data.")
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="Path to trace JSON directory.")
    parser.add_argument("--window-size", type=int, default=8,
                        help="Context window size (number of past tuples).")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output", type=str, default="phase_predict.pt",
                        help="Path to save the checkpoint.")
    parser.add_argument("--per-token", action="store_true",
                        help="Use one tuple per token (instead of per CPD segment).")
    args = parser.parse_args(argv)

    trace_dir = args.trace_dir or default_trace_dir()
    print(f"Loading traces from: {trace_dir}")  # noqa: T201

    all_tuples = _extract_tuples_from_traces(trace_dir, per_token=args.per_token)
    print(f"\nTotal tuples extracted: {len(all_tuples)}")  # noqa: T201

    if len(all_tuples) < args.window_size + 2:
        print(  # noqa: T201
            f"ERROR: Not enough tuples ({len(all_tuples)}) for window_size={args.window_size}. "
            "Try --per-token or add more traces.",
            file=sys.stderr,
        )
        sys.exit(1)

    model_cfg = ModelConfig(window_size=args.window_size)
    dataset = PhaseSequenceDataset(all_tuples, model_cfg)

    model = PhaseTransformer(model_cfg)
    train_cfg = TrainConfig(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        log_interval=10,
    )

    print(f"\nTraining PhaseTransformer for up to {args.epochs} epochs …")  # noqa: T201
    trainer = Trainer(model, train_cfg)
    history = trainer.fit(dataset)

    print(  # noqa: T201
        f"\nTraining complete. "
        f"Best val loss = {history.best_val_loss:.6f} at epoch {history.best_epoch}."
    )

    predictor = Predictor(model, mean=dataset.mean, std=dataset.std)
    predictor.save_checkpoint(args.output)
    print(f"Checkpoint saved to: {args.output}")  # noqa: T201

    # quick sanity-check prediction
    if len(all_tuples) >= args.window_size:
        context = all_tuples[-args.window_size:]
        result = predictor.predict(context)
        print(f"\nSample prediction (last {args.window_size} tuples → next):")  # noqa: T201
        print(f"  predicted: {result.predicted_tuple}")  # noqa: T201


if __name__ == "__main__":
    main()

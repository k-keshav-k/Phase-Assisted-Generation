"""Train and evaluate the PhaseTransformer on trace data.

Usage
-----
Run from the repository root::

    python scripts/train_phase_predict.py

Optional flags::

    --trace-dir PATH      path to trace JSON directory (default: phase_cpd default)
    --trace-jsonl PATH    path to trace JSONL file or directory of JSONL files
    --window-size N       context window size (default: 8)
    --whole-sequence      train on each full trace sequence instead of sliding windows
    --epochs N            max training epochs (default: 100)
    --lr LR               learning rate (default: 1e-3)
    --output PATH         where to save the trained model checkpoint (default: phase_predict.pt)
    --per-token           use one tuple per token instead of one per CPD segment

    Example with JSONL input:
    python scripts/train_phase_predict.py \
      --trace-jsonl /path/to/traces.jsonl \
      --whole-sequence \
      --epochs 50 \
      --lr 5e-4 \
      --output output/phase_predict_checkpoint.pt

The script:
    1. Loads traces from a phase_cpd trace directory or trace JSONL file(s).
    2. Extracts PhaseTuple sequences (one per CPD segment, one per token,
         or directly from JSONL token summaries).
    3. Concatenates sequences across all traces, or keeps one full sequence per
         trace when --whole-sequence is enabled.
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
from phase_predict.data_utils import tuple_sequences_from_trace_jsonl
from phase_predict.dataset import PhaseFullSequenceDataset
from phase_predict.dataset import PhaseSequenceDataset
from phase_predict.model import PhaseTransformer
from phase_predict.predict import Predictor
from phase_predict.schema import ModelConfig, PhaseTuple, TrainConfig
from phase_predict.train import Trainer


def _extract_tuples_from_traces(
    trace_dir: Path,
    *,
    per_token: bool = False,
    cpd_penalty: float = 0.1,
    cpd_min_segment: int = 2,
    cpd_smoothing: int = 3,
) -> list[PhaseTuple]:
    """Load all traces and extract PhaseTuple sequences."""
    entries = list_catalog_entries(trace_dir)
    all_tuples: list[PhaseTuple] = []

    detector = PeltDetector()
    cpd_params = CPDParameters(
        cost="l2",
        penalty=cpd_penalty,
        min_segment_length=cpd_min_segment,
        smoothing_window=cpd_smoothing,
    )

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


def _extract_sequences_from_traces(
    trace_dir: Path,
    *,
    per_token: bool = False,
    cpd_penalty: float = 0.1,
    cpd_min_segment: int = 2,
    cpd_smoothing: int = 3,
) -> list[list[PhaseTuple]]:
    """Load traces and return one tuple sequence per trace."""
    entries = list_catalog_entries(trace_dir)
    sequences: list[list[PhaseTuple]] = []

    detector = PeltDetector()
    cpd_params = CPDParameters(
        cost="l2",
        penalty=cpd_penalty,
        min_segment_length=cpd_min_segment,
        smoothing_window=cpd_smoothing,
    )

    for entry in entries:
        try:
            trace = load_trace(entry.path)
            if per_token:
                tuples = tuples_from_trace(trace)
            else:
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
            sequences.append(tuples)
            print(f"  Loaded {entry.trace_id}: {len(tuples)} tuples")  # noqa: T201
        except Exception as exc:
            print(f"  Skipping {entry.trace_id}: {exc}", file=sys.stderr)  # noqa: T201

    return sequences


def _extract_tuples_from_trace_jsonl(trace_path: Path) -> list[PhaseTuple]:
    """Load tuple sequences from trace JSONL file(s) and flatten them."""
    jsonl_paths = [trace_path] if trace_path.is_file() else sorted(trace_path.glob("*.jsonl"))
    if not jsonl_paths:
        msg = f"No JSONL trace files were found in {trace_path}"
        raise FileNotFoundError(msg)

    all_tuples: list[PhaseTuple] = []
    for path in jsonl_paths:
        sequences = tuple_sequences_from_trace_jsonl(path)
        file_tuple_count = sum(len(sequence) for sequence in sequences)
        print(f"  Loaded {path.name}: {file_tuple_count} tuples across {len(sequences)} sequences")  # noqa: T201
        for sequence in sequences:
            all_tuples.extend(sequence)

    return all_tuples


def _extract_sequences_from_trace_jsonl(trace_path: Path) -> list[list[PhaseTuple]]:
    """Load tuple sequences from trace JSONL file(s) without flattening."""
    jsonl_paths = [trace_path] if trace_path.is_file() else sorted(trace_path.glob("*.jsonl"))
    if not jsonl_paths:
        msg = f"No JSONL trace files were found in {trace_path}"
        raise FileNotFoundError(msg)

    sequences: list[list[PhaseTuple]] = []
    for path in jsonl_paths:
        file_sequences = tuple_sequences_from_trace_jsonl(path)
        file_tuple_count = sum(len(sequence) for sequence in file_sequences)
        print(f"  Loaded {path.name}: {file_tuple_count} tuples across {len(file_sequences)} sequences")  # noqa: T201
        sequences.extend(file_sequences)

    return sequences


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PhaseTransformer on phase_cpd data.")
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="Path to trace JSON directory.")
    parser.add_argument("--trace-jsonl", type=Path, default=None,
                        help="Path to trace JSONL file or directory of JSONL files.")
    parser.add_argument("--window-size", type=int, default=8,
                        help="Context window size (number of past tuples).")
    parser.add_argument("--whole-sequence", action="store_true",
                        help="Train on each full trace sequence instead of sliding windows.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output", type=str, default="phase_predict.pt",
                        help="Path to save the checkpoint.")
    parser.add_argument("--per-token", action="store_true",
                        help="Use one tuple per token (instead of per CPD segment).")
    parser.add_argument("--cpd-penalty", type=float, default=0.1,
                        help="CPD penalty for breakpoint detection (default: 0.1).")
    parser.add_argument("--cpd-min-segment", type=int, default=2,
                        help="CPD minimum segment length in tokens (default: 2).")
    parser.add_argument("--cpd-smoothing", type=int, default=3,
                        help="CPD smoothing window size (default: 3).")
    args = parser.parse_args(argv)

    if args.trace_jsonl is not None and args.trace_dir is not None:
        parser.error("Provide only one of --trace-dir or --trace-jsonl")

    if args.whole_sequence:
        if args.trace_jsonl is not None:
            trace_source = args.trace_jsonl
            print(f"Loading trace JSONL from: {trace_source}")  # noqa: T201
            sequences = _extract_sequences_from_trace_jsonl(trace_source)
        else:
            trace_dir = args.trace_dir or default_trace_dir()
            print(f"Loading traces from: {trace_dir}")  # noqa: T201
            sequences = _extract_sequences_from_traces(
                trace_dir,
                per_token=args.per_token,
                cpd_penalty=args.cpd_penalty,
                cpd_min_segment=args.cpd_min_segment,
                cpd_smoothing=args.cpd_smoothing,
            )

        if not sequences:
            print("ERROR: No sequences were extracted.", file=sys.stderr)  # noqa: T201
            sys.exit(1)

        sequence_lengths = {len(sequence) for sequence in sequences}
        if len(sequence_lengths) != 1:
            print(  # noqa: T201
                "ERROR: --whole-sequence requires equal-length sequences. "
                f"Found lengths: {sorted(sequence_lengths)}",
                file=sys.stderr,
            )
            sys.exit(1)

        inferred_window_size = next(iter(sequence_lengths)) - 1
        print(f"Inferred whole-sequence window size: {inferred_window_size}")  # noqa: T201
        all_tuples = [tuple_value for sequence in sequences for tuple_value in sequence]
    else:
        if args.trace_jsonl is not None:
            trace_source = args.trace_jsonl
            print(f"Loading trace JSONL from: {trace_source}")  # noqa: T201
            all_tuples = _extract_tuples_from_trace_jsonl(trace_source)
        else:
            trace_dir = args.trace_dir or default_trace_dir()
            print(f"Loading traces from: {trace_dir}")  # noqa: T201

            all_tuples = _extract_tuples_from_traces(
                trace_dir,
                per_token=args.per_token,
                cpd_penalty=args.cpd_penalty,
                cpd_min_segment=args.cpd_min_segment,
                cpd_smoothing=args.cpd_smoothing,
            )
    print(f"\nTotal tuples extracted: {len(all_tuples)}")  # noqa: T201

    effective_window_size = inferred_window_size if args.whole_sequence else args.window_size

    if len(all_tuples) < effective_window_size + 2:
        print(  # noqa: T201
            f"ERROR: Not enough tuples ({len(all_tuples)}) for window_size={effective_window_size}. "
            "Try --per-token or add more traces.",
            file=sys.stderr,
        )
        sys.exit(1)

    model_cfg = ModelConfig(window_size=effective_window_size)
    if args.whole_sequence:
        dataset = PhaseFullSequenceDataset(sequences, model_cfg)
    else:
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
    if len(all_tuples) >= effective_window_size:
        context = all_tuples[-effective_window_size:]
        result = predictor.predict(context)
        print(f"\nSample prediction (last {effective_window_size} tuples → next):")  # noqa: T201
        print(f"  predicted: {result.predicted_tuple}")  # noqa: T201


if __name__ == "__main__":
    main()

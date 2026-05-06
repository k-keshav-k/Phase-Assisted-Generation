"""Train and evaluate the PhaseTransformer on tuple-sequence data.

Usage
-----
Run from the repository root::

    python scripts/train_phase_predict.py

Optional flags::

    --train-jsonl PATH    path to phase_tuples training JSONL file or directory
    --test-jsonl PATH     path to phase_tuples test JSONL file or directory
    --trace-dir PATH      path to trace JSON directory (default: phase_cpd default)
    --trace-jsonl PATH    path to trace JSONL file or directory of JSONL files
    --window-size N       context window size (default: 8)
    --d-model N           transformer hidden size (default: 64)
    --n-heads N           attention heads (default: 4)
    --n-layers N          transformer encoder layers (default: 2)
    --dropout P           dropout probability (default: 0.1)
    --whole-sequence      train on each full trace sequence instead of sliding windows
    --epochs N            max training epochs (default: 100)
    --learning-rate LR    learning rate (default: 1e-3)
    --lr LR               alias for --learning-rate
    --device NAME         training device: auto, cpu, or cuda (default: auto)
    --output PATH         where to save the trained model checkpoint (default: phase_predict.pt)
    --per-token           use one tuple per token instead of one per CPD segment

    Example with JSONL input:
    python scripts/train_phase_predict.py \
            --train-jsonl traces/phase_tuples_train.jsonl \
            --test-jsonl traces/phase_tuples_test.jsonl \
      --epochs 50 \
      --lr 5e-4 \
      --output output/phase_predict_checkpoint.pt

The script:
    1. Loads PhaseTuple sequences from phase_tuples JSONL files when present.
    2. Falls back to legacy trace directories or trace JSONL files.
    3. Trains either a full-sequence or sliding-window dataset.
    4. Reports validation loss and saves the checkpoint.
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
from phase_predict.data_utils import tuple_sequences_from_phase_tuples_jsonl
from phase_predict.data_utils import tuple_sequences_from_trace_jsonl
from phase_predict.data_utils import extended_tuple_sequences_from_phase_tuples_jsonl
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


def _extract_sequences_from_phase_tuples_jsonl(
    jsonl_path: Path, *, block_field: str = "block_size", second_field: str = "nfe"
) -> list[list[PhaseTuple]]:
    """Load tuple sequences from phase_tuples JSONL file(s).

    Args:
        jsonl_path: path to file or directory.
        block_field: field name for block size in each tuple dict.
        second_field: field name for the second component (e.g. "nfe").
    """
    jsonl_paths = [jsonl_path] if jsonl_path.is_file() else sorted(jsonl_path.glob("*.jsonl"))
    if not jsonl_paths:
        msg = f"No JSONL trace files were found in {jsonl_path}"
        raise FileNotFoundError(msg)

    sequences: list[list[PhaseTuple]] = []
    for path in jsonl_paths:
        file_sequences = tuple_sequences_from_phase_tuples_jsonl(
            path, block_field=block_field, second_field=second_field
        )
        file_tuple_count = sum(len(sequence) for sequence in file_sequences)
        print(
            f"  Loaded {path.name}: {file_tuple_count} tuples across {len(file_sequences)} sequences"
        )  # noqa: T201
        sequences.extend(file_sequences)

    return sequences


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PhaseTransformer on phase_cpd data.")
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("traces/rich/stab_tuples_conf_train_rich.jsonl"),
        help="Path to phase_tuples training JSONL file or directory.",
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=Path("traces/rich/stab_tuples_conf_test_rich.jsonl"),
        help="Path to phase_tuples test JSONL file or directory.",
    )
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="Path to trace JSON directory.")
    parser.add_argument("--trace-jsonl", type=Path, default=None,
                        help="Path to trace JSONL file or directory of JSONL files.")
    parser.add_argument("--window-size", type=int, default=8,
                        help="Context window size (number of past tuples).")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Transformer hidden size.")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads.")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability.")
    parser.add_argument("--whole-sequence", action="store_true",
                        help="Train on each full trace sequence instead of sliding windows.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device to use.",
    )
    parser.add_argument(
        "--tuple-second-field",
        type=str,
        default="max_stab_step",
        help="Field name to use as the second tuple component.",
    )
    parser.add_argument(
        "--tuple-block-field",
        type=str,
        default="block_size",
        help="Field name to use as the block size field.",
    )
    parser.add_argument(
        "--input-features",
        type=str,
        nargs="+",
        default=["block_size", "nfe", "mean_stab_step", "max_stab_step",
                 "mean_ref_step", "max_ref_step", "mean_gap", "max_gap",
                 "mean_top1_confidence", "min_top1_confidence",
                 "digit_fraction", "delimiter_fraction"],
        help="List of field names to use as input features.",
    )
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
    parser.add_argument("--num-block-classes", type=int, default=128,
                        help="Number of block size classes for classification head.")
    parser.add_argument("--num-stab-thresholds", type=int, default=83,
                        help="Number of ordinal thresholds for stab step head.")
    args = parser.parse_args(argv)

    if args.train_jsonl is not None and not args.train_jsonl.exists():
        parser.error(f"Training JSONL path does not exist: {args.train_jsonl}")
    if args.test_jsonl is not None and not args.test_jsonl.exists():
        parser.error(f"Test JSONL path does not exist: {args.test_jsonl}")
    if args.trace_jsonl is not None and args.trace_dir is not None:
        parser.error("Provide only one of --trace-dir or --trace-jsonl")

    # if args.trace_dir is None and args.trace_jsonl is None:
    #     if args.train_jsonl is None and default_train_jsonl.exists():
    #         args.train_jsonl = default_train_jsonl
    #     if args.test_jsonl is None and default_test_jsonl.exists():
    #         args.test_jsonl = default_test_jsonl

    use_phase_tuples_jsonl = args.train_jsonl is not None
    use_sequence_mode = use_phase_tuples_jsonl or args.whole_sequence

    train_sequences: list[list[PhaseTuple]] = []
    val_sequences: list[list[PhaseTuple]] = []
    all_tuples: list[PhaseTuple] = []
    inferred_window_size = args.window_size

    if use_phase_tuples_jsonl:
        if args.input_features is not None:
            train_sequences = extended_tuple_sequences_from_phase_tuples_jsonl(
                args.train_jsonl,
                output_fields=(args.tuple_block_field, args.tuple_second_field),
                input_feature_fields=args.input_features,
            )
        else:
            train_sequences = _extract_sequences_from_phase_tuples_jsonl(
                args.train_jsonl,
                block_field=args.tuple_block_field,
                second_field=args.tuple_second_field,
            )

        if not train_sequences:
            print("ERROR: No training sequences were extracted.", file=sys.stderr)  # noqa: T201
            sys.exit(1)

        if args.test_jsonl is not None:
            if args.input_features is not None:
                val_sequences = extended_tuple_sequences_from_phase_tuples_jsonl(
                    args.test_jsonl,
                    output_fields=(args.tuple_block_field, args.tuple_second_field),
                    input_feature_fields=args.input_features,
                )
            else:
                val_sequences = _extract_sequences_from_phase_tuples_jsonl(
                    args.test_jsonl,
                    block_field=args.tuple_block_field,
                    second_field=args.tuple_second_field,
                )
        else:
            # 80/20 split of training sequences when no test JSONL provided
            if len(train_sequences) < 2:
                print(  # noqa: T201
                    "ERROR: phase_tuples training needs at least two sequences "
                    "when no --test-jsonl is provided.",
                    file=sys.stderr,
                )
                sys.exit(1)
            split_index = max(1, int(len(train_sequences) * 0.8))
            split_index = min(split_index, len(train_sequences) - 1)
            val_sequences = train_sequences[split_index:]
            train_sequences = train_sequences[:split_index]
            print(f"Auto-split: {len(train_sequences)} train, {len(val_sequences)} val")  # noqa: T201

        all_sequences = train_sequences + val_sequences
        inferred_window_size = max(len(sequence) for sequence in all_sequences) - 1
        print(f"Inferred whole-sequence window size: {inferred_window_size}")  # noqa: T201
        all_tuples = [tuple_value for sequence in all_sequences for tuple_value in sequence]
    elif args.whole_sequence:
        if args.trace_jsonl is not None:
            trace_source = args.trace_jsonl
            print(f"Loading trace JSONL from: {trace_source}")  # noqa: T201
            all_sequences = _extract_sequences_from_trace_jsonl(trace_source)
        else:
            trace_dir = args.trace_dir or default_trace_dir()
            print(f"Loading traces from: {trace_dir}")  # noqa: T201
            all_sequences = _extract_sequences_from_traces(
                trace_dir,
                per_token=args.per_token,
                cpd_penalty=args.cpd_penalty,
                cpd_min_segment=args.cpd_min_segment,
                cpd_smoothing=args.cpd_smoothing,
            )

        if not all_sequences:
            print("ERROR: No sequences were extracted.", file=sys.stderr)  # noqa: T201
            sys.exit(1)

        if len(all_sequences) < 2:
            print(  # noqa: T201
                "ERROR: --whole-sequence needs at least two sequences to train and validate.",
                file=sys.stderr,
            )
            sys.exit(1)

        split_index = max(1, int(len(all_sequences) * 0.8))
        split_index = min(split_index, len(all_sequences) - 1)
        train_sequences = all_sequences[:split_index]
        val_sequences = all_sequences[split_index:]
        inferred_window_size = max(len(sequence) for sequence in all_sequences) - 1
        print(f"Inferred whole-sequence window size: {inferred_window_size}")  # noqa: T201
        all_tuples = [tuple_value for sequence in all_sequences for tuple_value in sequence]
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
        train_sequences = []
        val_sequences = []
    print(f"\nTotal tuples extracted: {len(all_tuples)}")  # noqa: T201

    effective_window_size = inferred_window_size if use_sequence_mode else args.window_size

    if len(all_tuples) < effective_window_size + 2:
        print(  # noqa: T201
            f"ERROR: Not enough tuples ({len(all_tuples)}) for window_size={effective_window_size}. "
            "Try --per-token or add more traces.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine input tuple size based on specified input features
    # Default: 2 (block_size and second_field)
    input_tuple_size = 2
    if args.input_features is not None:
        input_tuple_size = len(args.input_features)
        print(f"Using {input_tuple_size} input features: {args.input_features}")  # noqa: T201

    model_cfg = ModelConfig(
        window_size=effective_window_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        input_tuple_size=input_tuple_size,
        output_tuple_size=2,
        num_block_classes=args.num_block_classes,
        num_stab_thresholds=args.num_stab_thresholds,
    )
    if use_sequence_mode:
        # pass feature/output field names to datasets when using extended tuples
        ff = args.input_features
        of = [args.tuple_block_field, args.tuple_second_field]
        train_dataset = PhaseFullSequenceDataset(train_sequences, model_cfg, feature_fields=ff, output_fields=of)
        if val_sequences:
            val_dataset = PhaseFullSequenceDataset(
                val_sequences,
                model_cfg,
                stats=(train_dataset.mean, train_dataset.std),
                input_stats=(train_dataset.input_mean, train_dataset.input_std),
                feature_fields=ff,
                output_fields=of,
            )
        else:
            val_dataset = None
        dataset = train_dataset
    else:
        dataset = PhaseSequenceDataset(all_tuples, model_cfg, feature_fields=args.input_features, output_fields=[args.tuple_block_field, args.tuple_second_field])

    model = PhaseTransformer(model_cfg)
    train_cfg = TrainConfig(
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_interval=10,
        batch_size=32 if use_sequence_mode else 32,
    )

    print(f"\nTraining PhaseTransformer for up to {args.epochs} epochs …")  # noqa: T201
    if args.device == "auto":
        trainer_device = None
    else:
        trainer_device = args.device
    trainer = Trainer(model, train_cfg, device=trainer_device)
    print(f"Using device: {trainer.device}")  # noqa: T201
    if use_sequence_mode and val_sequences:
        history = trainer.fit(dataset, val_dataset=val_dataset)
    else:
        history = trainer.fit(dataset)

    print(  # noqa: T201
        f"\nTraining complete. "
        f"Best val loss = {history.best_val_loss:.6f} at epoch {history.best_epoch}."
    )

    predictor = Predictor(
        model,
        mean=dataset.mean,
        std=dataset.std,
        input_mean=getattr(dataset, "input_mean", None),
        input_std=getattr(dataset, "input_std", None),
        input_fields=getattr(dataset, "feature_fields", None),
    )
    # append best validation loss to checkpoint filename
    out_path = Path(args.output)
    metric_tag = f"bestval={history.best_val_loss:.6f}"
    new_name = f"{out_path.stem}_{metric_tag}{out_path.suffix}"
    out_path = out_path.with_name(new_name)
    predictor.save_checkpoint(str(out_path))
    print(f"Checkpoint saved to: {out_path}")  # noqa: T201

    # quick sanity-check prediction
    if use_sequence_mode and all_sequences:
        context = all_sequences[-1]
    elif len(all_tuples) >= effective_window_size:
        context = all_tuples[-effective_window_size:]
    else:
        context = []

    if context:
        print(f"\nMaking a sample prediction using the context: {context} tuples as context …")
        result = predictor.predict(context)
        print(f"\nSample prediction (last {effective_window_size} tuples → next):")  # noqa: T201
        print(f"  predicted: {result.predicted_tuple}")  # noqa: T201


if __name__ == "__main__":
    main()

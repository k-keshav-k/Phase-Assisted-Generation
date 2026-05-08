"""Run hyperparameter ablation studies on PhaseTransformer.

This script systematically trains models with different hyperparameter
combinations, saves checkpoints for each, and logs results for comparison.

Usage
-----
Run from the repository root::

    python scripts/ablate_phase_predict.py

Optional flags::

    --train-jsonl PATH       path to training JSONL
    --test-jsonl PATH        path to test JSONL
    --output-dir PATH        where to save ablation results (default: output/ablations)
    --epochs N               max training epochs per run (default: 50)
    --preset NAME            predefined ablation grid (options: small, medium, large, xlarge)
    --whole-sequence         train on each full trace sequence instead of sliding windows
    --dry-run                print ablation plan without training

Examples::

    # Use default small ablation grid
    python scripts/ablate_phase_predict.py --epochs 50

    # Use larger ablation grid (more combinations)
    python scripts/ablate_phase_predict.py --preset large --epochs 100

    # Check the ablation plan first
    python scripts/ablate_phase_predict.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force line-buffered stdout so print statements appear immediately in sbatch /
# Singularity logs rather than being flushed only at script exit.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from phase_predict.data_utils import (  # noqa: E402
    extended_tuple_sequences_from_phase_tuples_jsonl,
    tuple_sequences_from_phase_tuples_jsonl,
)
from phase_predict.dataset import PhaseFullSequenceDataset, PhaseSequenceDataset  # noqa: E402
from phase_predict.model import PhaseTransformer  # noqa: E402
from phase_predict.predict import Predictor  # noqa: E402
from phase_predict.schema import ModelConfig, TrainConfig  # noqa: E402
from phase_predict.train import Trainer  # noqa: E402

DEFAULT_INPUT_FEATURES = [
    "block_size",
    "nfe",
    "mean_stab_step",
    "max_stab_step",
    "mean_ref_step",
    "max_ref_step",
    "mean_gap",
    "max_gap",
    "mean_top1_confidence",
    "min_top1_confidence",
    "digit_fraction",
    "delimiter_fraction",
]


@dataclass
class AblationResult:
    """Record of one ablation run."""

    run_id: str
    model_config: dict[str, Any]
    train_config: dict[str, Any]
    train_loss: float
    val_loss: float
    best_epoch: int
    training_time_sec: float
    checkpoint_path: str


# Predefined ablation grids
ABLATION_PRESETS = {
    "small": {
        "window_size": [4, 8],  # Placeholder, will be overridden to max sequence length
        "d_model": [32],
        "n_heads": [2, 4],
        "n_layers": [2, 4],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
    },
    "medium": {
        "window_size": [8],  # Placeholder, will be overridden to max sequence length
        "d_model": [64],
        "n_heads": [2, 4],
        "n_layers": [2, 4],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
    },
    "large": {
        "window_size": [8, 16],  # Placeholder, will be overridden to max sequence length
        "d_model": [128],
        "n_heads": [2, 4, 8],
        "n_layers": [2, 4],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
    },
    "xlarge": {
        "window_size": [8, 16],  # Placeholder, will be overridden to max sequence length
        "d_model": [256, 512],
        "n_heads": [2, 4, 8],
        "n_layers": [2, 4, 6],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [5e-4, 1e-4],
    },
}


def generate_ablation_configs(
    preset: str = "small",
) -> list[tuple[ModelConfig, TrainConfig]]:
    """Generate all combinations of hyperparameters from ablation grid.

    Args:
        preset: one of 'small', 'medium', 'large', 'xlarge'

    Returns:
        List of (ModelConfig, TrainConfig) tuples to test.
    """
    if preset not in ABLATION_PRESETS:
        msg = f"Unknown preset: {preset}. Choose from {list(ABLATION_PRESETS.keys())}"
        raise ValueError(msg)

    grid = ABLATION_PRESETS[preset]
    configs = []

    # Generate Cartesian product of all hyperparameters
    import itertools

    keys = list(grid.keys())
    model_keys = {"window_size", "d_model", "n_heads", "n_layers", "dropout"}
    train_keys = {"learning_rate"}

    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values, strict=True))

        # Validate ModelConfig constraints
        d_model = params["d_model"]
        n_heads = params["n_heads"]
        if d_model % n_heads != 0:
            continue  # Skip invalid combinations

        model_params = {k: v for k, v in params.items() if k in model_keys}
        train_params = {k: v for k, v in params.items() if k in train_keys}

        model_cfg = ModelConfig(**model_params)
        train_cfg = TrainConfig(learning_rate=train_params["learning_rate"])

        configs.append((model_cfg, train_cfg))

    return configs


def load_data(
    train_jsonl: Path,
    test_jsonl: Path | None,
    *,
    block_field: str = "block_size",
    second_field: str = "nfe",
    input_feature_fields: list[str] | None = None,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Load training and validation sequences from phase_tuples JSONL data.

    Args:
        train_jsonl: path to training JSONL file or directory.
        test_jsonl: path to test JSONL file or directory (optional).

    Returns:
        (train_sequences, val_sequences)
    """
    if not train_jsonl.exists():
        msg = f"Training JSONL not found: {train_jsonl}"
        raise FileNotFoundError(msg)

    print(f"Loading training data from: {train_jsonl}")  # noqa: T201
    if input_feature_fields is not None:
        train_sequences = extended_tuple_sequences_from_phase_tuples_jsonl(
            train_jsonl,
            output_fields=(block_field, second_field),
            input_feature_fields=input_feature_fields,
        )
    else:
        train_sequences = tuple_sequences_from_phase_tuples_jsonl(
            train_jsonl,
            block_field=block_field,
            second_field=second_field,
        )

    if test_jsonl and test_jsonl.exists():
        print(f"Loading validation data from: {test_jsonl}")  # noqa: T201
        if input_feature_fields is not None:
            val_sequences = extended_tuple_sequences_from_phase_tuples_jsonl(
                test_jsonl,
                output_fields=(block_field, second_field),
                input_feature_fields=input_feature_fields,
            )
        else:
            val_sequences = tuple_sequences_from_phase_tuples_jsonl(
                test_jsonl,
                block_field=block_field,
                second_field=second_field,
            )
    else:
        # 80/20 split
        split_idx = max(1, int(len(train_sequences) * 0.8))
        split_idx = min(split_idx, len(train_sequences) - 1)
        val_sequences = train_sequences[split_idx:]
        train_sequences = train_sequences[:split_idx]
        print(f"Auto-split: {len(train_sequences)} train, {len(val_sequences)} val")  # noqa: T201

    return train_sequences, val_sequences


def align_configs_to_training_shape(
    configs: list[tuple[ModelConfig, TrainConfig]],
    train_sequences: list[list[Any]],
    val_sequences: list[list[Any]],
    *,
    whole_sequence: bool,
    window_size: int,
) -> tuple[list[tuple[ModelConfig, TrainConfig]], int]:
    """Align ablation configs to the same context length used by training.

    Sliding-window training uses the CLI window size. Whole-sequence training
    infers the context length from the longest loaded sequence. This mirrors
    ``scripts/train_phase_predict.py``.
    """
    all_sequences = train_sequences + val_sequences
    effective_window_size = (
        max(len(sequence) for sequence in all_sequences) - 1 if whole_sequence else window_size
    )

    aligned_configs: list[tuple[ModelConfig, TrainConfig]] = []
    seen: set[tuple[Any, ...]] = set()
    for model_cfg, train_cfg in configs:
        aligned_model_cfg = replace(model_cfg, window_size=effective_window_size)
        key = (
            aligned_model_cfg.window_size,
            aligned_model_cfg.d_model,
            aligned_model_cfg.n_heads,
            aligned_model_cfg.n_layers,
            aligned_model_cfg.dropout,
            train_cfg.learning_rate,
        )
        if key in seen:
            continue
        seen.add(key)
        aligned_configs.append((aligned_model_cfg, train_cfg))

    return aligned_configs, effective_window_size


def run_ablation(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_sequences: list[list[Any]],
    val_sequences: list[list[Any]],
    run_id: str,
    output_dir: Path,
    *,
    whole_sequence: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    input_feature_fields: list[str] | None = None,
) -> AblationResult | None:
    """Train one model configuration.

    Args:
        model_cfg: ModelConfig instance
        train_cfg: TrainConfig instance
        train_sequences: training data
        val_sequences: validation data
        run_id: descriptive ID for this run
        output_dir: directory to save checkpoint

    Returns:
        AblationResult with metrics, or None if training failed.
    """
    try:
        print(f"\n{'=' * 70}")  # noqa: T201
        print(f"Run: {run_id}")  # noqa: T201
        print(
            f"  ModelConfig: window_size={model_cfg.window_size}, "
            f"d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, "
            f"n_layers={model_cfg.n_layers}, dropout={model_cfg.dropout}"
        )  # noqa: T201
        run_train_cfg = replace(
            train_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        print(
            f"  TrainConfig: lr={run_train_cfg.learning_rate}, "
            f"epochs={run_train_cfg.max_epochs}, batch_size={run_train_cfg.batch_size}"
        )  # noqa: T201

        if whole_sequence:
            train_dataset = PhaseFullSequenceDataset(
                train_sequences,
                model_cfg,
                feature_fields=input_feature_fields,
            )
            val_dataset = PhaseFullSequenceDataset(
                val_sequences,
                model_cfg,
                input_stats=(train_dataset.input_mean, train_dataset.input_std),
                feature_fields=input_feature_fields,
            )
            dataset = train_dataset
        else:
            all_tuples = [t for seq in (train_sequences + val_sequences) for t in seq]
            dataset = PhaseSequenceDataset(
                all_tuples,
                model_cfg,
                feature_fields=input_feature_fields,
            )
            val_dataset = None

        model = PhaseTransformer(model_cfg)
        trainer_device = None if device == "auto" else device
        trainer = Trainer(model, run_train_cfg, device=trainer_device)
        print(f"  Device: {trainer.device}")  # noqa: T201

        start_time = time.time()
        if whole_sequence:
            history = trainer.fit(dataset, val_dataset=val_dataset)
        else:
            history = trainer.fit(dataset)
        elapsed = time.time() - start_time

        predictor = Predictor(
            model,
            input_mean=getattr(dataset, "input_mean", None),
            input_std=getattr(dataset, "input_std", None),
            input_fields=getattr(dataset, "feature_fields", None),
        )
        metric_tag = f"bestval={history.best_val_loss:.6f}"
        checkpoint_name = f"{run_id}_{metric_tag}.pt"
        checkpoint_path = output_dir / checkpoint_name
        predictor.save_checkpoint(str(checkpoint_path))

        result = AblationResult(
            run_id=run_id,
            model_config=asdict(model_cfg),
            train_config=asdict(run_train_cfg),
            train_loss=float(history.train_losses[-1]) if history.train_losses else 0.0,
            val_loss=history.best_val_loss,
            best_epoch=history.best_epoch,
            training_time_sec=elapsed,
            checkpoint_path=str(checkpoint_path),
        )

        print(f"  ✓ Train loss: {result.train_loss:.6f}")  # noqa: T201
        print(f"  ✓ Val loss: {result.val_loss:.6f} (epoch {result.best_epoch})")  # noqa: T201
        print(f"  ✓ Time: {elapsed:.1f}s")  # noqa: T201
        print(f"  ✓ Checkpoint: {checkpoint_path}")  # noqa: T201

        return result

    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ FAILED: {exc}", file=sys.stderr)  # noqa: T201
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter ablations on PhaseTransformer."
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("traces/rich/stab_tuples_conf_train_rich.jsonl"),
        help="Path to phase_tuples training JSONL.",
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=Path("traces/rich/stab_tuples_conf_test_rich.jsonl"),
        help="Path to phase_tuples test JSONL (optional; else use 80/20 split).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ablations"),
        help="Directory to save ablation results.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs per run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (0 = main process only).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Context window size for sliding-window training.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(ABLATION_PRESETS.keys()),
        default="small",
        help="Predefined ablation grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ablation plan without training.",
    )
    parser.add_argument(
        "--whole-sequence",
        action="store_true",
        help="Train on each full trace sequence instead of sliding windows.",
    )
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
        help=(
            "Field name to use as the block size field when reading "
            "phase_tuples JSONL (default: block_size)."
        ),
    )
    parser.add_argument(
        "--input-features",
        type=str,
        nargs="+",
        default=DEFAULT_INPUT_FEATURES,
        help="List of field names to use as input features.",
    )
    parser.add_argument(
        "--num-block-classes",
        type=int,
        default=128,
        help="Number of block size classes for classification head.",
    )
    parser.add_argument(
        "--num-stab-thresholds",
        type=int,
        default=83,
        help="Number of ordinal thresholds for stab step head.",
    )
    args = parser.parse_args(argv)

    if not args.train_jsonl.exists():
        parser.error(f"Training JSONL path does not exist: {args.train_jsonl}")
    if args.test_jsonl is not None and not args.test_jsonl.exists():
        parser.error(f"Test JSONL path does not exist: {args.test_jsonl}")

    # Generate configurations
    configs = generate_ablation_configs(args.preset)

    # Load data
    train_sequences, val_sequences = load_data(
        args.train_jsonl,
        args.test_jsonl,
        block_field=args.tuple_block_field,
        second_field=args.tuple_second_field,
        input_feature_fields=args.input_features,
    )

    all_tuples = [t for seq in (train_sequences + val_sequences) for t in seq]
    if len(all_tuples) < args.window_size + 2 and not args.whole_sequence:
        parser.error(f"Not enough tuples ({len(all_tuples)}) for window_size={args.window_size}.")

    if args.input_features is not None:
        configs = [
            (
                replace(
                    model_cfg,
                    input_tuple_size=len(args.input_features),
                    output_tuple_size=2,
                    num_block_classes=args.num_block_classes,
                    num_stab_thresholds=args.num_stab_thresholds,
                ),
                train_cfg,
            )
            for model_cfg, train_cfg in configs
        ]
    else:
        configs = [
            (
                replace(
                    model_cfg,
                    input_tuple_size=2,
                    output_tuple_size=2,
                    num_block_classes=args.num_block_classes,
                    num_stab_thresholds=args.num_stab_thresholds,
                ),
                train_cfg,
            )
            for model_cfg, train_cfg in configs
        ]

    configs, effective_window_size = align_configs_to_training_shape(
        configs,
        train_sequences,
        val_sequences,
        whole_sequence=args.whole_sequence,
        window_size=args.window_size,
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating ablation grid: {args.preset}")  # noqa: T201
    mode = "whole-sequence" if args.whole_sequence else "sliding-window"
    print(f"  Training mode: {mode}")  # noqa: T201
    print(f"  Effective window_size: {effective_window_size}")  # noqa: T201
    if args.input_features is not None:
        print(f"  Input features ({len(args.input_features)}): {args.input_features}")  # noqa: T201
    print(f"  Total configurations: {len(configs)}")  # noqa: T201

    if args.dry_run:
        print("\nAblation plan (dry-run, no training):")  # noqa: T201
        for i, (model_cfg, _train_cfg) in enumerate(configs, 1):
            run_id = (
                f"{args.preset}_"
                f"ws{model_cfg.window_size}_"
                f"d{model_cfg.d_model}_"
                f"h{model_cfg.n_heads}_"
                f"l{model_cfg.n_layers}_"
                f"dp{int(model_cfg.dropout * 100)}_"
                f"lr{_train_cfg.learning_rate * 1000}"
            )
            print(f"  {i:3d}. {run_id}")  # noqa: T201
        return

    # Run ablations
    results: list[AblationResult] = []
    start_time = time.time()

    for i, (model_cfg, train_cfg) in enumerate(configs, 1):
        run_id = (
            f"{args.preset}_"
            f"ws{model_cfg.window_size}_"
            f"d{model_cfg.d_model}_"
            f"h{model_cfg.n_heads}_"
            f"l{model_cfg.n_layers}_"
            f"dp{int(model_cfg.dropout * 100)}_"
            f"lr{train_cfg.learning_rate * 1000}"
        )

        train_cfg = replace(train_cfg, max_epochs=args.epochs)

        print(f"\n[{i}/{len(configs)}] ", end="")  # noqa: T201
        result = run_ablation(
            model_cfg,
            train_cfg,
            train_sequences,
            val_sequences,
            run_id,
            args.output_dir,
            whole_sequence=args.whole_sequence,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            input_feature_fields=args.input_features,
        )

        if result:
            results.append(result)

    total_time = time.time() - start_time

    # Save results
    print(f"\n{'=' * 70}")  # noqa: T201
    print(f"Ablation complete: {len(results)}/{len(configs)} runs succeeded")  # noqa: T201
    print(f"Total time: {total_time / 60:.1f} minutes")  # noqa: T201

    # Save results as CSV
    csv_path = args.output_dir / "ablation_results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "run_id",
                "window_size",
                "d_model",
                "n_heads",
                "n_layers",
                "dropout",
                "learning_rate",
                "batch_size",
                "num_workers",
                "train_loss",
                "val_loss",
                "best_epoch",
                "training_time_sec",
                "checkpoint_path",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "run_id": result.run_id,
                    "window_size": result.model_config["window_size"],
                    "d_model": result.model_config["d_model"],
                    "n_heads": result.model_config["n_heads"],
                    "n_layers": result.model_config["n_layers"],
                    "dropout": result.model_config["dropout"],
                    "learning_rate": result.train_config["learning_rate"],
                    "batch_size": result.train_config["batch_size"],
                    "num_workers": result.train_config["num_workers"],
                    "train_loss": result.train_loss,
                    "val_loss": result.val_loss,
                    "best_epoch": result.best_epoch,
                    "training_time_sec": result.training_time_sec,
                    "checkpoint_path": result.checkpoint_path,
                }
                writer.writerow(row)

        print(f"Results saved to: {csv_path}")  # noqa: T201

    # Save full results as JSON
    json_path = args.output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json_results = [asdict(r) for r in results]
        json.dump(
            {
                "preset": args.preset,
                "training_mode": mode,
                "input_features": args.input_features,
                "tuple_block_field": args.tuple_block_field,
                "tuple_second_field": args.tuple_second_field,
                "total_configs": len(configs),
                "successful_runs": len(results),
                "total_time_sec": total_time,
                "results": json_results,
            },
            f,
            indent=2,
        )

    print(f"Full results saved to: {json_path}")  # noqa: T201

    # Print summary: top 5 by validation loss
    if results:
        sorted_results = sorted(results, key=lambda r: r.val_loss)
        print("\nTop 5 configurations by validation loss:")  # noqa: T201
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {result.run_id}")  # noqa: T201
            print(f"     val_loss={result.val_loss:.6f}, train_loss={result.train_loss:.6f}")  # noqa: T201


if __name__ == "__main__":
    main()

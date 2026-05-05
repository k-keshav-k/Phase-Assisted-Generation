"""Run hyperparameter ablation studies on PhaseTransformer.

This script systematically trains models with different hyperparameter
combinations, saves checkpoints for each, and logs results for comparison.

Usage
-----
Run from the repository root::

    python scripts/ablate_phase_predict.py

Optional flags::

    --train-jsonl PATH       path to training JSONL (default: traces/phase_tuples_train.jsonl)
    --test-jsonl PATH        path to test JSONL (default: traces/phase_tuples_test.jsonl)
    --output-dir PATH        where to save ablation results (default: output/ablations)
    --epochs N               max training epochs per run (default: 50)
    --preset NAME            predefined ablation grid (options: small, medium, large, xlarge)
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

from phase_predict.data_utils import extended_tuple_sequences_from_phase_tuples_jsonl
from phase_predict.data_utils import tuple_sequences_from_phase_tuples_jsonl
from phase_predict.dataset import PhaseFullSequenceDataset
from phase_predict.model import PhaseTransformer
from phase_predict.predict import Predictor
from phase_predict.schema import ModelConfig, TrainConfig
from phase_predict.train import Trainer


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
        "window_size": [4],  # Placeholder, will be overridden to max sequence length
        "d_model": [32, 64],
        "n_heads": [2, 4],
        "n_layers": [1, 2],
        "dropout": [0.0, 0.1],
        "learning_rate": [1e-3, 5e-4],
    },
    "medium": {
        "window_size": [4], # Placeholder, will be overridden to max sequence length
        "d_model": [64, 128],
        "n_heads": [2, 4],
        "n_layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
    },
    "large": {
        "window_size": [4], # Placeholder, will be overridden to max sequence length
        "d_model": [128, 256],
        "n_heads": [2, 4, 8],
        "n_layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
    },
    "xlarge": {
        "window_size": [8], # Placeholder, will be overridden to max sequence length
        "d_model": [256, 512],
        "n_heads": [4, 8, 16],
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
        params = dict(zip(keys, values))

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
    """Load training and validation sequences.

    Args:
        train_jsonl: path to training JSONL file
        test_jsonl: path to test JSONL file (optional)

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


def align_configs_to_sequence_length(
    configs: list[tuple[ModelConfig, TrainConfig]],
    train_sequences: list[list[Any]],
    val_sequences: list[list[Any]],
) -> tuple[list[tuple[ModelConfig, TrainConfig]], int]:
    """Align ablation configs to the full-sequence context length.

    Phase tuple JSONL training uses one full history per sequence and the
    model context must therefore match the longest sequence length seen by
    the dataset. This mirrors ``scripts/train_phase_predict.py``.
    """
    all_sequences = train_sequences + val_sequences
    inferred_window_size = max(len(sequence) for sequence in all_sequences) - 1

    aligned_configs: list[tuple[ModelConfig, TrainConfig]] = []
    seen: set[tuple[Any, ...]] = set()
    for model_cfg, train_cfg in configs:
        aligned_model_cfg = replace(model_cfg, window_size=inferred_window_size)
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

    return aligned_configs, inferred_window_size


def run_ablation(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_sequences: list[list[Any]],
    val_sequences: list[list[Any]],
    run_id: str,
    output_dir: Path,
    *,
    input_feature_fields: list[str] | None = None,
    output_fields: list[str] | None = None,
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
        print(f"\n{'='*70}")  # noqa: T201
        print(f"Run: {run_id}")  # noqa: T201
        print(f"  ModelConfig: window_size={model_cfg.window_size}, "
              f"d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, "
              f"n_layers={model_cfg.n_layers}, dropout={model_cfg.dropout}")  # noqa: T201
        print(f"  TrainConfig: lr={train_cfg.learning_rate}, "
              f"epochs={train_cfg.max_epochs}, batch_size={train_cfg.batch_size}")  # noqa: T201

        # Create datasets
        train_dataset = PhaseFullSequenceDataset(
            train_sequences,
            model_cfg,
            feature_fields=input_feature_fields,
            output_fields=output_fields,
        )
        val_dataset = PhaseFullSequenceDataset(
            val_sequences,
            model_cfg,
            stats=(train_dataset.mean, train_dataset.std),
            input_stats=(train_dataset.input_mean, train_dataset.input_std),
            feature_fields=input_feature_fields,
            output_fields=output_fields,
        )

        # Train
        model = PhaseTransformer(model_cfg)
        trainer = Trainer(model, train_cfg, device=None)

        start_time = time.time()
        history = trainer.fit(train_dataset, val_dataset=val_dataset)
        elapsed = time.time() - start_time

        # Save checkpoint (include best validation loss in filename)
        predictor = Predictor(
            model,
            mean=train_dataset.mean,
            std=train_dataset.std,
            input_mean=getattr(train_dataset, "input_mean", None),
            input_std=getattr(train_dataset, "input_std", None),
            input_fields=getattr(train_dataset, "feature_fields", None),
        )
        metric_tag = f"bestval={history.best_val_loss:.6f}"
        checkpoint_name = f"{run_id}_{metric_tag}.pt"
        checkpoint_path = output_dir / checkpoint_name
        predictor.save_checkpoint(str(checkpoint_path))

        result = AblationResult(
            run_id=run_id,
            model_config=asdict(model_cfg),
            train_config={"learning_rate": train_cfg.learning_rate},
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
        default=Path("traces/phase_tuples_train.jsonl"),
        help="Path to phase_tuples training JSONL.",
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=None,
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
        default=50,
        help="Max training epochs per run.",
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
        "--tuple-second-field",
        type=str,
        default="nfe",
        help="Field name to use as the second tuple component when reading phase_tuples JSONL (default: nfe).",
    )
    parser.add_argument(
        "--tuple-block-field",
        type=str,
        default="block_size",
        help="Field name to use as the block size field when reading phase_tuples JSONL (default: block_size).",
    )
    parser.add_argument(
        "--input-features",
        type=str,
        nargs="+",
        default=None,
        help="List of field names to use as input features (e.g. 'block_size nfe max_stab_step').",
    )
    args = parser.parse_args(argv)

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

    if args.input_features is not None:
        configs = [
            (
                replace(model_cfg, input_tuple_size=len(args.input_features), output_tuple_size=2),
                train_cfg,
            )
            for model_cfg, train_cfg in configs
        ]
    configs, inferred_window_size = align_configs_to_sequence_length(
        configs,
        train_sequences,
        val_sequences,
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating ablation grid: {args.preset}")  # noqa: T201
    print(f"  Sequence-aligned window_size: {inferred_window_size}")  # noqa: T201
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
                f"dp{int(model_cfg.dropout*100)}_"
                f"lr{_train_cfg.learning_rate*1000}"
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
            f"dp{int(model_cfg.dropout*100)}_"
            f"lr{train_cfg.learning_rate*1000}"
        )

        # Override epochs from CLI
        train_cfg.max_epochs = args.epochs

        print(f"\n[{i}/{len(configs)}] ", end="")  # noqa: T201
        result = run_ablation(
            model_cfg,
            train_cfg,
            train_sequences,
            val_sequences,
            run_id,
            args.output_dir,
            input_feature_fields=args.input_features,
            output_fields=[args.tuple_block_field, args.tuple_second_field],
        )

        if result:
            results.append(result)

    total_time = time.time() - start_time

    # Save results
    print(f"\n{'='*70}")  # noqa: T201
    print(f"Ablation complete: {len(results)}/{len(configs)} runs succeeded")  # noqa: T201
    print(f"Total time: {total_time/60:.1f} minutes")  # noqa: T201

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
        print(f"\nTop 5 configurations by validation loss:")  # noqa: T201
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {result.run_id}")  # noqa: T201
            print(f"     val_loss={result.val_loss:.6f}, "
                  f"train_loss={result.train_loss:.6f}")  # noqa: T201


if __name__ == "__main__":
    main()

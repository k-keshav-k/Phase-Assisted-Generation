#!/usr/bin/env python3
"""Runnable training script for the block/stabilising-step RF predictor.

Usage::

    python -m block_stab_predict.train                          
        --train-path traces/stab_tuples_conf_train.jsonl         
        --test-path traces/stab_tuples_conf_test.jsonl           
        --output-dir block_stab_predict/models
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from block_stab_predict.dataset import (
    build_X_y,
    filter_tuples,
    load_jsonl,
    train_test_split_by_sample,
)
from block_stab_predict.model import BlockStabPredictor
from block_stab_predict.schema import RFConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the block/stabilising-step RF predictor."
    )
    parser.add_argument(
        "--train-path",
        default="traces/stab_tuples_conf_train.jsonl",
        type=str,
        help="Path to training JSONL (default: traces/stab_tuples_conf_train.jsonl)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        help="Optional held-out JSONL for final evaluation. "
        "If omitted, an 80/20 split of --train-path is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="block_stab_predict/models",
        type=str,
        help="Directory to save the trained model (default: block_stab_predict/models).",
    )
    # RF hyper-parameters
    parser.add_argument("--window-size", type=int, default=20, help="Context window")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of trees (default: 200).")
    parser.add_argument("--max-depth", type=int, default=15,
                        help="Max tree depth (default: 15).")
    parser.add_argument("--min-samples-leaf", type=int, default=5,
                        help="Min samples per leaf (default: 5).")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Validation fraction (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs (default: -1 = all CPUs).")
    # Data filtering
    parser.add_argument(
        "--filter-mode",
        default="none",
        choices=["none", "no-eog", "no-sentinel"],
        help="Filter mode for training tuples (default: none).",
    )
    parser.add_argument(
        "--tag",
        default="v1",
        type=str,
        help="Model version tag (default: v1). Saved as rf_{tag}.joblib.",
    )
    return parser.parse_args(argv)


# ── Evaluation helpers ────────────────────────────────────────────────


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def _r2(preds: np.ndarray, targets: np.ndarray) -> float:
    ss_res = ((preds - targets) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-15))


def _accuracy_within(preds: np.ndarray, targets: np.ndarray, tol: int) -> float:
    return float((np.abs(preds - targets) <= tol).mean())


def _print_metrics(
    split_name: str,
    preds: np.ndarray,
    targets: np.ndarray,
    names: list[str],
) -> None:
    """Print per-target and aggregate metrics for one split."""
    print(f"\n{'=' * 60}")
    print(f"  {split_name}")
    print(f"{'=' * 60}")
    for col_idx, name in enumerate(names):
        p = preds[:, col_idx]
        t = targets[:, col_idx]
        print(f"\n  {name}:")
        print(f"    MAE : {_mae(p, t):.4f}")
        print(f"    RMSE: {_rmse(p, t):.4f}")
        print(f"    R²  : {_r2(p, t):.4f}")
        if name == "block_size":
            print(f"    ±1  : {_accuracy_within(p, t, 1) * 100:.1f}%")
            print(f"    ±5  : {_accuracy_within(p, t, 5) * 100:.1f}%")
        elif name == "max_stab_step":
            print(f"    exact: {_accuracy_within(p, t, 0) * 100:.1f}%")
            print(f"    ±1   : {_accuracy_within(p, t, 1) * 100:.1f}%")


def _compute_baselines(
    X: np.ndarray,
    Y: np.ndarray,
    col_idx: int,
    name: str,
    feature_names: list[str],
) -> dict[str, float]:
    """Naive baselines for comparison."""
    # Baseline A: predict the last seen value for the corresponding feature.
    last_feat = f"{name}_last"
    try:
        last_idx = feature_names.index(last_feat)
    except ValueError:
        last_idx = None
    if last_idx is not None:
        last_preds = X[:, last_idx]
    else:
        last_preds = np.full(Y.shape[0], Y[:, col_idx].mean())
    mean_preds = np.full(Y.shape[0], Y[:, col_idx].mean())

    return {
        "last_mae": _mae(last_preds, Y[:, col_idx]),
        "mean_mae": _mae(mean_preds, Y[:, col_idx]),
    }


# ── Main ──────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> dict:
    """Run the full training pipeline and return a metrics summary."""

    # 1. Config
    config = RFConfig(
        window_size=args.window_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        test_size=args.test_size,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    print(f"Config:\n  {config}\n")

    # 2. Load data
    print(f"Loading training data from {args.train_path} ...")
    train_samples = load_jsonl(args.train_path)
    print(f"  {len(train_samples)} samples loaded.")

    # 2b. Filter tuples
    filter_mode = args.filter_mode
    if filter_mode != "none":
        before = sum(len(s["tuples"]) for s in train_samples)
        train_samples = filter_tuples(train_samples, filter_mode)
        after = sum(len(s["tuples"]) for s in train_samples)
        print(f"  Filter mode '{filter_mode}': {before} tuples → {after} "
              f"({after / max(before, 1) * 100:.1f}% retained) across "
              f"{len(train_samples)} samples.")

    # 3. Split or load held-out test set
    if args.test_path:
        print(f"Loading held-out test data from {args.test_path} ...")
        test_samples = load_jsonl(args.test_path)
        print(f"  {len(test_samples)} samples loaded.")
        if filter_mode != "none":
            before_te = sum(len(s["tuples"]) for s in test_samples)
            test_samples = filter_tuples(test_samples, filter_mode)
            after_te = sum(len(s["tuples"]) for s in test_samples)
            print(f"  Test filter '{filter_mode}': {before_te} tuples → {after_te} "
                  f"({after_te / max(before_te, 1) * 100:.1f}% retained) across "
                  f"{len(test_samples)} samples.")
        X_tr, Y_tr, names = build_X_y(train_samples, config)
        X_te, Y_te, _ = build_X_y(test_samples, config)
        split_type = "held-out"
    else:
        print("No --test-path provided; splitting training data 80/20 by sample ...")
        X_tr, X_te, Y_tr, Y_te, names = train_test_split_by_sample(train_samples, config)
        split_type = "80/20-by-sample"
        test_samples = []  # not used further

    print(f"  Train examples: {X_tr.shape[0]}, Test examples: {X_te.shape[0]}")
    print(f"  Features: {len(names)} ({', '.join(names)})")

    # 4. Train
    print("\nTraining Random Forest ...")
    t0 = time.perf_counter()
    predictor = BlockStabPredictor(config).fit(X_tr, Y_tr, feature_names=names)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s.")

    # 5. Evaluate on test set
    preds_te = predictor.predict(X_te)
    _print_metrics(f"Test ({split_type})", preds_te, Y_te, list(config.target_fields))

    # 6. Naive baselines on test set
    print("\n  Baselines (test set):")
    for col_idx, name in enumerate(config.target_fields):
        bl = _compute_baselines(X_te, Y_te, col_idx, name, names)
        print(f"    {name}:")
        print(f"      predict-last MAE: {bl['last_mae']:.4f}")
        print(f"      predict-mean MAE: {bl['mean_mae']:.4f}")

    # 7. Feature importance
    imp, imp_names = predictor.feature_importances()
    print(f"\n{'─' * 60}")
    print("  Feature importance")
    print(f"{'─' * 60}")
    for name, val in zip(imp_names, imp, strict=True):
        print(f"    {name:30s} {val:.4f}")

    # 8. Save model
    output_dir = Path(args.output_dir)
    model_name = f"rf_{args.tag}.joblib"
    model_path = output_dir / model_name
    predictor.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # 9. Return metrics summary
    metrics = {
        "config": {
            "window_size": config.window_size,
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "feature_fields": list(config.feature_fields),
            "target_fields": list(config.target_fields),
        },
        "train_examples": int(X_tr.shape[0]),
        "test_examples": int(X_te.shape[0]),
        "train_time_sec": round(elapsed, 2),
        "test_metrics": {
            name: {
                "mae": round(_mae(preds_te[:, i], Y_te[:, i]), 4),
                "rmse": round(_rmse(preds_te[:, i], Y_te[:, i]), 4),
                "r2": round(_r2(preds_te[:, i], Y_te[:, i]), 4),
            }
            for i, name in enumerate(config.target_fields)
        },
        "feature_importances": {
            name: round(float(val), 4) for name, val in zip(imp_names, imp, strict=True)
        },
        "model_path": str(model_path),
    }

    # Also write metrics as JSON
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to {metrics_path}")

    return metrics


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()

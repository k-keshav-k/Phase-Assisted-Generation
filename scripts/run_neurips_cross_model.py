from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from pag.experiments.cross_model_config import load_cross_model_config
from pag.experiments.cross_model_orchestrator import CrossModelOrchestrator
from pag.experiments.orchestrator import ControlledStop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen LLaDA/Dream residual-PAG evaluation.")
    parser.add_argument(
        "--config",
        default="configs/experiments/neurips_cross_model.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="artifacts/neurips_cross_model")
    parser.add_argument(
        "--llada-trace",
        default="traces/rich/stab_tuples_conf_train_rich.jsonl",
    )
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--gpu-rate", type=float, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _preflight(args: argparse.Namespace) -> tuple[object, Path]:
    config = load_cross_model_config(args.config)
    trace = Path(args.llada_trace).resolve()
    if not trace.is_file():
        raise FileNotFoundError(f"LLaDA trace not found: {trace}")
    prior_manifest = Path(config.math500.prior_selection_manifest)
    if not prior_manifest.is_file():
        raise FileNotFoundError(f"prior MATH selection not found: {prior_manifest}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    budget = config.budget_usd if args.budget_usd is None else args.budget_usd
    rate = config.gpu_rate if args.gpu_rate is None else args.gpu_rate
    if not 0 < budget <= 19:
        raise ValueError("budget-usd must be in (0, 19]")
    if rate <= 0:
        raise ValueError("gpu-rate must be positive")
    return config, trace


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args()
    config, trace = _preflight(args)
    if args.preflight_only:
        print("Preflight passed: local config, trace, manifest, device, and budget checks.")
        return 0
    run_id = f"cross-model-{config.config_hash[:12]}"
    run_dir = Path(args.output_root).resolve() / run_id
    print(f"Run ID: {run_id}")
    print(f"Artifacts: {run_dir}")
    runner = CrossModelOrchestrator(
        config=config,
        run_dir=run_dir,
        device=args.device,
        llada_trace_path=trace,
        budget_usd=args.budget_usd,
        gpu_rate=args.gpu_rate,
    )
    try:
        runner.run()
    except ControlledStop as exc:
        print(f"Controlled stop: {exc}")
        return 2
    print(f"Generation complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

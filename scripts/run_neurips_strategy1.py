from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pag.experiments.config import load_experiment_config
from pag.experiments.orchestrator import (
    ControlledStop,
    StrategyOneOrchestrator,
    protocol_summary,
)
from pag.experiments.runtime import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "neurips_strategy1.yaml"
DEFAULT_TRACE = ROOT / "traces" / "rich" / "stab_tuples_conf_train_rich.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PAG NeurIPS Strategy 1 evidence package.")
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--predictor-ckpt", type=Path)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--budget-usd", type=float, default=20.0)
    parser.add_argument("--gpu-rate", type=float, default=0.35)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts" / "neurips_strategy1"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    args = build_parser().parse_args()
    config = load_experiment_config(_resolve(args.config))
    if args.dry_run:
        print(json.dumps(protocol_summary(config, args.budget_usd, args.gpu_rate), indent=2))
        return 0
    if args.predictor_ckpt is None:
        raise SystemExit("--predictor-ckpt is required unless --dry-run is used")
    checkpoint = _resolve(args.predictor_ckpt)
    checkpoint_hash = sha256_file(checkpoint)
    run_id = args.run_id or (
        f"strategy1-{config.config_hash[:8]}-{checkpoint_hash[:8]}-"
        f"{hashlib.sha256(args.model_path.encode()).hexdigest()[:8]}"
    )
    orchestrator = StrategyOneOrchestrator(
        config=config,
        model_path=args.model_path,
        predictor_ckpt=checkpoint,
        trace_path=_resolve(args.trace_path),
        device=args.device,
        dtype=args.dtype,
        output_root=_resolve(args.output_root),
        run_id=run_id,
        budget_usd=args.budget_usd,
        gpu_rate=args.gpu_rate,
    )
    print(f"Run ID: {run_id}", flush=True)
    print(f"Artifacts: {orchestrator.run_dir}", flush=True)
    try:
        output = orchestrator.run()
    except ControlledStop as exc:
        print(f"Controlled stop: {exc}", flush=True)
        return 75
    print(f"Complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

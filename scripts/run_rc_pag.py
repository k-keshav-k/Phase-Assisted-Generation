from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from pag.experiments.orchestrator import ControlledStop
from pag.experiments.rc_pag_config import RCPAGConfig, load_rc_pag_config
from pag.experiments.rc_pag_orchestrator import MockRCPAGRuntime, RCPAGOrchestrator

DEFAULT_CONFIG = Path("configs/experiments/rc_pag_neurips_workshop_v9.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered risk-calibrated PAG experiment funnel."
    )
    parser.add_argument("stage", choices=(*RCPAGOrchestrator.ALL_STAGES, "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rc_pag"))
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-confirmatory", action="store_true")
    parser.add_argument(
        "--reuse-development-from",
        type=Path,
        help=(
            "Reuse compatible development artifacts. v9 imports only paired AdaBlock audit "
            "references from a parity-validated v8/v9 pilot; all numerical audits, policy "
            "rules, tuning, calibration, and confirmation remain fresh."
        ),
    )
    return parser


def _real_runtime_factory(
    config: RCPAGConfig,
    *,
    device: str,
    run_dir: Path,
):
    try:
        from pag.experiments.rc_pag_runtime import UnifiedRCPAGRuntime
    except ImportError as exc:
        raise RuntimeError(
            "the real RC-PAG adapter is unavailable; run local verification with --mock"
        ) from exc

    return lambda model: UnifiedRCPAGRuntime(
        config=config,
        model=model,
        device=device,
        run_dir=run_dir,
    )


def _resume_command(args: argparse.Namespace, *, run_id: str) -> str:
    command = [
        sys.executable,
        "scripts/run_rc_pag.py",
        args.stage,
        "--config",
        str(args.config),
        "--output-root",
        str(args.output_root),
        "--run-id",
        run_id,
        "--device",
        args.device,
        "--resume",
    ]
    if args.mock:
        command.append("--mock")
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.allow_confirmatory:
        command.append("--allow-confirmatory")
    if args.reuse_development_from is not None:
        command.extend(("--reuse-development-from", str(args.reuse_development_from)))
    return shlex.join(command)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    includes_confirmation = args.stage in {"confirm", "all"}
    if includes_confirmation and not args.mock and not args.allow_confirmatory:
        parser.error("confirmation requires --allow-confirmatory")
    if includes_confirmation and not args.mock and args.limit is not None:
        parser.error("confirmatory execution rejects --limit")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    config = load_rc_pag_config(args.config)
    run_id = args.run_id or f"rc-pag-{config.config_hash[:12]}"
    run_dir = args.output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        parser.error(f"run directory already exists; pass --resume: {run_dir}")
    if args.mock:
        runtime = MockRCPAGRuntime(calibration_repetitions=300)

        def runtime_factory(model: str) -> MockRCPAGRuntime:
            del model
            return runtime

        development_limit = args.limit if args.limit is not None else 2
        device = "cpu" if args.device == "cuda" else args.device
    else:
        runtime_factory = _real_runtime_factory(config, device=args.device, run_dir=run_dir)
        development_limit = args.limit
        device = args.device
    runner = RCPAGOrchestrator(
        config,
        run_dir,
        device=device,
        runtime_factory=runtime_factory,
        development_limit=development_limit,
        mock_mode=args.mock,
        reuse_development_from=args.reuse_development_from,
    )
    print(f"Run ID: {run_id}")
    print(f"Config hash: {config.config_hash}")
    print(f"Artifacts: {run_dir}")
    try:
        if args.stage == "all":
            runner.run_through("paper")
            completed = "All stages complete"
        else:
            runner.run_stage(args.stage)
            completed = f"{args.stage.capitalize()} complete"
    except ControlledStop as exc:
        print(f"Controlled stop: {exc}", file=sys.stderr)
        print(f"Resume command: {_resume_command(args, run_id=run_id)}")
        return 2
    projection_path = run_dir / "compute_projection.json"
    if projection_path.is_file():
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        print(f"Projected A100-hours: {projection['projected_a100_hours']:.3f}")
        if "projected_storage_bytes" in projection:
            print(f"Projected storage bytes: {projection['projected_storage_bytes']}")
    print(completed)
    print(f"Resume command: {_resume_command(args, run_id=run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

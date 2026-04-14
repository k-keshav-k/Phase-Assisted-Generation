from __future__ import annotations

import argparse
from collections.abc import Sequence

from pag.baselines import run_baseline
from pag.config import load_run_config, load_samples
from pag.evaluation import evaluate_runs
from pag.orchestration.pipeline import run_pipeline
from pag.phases import run_phase_analysis
from pag.scheduler import run_adaptive_decoding
from pag.utils.io import (
    read_adaptive_artifacts,
    read_baseline_artifacts,
    read_phase_artifacts,
    write_adaptive_artifacts,
    write_baseline_artifacts,
    write_evaluation_artifacts,
    write_phase_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_config = load_run_config(args.config)

    if args.command == "baseline":
        samples = load_samples(run_config.dataset_path)
        artifacts = run_baseline(run_config, samples)
        write_baseline_artifacts(artifacts)
        return 0

    if args.command == "phases":
        baseline_artifacts = read_baseline_artifacts(run_config)
        artifacts = run_phase_analysis(run_config, baseline_artifacts)
        write_phase_artifacts(artifacts)
        return 0

    if args.command == "adaptive":
        baseline_artifacts = read_baseline_artifacts(run_config)
        phase_artifacts = read_phase_artifacts(run_config)
        artifacts = run_adaptive_decoding(run_config, baseline_artifacts, phase_artifacts)
        write_adaptive_artifacts(artifacts)
        return 0

    if args.command == "evaluate":
        baseline_artifacts = read_baseline_artifacts(run_config)
        adaptive_artifacts = read_adaptive_artifacts(run_config)
        artifacts = evaluate_runs(run_config, baseline_artifacts, adaptive_artifacts)
        write_evaluation_artifacts(artifacts)
        return 0

    samples = load_samples(run_config.dataset_path)
    run_pipeline(run_config, samples, persist=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PAG scaffold CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("baseline", "phases", "adaptive", "evaluate", "pipeline"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, help="Path to a run config YAML file.")
    return parser

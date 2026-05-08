from __future__ import annotations

from pathlib import Path

from pag.contracts.enums import StageName
from pag.contracts.schemas import RunConfig


def run_directory(run_config: RunConfig) -> Path:
    return Path(run_config.output_root) / run_config.run_id


def stage_directory(run_config: RunConfig, stage: StageName) -> Path:
    return run_directory(run_config) / stage.value


def baseline_paths(run_config: RunConfig) -> dict[str, Path]:
    base = stage_directory(run_config, StageName.BASELINE)
    return {
        "requests": base / "requests.jsonl",
        "traces": base / "traces.jsonl",
        "token_signals": base / "token_signals.jsonl",
        "completions": base / "completions.jsonl",
        "summary": base / "run_summary.json",
    }


def phase_paths(run_config: RunConfig) -> dict[str, Path]:
    base = stage_directory(run_config, StageName.PHASES)
    return {
        "phase_annotations": base / "phase_annotations.jsonl",
        "predictor_dataset": base / "predictor_dataset.jsonl",
        "predictions": base / "predictions.jsonl",
        "predictor_metadata": base / "predictor_metadata.json",
        "summary": base / "run_summary.json",
    }


def adaptive_paths(run_config: RunConfig) -> dict[str, Path]:
    base = stage_directory(run_config, StageName.SCHEDULER)
    return {
        "schedule_decisions": base / "schedule_decisions.jsonl",
        "schedule_plans": base / "schedule_plans.jsonl",
        "adaptive_results": base / "adaptive_results.jsonl",
        "comparison_metrics": base / "comparison_metrics.json",
        "summary": base / "run_summary.json",
    }


def evaluation_paths(run_config: RunConfig) -> dict[str, Path]:
    base = stage_directory(run_config, StageName.EVALUATION)
    return {
        "records": base / "records.jsonl",
        "summary": base / "run_summary.json",
    }

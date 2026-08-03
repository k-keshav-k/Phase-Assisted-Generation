#!/usr/bin/env python3
"""Inspect RC-PAG screen records: where (and how early) each candidate stopped.

Scans the screen-stage records of a run directory and reports, for every
serialized policy stop, the block's budgeted vs actual NFE and the step index
at which the gate fired, plus a per-method summary.

This is the first diagnostic for "the gate fires but NFE savings are small":
- stops at a step_index near the budget (ratio ~1.0) mean the risk estimator
  only certifies safety at the end of a block -> late stops, little savings;
- stops at low ratios that are rare mean the blocker is patience/min_steps,
  not the score threshold.

Records are plain JSON written by RecordStore under
    <run_dir>/screen/<model>/<method>/<sample_id>.json
Only the standard library is required; the RC-PAG package is not imported.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any


def _find_newest_run(artifacts_root: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(artifacts_root, "rc-pag-*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"no rc-pag-* run directories under {artifacts_root}")
    return candidates[0]


def _iter_records(run_dir: str, model: str, method: str):
    pattern = os.path.join(run_dir, "screen", model, method, "*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            yield json.load(handle)


def _step_index(step: dict[str, Any]) -> int | None:
    value = step.get("step_index")
    if value is None:
        observation = step.get("observation") or {}
        value = observation.get("step_index")
    return None if value is None else int(value)


def _scan(run_dir: str, model: str, method: str) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    blocks = 0
    prompts = 0
    for record in _iter_records(run_dir, model, method):
        prompts += 1
        for block in record.get("schedule_history", ()):
            blocks += 1
            budget = block.get("budgeted_refinement_steps")
            nfe = block.get("actual_nfe_used")
            for step in block.get("risk_steps", ()):
                if not step.get("should_stop"):
                    continue
                rows.append(
                    {
                        "model": model,
                        "sample": record.get("sample_id", "?"),
                        "block": block.get("block_index", "?"),
                        "step_index": _step_index(step),
                        "budget": None if budget is None else int(budget),
                        "nfe_used": None if nfe is None else int(nfe),
                        "risk_score": step.get("risk_score"),
                        "safe_streak": step.get("safe_streak"),
                        "reason": step.get("reason", "?"),
                    }
                )
    return rows, blocks, prompts


def _discover_methods(run_dir: str, model: str) -> list[str]:
    directory = os.path.join(run_dir, "screen", model)
    if not os.path.isdir(directory):
        return []
    return sorted(
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    )


def _print_summary(rows: list[dict[str, Any]], blocks: int, prompts: int) -> None:
    print(f"  stops: {len(rows)} | prompts with >=1 stop: "
          f"{len({row['sample'] for row in rows})}/{prompts} | blocks: {blocks}")
    if not rows:
        return
    indices = [row["step_index"] for row in rows if row["step_index"] is not None]
    if indices:
        print("  stop step_index distribution:", dict(sorted(Counter(indices).items())))
    ratios = [
        row["step_index"] / row["budget"]
        for row in rows
        if row["step_index"] is not None and row["budget"]
    ]
    if ratios:
        print(f"  mean stop_idx/budget ratio: {sum(ratios) / len(ratios):.2f} "
              f"(1.0 = stop at the last budgeted step = ~zero savings)")
        print(f"  early stops (ratio < 0.5): {sum(r < 0.5 for r in ratios)} | "
              f"near-budget stops (ratio >= 0.9): {sum(r >= 0.9 for r in ratios)}")
    reasons = Counter(row["reason"] for row in rows)
    if reasons:
        print("  stop reasons:", dict(reasons))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show where RC-PAG candidates stopped during the screen stage."
    )
    parser.add_argument(
        "--run-dir",
        help="Run directory; defaults to the newest rc-pag-* under --artifacts-root.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/rc_pag",
        help="Root used for run-dir auto-detection (default: artifacts/rc_pag).",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Method(s) to inspect (e.g. rgate_t20_p3_v7). Default: all discovered.",
    )
    parser.add_argument(
        "--models",
        default="llada,dream",
        help="Comma-separated models (default: llada,dream).",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir or _find_newest_run(args.artifacts_root)
    print(f"Run directory: {run_dir}")
    models = [name.strip() for name in args.models.split(",") if name.strip()]

    for model in models:
        methods = args.method or _discover_methods(run_dir, model)
        if not methods:
            print(f"\n[{model}] no screen methods found under {run_dir}/screen/{model}")
            continue
        for method in methods:
            rows, blocks, prompts = _scan(run_dir, model, method)
            print(f"\n[{model} / {method}]")
            if rows:
                print("  model\tsample\tblock\tstop_idx\tbudget\tnfe_used\tratio\trisk\tsafe\treason")
                for row in sorted(rows, key=lambda r: (r["model"], str(r["sample"]))):
                    ratio = (
                        f"{row['step_index'] / row['budget']:.2f}"
                        if row["step_index"] is not None and row["budget"]
                        else "-"
                    )
                    print(
                        "\t".join(
                            str(row[key])
                            for key in ("model", "sample", "block", "step_index", "budget", "nfe_used")
                        )
                        + f"\t{ratio}"
                        + f"\t{row['risk_score']}"
                        + f"\t{row['safe_streak']}"
                        + f"\t{row['reason']}"
                    )
            _print_summary(rows, blocks, prompts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

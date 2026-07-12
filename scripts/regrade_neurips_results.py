from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pag.experiments.grading import grade_gsm8k, grade_math500
from pag.experiments.orchestrator import select_history_free
from pag.experiments.report import write_report

STAGE_METHODS = {
    "development": (
        "adablock",
        "gates_only",
        "constant_budget",
        "size_lookup",
        "previous_nfe",
        "random_forest",
        "pag_hard_cap",
        "pag",
    ),
    "gsm8k_test": ("adablock", "gates_only", "size_lookup", "pag"),
    "math500": ("adablock", "pag"),
    "timing": ("adablock", "pag"),
}


def _load_regraded(run_dir: Path, stage: str, method: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted((run_dir / stage / method).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        grade = (
            grade_gsm8k(str(record["generated_text"]), str(record["grade"]["gold_answer"]))
            if record["dataset"] == "gsm8k"
            else grade_math500(str(record["generated_text"]), str(record["grade"]["gold_answer"]))
        )
        record["grade"] = asdict(grade)
        record["grade_version"] = 2
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Regrade completed NeurIPS records without a GPU.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    selected = json.loads((run_dir / "selected_samples.json").read_text(encoding="utf-8"))
    stages = {
        stage: {method: _load_regraded(run_dir, stage, method) for method in methods}
        for stage, methods in STAGE_METHODS.items()
    }
    selection = select_history_free(
        stages["development"],
        max_correct_loss=int(manifest["resolved_config"]["promotion"]["max_correct_loss"]),
    )
    (run_dir / "selection_regraded.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        run_dir / "report_regraded",
        stages=stages,
        bootstrap_samples=int(manifest["resolved_config"]["statistics"]["bootstrap_samples"]),
        seed=int(manifest["resolved_config"]["seed"]),
        confirmatory_ids=set(selected["confirmatory"]),
    )
    print(f"Regraded report: {run_dir / 'report_regraded'}")
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

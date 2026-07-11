from __future__ import annotations

from pag.experiments.datasets import ExperimentSample
from pag.experiments.orchestrator import (
    inspect_stage_work,
    protocol_summary,
    select_history_free,
)
from pag.experiments.records import RecordStore


def _rows(method: str, correct: int, nfes: list[int]):
    return [
        {
            "sample_id": f"sample-{index}",
            "method": method,
            "grade": {"is_correct": index < correct},
            "total_nfe": nfe,
        }
        for index, nfe in enumerate(nfes)
    ]


def test_promotion_uses_lowest_nfe_eligible_history_free_method() -> None:
    records = {
        "adablock": _rows("adablock", 4, [10, 10, 10, 10]),
        "constant_budget": _rows("constant_budget", 2, [6, 6, 6, 6]),
        "size_lookup": _rows("size_lookup", 4, [8, 8, 8, 8]),
    }
    selection = select_history_free(records, max_correct_loss=1)
    assert selection["selected"] == "size_lookup"
    assert not selection["fallback"]


def test_protocol_summary_has_seven_stages(strategy_config) -> None:
    summary = protocol_summary(strategy_config, 20, 0.35)
    assert len(summary["stages"]) == 7
    assert len(summary["development_methods"]) == 8
    assert summary["usable_budget_usd"] == 18


def test_resume_projection_counts_only_missing_records(tmp_path) -> None:
    store = RecordStore(tmp_path, {"config_hash": "abc"})
    samples = [ExperimentSample(f"sample-{index}", "gsm8k", "prompt", "1") for index in range(3)]
    for sample in samples[:2]:
        store.write(
            "gsm8k_test",
            "adablock",
            sample.sample_id,
            {"elapsed_sec": 10.0},
        )
        store.write(
            "gsm8k_test",
            "pag",
            sample.sample_id,
            {"elapsed_sec": 20.0},
        )

    work = inspect_stage_work(store, "gsm8k_test", samples, ("adablock", "pag"))

    assert work.remaining_runs == 2
    assert work.observed_seconds_per_run == 15.0

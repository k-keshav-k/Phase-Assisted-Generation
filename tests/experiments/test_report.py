from __future__ import annotations

from pag.experiments.cross_model_report import write_cross_model_report
from pag.experiments.records import RecordStore
from pag.experiments.report import write_report


def _record(sample_id: str, method: str, *, correct: bool, nfe: int, elapsed: float):
    return {
        "sample_id": sample_id,
        "method": method,
        "grade": {"is_correct": correct},
        "total_nfe": nfe,
        "elapsed_sec": elapsed,
    }


def test_report_writes_statistics_tables_and_figure(tmp_path) -> None:
    stages = {
        "gsm8k_test": {
            "adablock": [_record("a", "adablock", correct=True, nfe=10, elapsed=1.0)],
            "pag": [_record("a", "pag", correct=True, nfe=8, elapsed=0.9)],
        }
    }
    summary = write_report(
        tmp_path,
        stages=stages,
        bootstrap_samples=100,
        seed=20260710,
        confirmatory_ids={"a"},
    )
    assert summary["gsm8k_test"]["pag"]["nfe_difference"]["estimate"] == -2
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "tables" / "gsm8k_test.tex").exists()
    assert (tmp_path / "tables" / "paired_gsm8k.tex").exists()
    assert (tmp_path / "figures" / "nfe_deltas.pdf").exists()
    assert (tmp_path / "figures" / "nfe_parity.pdf").exists()


def test_cross_model_report_emits_locked_claim_audit(tmp_path) -> None:
    store = RecordStore(tmp_path, {"config_hash": "abc"})
    for model in ("llada", "dream"):
        for dataset in ("test_gsm8k", "test_math500"):
            stage = f"{dataset}/{model}"
            for index in range(4):
                sample_id = f"{dataset}-{index}"
                for method, nfe in (
                    ("adablock", 10),
                    ("size_lookup", 9),
                    ("residual_pag", 8),
                ):
                    store.write(
                        stage,
                        method,
                        sample_id,
                        {
                            "grade": {"is_correct": True},
                            "total_nfe": nfe,
                            "elapsed_sec": float(nfe),
                        },
                    )
    audit = write_cross_model_report(
        tmp_path,
        identity=store.identity,
        bootstrap_samples=1000,
        seed=7,
        thresholds={
            "minimum_nfe_reduction": 0.10,
            "minimum_lookup_reduction": 0.03,
            "minimum_accuracy_ci": -0.02,
        },
    )
    assert audit["headline_eligible"] is True
    assert (tmp_path / "report" / "claim_audit.json").is_file()
    assert (tmp_path / "report" / "tables" / "cross_model.tex").is_file()
    assert (tmp_path / "report" / "figures" / "nfe_reduction.pdf").is_file()

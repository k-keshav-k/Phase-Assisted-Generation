from __future__ import annotations

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

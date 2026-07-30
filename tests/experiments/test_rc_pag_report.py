from __future__ import annotations

import json

import pytest

from pag.experiments.rc_pag_report import write_rc_pag_report


def _certificate(*, certified: bool = True, mock: bool = False) -> dict:
    candidates = []
    for model in ("llada", "dream"):
        for name, risk, nfe in (
            ("local_t025_p2", 0.01, 60.0),
            ("history_t025_p2", 0.01, 55.0),
        ):
            candidates.append(
                {
                    "name": f"{model}/{name}",
                    "failures": 2,
                    "count": 300,
                    "empirical_risk": risk,
                    "upper_risk_bound": 0.04,
                    "pvalue": 0.001 if certified else 0.5,
                    "corrected_cutoff": 0.0042,
                    "certified": certified,
                    "mean_nfe": nfe,
                }
            )
    return {
        "alpha": 0.05,
        "familywise_delta": 0.05,
        "selected": "history_t025_p2" if certified else "full_budget",
        "fallback": not certified,
        "candidates": candidates,
        "mock": mock,
    }


def _records(*, history_nfe: float = 55.0) -> dict:
    result = {}
    for model in ("llada", "dream"):
        result[model] = {}
        for dataset in ("gsm8k_test", "math500", "mbpp_sanitized", "humaneval"):
            methods = {}
            for method, nfe in (
                ("adablock", 80.0),
                ("best_nonlearned", 66.0),
                ("rc_pag_local", 60.0),
                ("rc_pag_history", history_nfe),
            ):
                methods[method] = [
                    {
                        "sample_id": f"{dataset}-{index}",
                        "total_nfe": nfe + (index % 2),
                        "elapsed_sec": nfe / 20 + index * 0.001,
                        "is_correct": index != 0,
                    }
                    for index in range(8)
                ]
            result[model][dataset] = methods
    return result


def _workshop_records() -> dict:
    records = _records()
    for model in records.values():
        for dataset in model.values():
            dataset.pop("rc_pag_local")
    return records


def test_failed_risk_certificate_blocks_headline(tmp_path):
    audit = write_rc_pag_report(
        tmp_path,
        records=_records(),
        certificate=_certificate(certified=False),
        bootstrap_samples=200,
        seed=7,
    )

    assert not audit["headline_eligible"]
    assert "risk_certificate" in audit["failed_gates"]
    assert "passed" not in (tmp_path / "report" / "headline.tex").read_text().lower()


def test_positive_evidence_writes_complete_publication_artifacts(tmp_path):
    audit = write_rc_pag_report(
        tmp_path,
        records=_records(),
        certificate=_certificate(),
        bootstrap_samples=200,
        seed=7,
    )

    assert audit["headline_eligible"]
    for name in ("main_results.tex", "calibration.tex", "ablations.tex", "headline.tex"):
        assert (tmp_path / "report" / "tables" / name).is_file()
    for name in ("nfe_accuracy.pdf", "risk_compute.pdf", "reliability.pdf"):
        assert (tmp_path / "report" / "figures" / name).stat().st_size > 0
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["normalized_cross_model"]["rc_pag_history"] < 1.0


def test_history_credit_requires_interval_excluding_zero(tmp_path):
    audit = write_rc_pag_report(
        tmp_path,
        records=_records(history_nfe=60.0),
        certificate=_certificate(),
        bootstrap_samples=200,
        seed=7,
    )

    assert not audit["headline_eligible"]
    assert "history_frontier" in audit["failed_gates"]


def test_missing_paired_coverage_is_an_error(tmp_path):
    records = _records()
    records["llada"]["gsm8k_test"]["rc_pag_history"].pop()

    with pytest.raises(ValueError, match="coverage"):
        write_rc_pag_report(
            tmp_path,
            records=records,
            certificate=_certificate(),
            bootstrap_samples=100,
            seed=7,
        )


def test_workshop_report_accepts_single_certified_variant(tmp_path):
    audit = write_rc_pag_report(
        tmp_path,
        records=_workshop_records(),
        certificate=_certificate(),
        bootstrap_samples=200,
        seed=7,
        methods=("adablock", "best_nonlearned", "rc_pag_history"),
        primary_method="rc_pag_history",
        require_history_frontier_ci=False,
    )

    assert audit["headline_eligible"]
    assert "history_frontier" not in audit["gates"]

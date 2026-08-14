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


def _v2_certificate(*, certified: bool = True) -> dict:
    selected = {"llada": "local_q20_tail25_p2", "dream": "local_q05_tail25_p2"}
    return {
        "alpha": 0.02,
        "familywise_delta": 0.05,
        "selected": "per_model_frozen_policy" if certified else "full_budget",
        "fallback": not certified,
        "selected_by_model": selected,
        "candidates": [
            {
                "name": f"{model}/{name}",
                "failures": 0,
                "count": 300,
                "empirical_risk": 0.0,
                "upper_risk_bound": 0.018,
                "pvalue": 0.002 if certified else 0.5,
                "corrected_cutoff": 0.025,
                "certified": certified,
                "mean_nfe": 55.0,
            }
            for model, name in selected.items()
        ],
    }


def _v4_certificate(*, certified: bool = True) -> dict:
    payload = _v2_certificate(certified=certified)
    payload["minimum_nfe_reduction"] = 0.05
    payload["hypotheses"] = 4
    payload["certificate_mode"] = "joint_harm_and_compute"
    for row in payload["candidates"]:
        row.update(
            {
                "harm_pvalue": row["pvalue"],
                "compute_pvalue": 0.001 if certified else 0.5,
                "harm_certified": certified,
                "compute_certified": certified,
                "empirical_nfe_reduction": 0.15,
                "lower_nfe_reduction_bound": 0.08 if certified else 0.01,
            }
        )
    return payload


def _v2_records() -> dict:
    records = _records()
    for model in records.values():
        for dataset in model.values():
            dataset["rc_pag_selected"] = dataset.pop("rc_pag_history")
            dataset.pop("rc_pag_local")
            for index, row in enumerate(dataset["adablock"]):
                row["generated_ids"] = [index, 1]
            for index, row in enumerate(dataset["rc_pag_selected"]):
                row["generated_ids"] = [index, 1]
    return records


def _v9_certificate() -> dict:
    payload = _v2_certificate()
    payload.update(
        {
            "certificate_mode": "hardware_scoped_execution_equivalence",
            "mock": False,
        }
    )
    return payload


def _v9_records() -> dict:
    records = _v2_records()
    for model in records.values():
        for dataset in model.values():
            for index, baseline in enumerate(dataset["adablock"]):
                baseline.update(
                    {
                        "state_trajectory_digest": [f"state-{index}"],
                        "evaluated_rows": 80,
                        "serial_forward_calls": 80,
                        "model_time_sec": 4.0,
                    }
                )
            for index, candidate in enumerate(dataset["rc_pag_selected"]):
                candidate.update(
                    {
                        "elapsed_sec": dataset["adablock"][index]["elapsed_sec"] * 0.8,
                        "state_trajectory_digest": [f"state-{index}"],
                        "evaluated_rows": 70,
                        "serial_forward_calls": 60,
                        "model_time_sec": 3.2,
                        "schedule_history": [
                            {
                                "verified_sequence_safe": True,
                                "speculation_steps": [
                                    {
                                        "accepted_draft_edges": 1,
                                        "verified_transitions": 2,
                                        "evaluated_nodes": 2,
                                        "nfe_saved": 1,
                                        "guard_passed": True,
                                        "reference_checked": False,
                                        "canonical_fallback_rows": 0,
                                    }
                                ],
                            }
                        ],
                    }
                )
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


def test_v2_report_audits_frozen_per_model_policies_and_end_to_end_harm(tmp_path):
    audit = write_rc_pag_report(
        tmp_path,
        records=_v2_records(),
        certificate=_v2_certificate(),
        bootstrap_samples=200,
        seed=7,
        methods=("adablock", "best_nonlearned", "rc_pag_selected"),
        primary_method="rc_pag_selected",
        require_history_frontier_ci=False,
    )

    assert audit["headline_eligible"]
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    comparison = summary["llada"]["gsm8k_test"]["comparisons"]
    assert comparison["rc_pag_selected_vs_adablock"]["harmful_regressions"] == 0
    assert comparison["rc_pag_selected_vs_adablock"]["sequence_disagreements"] == 0


def test_v4_report_writes_joint_harm_and_compute_certificate(tmp_path):
    audit = write_rc_pag_report(
        tmp_path,
        records=_v2_records(),
        certificate=_v4_certificate(),
        bootstrap_samples=200,
        seed=7,
        methods=("adablock", "best_nonlearned", "rc_pag_selected"),
        primary_method="rc_pag_selected",
        require_history_frontier_ci=False,
    )

    table = (tmp_path / "report" / "tables" / "calibration.tex").read_text()
    diagnostics = json.loads((tmp_path / "report" / "risk_diagnostics.json").read_text())
    assert audit["headline_eligible"]
    assert "Saving LCB" in table
    assert "$p_H$" in table
    assert diagnostics["candidates"][0]["compute_certified"]


def test_v6_requires_positive_model_level_compute_lower_bounds(tmp_path):
    audit = write_rc_pag_report(
        tmp_path,
        records=_v2_records(),
        certificate=_v2_certificate(),
        bootstrap_samples=200,
        seed=7,
        methods=("adablock", "best_nonlearned", "rc_pag_selected"),
        primary_method="rc_pag_selected",
        require_history_frontier_ci=False,
        minimum_model_nfe_reduction_lower_ci=0.05,
    )

    assert audit["gates"]["model_nfe_reduction_lower_ci"]
    assert all(
        interval["lower"] > 0.05 for interval in audit["details"]["model_nfe_reduction"].values()
    )


def test_v6_compute_gate_rejects_a_small_nfe_improvement(tmp_path):
    records = _v2_records()
    for model in records.values():
        for dataset in model.values():
            for candidate, baseline in zip(
                dataset["rc_pag_selected"],
                dataset["adablock"],
                strict=True,
            ):
                candidate["total_nfe"] = float(baseline["total_nfe"]) - 2.0
    audit = write_rc_pag_report(
        tmp_path,
        records=records,
        certificate=_v2_certificate(),
        bootstrap_samples=200,
        seed=7,
        methods=("adablock", "best_nonlearned", "rc_pag_selected"),
        primary_method="rc_pag_selected",
        require_history_frontier_ci=False,
        minimum_model_nfe_reduction_lower_ci=0.05,
    )

    assert not audit["headline_eligible"]
    assert "model_nfe_reduction_lower_ci" in audit["failed_gates"]


def test_v9_report_certifies_equivalence_latency_and_honest_work(tmp_path):
    audit = write_rc_pag_report(
        tmp_path,
        records=_v9_records(),
        certificate=_v9_certificate(),
        bootstrap_samples=200,
        seed=7,
        methods=("adablock", "best_nonlearned", "rc_pag_selected"),
        primary_method="rc_pag_selected",
        require_history_frontier_ci=False,
        minimum_model_latency_reduction_lower_ci=0.05,
        require_evaluated_row_nonincrease=True,
        require_trajectory_equivalence=True,
    )

    assert audit["headline_eligible"]
    assert audit["gates"]["exact_sequence_equivalence"]
    assert audit["gates"]["exact_trajectory_equivalence"]
    assert audit["gates"]["model_latency_reduction_lower_ci"]
    assert audit["gates"]["evaluated_row_nonincrease"]
    assert "model_nfe_reduction_lower_ci" not in audit["gates"]
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    values = summary["llada"]["gsm8k_test"]["methods"]["rc_pag_selected"]
    assert values["mean_serial_forward_calls"] == 60.0
    assert values["mean_evaluated_rows"] == 70.0
    assert values["speculation"]["guard_evidence_complete"]

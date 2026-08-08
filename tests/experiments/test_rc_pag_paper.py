from pathlib import Path


def test_workshop_paper_defers_numbers_and_states_v8_equivalence_claim() -> None:
    paper = Path("writeup/rc_pag_workshop.tex").read_text(encoding="utf-8")

    assert "F(s)=\\operatorname{Transfer}" in paper
    assert "Trajectory equivalence" in paper
    assert "$u$, not $d_{j+1}$" in paper
    assert "Risk controls capacity, never acceptance" in paper
    assert "(2,1,.05,.15,.75)" in paper
    assert "least 5\\% physical NFE" in paper
    assert "zero token-sequence disagreements" in paper
    assert "SSD" in paper
    assert "S2D2" in paper
    assert "generated/headline.tex" in paper
    assert "Confirmatory results pending" in paper
    assert "RC-PAG-MAIN-PAGES" in paper
    assert "500 GSM8K" in paper
    assert "200 MATH-500" in paper
    assert "100 sanitized MBPP" in paper
    assert "64 remaining HumanEval" in paper


def test_paper_builder_guards_mock_results_and_page_limit() -> None:
    builder = Path("scripts/build_rc_pag_paper.sh").read_text(encoding="utf-8")

    assert "refusing to build numerical paper results from mock evidence" in builder
    assert "len(names) != 2" in builder
    assert "adablock_correct_candidate_wrong" in builder
    assert "exact_trajectory_with_paired_compute_evidence" in builder
    assert 'audit.get("gates", {}).get("exact_sequence_equivalence")' in builder
    assert 'audit.get("gates", {}).get("verified_transition_evidence")' in builder
    assert 'certificate.get("minimum_nfe_reduction") is not None' in builder
    assert 'required_model_nfe_reduction_lower_ci") == 0.05' in builder
    assert "MAIN_PAGES > 8" in builder
    assert "neurips_2026.sty" in builder


def test_runbook_documents_manual_and_one_command_execution() -> None:
    runbook = Path("docs/rc_pag_runbook.md").read_text(encoding="utf-8")

    assert "CPU estimator fit" in runbook
    assert "CPU report and paper manifest" in runbook
    assert "Prefer the stage-by-stage commands" in runbook
    assert "submit_rc_pag_all.sh" in runbook
    assert "CPU-only stages on the allocated node" in runbook
    assert "RC_PAG_ALLOW_CONFIRMATORY=1" in runbook

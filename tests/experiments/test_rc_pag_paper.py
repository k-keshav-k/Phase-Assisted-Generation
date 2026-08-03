from pathlib import Path


def test_workshop_paper_defers_numbers_and_states_v6_harm_certificate() -> None:
    paper = Path("writeup/rc_pag_workshop.tex").read_text(encoding="utf-8")

    assert "hence $J=2$" in paper
    assert "A(G_0(X),X)=1" in paper
    assert "\\Delta_\\lambda(X)=C_\\lambda(X)-C_0(X)" in paper
    assert "p_\\lambda=" in paper
    assert "p_\\lambda^C" not in paper
    assert "p_{b,t}=\\widehat q(S_{b,t})" in paper
    assert "(0.02,1,4),(0.05,2,3),(0.10,3,2)" in paper
    assert "Q\\leftarrow Q+p_{b,t}" in paper
    assert "every position that remains masked has the same proposal" in paper
    assert "saves at least 8\\% NFE" in paper
    assert "AUROC 0.456 on Dream and 0.372 on LLaDA" in paper
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
    assert (
        'certificate.get("certificate_mode") != "harm_only_with_paired_compute_evidence"' in builder
    )
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

from pathlib import Path


def test_workshop_paper_defers_numbers_and_states_strict_risk() -> None:
    paper = Path("writeup/rc_pag_workshop.tex").read_text(encoding="utf-8")

    assert "M=2|\\family|=12" in paper
    assert "same decoder states" in paper
    assert "not whether either answer is semantically correct" in paper
    assert "generated/headline.tex" in paper
    assert "Confirmatory results pending" in paper
    assert "RC-PAG-MAIN-PAGES" in paper


def test_paper_builder_guards_mock_results_and_page_limit() -> None:
    builder = Path("scripts/build_rc_pag_paper.sh").read_text(encoding="utf-8")

    assert "refusing to build numerical paper results from mock evidence" in builder
    assert 'len(names) != 12' in builder
    assert "MAIN_PAGES > 8" in builder
    assert "neurips_2026.sty" in builder


def test_runbook_keeps_cpu_stages_off_a100() -> None:
    runbook = Path("docs/rc_pag_runbook.md").read_text(encoding="utf-8")

    assert "CPU estimator fit" in runbook
    assert "CPU report and paper manifest" in runbook
    assert "Never submit `all`" in runbook
    assert "RC_PAG_ALLOW_CONFIRMATORY=1" in runbook

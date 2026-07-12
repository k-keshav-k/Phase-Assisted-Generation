from pathlib import Path


def test_paper_does_not_retain_invalidated_nfe_claim() -> None:
    paper = Path("writeup/final_report.tex").read_text(encoding="utf-8")
    assert r"21.4\%" not in paper
    assert "matches AdaBlock answer accuracy while reducing" not in paper
    assert "significant response time savings" not in paper

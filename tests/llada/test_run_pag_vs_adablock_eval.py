from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

run_eval = importlib.import_module("run_pag_vs_adablock_eval")


def test_load_eval_prompts_reads_expected_substrings(tmp_path) -> None:
    prompt_file = tmp_path / "eval.jsonl"
    prompt_file.write_text(
        '{"id":"math","category":"reasoning","tags":["a"],'
        '"expected_contains":["42"],"expected_answers":["42"],'
        '"prompt":"What is the answer?"}\n',
        encoding="utf-8",
    )

    records = run_eval.load_eval_prompts(prompt_file)

    assert len(records) == 1
    assert records[0].prompt_id == "math"
    assert records[0].category == "reasoning"
    assert records[0].tags == ["a"]
    assert records[0].expected_contains == ["42"]
    assert records[0].expected_answers == ["42"]


def test_substring_score_is_case_insensitive() -> None:
    score = run_eval._substring_score("The answer is Forty Two and 42.", ["42", "answer"])

    assert score == {
        "expected_contains": ["42", "answer"],
        "matched": ["42", "answer"],
        "missing": [],
        "score": 1.0,
    }


def test_answer_score_rejects_embedded_numeric_match() -> None:
    score = run_eval._answer_score("The model produced 772 kilometers.", ["72"])

    assert score == {
        "expected_answers": ["72"],
        "matched": [],
        "missing": ["72"],
        "is_correct": False,
        "score": 0.0,
    }


def test_answer_score_accepts_any_answer_variant() -> None:
    score = run_eval._answer_score("The final price is $97.20.", ["97.2", "97.20"])

    assert score["matched"] == ["97.20"]
    assert score["is_correct"] is True
    assert score["score"] == 1.0


def test_comparison_delta_reports_nfe_ratio() -> None:
    pag = {
        "metrics": {
            "total_nfe": 10,
            "elapsed_sec": 2.0,
            "num_blocks": 4,
            "substring_check": {"score": 0.5},
            "answer_check": {"score": 1.0},
        }
    }
    adablock = {
        "metrics": {
            "total_nfe": 20,
            "elapsed_sec": 3.5,
            "num_blocks": 3,
            "substring_check": {"score": 1.0},
            "answer_check": {"score": 0.0},
        }
    }

    delta = run_eval._comparison_delta(pag, adablock)

    assert delta["nfe_delta_pag_minus_adablock"] == -10
    assert delta["nfe_ratio_pag_over_adablock"] == 0.5
    assert delta["elapsed_delta_sec_pag_minus_adablock"] == -1.5
    assert delta["block_count_delta_pag_minus_adablock"] == 1
    assert delta["substring_score_delta_pag_minus_adablock"] == -0.5
    assert delta["answer_score_delta_pag_minus_adablock"] == 1.0

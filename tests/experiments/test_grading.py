from __future__ import annotations

import pytest

from pag.experiments.grading import grade_gsm8k, grade_math500


@pytest.mark.parametrize(
    ("text", "gold", "correct"),
    [
        ("Final answer: 1,234", "1234", True),
        ("I considered 72. Final answer: 772", "72", False),
        ("Final answer: -3/2", "-1.5", True),
        ("The arithmetic contains 42 but no final marker", "42", False),
    ],
)
def test_grade_gsm8k(text: str, gold: str, correct: bool) -> None:
    assert grade_gsm8k(text, gold).is_correct is correct


def test_grade_math500_equivalent_expression() -> None:
    result = grade_math500("The result is \\boxed{1/2}.", "\\frac{2}{4}")
    assert result.is_correct

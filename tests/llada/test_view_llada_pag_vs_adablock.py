from __future__ import annotations

import importlib

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("streamlit")
ui = importlib.import_module("scripts.view_llada_pag_vs_adablock")


def test_render_blocked_output_uses_single_inline_element_per_block(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ui.st, "markdown", lambda value, **kwargs: calls.append(value))
    monkeypatch.setattr(ui.st, "write", lambda value: calls.append(str(value)))
    monkeypatch.setattr(ui.st, "caption", lambda value: calls.append(str(value)))

    ui._render_blocked_output(
        label="PAG",
        output={
            "generated_text": "a\nb",
            "block_visualization": [
                {
                    "block_index": 0,
                    "applied_block_size": 2,
                    "budgeted_refinement_steps": 4,
                    "actual_nfe_used": 2,
                    "block_text": "a\nb </div>",
                }
            ],
        },
        color=ui.PAG_BLOCK_COLOR,
    )

    html = calls[-1]
    assert "<div" not in html
    assert "</div>" not in html
    assert html.count("class='generation-block'") == 1
    assert "<br>" in html
    assert "&lt;/div&gt;" in html
    assert "data-refinement='4'" in html


def test_validate_graph_data_accepts_consistent_rows() -> None:
    method_df = pd.DataFrame(
        [
            {
                "record_index": 0,
                "method": "PAG",
                "total_nfe": 10,
                "answer_score": 0.0,
                "answer_correct": False,
            },
            {
                "record_index": 0,
                "method": "AdaBlock",
                "total_nfe": 12,
                "answer_score": 1.0,
                "answer_correct": True,
            },
        ]
    )
    delta_df = pd.DataFrame(
        [
            {
                "record_index": 0,
                "nfe_delta_pag_minus_adablock": -2,
                "answer_score_delta_pag_minus_adablock": -1.0,
            }
        ]
    )

    assert ui.validate_graph_data(method_df, delta_df) == []


def test_validate_graph_data_flags_wrong_answer_delta() -> None:
    method_df = pd.DataFrame(
        [
            {
                "record_index": 0,
                "method": "PAG",
                "total_nfe": 10,
                "answer_score": 0.0,
                "answer_correct": False,
            },
            {
                "record_index": 0,
                "method": "AdaBlock",
                "total_nfe": 12,
                "answer_score": 1.0,
                "answer_correct": True,
            },
        ]
    )
    delta_df = pd.DataFrame(
        [
            {
                "record_index": 0,
                "nfe_delta_pag_minus_adablock": -2,
                "answer_score_delta_pag_minus_adablock": 1.0,
            }
        ]
    )

    issues = ui.validate_graph_data(method_df, delta_df)

    assert any("Answer delta mismatch" in issue for issue in issues)

from __future__ import annotations

import importlib

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("streamlit")
ui = importlib.import_module("scripts.view_llada_pag_vs_adablock")


def test_render_blocked_output_escapes_inline_block_text(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ui.st, "markdown", lambda value, **kwargs: calls.append(value))
    monkeypatch.setattr(ui.st, "write", lambda value: calls.append(str(value)))
    monkeypatch.setattr(ui.st, "caption", lambda value: calls.append(str(value)))
    component_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        ui.components,
        "html",
        lambda body, **kwargs: component_calls.append({"body": body, **kwargs}),
    )

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
                    "block_text": "a\nb </div> </span>",
                }
            ],
        },
        color=ui.PAG_BLOCK_COLOR,
    )

    assert calls[0] == "**PAG Output**"
    assert len(component_calls) == 1
    body = str(component_calls[0]["body"])
    assert "class=\"generation-block\"" in body
    assert "data-refinement=\"4\"" in body
    assert "a<br>b &lt;/div&gt; &lt;/span&gt;" in body
    assert "a\nb </div> </span>" not in body


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


def test_flatten_methods_exposes_split_timing_metrics() -> None:
    records = [
        {
            "prompt_id": "p",
            "pag": {
                "metrics": {
                    "elapsed_sec": 1.0,
                    "total_elapsed_sec": 1.0,
                    "scheduler_predict_time_sec": 0.1,
                    "llada_decode_time_sec": 0.9,
                    "substring_check": {},
                    "answer_check": {},
                }
            },
            "adablock": {
                "metrics": {
                    "elapsed_sec": 1.2,
                    "total_elapsed_sec": 1.2,
                    "scheduler_predict_time_sec": 0.0,
                    "llada_decode_time_sec": 1.2,
                    "substring_check": {},
                    "answer_check": {},
                }
            },
        }
    ]

    method_df = ui.flatten_methods(records)
    pag_row = method_df[method_df["method"] == "PAG"].iloc[0]

    assert pag_row["total_elapsed_sec"] == 1.0
    assert pag_row["scheduler_predict_time_sec"] == 0.1
    assert pag_row["llada_decode_time_sec"] == 0.9


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

from __future__ import annotations

import json
import sys
from pathlib import Path

from phase_cpd.collect_traces import main as collect_traces_main
from phase_cpd.importers.common import load_step_dump_as_trace


def test_dream_importer_converts_step_dump_into_trace(tmp_path: Path) -> None:
    raw_path = tmp_path / "dream_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "trace_id": "dream-sample-001",
                "prompt": "Explain why adaptive diffusion decoding can help.",
                "model_name": "dream-7b",
                "decoding_metadata": {"run_id": "dream-run-001"},
                "steps": [
                    {
                        "step_index": 0,
                        "tokens": [
                            {
                                "token_index": 0,
                                "token_text": "Adaptive",
                                "top1_prob": 0.62,
                                "selected_logit": 2.4,
                            },
                            {
                                "token_index": 1,
                                "token_text": " decoding",
                                "top1_prob": 0.58,
                                "selected_logit": 2.2,
                            },
                        ],
                    },
                    {
                        "step_index": 1,
                        "tokens": [
                            {
                                "token_index": 0,
                                "token_text": "Adaptive",
                                "top1_prob": 0.83,
                                "selected_logit": 3.1,
                            },
                            {
                                "token_index": 1,
                                "token_text": " decoding",
                                "top1_prob": 0.79,
                                "selected_logit": 2.9,
                            },
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    trace = load_step_dump_as_trace(
        raw_path,
        backend="dream",
        default_model_name="dream-7b",
    )

    assert trace.trace_id == "dream-sample-001"
    assert trace.final_text == "Adaptive decoding"
    assert len(trace.tokens) == 2
    assert [observation.step_index for observation in trace.tokens[0].observations] == [0, 1]
    assert trace.tokens[0].observations[-1].top1_prob == 0.83


def test_collect_traces_writes_converted_real_backend_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample.json").write_text(
        json.dumps(
            {
                "trace_id": "dream-sample-002",
                "prompt": "Describe phase changes.",
                "steps": [
                    {
                        "step_index": 0,
                        "tokens": [
                            {"token_index": 0, "token_text": "Phase", "top1_prob": 0.6},
                            {"token_index": 1, "token_text": " changes", "top1_prob": 0.55},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "converted"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_traces.py",
            "--backend",
            "dream",
            "--source",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = collect_traces_main()

    assert exit_code == 0
    assert (output_dir / "dream-sample-002.json").exists()


def test_dream_importer_uses_latest_token_text_for_final_trace(tmp_path: Path) -> None:
    raw_path = tmp_path / "dream_latest_step.json"
    raw_path.write_text(
        json.dumps(
            {
                "trace_id": "dream-sample-003",
                "prompt": "Explain the answer.",
                "steps": [
                    {
                        "step_index": 0,
                        "tokens": [
                            {"token_index": 0, "token_text": "<mask>", "top1_prob": 0.1},
                            {"token_index": 1, "token_text": " guess", "top1_prob": 0.2},
                        ],
                    },
                    {
                        "step_index": 1,
                        "tokens": [
                            {"token_index": 0, "token_text": "Final", "top1_prob": 0.8},
                            {"token_index": 1, "token_text": " answer", "top1_prob": 0.7},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    trace = load_step_dump_as_trace(
        raw_path,
        backend="dream",
        default_model_name="dream-7b",
    )

    assert trace.final_text == "Final answer"
    assert [token.token_text for token in trace.tokens] == ["Final", " answer"]

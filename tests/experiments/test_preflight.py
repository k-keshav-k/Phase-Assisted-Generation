from __future__ import annotations

import json

import torch

from pag.experiments.orchestrator import run_preflight

FIELDS = [
    "block_size",
    "nfe",
    "mean_top1_confidence",
    "min_top1_confidence",
    "digit_fraction",
    "delimiter_fraction",
]


def _trace(path) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for index in range(5000):
            file_obj.write(
                json.dumps({"sample_id": f"gsm8k-train-{index:04d}", "tuples": []}) + "\n"
            )


def test_preflight_accepts_compatible_checkpoint_and_trace(tmp_path, strategy_config) -> None:
    checkpoint = tmp_path / "predictor.pt"
    torch.save({"input_fields": FIELDS}, checkpoint)
    trace = tmp_path / "trace.jsonl"
    _trace(trace)
    result = run_preflight(
        config=strategy_config,
        predictor_ckpt=checkpoint,
        trace_path=trace,
        output_root=tmp_path,
        device="cpu",
    )
    assert result.ok, result.errors


def test_preflight_rejects_incompatible_checkpoint(tmp_path, strategy_config) -> None:
    checkpoint = tmp_path / "predictor.pt"
    torch.save({"input_fields": ["block_size"]}, checkpoint)
    trace = tmp_path / "trace.jsonl"
    _trace(trace)
    result = run_preflight(
        config=strategy_config,
        predictor_ckpt=checkpoint,
        trace_path=trace,
        output_root=tmp_path,
        device="cpu",
    )
    assert not result.ok
    assert any("six-feature schema" in error for error in result.errors)

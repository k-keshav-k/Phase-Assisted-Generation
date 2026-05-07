"""Unit tests for phase_predict.predict."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from phase_predict.dataset import PhaseSequenceDataset
from phase_predict.model import PhaseTransformer
from phase_predict.predict import Predictor
from phase_predict.schema import ModelConfig, PhaseTuple


def _make_sequence(n: int = 20) -> list[PhaseTuple]:
    return [PhaseTuple((i % 8) + 1, i % 6) for i in range(n)]


def _make_predictor() -> tuple[Predictor, PhaseSequenceDataset]:
    cfg = ModelConfig(window_size=4, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    seq = _make_sequence(20)
    ds = PhaseSequenceDataset(seq, cfg)
    model = PhaseTransformer(cfg)
    predictor = Predictor(
        model,
        input_mean=ds.input_mean,
        input_std=ds.input_std,
    )
    return predictor, ds


class TestPredictor:
    def test_predict_returns_phase_tuple(self) -> None:
        predictor, ds = _make_predictor()
        seq = _make_sequence(20)
        result = predictor.predict(seq[-4:])
        assert isinstance(result.predicted_tuple, PhaseTuple)

    def test_predicted_values_non_negative(self) -> None:
        predictor, ds = _make_predictor()
        seq = _make_sequence(20)
        result = predictor.predict(seq[-4:])
        t = result.predicted_tuple
        assert t.block_size >= 0
        assert t.refinement_steps >= 0

    def test_raw_output_length(self) -> None:
        predictor, ds = _make_predictor()
        seq = _make_sequence(20)
        result = predictor.predict(seq[-4:])
        assert len(result.raw_output) == 128

    def test_short_context_window_padded(self) -> None:
        """Predict should accept context shorter than window_size."""
        predictor, _ = _make_predictor()
        short_ctx = [PhaseTuple(4, 3)]  # only 1 tuple; window_size is 4
        result = predictor.predict(short_ctx)
        assert isinstance(result.predicted_tuple, PhaseTuple)

    def test_long_context_window_truncated(self) -> None:
        """Predict should accept context longer than window_size."""
        predictor, _ = _make_predictor()
        long_ctx = _make_sequence(20)  # 20 tuples; window_size is 4
        result = predictor.predict(long_ctx)
        assert result.metadata["window_size_used"] == 4

    def test_deterministic_in_eval_mode(self) -> None:
        predictor, _ = _make_predictor()
        ctx = _make_sequence(4)
        r1 = predictor.predict(ctx)
        r2 = predictor.predict(ctx)
        assert r1.predicted_tuple == r2.predicted_tuple

    def test_save_and_load_checkpoint(self) -> None:
        predictor, _ = _make_predictor()
        ctx = _make_sequence(4)
        original_pred = predictor.predict(ctx).predicted_tuple

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "model.pt")
            predictor.save_checkpoint(ckpt_path)

            loaded = Predictor.from_checkpoint(ckpt_path)
            loaded_pred = loaded.predict(ctx).predicted_tuple

        assert original_pred == loaded_pred

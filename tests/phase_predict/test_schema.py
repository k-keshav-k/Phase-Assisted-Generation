"""Unit tests for phase_predict.schema."""

from __future__ import annotations

import pytest

from phase_predict.schema import ModelConfig, PhaseTuple, PredictionResult, TrainConfig


class TestPhaseTuple:
    def test_named_fields(self) -> None:
        t = PhaseTuple(block_size=4, refinement_steps=3)
        assert t.block_size == 4
        assert t.refinement_steps == 3

    def test_positional_construction(self) -> None:
        t = PhaseTuple(8, 10)
        assert t[0] == 8
        assert t[1] == 10

    def test_equality(self) -> None:
        assert PhaseTuple(1, 3) == PhaseTuple(1, 3)
        assert PhaseTuple(1, 3) != PhaseTuple(1, 4)

    def test_iterable(self) -> None:
        t = PhaseTuple(4, 3)
        assert list(t) == [4, 3]


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.window_size == 8
        assert cfg.d_model == 64
        assert cfg.n_heads == 4
        assert cfg.n_layers == 2
        assert cfg.tuple_size is None

    def test_custom_values(self) -> None:
        cfg = ModelConfig(window_size=4, d_model=32, n_heads=2, n_layers=1, tuple_size=2)
        assert cfg.window_size == 4
        assert cfg.d_model == 32
        assert cfg.n_heads == 2

    def test_d_model_not_divisible_by_n_heads_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(d_model=33, n_heads=4)

    def test_window_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            ModelConfig(window_size=0)

    def test_tuple_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="tuple_size"):
            ModelConfig(tuple_size=0)


class TestTrainConfig:
    def test_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.max_epochs == 100
        assert cfg.batch_size == 32
        assert 0.0 < cfg.val_fraction < 1.0

    def test_val_fraction_boundary_raises(self) -> None:
        with pytest.raises(ValueError, match="val_fraction"):
            TrainConfig(val_fraction=0.0)
        with pytest.raises(ValueError, match="val_fraction"):
            TrainConfig(val_fraction=1.0)

    def test_max_epochs_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_epochs"):
            TrainConfig(max_epochs=0)


class TestPredictionResult:
    def test_construction(self) -> None:
        result = PredictionResult(
            predicted_tuple=PhaseTuple(4, 3),
            raw_output=[3.7, 2.8],
        )
        assert result.predicted_tuple.block_size == 4
        assert len(result.raw_output) == 2
        assert result.metadata == {}

    def test_metadata(self) -> None:
        result = PredictionResult(
            predicted_tuple=PhaseTuple(1, 1),
            raw_output=[1.0, 0.9],
            metadata={"window_size_used": 4},
        )
        assert result.metadata["window_size_used"] == 4

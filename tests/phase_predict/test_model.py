"""Unit tests for phase_predict.model."""

from __future__ import annotations

import torch

from phase_predict.model import PhaseTransformer
from phase_predict.schema import ModelConfig


def _make_cfg(**kwargs) -> ModelConfig:
    defaults = dict(window_size=4, d_model=16, n_heads=2, n_layers=1, tuple_size=2, dropout=0.0)
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class TestPhaseTransformer:
    def test_output_shape(self) -> None:
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        batch = 8
        x = torch.randn(batch, cfg.window_size, cfg.tuple_size)
        out = model(x)
        assert out.shape == (batch, cfg.tuple_size)

    def test_output_dtype_float32(self) -> None:
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        x = torch.randn(2, cfg.window_size, cfg.tuple_size)
        out = model(x)
        assert out.dtype == torch.float32

    def test_single_sample_batch(self) -> None:
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        x = torch.randn(1, cfg.window_size, cfg.tuple_size)
        out = model(x)
        assert out.shape == (1, cfg.tuple_size)

    def test_no_nan_in_output(self) -> None:
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        x = torch.randn(4, cfg.window_size, cfg.tuple_size)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_eval_mode_deterministic(self) -> None:
        """Model output should be identical on two forward passes in eval mode."""
        cfg = _make_cfg(dropout=0.1)
        model = PhaseTransformer(cfg)
        model.eval()
        x = torch.randn(3, cfg.window_size, cfg.tuple_size)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_gradients_flow(self) -> None:
        """All parameters should receive a gradient after one backward pass."""
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        x = torch.randn(4, cfg.window_size, cfg.tuple_size)
        loss = model(x).sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_custom_tuple_size(self) -> None:
        """Model should work for tuple_size != 2."""
        cfg = _make_cfg(tuple_size=5, d_model=20, n_heads=2)
        model = PhaseTransformer(cfg)
        x = torch.randn(2, cfg.window_size, 5)
        out = model(x)
        assert out.shape == (2, 5)

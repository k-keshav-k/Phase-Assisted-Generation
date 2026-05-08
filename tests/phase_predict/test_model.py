"""Unit tests for phase_predict.model."""
from __future__ import annotations

import torch

from phase_predict.model import PhaseTransformer
from phase_predict.schema import ModelConfig


def _make_cfg(**kwargs) -> ModelConfig:
    defaults = dict(window_size=4, d_model=16, n_heads=2, n_layers=1,
                    input_tuple_size=2, num_stab_thresholds=10, dropout=0.0)
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class TestPhaseTransformer:
    def test_model_returns_single_output(self) -> None:
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        x = torch.randn(2, cfg.window_size, cfg.input_tuple_size)
        out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_stab_logits_shape(self) -> None:
        cfg = _make_cfg(num_stab_thresholds=83)
        model = PhaseTransformer(cfg)
        x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
        stab_logits = model(x)
        assert stab_logits.shape == (4, 83)

    def test_single_sample_batch(self) -> None:
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        x = torch.randn(1, cfg.window_size, cfg.input_tuple_size)
        out = model(x)
        assert out.shape == (1, cfg.num_stab_thresholds)

    def test_no_nan_in_output(self) -> None:
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_eval_mode_deterministic(self) -> None:
        cfg = _make_cfg(dropout=0.1)
        model = PhaseTransformer(cfg)
        model.eval()
        x = torch.randn(3, cfg.window_size, cfg.input_tuple_size)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        assert torch.allclose(o1, o2)

    def test_gradients_flow(self) -> None:
        cfg = _make_cfg()
        model = PhaseTransformer(cfg)
        x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_custom_input_tuple_size(self) -> None:
        cfg = _make_cfg(input_tuple_size=12, d_model=32, n_heads=2)
        model = PhaseTransformer(cfg)
        x = torch.randn(2, cfg.window_size, 12)
        out = model(x)
        assert out.shape == (2, cfg.num_stab_thresholds)

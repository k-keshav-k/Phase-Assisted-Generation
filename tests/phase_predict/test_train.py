"""Unit tests for phase_predict.train."""

from __future__ import annotations

from phase_predict.dataset import PhaseSequenceDataset
from phase_predict.model import PhaseTransformer
from phase_predict.schema import ModelConfig, PhaseTuple, TrainConfig
from phase_predict.train import Trainer, evaluate, train_epoch


def _make_sequence(n: int = 40) -> list[PhaseTuple]:
    return [PhaseTuple((i % 8) + 1, i % 6) for i in range(n)]


def _small_setup():
    """Return (model, dataset, train_config) for fast tests."""
    cfg = ModelConfig(window_size=4, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    seq = _make_sequence(40)
    ds = PhaseSequenceDataset(seq, cfg)
    model = PhaseTransformer(cfg)
    train_cfg = TrainConfig(max_epochs=5, batch_size=8, log_interval=0, patience=10)
    return model, ds, train_cfg


class TestTrainer:
    def test_fit_returns_history(self) -> None:
        model, ds, train_cfg = _small_setup()
        trainer = Trainer(model, train_cfg)
        history = trainer.fit(ds)
        assert len(history.train_losses) > 0
        assert len(history.val_losses) == len(history.train_losses)

    def test_loss_decreases_over_epochs(self) -> None:
        """Loss should be lower at the end than at the beginning."""
        model, ds, train_cfg = _small_setup()
        train_cfg.max_epochs = 20
        train_cfg.learning_rate = 5e-3
        trainer = Trainer(model, train_cfg)
        history = trainer.fit(ds)
        assert history.train_losses[-1] <= history.train_losses[0]

    def test_best_epoch_tracked(self) -> None:
        model, ds, train_cfg = _small_setup()
        trainer = Trainer(model, train_cfg)
        history = trainer.fit(ds)
        assert 1 <= history.best_epoch <= train_cfg.max_epochs
        assert history.best_val_loss < float("inf")

    def test_early_stopping_triggers(self) -> None:
        """With patience=2 the loop should stop before max_epochs."""
        cfg = ModelConfig(window_size=4, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
        # tiny train set with a repeating pattern – easy to overfit
        train_seq = [PhaseTuple((i % 2) + 1, 0) for i in range(10)]
        # val set with a very different scale – model cannot improve on it
        val_seq = [PhaseTuple(100 + i, 300) for i in range(10)]
        train_ds = PhaseSequenceDataset(train_seq, cfg)
        val_ds = PhaseSequenceDataset(
            val_seq,
            cfg,
            stats=(train_ds.mean, train_ds.std),  # normalised using train stats
        )
        model = PhaseTransformer(cfg)
        # high lr → quickly overfits train, val loss degrades → early stopping
        train_cfg = TrainConfig(
            max_epochs=200,
            batch_size=4,
            learning_rate=1.0,
            patience=2,
            log_interval=0,
        )
        trainer = Trainer(model, train_cfg)
        history = trainer.fit(train_ds, val_dataset=val_ds)
        assert len(history.train_losses) < 200

    def test_explicit_val_dataset(self) -> None:
        cfg = ModelConfig(window_size=4, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
        train_seq = _make_sequence(40)
        val_seq = _make_sequence(20)
        train_ds = PhaseSequenceDataset(train_seq, cfg)
        val_ds = PhaseSequenceDataset(val_seq, cfg)
        model = PhaseTransformer(cfg)
        trainer = Trainer(model, TrainConfig(max_epochs=3, log_interval=0))
        history = trainer.fit(train_ds, val_dataset=val_ds)
        assert len(history.val_losses) == 3

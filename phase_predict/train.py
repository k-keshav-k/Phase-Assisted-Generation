"""Training loop for the PhaseTransformer model.

Usage example::

    from phase_predict.schema import ModelConfig, TrainConfig, PhaseTuple
    from phase_predict.dataset import PhaseSequenceDataset
    from phase_predict.model import PhaseTransformer
    from phase_predict.train import Trainer

    tuples = [PhaseTuple(4, 3), PhaseTuple(8, 4), ...]
    model_cfg = ModelConfig()
    dataset = PhaseSequenceDataset(tuples, model_cfg)
    model = PhaseTransformer(model_cfg)
    trainer = Trainer(model, TrainConfig())
    history = trainer.fit(dataset)
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phase_predict.dataset import PhaseSequenceDataset, split_dataset
from phase_predict.model import PhaseTransformer
from phase_predict.schema import TrainConfig


@dataclass
class TrainHistory:
    """Collects per-epoch training and validation losses."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


def train_epoch(
    model: PhaseTransformer,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig | None = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Run one training epoch.

    Args:
        model:     the :class:`~phase_predict.model.PhaseTransformer`.
        loader:    DataLoader yielding ``(input, target)`` batches.
        optimizer: PyTorch optimiser.
        criterion: loss function (MSELoss).
        device:    compute device.
        config:    optional :class:`~phase_predict.schema.TrainConfig` used for
                   gradient clipping (``max_grad_norm``).

    Returns:
        Mean loss over all batches in this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        autocast_context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with autocast_context:
            preds = model(inputs)
            loss = criterion(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if config is not None and config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config is not None and config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: PhaseTransformer,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Evaluate the model on a DataLoader without updating weights.

    Args:
        model:     the :class:`~phase_predict.model.PhaseTransformer`.
        loader:    DataLoader yielding ``(input, target)`` batches.
        criterion: loss function.
        device:    compute device.

    Returns:
        Mean loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        autocast_context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with autocast_context:
            preds = model(inputs)
            loss = criterion(preds, targets)
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


class Trainer:
    """High-level training wrapper for :class:`~phase_predict.model.PhaseTransformer`.

    Args:
        model:       the model to train.
        train_config: :class:`~phase_predict.schema.TrainConfig` with
                     hyper-parameters.
        device:      explicit ``torch.device``; defaults to CUDA if available,
                     otherwise CPU.
    """

    def __init__(
        self,
        model: PhaseTransformer,
        train_config: TrainConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.config = train_config or TrainConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

    def fit(
        self,
        dataset: PhaseSequenceDataset,
        *,
        val_dataset: PhaseSequenceDataset | None = None,
    ) -> TrainHistory:
        """Train the model on *dataset*.

        If *val_dataset* is not provided, a held-out validation set is
        carved out from the end of *dataset* using
        :func:`~phase_predict.dataset.split_dataset`.

        Args:
            dataset:     full training (or train+val) dataset.
            val_dataset: optional pre-split validation dataset.

        Returns:
            :class:`TrainHistory` with per-epoch loss curves.
        """
        torch.manual_seed(self.config.seed)

        if val_dataset is None:
            train_set, val_set = split_dataset(
                dataset,
                val_fraction=self.config.val_fraction,
                seed=self.config.seed,
            )
        else:
            train_set, val_set = dataset, val_dataset

        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()
        # Mixed precision: create scaler if using CUDA
        scaler: Optional[torch.amp.GradScaler]
        if self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = None
        history = TrainHistory()
        epochs_no_improve = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = train_epoch(
                self.model, train_loader, optimizer, criterion, self.device, self.config, scaler
            )
            val_loss = evaluate(self.model, val_loader, criterion, self.device, scaler)

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)

            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                epochs_no_improve = 0
                # store best weights in memory
                self._best_state: dict[str, Any] = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                epochs_no_improve += 1

            if (
                self.config.log_interval > 0
                and epoch % self.config.log_interval == 0
            ):
                print(  # noqa: T201
                    f"Epoch {epoch:4d}/{self.config.max_epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f} | "
                    f"best_val={history.best_val_loss:.6f} (epoch {history.best_epoch})"
                )

            if epochs_no_improve >= self.config.patience:
                if self.config.log_interval > 0:
                    print(  # noqa: T201
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.config.patience} epochs)"
                    )
                break

        # restore best weights
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_state.items()}
            )

        return history

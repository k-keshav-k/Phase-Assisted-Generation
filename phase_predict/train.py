"""Training loop for the PhaseTransformer model.

Usage example::

    from phase_predict.schema import ModelConfig, TrainConfig, PhaseTuple
    from phase_predict.dataset import PhaseSequenceDataset
    from phase_predict.model import PhaseTransformer
    from phase_predict.train import Trainer

    tuples = [PhaseTuple(4, 2, 3), PhaseTuple(8, 3, 4), ...]
    model_cfg = ModelConfig()
    dataset = PhaseSequenceDataset(tuples, model_cfg)
    model = PhaseTransformer(model_cfg)
    trainer = Trainer(model, TrainConfig())
    history = trainer.fit(dataset)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
) -> float:
    """Run one training epoch.

    Args:
        model:     the :class:`~phase_predict.model.PhaseTransformer`.
        loader:    DataLoader yielding ``(input, target)`` batches.
        optimizer: PyTorch optimiser.
        criterion: loss function (MSELoss).
        device:    compute device.

    Returns:
        Mean loss over all batches in this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        # gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
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
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = train_config or TrainConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        history = TrainHistory()
        epochs_no_improve = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = train_epoch(
                self.model, train_loader, optimizer, criterion, self.device
            )
            val_loss = evaluate(self.model, val_loader, criterion, self.device)

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

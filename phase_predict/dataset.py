"""Dataset utilities for phase-tuple sequence prediction.

Provides:
  - ``build_windows``: convert a flat sequence of PhaseTuples into
    (input_window, target) pairs suitable for supervised training.
  - ``PhaseSequenceDataset``: PyTorch Dataset wrapping the windowed pairs.
  - ``split_dataset``: reproducible train / validation split.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from phase_predict.schema import ModelConfig, PhaseTuple

# Minimum standard deviation used in normalisation to avoid division by zero.
_MIN_STD_EPSILON: float = 1e-6


def build_windows(
    sequence: Sequence[PhaseTuple],
    window_size: int,
) -> list[tuple[list[PhaseTuple], PhaseTuple]]:
    """Slide a fixed-size window over *sequence* and collect (input, target) pairs.

    Each sample consists of ``window_size`` consecutive tuples as input and
    the immediately following tuple as the prediction target.

    Args:
        sequence:    ordered list of :class:`~phase_predict.schema.PhaseTuple`
                     values representing the full observed history.
        window_size: number of past tuples fed as context; must be >= 1.

    Returns:
        A list of ``(context_window, next_tuple)`` pairs.  The list is empty
        when ``len(sequence) <= window_size``.

    Raises:
        ValueError: if *window_size* is less than 1.
    """
    if window_size < 1:
        msg = "window_size must be >= 1"
        raise ValueError(msg)

    samples: list[tuple[list[PhaseTuple], PhaseTuple]] = []
    for i in range(len(sequence) - window_size):
        context = list(sequence[i : i + window_size])
        target = sequence[i + window_size]
        samples.append((context, target))
    return samples


class PhaseSequenceDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset of windowed phase-tuple sequences.

    Converts integer tuples to float tensors internally so the model can
    directly consume the output.

    Args:
        sequence:    full ordered sequence of
                     :class:`~phase_predict.schema.PhaseTuple` values.
        model_config: :class:`~phase_predict.schema.ModelConfig` whose
                     ``window_size`` and ``tuple_size`` are used when
                     building windows and tensors.
        normalize:   when *True* (default) each tuple field is standardised
                     using the per-field mean and standard deviation computed
                     from *sequence*.  Set to *False* to skip normalisation
                     (e.g. when using pre-fitted statistics from training).
        stats:       optional ``(mean, std)`` tensors of shape
                     ``(tuple_size,)`` to use instead of computing them from
                     *sequence* (useful for applying training statistics to a
                     held-out set).
    """

    def __init__(
        self,
        sequence: Sequence[PhaseTuple],
        model_config: ModelConfig,
        *,
        normalize: bool = True,
        stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.window_size = model_config.window_size
        self.tuple_size = model_config.tuple_size

        raw = torch.tensor(
            [[t.block_size, t.stabilizing_steps, t.refinement_steps] for t in sequence],
            dtype=torch.float32,
        )  # (N, tuple_size)

        if normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                self.mean = raw.mean(dim=0)
                self.std = raw.std(dim=0).clamp(min=_MIN_STD_EPSILON)
            self._raw_normalised = (raw - self.mean) / self.std
        else:
            self.mean = torch.zeros(self.tuple_size)
            self.std = torch.ones(self.tuple_size)
            self._raw_normalised = raw

        self._windows = build_windows(sequence, self.window_size)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_tensor, target_tensor)`` for sample *idx*.

        Returns:
            input_tensor:  float32 tensor of shape ``(window_size, tuple_size)``
            target_tensor: float32 tensor of shape ``(tuple_size,)``
        """
        context, target = self._windows[idx]
        start = idx  # context starts at position idx in the full sequence
        input_tensor = self._raw_normalised[start : start + self.window_size]  # (W, T)
        target_tensor = self._raw_normalised[start + self.window_size]  # (T,)
        return input_tensor, target_tensor

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Map normalised float values back to the original integer scale.

        Args:
            tensor: float tensor of shape ``(..., tuple_size)``.

        Returns:
            Tensor in the original (un-normalised) scale.
        """
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


def split_dataset(
    dataset: PhaseSequenceDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[PhaseSequenceDataset, PhaseSequenceDataset]:
    """Deterministically split *dataset* into train and validation subsets.

    The split respects temporal order: validation samples come from the
    **end** of the sequence so that the model is evaluated on unseen
    future data rather than interpolating between train samples.

    Args:
        dataset:      the full :class:`PhaseSequenceDataset`.
        val_fraction: fraction of samples to reserve for validation.
        seed:         unused (kept for API symmetry); the split is purely
                      positional, not random.

    Returns:
        ``(train_dataset, val_dataset)`` as
        :class:`torch.utils.data.Subset` views.
    """
    from torch.utils.data import Subset

    n = len(dataset)
    if n < 2:
        msg = "Dataset must contain at least 2 windows to split"
        raise ValueError(msg)

    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n))

    return Subset(dataset, train_indices), Subset(dataset, val_indices)  # type: ignore[return-value]

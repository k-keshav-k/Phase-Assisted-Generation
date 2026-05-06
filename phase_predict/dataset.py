"""Dataset utilities for phase-tuple sequence prediction.

Provides:
  - ``build_windows``: convert a flat sequence of PhaseTuples into
    (input_window, target) pairs suitable for supervised training.
  - ``PhaseSequenceDataset``: PyTorch Dataset wrapping the windowed pairs.
    - ``PhaseFullSequenceDataset``: PyTorch Dataset wrapping one full
        context/target pair per trace sequence.
  - ``split_dataset``: reproducible train / validation split.

Supports both standard PhaseTuple and extended multi-feature training using
ExtendedPhaseTuple with configurable input and output tuple sizes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from phase_predict.schema import ExtendedPhaseTuple, ModelConfig, PhaseTuple

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


def _sequence_tensor(sequence: Sequence[PhaseTuple]) -> torch.Tensor:
    """Convert a PhaseTuple sequence to a float tensor."""
    return torch.tensor(
        [[t.block_size, t.refinement_steps] for t in sequence],
        dtype=torch.float32,
    )


def _extended_sequence_tensor(
    sequence: Sequence[Any],
    feature_fields: list[str],
) -> torch.Tensor:
    """Convert an ExtendedPhaseTuple sequence to a float tensor.

    Args:
        sequence: sequence of ExtendedPhaseTuple objects.
        feature_fields: list of feature field names in order.

    Returns:
        Tensor of shape (len(sequence), len(feature_fields)) with feature values.
    """
    return torch.tensor(
        [t.as_list(feature_fields) for t in sequence],
        dtype=torch.float32,
    )


def _extended_output_tensor_from_extended(
    sequence: Sequence[Any],
    output_fields: list[str],
) -> torch.Tensor:
    """Extract output (block, refinement) tensor from ExtendedPhaseTuple seq.

    Args:
        sequence: list of ExtendedPhaseTuple
        output_fields: list of two field names [block_field, second_field]

    Returns:
        Tensor of shape (len(sequence), 2)
    """
    return torch.tensor(
        [
            [getattr(t, "values", {}).get(output_fields[0], 0), getattr(t, "values", {}).get(output_fields[1], 0)]
            for t in sequence
        ],
        dtype=torch.float32,
    )


class PhaseSequenceDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset of windowed phase-tuple sequences.

    Converts integer tuples to float tensors internally so the model can
    directly consume the output. Supports both standard PhaseTuple sequences
    and multi-feature extended sequences.

    Args:
        sequence:    full ordered sequence of
                     :class:`~phase_predict.schema.PhaseTuple` values.
        model_config: :class:`~phase_predict.schema.ModelConfig` whose
                     ``window_size``, ``input_tuple_size``, and
                     ``output_tuple_size`` are used when building tensors.
        normalize:   when *True* (default) each tuple field is standardised
                     using the per-field mean and standard deviation computed
                     from *sequence*.  Set to *False* to skip normalisation
                     (e.g. when using pre-fitted statistics from training).
        stats:       optional ``(mean, std)`` tensors of shape
                     ``(input_tuple_size,)`` to use instead of computing them from
                     *sequence* (useful for applying training statistics to a
                     held-out set).
    """

    def __init__(
        self,
        sequence: Sequence[Any],
        model_config: ModelConfig,
        *,
        normalize: bool = True,
        stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        feature_fields: list[str] | None = None,
        output_fields: list[str] | None = None,
    ) -> None:
        self.window_size = model_config.window_size
        self.input_tuple_size = model_config.input_tuple_size
        self.output_tuple_size = model_config.output_tuple_size
        self.tuple_size = self.output_tuple_size
        self.model_config = model_config

        self.feature_fields = feature_fields
        self.output_fields = output_fields

        if feature_fields is not None:
            input_raw = _extended_sequence_tensor(sequence, feature_fields)
            out_fields = output_fields or feature_fields[: self.output_tuple_size]
            output_raw = _extended_output_tensor_from_extended(sequence, out_fields)
        else:
            input_raw = _sequence_tensor(sequence)
            output_raw = input_raw

        if normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                self.mean = output_raw.mean(dim=0)
                self.std = output_raw.std(dim=0).clamp(min=_MIN_STD_EPSILON)
            self.input_mean = input_raw.mean(dim=0)
            self.input_std = input_raw.std(dim=0).clamp(min=_MIN_STD_EPSILON)
            input_norm = (input_raw - self.input_mean) / self.input_std
        else:
            self.mean = torch.zeros(self.output_tuple_size)
            self.std = torch.ones(self.output_tuple_size)
            self.input_mean = torch.zeros(self.input_tuple_size)
            self.input_std = torch.ones(self.input_tuple_size)
            input_norm = input_raw

        self._windows: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = []
        for i in range(len(sequence) - self.window_size):
            context_input = input_norm[i : i + self.window_size]
            raw_next = sequence[i + self.window_size]

            if hasattr(raw_next, "values"):
                block_val = raw_next.values.get("block_size", 0)
                stab_val = raw_next.values.get("max_stab_step", raw_next.values.get("nfe", 0))
            else:
                block_val = raw_next.block_size
                stab_val = raw_next.refinement_steps

            block_target = torch.tensor(max(0, int(block_val) - 1), dtype=torch.long)
            n_thresh = model_config.num_stab_thresholds
            stab_target = torch.zeros(n_thresh, dtype=torch.float32)
            clamped = min(max(0, int(stab_val)), n_thresh)
            if clamped > 0:
                stab_target[:clamped] = 1.0

            self._windows.append((context_input, (block_target, stab_target)))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        context, target = self._windows[idx]
        return context, target

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Map normalised float values back to the original integer scale.

        Args:
            tensor: float tensor of shape ``(..., output_tuple_size)``.

        Returns:
            Tensor in the original (un-normalised) scale.
        """
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


class PhaseFullSequenceDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset of one full context/target pair per sequence.

    Each item uses the entire tuple history of a trace as context and the
    final tuple as the target. Contexts are left-padded to a shared
    ``window_size`` so batches can be stacked by the default DataLoader.

    Supports both standard PhaseTuple and multi-feature extended sequences.

    Args:
        sequences:   ordered list of PhaseTuple sequences, one per trace.
        model_config: :class:`~phase_predict.schema.ModelConfig` whose
                     ``output_tuple_size`` and ``input_tuple_size`` are used.
        normalize:   when *True* (default) each tuple field is standardised
                 using the per-field mean and standard deviation computed
                 from all tuples across *sequences*.
        stats:       optional ``(mean, std)`` tensors of shape
                     ``(output_tuple_size,)`` to use instead of computing from
                     *sequences*.
    """

    def __init__(
        self,
        sequences: Sequence[Sequence[Any]],
        model_config: ModelConfig,
        *,
        normalize: bool = True,
        stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        input_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        feature_fields: list[str] | None = None,
        output_fields: list[str] | None = None,
    ) -> None:
        self.output_tuple_size = model_config.output_tuple_size
        self.input_tuple_size = model_config.input_tuple_size
        self.tuple_size = self.output_tuple_size
        self.model_config = model_config

        if not sequences:
            msg = "PhaseFullSequenceDataset requires at least one sequence"
            raise ValueError(msg)

        lengths = [len(sequence) for sequence in sequences]
        if any(length < 2 for length in lengths):
            msg = "Each sequence must contain at least 2 tuples"
            raise ValueError(msg)

        self.window_size = max(lengths) - 1

        self.feature_fields = feature_fields
        self.output_fields = output_fields

        if feature_fields is not None:
            input_seqs = [_extended_sequence_tensor(sequence, feature_fields) for sequence in sequences]
            out_fields = output_fields or feature_fields[: self.output_tuple_size]
            output_seqs = [
                _extended_output_tensor_from_extended(sequence, out_fields) for sequence in sequences
            ]
        else:
            input_seqs = [_sequence_tensor(sequence) for sequence in sequences]
            output_seqs = input_seqs

        raw_all_outputs = torch.cat(output_seqs, dim=0)

        if normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                self.mean = raw_all_outputs.mean(dim=0)
                self.std = raw_all_outputs.std(dim=0).clamp(min=_MIN_STD_EPSILON)
            if input_stats is not None:
                self.input_mean, self.input_std = input_stats
            else:
                all_inputs = torch.cat(input_seqs, dim=0)
                self.input_mean = all_inputs.mean(dim=0)
                self.input_std = all_inputs.std(dim=0).clamp(min=_MIN_STD_EPSILON)
            norm_input_seqs = [(raw - self.input_mean) / self.input_std for raw in input_seqs]
        else:
            self.mean = torch.zeros(self.output_tuple_size)
            self.std = torch.ones(self.output_tuple_size)
            self.input_mean = torch.zeros(self.input_tuple_size)
            self.input_std = torch.ones(self.input_tuple_size)
            norm_input_seqs = input_seqs

        self._samples: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = []
        for seq_idx in range(len(sequences)):
            raw_seq = sequences[seq_idx]
            context = norm_input_seqs[seq_idx][:-1]
            if context.size(0) < self.window_size:
                pad_len = self.window_size - context.size(0)
                context = F.pad(context, (0, 0, pad_len, 0))

            raw_next = raw_seq[-1]
            if hasattr(raw_next, "values"):
                block_val = raw_next.values.get("block_size", 0)
                stab_val = raw_next.values.get("max_stab_step", raw_next.values.get("nfe", 0))
            else:
                block_val = raw_next.block_size
                stab_val = raw_next.refinement_steps

            block_target = torch.tensor(max(0, int(block_val) - 1), dtype=torch.long)
            n_thresh = model_config.num_stab_thresholds
            stab_target = torch.zeros(n_thresh, dtype=torch.float32)
            clamped = min(max(0, int(stab_val)), n_thresh)
            if clamped > 0:
                stab_target[:clamped] = 1.0

            self._samples.append((context, (block_target, stab_target)))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self._samples[idx]

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Map normalised float values back to the original integer scale."""
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

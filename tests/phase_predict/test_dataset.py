"""Unit tests for phase_predict.dataset."""

from __future__ import annotations

import pytest
import torch

from phase_predict.dataset import PhaseSequenceDataset, build_windows, split_dataset
from phase_predict.schema import ModelConfig, PhaseTuple


def _make_sequence(n: int = 20) -> list[PhaseTuple]:
    """Return a deterministic synthetic sequence of length *n*."""
    return [PhaseTuple(block_size=(i % 8) + 1, refinement_steps=i % 6)
            for i in range(n)]


class TestBuildWindows:
    def test_correct_number_of_windows(self) -> None:
        seq = _make_sequence(10)
        windows = build_windows(seq, window_size=3)
        # expect 10 - 3 = 7 samples
        assert len(windows) == 7

    def test_window_and_target_shapes(self) -> None:
        seq = _make_sequence(10)
        windows = build_windows(seq, window_size=3)
        context, target = windows[0]
        assert len(context) == 3
        assert isinstance(target, PhaseTuple)

    def test_targets_are_consecutive(self) -> None:
        seq = _make_sequence(5)
        windows = build_windows(seq, window_size=2)
        for i, (ctx, tgt) in enumerate(windows):
            assert ctx == list(seq[i : i + 2])
            assert tgt == seq[i + 2]

    def test_empty_when_sequence_too_short(self) -> None:
        seq = _make_sequence(3)
        assert build_windows(seq, window_size=3) == []

    def test_window_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            build_windows(_make_sequence(5), window_size=0)


class TestPhaseSequenceDataset:
    def test_len(self) -> None:
        seq = _make_sequence(20)
        cfg = ModelConfig(window_size=4)
        ds = PhaseSequenceDataset(seq, cfg)
        assert len(ds) == 20 - 4

    def test_item_shapes(self) -> None:
        seq = _make_sequence(20)
        cfg = ModelConfig(window_size=4, input_tuple_size=2)
        ds = PhaseSequenceDataset(seq, cfg)
        inp, target = ds[0]
        assert inp.shape == (4, 2)
        block_target, stab_target = target
        assert block_target.shape == ()
        assert block_target.dtype == torch.long
        assert stab_target.shape == (cfg.num_stab_thresholds,)
        assert stab_target.dtype == torch.float32

    def test_block_target_is_correct_class(self) -> None:
        seq = _make_sequence(20)
        cfg = ModelConfig(window_size=4)
        ds = PhaseSequenceDataset(seq, cfg, normalize=False)
        _, target = ds[0]
        block_target, _ = target
        # window 0: seq[0:4] -> target seq[4] = PhaseTuple(5, 4)
        # block_size=5 -> class_id=4
        assert block_target.item() == 4

    def test_stab_target_first_elements_are_one(self) -> None:
        seq = _make_sequence(20)
        cfg = ModelConfig(window_size=4)
        ds = PhaseSequenceDataset(seq, cfg, normalize=False)
        _, target = ds[0]
        _, stab_target = target
        # window 0: target seq[4] = PhaseTuple(5, 4) -> stab_steps=4
        # first 4 elements should be 1, rest 0
        assert stab_target[:4].sum().item() == 4.0
        assert stab_target[4:].sum().item() == 0.0

    def test_normalisation_mean_near_zero(self) -> None:
        seq = _make_sequence(50)
        cfg = ModelConfig(window_size=4)
        ds = PhaseSequenceDataset(seq, cfg)
        all_inputs = torch.stack([ds[i][0] for i in range(len(ds))])
        col_means = all_inputs.mean(dim=(0, 1))
        assert col_means.abs().max().item() < 1.5

    def test_skip_normalisation(self) -> None:
        seq = _make_sequence(20)
        cfg = ModelConfig(window_size=4)
        ds = PhaseSequenceDataset(seq, cfg, normalize=False)
        inp, _ = ds[0]
        assert inp[0, 0].item() == pytest.approx(1.0)

    def test_external_stats_applied(self) -> None:
        seq = _make_sequence(20)
        cfg = ModelConfig(window_size=4)
        ds_ref = PhaseSequenceDataset(seq, cfg)
        ds2 = PhaseSequenceDataset(seq, cfg, stats=(ds_ref.mean, ds_ref.std))
        inp_ref, _ = ds_ref[0]
        inp2, _ = ds2[0]
        assert torch.allclose(inp_ref, inp2)


class TestSplitDataset:
    def test_split_sizes(self) -> None:
        seq = _make_sequence(30)
        cfg = ModelConfig(window_size=4)
        ds = PhaseSequenceDataset(seq, cfg)
        train, val = split_dataset(ds, val_fraction=0.2)
        total = len(train) + len(val)
        assert total == len(ds)
        assert len(val) == max(1, int(len(ds) * 0.2))

    def test_temporal_order_preserved(self) -> None:
        """Validation indices should come after all training indices."""
        seq = _make_sequence(30)
        cfg = ModelConfig(window_size=4)
        ds = PhaseSequenceDataset(seq, cfg)
        train, val = split_dataset(ds, val_fraction=0.2)
        assert max(train.indices) < min(val.indices)  # type: ignore[union-attr]

    def test_too_small_dataset_raises(self) -> None:
        seq = _make_sequence(5)
        cfg = ModelConfig(window_size=4)
        ds = PhaseSequenceDataset(seq, cfg)
        # only 1 window – cannot split
        with pytest.raises(ValueError, match="at least 2"):
            split_dataset(ds)

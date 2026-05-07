from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Predictor = importlib.import_module("phase_predict.predict").Predictor
PhaseTuple = importlib.import_module("phase_predict.schema").PhaseTuple
ExtendedPhaseTuple = importlib.import_module("phase_predict.schema").ExtendedPhaseTuple


@dataclass(slots=True)
class ScheduledBlock:
    predicted_tuple: PhaseTuple
    applied_block_size: int
    budgeted_refinement_steps: int


class PAGTupleScheduler:
    """Predict one PAG tuple per block using a trained phase predictor."""

    def __init__(
        self,
        *,
        predictor_ckpt: str | Path | None = None,
        seed_block_length: int,
        seed_refinement_steps: int,
        predictor_device: str | torch.device = "cpu",
        predictor: Predictor | None = None,
        context_seed_block_length: int | None = None,
        context_seed_stabilizing_steps: int | None = None,
        min_refinement_steps: int = 3,
        context_mean_confidence: float = 1.0,
        context_min_confidence: float = 1.0,
        context_digit_fraction: float = 0.0,
        context_delimiter_fraction: float = 0.0,
    ) -> None:
        self.seed_tuple = PhaseTuple(
            block_size=max(1, int(seed_block_length)),
            refinement_steps=max(1, int(seed_refinement_steps)),
        )
        self.context_seed_tuple = ExtendedPhaseTuple(values={
            "block_size": max(1, int(
                seed_block_length if context_seed_block_length is None
                else context_seed_block_length
            )),
            "nfe": max(0, int(
                seed_refinement_steps - 1
                if context_seed_stabilizing_steps is None
                else context_seed_stabilizing_steps
            )),
            "mean_top1_confidence": float(context_mean_confidence),
            "min_top1_confidence": float(context_min_confidence),
            "digit_fraction": float(context_digit_fraction),
            "delimiter_fraction": float(context_delimiter_fraction),
        })
        self.min_refinement_steps = max(1, int(min_refinement_steps))

        if predictor is None:
            if predictor_ckpt is None:
                msg = "predictor_ckpt is required when predictor is not provided"
                raise ValueError(msg)
            device = (
                predictor_device
                if isinstance(predictor_device, torch.device)
                else torch.device(str(predictor_device))
            )
            predictor = Predictor.from_checkpoint(str(predictor_ckpt), device=device)

        self.predictor = predictor
        self.reset()

    def reset(self) -> None:
        self._history: list[ExtendedPhaseTuple] = []
        self._block_index = 0

    def _padded_context(self) -> list[ExtendedPhaseTuple]:
        window_size = int(self.predictor.config.window_size)
        history = self._history[-window_size:]
        pad_count = max(0, window_size - len(history))
        return ([self.context_seed_tuple] * pad_count) + history

    def next_schedule(
        self,
        *,
        remaining_tokens: int,
        max_block_length: int,
        max_refinement_steps: int,
    ) -> ScheduledBlock:
        if remaining_tokens < 1:
            msg = "remaining_tokens must be positive"
            raise ValueError(msg)

        is_seed_block = self._block_index == 0
        if is_seed_block:
            predicted_tuple = self.seed_tuple
        else:
            raw_predicted_tuple = (
                self.predictor.predict(self._padded_context()).predicted_tuple
            )
            predicted_tuple = PhaseTuple(
                block_size=int(raw_predicted_tuple.block_size),
                refinement_steps=int(raw_predicted_tuple.refinement_steps) + 1,
            )

        self._block_index += 1
        applied_block_size = max(
            1,
            min(
                int(predicted_tuple.block_size),
                min(int(max_block_length), int(remaining_tokens)),
            ),
        )
        budget_floor = (
            1
            if is_seed_block
            else min(self.min_refinement_steps, int(max_refinement_steps))
        )
        budgeted_refinement_steps = min(
            int(max_refinement_steps),
            max(budget_floor, int(predicted_tuple.refinement_steps)),
        )
        return ScheduledBlock(
            predicted_tuple=PhaseTuple(
                block_size=int(predicted_tuple.block_size),
                refinement_steps=int(predicted_tuple.refinement_steps),
            ),
            applied_block_size=applied_block_size,
            budgeted_refinement_steps=budgeted_refinement_steps,
        )

    def record_realized(
        self,
        applied_block_size: int,
        actual_nfe_used: int,
        mean_confidence: float = 1.0,
        min_confidence: float = 1.0,
        digit_fraction: float = 0.0,
        delimiter_fraction: float = 0.0,
    ) -> None:
        self._history.append(
            ExtendedPhaseTuple(values={
                "block_size": max(1, int(applied_block_size)),
                "nfe": max(0, int(actual_nfe_used)),
                "mean_top1_confidence": float(mean_confidence),
                "min_top1_confidence": float(min_confidence),
                "digit_fraction": float(digit_fraction),
                "delimiter_fraction": float(delimiter_fraction),
            })
        )

    @property
    def history(self) -> list[ExtendedPhaseTuple]:
        return list(self._history)

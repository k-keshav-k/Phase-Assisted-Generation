"""Scheduler that uses the trained Random Forest to predict (block_size, stabilising_steps).

Drop-in replacement for :class:`pag_predictor.PAGTupleScheduler`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from block_stab_predict.predict import InferencePredictor
from phase_predict.schema import PhaseTuple

# ── Re-export for callers that use it as a return type ────────────────


@dataclass(slots=True)
class ScheduledBlock:
    predicted_tuple: PhaseTuple
    applied_block_size: int
    budgeted_refinement_steps: int


# ── Scheduler ─────────────────────────────────────────────────────────


class RFTupleScheduler:
    """Predict block size and stabilising steps using a trained Random Forest.

    Block 0 uses an explicit seed tuple; subsequent blocks use the RF
    model, consuming Phase 2 per-block metrics recorded by the generation
    loop.
    """

    def __init__(
        self,
        *,
        rf_model_path: str | Path,
        seed_block_length: int,
        seed_refinement_steps: int,
        min_refinement_steps: int = 3,
        context_seed_block_length: int | None = None,
        context_seed_nfe: int | None = None,
    ) -> None:
        self.seed_tuple = PhaseTuple(
            block_size=max(1, int(seed_block_length)),
            refinement_steps=max(1, int(seed_refinement_steps)),
        )
        self.min_refinement_steps = max(1, int(min_refinement_steps))

        self._rf = InferencePredictor(Path(rf_model_path))

        self._block_index = 0
        self.prediction_trace: list[dict[str, object]] = []
        self.scheduler_predict_time_sec = 0.0

        # Pre-compute context-seed values so reset() can re-apply them.
        _cs_bl = context_seed_block_length
        self._context_seed_bl = seed_block_length if _cs_bl is None else int(_cs_bl)
        _cs_nfe = context_seed_nfe
        if _cs_nfe is None:
            self._context_seed_nfe = max(0, int(seed_refinement_steps) - 1)
        else:
            self._context_seed_nfe = int(_cs_nfe)
        self.reset()

    # ── Interface expected by generate_pag ────────────────────────────

    def reset(self) -> None:
        self._rf.reset()
        self._block_index = 0
        self.prediction_trace = []
        self.scheduler_predict_time_sec = 0.0
        # Restore the context-seed padding so the RF always has at least
        # one buffer entry before its first prediction.
        self._rf.record(block_size=self._context_seed_bl, nfe=self._context_seed_nfe)

    def next_schedule(
        self,
        *,
        remaining_tokens: int,
        max_block_length: int,
        max_refinement_steps: int,
    ) -> ScheduledBlock:
        if remaining_tokens < 1:
            raise ValueError("remaining_tokens must be positive")

        is_seed_block = self._block_index == 0
        block_index = self._block_index
        self._block_index += 1

        if is_seed_block:
            predicted_tuple = self.seed_tuple
            source = "seed"
            predict_time_sec = 0.0
        else:
            predict_start = time.perf_counter()
            rf_block_size, rf_stab = self._rf.predict()
            predict_time_sec = time.perf_counter() - predict_start
            self.scheduler_predict_time_sec += predict_time_sec

            predicted_tuple = PhaseTuple(
                block_size=int(rf_block_size),
                refinement_steps=int(rf_stab) + 1,  # +1 for the commit step
            )
            source = "rf_checkpoint"

        applied_block_size = max(
            1,
            min(
                int(predicted_tuple.block_size),
                min(int(max_block_length), int(remaining_tokens)),
            ),
        )
        budget_floor = (
            1 if is_seed_block
            else min(self.min_refinement_steps, int(max_refinement_steps))
        )
        budgeted_refinement_steps = min(
            int(max_refinement_steps),
            max(budget_floor, int(predicted_tuple.refinement_steps)),
        )

        self.prediction_trace.append(
            {
                "block_index": block_index,
                "source": source,
                "predicted_tuple": {
                    "block_size": int(predicted_tuple.block_size),
                    "refinement_steps": int(predicted_tuple.refinement_steps),
                },
                "remaining_tokens": int(remaining_tokens),
                "applied_block_size": int(applied_block_size),
                "budgeted_refinement_steps": int(budgeted_refinement_steps),
                "predict_time_sec": float(predict_time_sec),
            }
        )

        return ScheduledBlock(
            predicted_tuple=predicted_tuple,
            applied_block_size=applied_block_size,
            budgeted_refinement_steps=budgeted_refinement_steps,
        )

    def record_realized(
        self, applied_block_size: int, actual_nfe_used: int, **kwargs: object
    ) -> None:
        # Pass through all realised fields so Phase 2 extra fields
        # (max_stab_step, mean_ref_step, etc.) reach the RF buffer.
        extra: dict[str, float] = {}
        for k, v in kwargs.items():
            try:
                extra[k] = float(v)
            except (TypeError, ValueError):
                pass
        self._rf.record(block_size=int(applied_block_size), nfe=int(actual_nfe_used), **extra)

    # ── Read-only helpers for callers ─────────────────────────────────

    @property
    def history(self) -> list[dict[str, float]]:
        return self._rf.buffer

    @property
    def predictor(self):
        return self._rf

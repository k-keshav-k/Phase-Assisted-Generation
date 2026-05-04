"""Inference-time predictor with a rolling buffer of realised blocks.

Maintains a running history of completed blocks and calls the trained
Random Forest to predict ``(block_size, max_stab_step)`` for the next
block.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from block_stab_predict.features import compute_features
from block_stab_predict.model import BlockStabPredictor
from block_stab_predict.schema import RFConfig


class InferencePredictor:
    """Rolling-buffer predictor for online inference during decoding.

    Typical usage::

        predictor = InferencePredictor("models/rf_v1.joblib")
        predictor.reset()

        while decoding:
            block_size, stab_steps = predictor.predict()
            schedule = clamp(block_size, stab_steps, ...)
            # ... generate block ...
            predictor.record(block_size=schedule.applied_block_size, nfe=nfe)

    Args:
        model_or_path: Trained :class:`BlockStabPredictor` instance or path
            to a ``.joblib`` checkpoint.
        config: Optional :class:`RFConfig`.  Only needed when *model_or_path*
            is an existing predictor (checkpoints carry their own config).
    """

    def __init__(
        self,
        model_or_path: BlockStabPredictor | str | Path,
        config: RFConfig | None = None,
        fallback: tuple[int, int] = (16, 1),
    ) -> None:
        self._fallback = fallback
        if isinstance(model_or_path, BlockStabPredictor):
            self._predictor = model_or_path
        else:
            self._predictor = BlockStabPredictor.load(model_or_path)

        # Use the loaded model's config unless overridden.
        if config is not None:
            _c = self._predictor.config
            if config.feature_fields != _c.feature_fields:
                raise ValueError(
                    f"Config feature_fields {config.feature_fields} does not match "
                    f"training config {_c.feature_fields}"
                )
            if config.target_fields != _c.target_fields:
                raise ValueError(
                    f"Config target_fields {config.target_fields} does not match "
                    f"training config {_c.target_fields}"
                )
            if config.window_size != _c.window_size:
                raise ValueError(
                    f"Config window_size {config.window_size} does not match "
                    f"training config {_c.window_size}"
                )
            self.config = config
        else:
            self.config = self._predictor.config

        self._buffer: list[dict] = []

    # ── State management ──────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the history buffer (call before a new generation)."""
        self._buffer.clear()

    def record(self, block_size: int, nfe: int, **extra_fields: float) -> None:
        """Append a completed block's realised metrics to the buffer.

        Args:
            block_size: Applied block size.
            nfe: Actual number of forward passes used.
            **extra_fields: Additional realised fields (e.g.
                ``max_stab_step``, ``mean_ref_step``).  Ignored in
                Phase 1; used in Phase 2 when features are richer.
        """
        entry: dict[str, float] = {"block_size": float(block_size), "nfe": float(nfe)}
        entry.update(extra_fields)
        self._buffer.append(entry)

    # ── Prediction ────────────────────────────────────────────────────

    def predict(self) -> tuple[int, int]:
        """Predict the next block's ``(block_size, max_stab_step)``.

        If the buffer is empty or insufficiently populated, ``_fallback``
        (``(16, 1)`` by default) is returned.

        Returns:
            ``(block_size, max_stab_step)`` as non-negative integers.
        """
        if not self._buffer:
            return self._fallback

        # Use the most recent window_size entries (or fewer near the start).
        window = self._buffer[-self.config.window_size :]

        feat = compute_features(window, self.config)  # shape (n_features,)
        pred = self._predictor.predict(feat.reshape(1, -1))  # (1, 2)
        bs = max(0, round(float(pred[0, 0])))
        ss = max(0, round(float(pred[0, 1])))
        return (bs, ss)

    # ── Buffer introspection ──────────────────────────────────────────

    @property
    def buffer(self) -> Sequence[dict]:
        """Read-only view of the current history buffer."""
        return list(self._buffer)

    @property
    def buffer_size(self) -> int:
        """Number of completed blocks recorded so far."""
        return len(self._buffer)

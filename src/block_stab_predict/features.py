"""Rolling-window feature engineering for the block/stabilising-step predictor.

Transforms a sequence of past realised-block dicts into a flat feature
vector by computing summary statistics over a sliding window.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from block_stab_predict.schema import FIELD_STATS, RFConfig

# ── Per-field helpers ─────────────────────────────────────────────────


def _window_stats(tuples: Sequence[dict], field: str) -> dict[str, float]:
    """Compute rolling-window statistics for a single *field*.

    Args:
        tuples: Sequence of realised-block dicts (window in reverse
            chronological order, i.e. ``tuples[-1]`` is the most recent).
        field:  Key into each dict.

    Returns:
        ``{"last": …, "mean": …, "std": …, "min": …, "max": …, "trend": …}``
        where *trend* = last − mean.  Missing fields default to ``0.0``.
    """
    vals = np.array([t.get(field, 0.0) for t in tuples], dtype=np.float64)
    last = float(vals[-1])
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))  # population std
    mn = float(vals.min())
    mx = float(vals.max())
    trend = last - mean
    return dict(last=last, mean=mean, std=std, min=mn, max=mx, trend=trend)


# ── Context helpers ───────────────────────────────────────────────────


def _context_features(past_tuples: Sequence[dict]) -> dict[str, float]:
    """Compute meta features about the context window.

    Requires every dict in *past_tuples* to carry a ``"block_size"`` key.
    """
    return {
        "num_past_blocks": float(len(past_tuples)),
        "total_tokens_so_far": float(sum(t["block_size"] for t in past_tuples)),
    }


# ── Public API ────────────────────────────────────────────────────────


def compute_features(
    past_tuples: Sequence[dict],
    config: RFConfig,
) -> np.ndarray:
    """Build a flat feature vector from a window of past realised blocks.

    If *past_tuples* is shorter than ``config.window_size`` it is
    left-padded by repeating the first element.  The result is a single
    1-D float array of length:

        len(config.feature_fields) * len(FIELD_STATS) + 2

    where the trailing two features are ``num_past_blocks`` and
    ``total_tokens_so_far``.

    Args:
        past_tuples: Window of realised-block dicts (most recent last).
        config:      Predictor configuration (defines which fields to use
                     and the expected window size).

    Returns:
        Float array of shape ``(n_features,)``.
    """
    if not past_tuples:
        # No history at all — pad with a single synthetic zero-valued block.
        pad: dict[str, float] = {f: 0.0 for f in config.feature_fields}
        # Defensive: _context_features always reads block_size.
        # Even though default FEATURE_FIELDS includes it, this protects
        # against future reconfiguration that might drop it.
        pad.setdefault("block_size", 0.0)
        window = [pad]
    else:
        window = list(past_tuples)

    # Left-pad to config.window_size by repeating the first element.
    if len(window) < config.window_size:
        pad_count = config.window_size - len(window)
        window = [window[0]] * pad_count + window

    # Truncate to the rolling window so all features see the same context.
    window = window[-config.window_size :]

    # Build feature vector.
    parts: list[float] = []
    for field in config.feature_fields:
        stats = _window_stats(window, field)
        parts.extend(stats[s] for s in FIELD_STATS)

    ctx = _context_features(window)
    parts.extend(ctx.values())

    return np.array(parts, dtype=np.float32)


def feature_names(config: RFConfig) -> list[str]:
    """Ordered list of feature names corresponding to
    :func:`compute_features` output.

    Useful for DataFrame column labels and feature-importance plots.
    """
    names: list[str] = []
    for field in config.feature_fields:
        for stat in FIELD_STATS:
            names.append(f"{field}_{stat}")
    names.append("num_past_blocks")
    names.append("total_tokens_so_far")
    return names

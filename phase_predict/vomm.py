"""Variable-Order Markov Model baseline for phase-tuple prediction.

This is a simple, interpretable baseline to compare against the
PhaseTransformer. It models the conditional distribution of the next
`PhaseTuple` given up to `max_order` previous tuples using observed
counts and returns the empirical mean (rounded) as prediction.

The state keys are flattened tuples of integers so contexts of varying
lengths are naturally supported.
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from typing import Iterable, List, Tuple

from phase_predict.schema import PhaseTuple

TupleLike = Tuple[int, int]


class VariableOrderMarkovModel:
    """Simple Variable-Order Markov Model (VOMM) for next-tuple prediction.

    Args:
        max_order: maximum context length (number of preceding tuples) to
                   consider when predicting the next tuple.
    """

    def __init__(self, max_order: int = 4) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        self.max_order = max_order
        # counts: context_key -> dict[next_tuple -> count]
        self.counts: dict[tuple[int, ...], dict[TupleLike, int]] = {}
        # marginal counts for unconditional fallback
        self.marginal: dict[TupleLike, int] = defaultdict(int)

    @staticmethod
    def _flatten_context(context: Iterable[TupleLike]) -> tuple[int, ...]:
        """Flatten a sequence of (a,b) tuples into a flat integer tuple.

        Example: [(4,2),(8,3)] -> (4,2,8,3)
        """
        flat: list[int] = []
        for a, b in context:
            flat.append(int(a))
            flat.append(int(b))
        return tuple(flat)

    def fit(self, sequences: Iterable[List[PhaseTuple]]) -> None:
        """Fit counts from an iterable of PhaseTuple sequences.

        We record conditional counts for all orders up to `max_order` and
        also the global marginal distribution for fallback.
        """
        self.counts = {}
        self.marginal = defaultdict(int)

        for seq in sequences:
            # record marginals for all observed next tuples
            for t in seq[1:]:
                key = (int(t.block_size), int(t.refinement_steps))
                self.marginal[key] += 1

            # record conditional counts
            for i in range(1, len(seq)):
                next_t = (int(seq[i].block_size), int(seq[i].refinement_steps))
                # consider all orders up to max_order
                for order in range(1, self.max_order + 1):
                    start = max(0, i - order)
                    context = seq[start:i]
                    ctx_key = self._flatten_context(context)
                    bucket = self.counts.setdefault(ctx_key, defaultdict(int))
                    bucket[next_t] += 1

        # Convert default dicts to normal dicts for pickle friendliness
        self.counts = {k: dict(v) for k, v in self.counts.items()}
        self.marginal = dict(self.marginal)

    def _choose_mean(self, counts: dict[TupleLike, int]) -> TupleLike:
        """Return the weighted mean (rounded) of next tuples for given counts."""
        total = sum(counts.values())
        if total == 0:
            return (0, 0)
        sum_a = 0.0
        sum_b = 0.0
        for (a, b), c in counts.items():
            sum_a += a * c
            sum_b += b * c
        mean_a = sum_a / total
        mean_b = sum_b / total
        # round and clamp non-negative
        return (max(0, int(round(mean_a))), max(0, int(round(mean_b))))

    def predict(self, context: List[PhaseTuple]) -> PhaseTuple:
        """Predict the next PhaseTuple given `context`.

        The method searches for the longest matching suffix context (up to
        `max_order`). If no context matches, the unconditional marginal is
        used.
        """
        # Try from highest order down to 1
        for order in range(self.max_order, 0, -1):
            if len(context) < 1:
                break
            ctx = context[-order:]
            key = self._flatten_context(ctx)
            if key in self.counts:
                mean_pair = self._choose_mean(self.counts[key])
                return PhaseTuple(*mean_pair)

        # fallback to marginal
        if self.marginal:
            mean_pair = self._choose_mean(self.marginal)
            return PhaseTuple(*mean_pair)

        # no data at all
        return PhaseTuple(0, 0)

    def evaluate_mse(self, sequences: Iterable[List[PhaseTuple]]) -> float:
        """Compute mean squared error on provided sequences.

        For each sequence we predict the last tuple using the all-but-last as
        context (same evaluation behaviour used by full-sequence training).
        """
        mse_sum = 0.0
        n = 0
        for seq in sequences:
            if len(seq) < 2:
                continue
            context = seq[:-1]
            target = seq[-1]
            pred = self.predict(context)
            da = float(pred.block_size - target.block_size)
            db = float(pred.refinement_steps - target.refinement_steps)
            mse_sum += (da * da + db * db) / 2.0
            n += 1
        return mse_sum / n if n > 0 else float("nan")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "max_order": self.max_order,
                    "counts": self.counts,
                    "marginal": self.marginal,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "VariableOrderMarkovModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(data["max_order"])  # type: ignore[arg-type]
        model.counts = data["counts"]
        model.marginal = data["marginal"]
        return model

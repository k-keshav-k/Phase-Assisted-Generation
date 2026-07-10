from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    reason: str | None
    elapsed_seconds: float
    projected_total_seconds: float
    estimated_spend_usd: float


class BudgetGuard:
    def __init__(
        self,
        *,
        budget_usd: float,
        hourly_rate: float,
        reserve_fraction: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if budget_usd <= 0 or hourly_rate <= 0:
            raise ValueError("budget_usd and hourly_rate must be positive")
        if not 0 <= reserve_fraction < 1:
            raise ValueError("reserve_fraction must be in [0, 1)")
        self.budget_usd = float(budget_usd)
        self.hourly_rate = float(hourly_rate)
        self.reserve_fraction = float(reserve_fraction)
        self._clock = clock
        self._started_at = clock()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    @property
    def estimated_spend_usd(self) -> float:
        return self.elapsed_seconds / 3600 * self.hourly_rate

    @property
    def usable_budget_usd(self) -> float:
        return self.budget_usd * (1 - self.reserve_fraction)

    def can_start(self, *, stage: str, projected_seconds: float) -> BudgetDecision:
        del stage
        projected_total = self.elapsed_seconds + max(0.0, float(projected_seconds))
        projected_spend = projected_total / 3600 * self.hourly_rate
        allowed = projected_spend <= self.usable_budget_usd
        return BudgetDecision(
            allowed=allowed,
            reason=None if allowed else "budget_reserve",
            elapsed_seconds=self.elapsed_seconds,
            projected_total_seconds=projected_total,
            estimated_spend_usd=projected_spend,
        )

    @staticmethod
    def project_stage(*, completed: int, elapsed_seconds: float, remaining: int) -> float:
        if remaining <= 0:
            return 0.0
        if completed <= 0 or elapsed_seconds <= 0:
            raise ValueError("a live timing sample is required before stage projection")
        return elapsed_seconds / completed * remaining * 1.25

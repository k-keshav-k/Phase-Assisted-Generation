from __future__ import annotations

from pag.experiments.budget import BudgetGuard


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, *, hours: float) -> None:
        self.value += hours * 3600


def test_stage_rejected_when_projection_crosses_reserve() -> None:
    clock = FakeClock()
    guard = BudgetGuard(
        budget_usd=20,
        hourly_rate=0.35,
        reserve_fraction=0.10,
        clock=clock,
    )
    clock.advance(hours=50)
    decision = guard.can_start(stage="final", projected_seconds=3 * 3600)
    assert not decision.allowed
    assert decision.reason == "budget_reserve"


def test_projection_has_variance_margin() -> None:
    assert BudgetGuard.project_stage(completed=2, elapsed_seconds=20, remaining=4) == 50

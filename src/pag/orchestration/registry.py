from __future__ import annotations

from pag.contracts.protocols import BaselineRunner, Evaluator, PhaseRunner, SchedulerRunner

_baseline_runners: dict[str, BaselineRunner] = {}
_phase_runners: dict[str, PhaseRunner] = {}
_scheduler_runners: dict[str, SchedulerRunner] = {}
_evaluators: dict[str, Evaluator] = {}


def register_baseline_runner(name: str, runner: BaselineRunner) -> None:
    _baseline_runners[name] = runner


def register_phase_runner(name: str, runner: PhaseRunner) -> None:
    _phase_runners[name] = runner


def register_scheduler_runner(name: str, runner: SchedulerRunner) -> None:
    _scheduler_runners[name] = runner


def register_evaluator(name: str, evaluator: Evaluator) -> None:
    _evaluators[name] = evaluator


def get_baseline_runner(name: str) -> BaselineRunner | None:
    return _baseline_runners.get(name)


def get_phase_runner(name: str) -> PhaseRunner | None:
    return _phase_runners.get(name)


def get_scheduler_runner(name: str) -> SchedulerRunner | None:
    return _scheduler_runners.get(name)


def get_evaluator(name: str) -> Evaluator | None:
    return _evaluators.get(name)


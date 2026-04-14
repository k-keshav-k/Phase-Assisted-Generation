# Teammate Workflow

## Ownership split

- Team 1: `src/pag/baselines`, `src/pag/evaluation`
- Team 2: `src/pag/phases`
- Team 3: `src/pag/scheduler`

## What each team must preserve

- Keep the public stage entrypoint signature intact.
- Return the shared artifact dataclasses from `pag.contracts`.
- Keep the stage artifact files serializable through the provided helpers.
- Pass that team’s test subset plus `tests/contracts`.

## What each team can change freely

- Internal file structure inside its package.
- Service/class/function decomposition.
- Modeling approach, heuristics, learned predictors, or scheduling policy.
- Additional module-local tests and helpers.

## Suggested local workflow

1. Bootstrap the repo once with `uv sync`.
2. Start from the default stub implementation in your package.
3. Replace only your package internals first.
4. Keep running your subset:
   - baseline team: `make test-baselines`
   - phase team: `make test-phases`
   - scheduler team: `make test-scheduler`
5. Run `make test-integration` before handing off.

## Handoff expectations

- Baseline handoff: traces, token signals, completions, and run summary stay contract-compatible.
- Phase handoff: phase annotations, predictor dataset items, predictions, and metadata stay contract-compatible.
- Scheduler handoff: schedule decisions, plans, adaptive results, and comparison metrics stay contract-compatible.

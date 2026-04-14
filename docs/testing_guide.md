# Testing Guide

The test suite is contract-first. It verifies shape compatibility, serialization, and pipeline wiring, not internal architecture.

## Environment bootstrap

- `uv sync` creates `.venv/` and installs the default `dev` dependency group.
- `make install` is a thin wrapper around the same uv sync step.

## Test subsets

- `make test`: full suite
- `make test-baselines`: contracts + baseline stage tests
- `make test-phases`: contracts + phase stage tests
- `make test-scheduler`: contracts + scheduler stage tests
- `make test-integration`: end-to-end mock pipeline wiring

## What the tests enforce

- Required shared dataclasses can be constructed and validated.
- Artifacts serialize to JSON and JSONL cleanly.
- Stage outputs can be consumed by the next stage without shape mismatches.
- The default mock pipeline runs end to end.

## What the tests do not enforce

- Specific inheritance hierarchies.
- Specific class names or internal service patterns.
- Any real research/model/training logic.

## Recommended team workflow

1. Run your subset during local development.
2. Add module-local tests freely.
3. Run `make test-integration` before merging.
4. Run `make lint` and `make format` when `ruff` is available.

.PHONY: install test test-baselines test-phases test-scheduler test-integration lint format

UV ?= uv
PYTHON_VERSION ?= 3.11
UV_CACHE_DIR ?= .uv-cache

install:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --python $(PYTHON_VERSION)

test:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest

test-baselines:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest tests/contracts tests/baselines

test-phases:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest tests/contracts tests/phases

test-scheduler:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest tests/contracts tests/scheduler

test-integration:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest tests/integration

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run ruff check src tests scripts

format:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run ruff format src tests scripts

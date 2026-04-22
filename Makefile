.PHONY: install install-phase-cpd run-phase-cpd test test-baselines test-phases test-scheduler test-integration test-phase-cpd lint format

UV ?= uv
PYTHON_VERSION ?= 3.11
UV_CACHE_DIR ?= .uv-cache

install:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --python $(PYTHON_VERSION)

install-phase-cpd:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --python $(PYTHON_VERSION) --group phase_cpd

run-phase-cpd:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --group phase_cpd streamlit run phase_cpd/app.py

test:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --group phase_cpd pytest

test-baselines:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest tests/contracts tests/baselines

test-phases:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest tests/contracts tests/phases

test-scheduler:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest tests/contracts tests/scheduler

test-integration:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest tests/integration

test-phase-cpd:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --group phase_cpd pytest tests/phase_cpd

test-phase-predict:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --group phase_predict pytest tests/phase_predict

clone-adablock:
	git clone https://github.com/lgxi24/AdaBlock-dLLM.git AdaBlock-dLLM

probe-adablock:
	python scripts/probe_adablock_llada.py \
		--prompts phase_cpd/data/prompts/research_prompts.jsonl \
		--output-dir traces/adablock \
		--gen-length 128 \
		--init-block-length 16 \
		--delimiter-threshold 0.3 \
		--threshold 0.9

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run ruff check src tests scripts phase_cpd phase_predict

format:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run ruff format src tests scripts phase_cpd phase_predict

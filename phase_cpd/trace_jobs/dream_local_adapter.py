from __future__ import annotations

import gc
from typing import Any

from phase_cpd.trace_jobs.dream_runtime import DreamGenerationConfig, DreamTraceCollector

_CollectorCacheKey = tuple[
    str,
    int,
    int,
    float | None,
    float | None,
    int | None,
    str,
    float | None,
    str | None,
    str,
    str,
    int,
    tuple[str, ...],
]
_COLLECTOR_CACHE: dict[_CollectorCacheKey, DreamTraceCollector] = {}


def collect_trace(
    prompt_record: dict[str, Any],
    config: DreamGenerationConfig,
) -> dict[str, Any]:
    """Run Dream inference for one prompt and return a raw step-dump payload."""

    collector = _get_or_create_collector(config)
    return collector.collect(prompt_record)


def clear_collector_cache() -> None:
    collectors = list(_COLLECTOR_CACHE.values())
    _COLLECTOR_CACHE.clear()
    for collector in collectors:
        torch_module = getattr(collector, "_torch", None)
        if torch_module is None:
            continue
        cuda_module = getattr(torch_module, "cuda", None)
        if cuda_module is None or not cuda_module.is_available():
            continue
        cuda_module.empty_cache()
    gc.collect()


def _get_or_create_collector(config: DreamGenerationConfig) -> DreamTraceCollector:
    cache_key = (
        config.model_name,
        config.max_new_tokens,
        config.steps,
        config.temperature,
        config.top_p,
        config.top_k,
        config.alg,
        config.alg_temp,
        config.device,
        config.torch_dtype,
        config.trace_profile,
        config.seed,
        tuple(config.delimiter_texts),
    )
    collector = _COLLECTOR_CACHE.get(cache_key)
    if collector is None:
        collector = DreamTraceCollector(config)
        _COLLECTOR_CACHE[cache_key] = collector
    return collector

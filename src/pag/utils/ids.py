from __future__ import annotations


def build_request_id(run_id: str, sample_id: str) -> str:
    return f"{run_id}:{sample_id}"


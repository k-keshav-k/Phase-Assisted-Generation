from __future__ import annotations

import json
import os
import sys
from typing import Any

import torch

CUDA_UNAVAILABLE_EXIT = 75


def check_cuda(torch_module: Any = torch) -> dict[str, Any]:
    """Force CUDA context creation, allocation, execution, and synchronization."""
    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError("torch.cuda.is_available() is false")
    device_count = int(torch_module.cuda.device_count())
    if device_count < 1:
        raise RuntimeError("CUDA reports no visible devices")
    probe = torch_module.empty(1, device="cuda:0")
    probe.add_(1)
    torch_module.cuda.synchronize()
    return {
        "ok": True,
        "device": "cuda:0",
        "device_name": str(torch_module.cuda.get_device_name(0)),
        "visible_device_count": device_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "torch_version": str(torch_module.__version__),
    }


def main(*, torch_module: Any = torch) -> int:
    try:
        diagnostics = check_cuda(torch_module)
    except Exception as error:
        diagnostics = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
        }
        print(json.dumps(diagnostics, sort_keys=True), file=sys.stderr)
        return CUDA_UNAVAILABLE_EXIT
    print(json.dumps(diagnostics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

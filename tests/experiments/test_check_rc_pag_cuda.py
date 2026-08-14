from __future__ import annotations

import json

from scripts.check_rc_pag_cuda import CUDA_UNAVAILABLE_EXIT, check_cuda, main


class _FakeTensor:
    def add_(self, value: int) -> _FakeTensor:
        assert value == 1
        return self


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronized = False

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "Fake A100"

    def synchronize(self) -> None:
        self.synchronized = True


class _FakeTorch:
    __version__ = "test"

    def __init__(self, *, allocation_error: Exception | None = None) -> None:
        self.cuda = _FakeCuda()
        self.allocation_error = allocation_error

    def empty(self, size: int, *, device: str) -> _FakeTensor:
        assert size == 1
        assert device == "cuda:0"
        if self.allocation_error is not None:
            raise self.allocation_error
        return _FakeTensor()


def test_check_cuda_performs_a_real_allocation_and_synchronization() -> None:
    torch_module = _FakeTorch()

    diagnostics = check_cuda(torch_module)

    assert diagnostics["ok"]
    assert diagnostics["device_name"] == "Fake A100"
    assert diagnostics["visible_device_count"] == 1
    assert torch_module.cuda.synchronized


def test_main_returns_retryable_exit_and_json_when_cuda_allocation_fails(capsys) -> None:
    torch_module = _FakeTorch(
        allocation_error=RuntimeError("CUDA-capable device(s) is/are busy or unavailable")
    )

    exit_code = main(torch_module=torch_module)

    assert exit_code == CUDA_UNAVAILABLE_EXIT
    diagnostics = json.loads(capsys.readouterr().err)
    assert not diagnostics["ok"]
    assert diagnostics["error_type"] == "RuntimeError"
    assert "busy or unavailable" in diagnostics["error"]

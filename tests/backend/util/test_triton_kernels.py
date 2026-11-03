"""Failure policy shared by the optional Triton-backed kernels (`backend.util.triton_kernels`).

SageAttention's and int8 compute's own tests cover how each caller reacts; these pin the bookkeeping both rely on.
"""

import math

import pytest
import torch

from invokeai.backend.util.triton_kernels import (
    MAX_CACHE_FAILURES,
    FailureOutcome,
    TritonKernelHealth,
    relative_error,
)

CACHE_ERROR = PermissionError("[WinError 5] Access is denied: 'triton\\cache\\tmp.pid_123'")


class TestCacheFailures:
    def test_only_failures_in_a_row_retire_the_device(self):
        health = TritonKernelHealth()
        for _ in range(MAX_CACHE_FAILURES - 1):
            assert health.record_failure(0, CACHE_ERROR) is FailureOutcome.TRANSIENT
        health.record_success(0)
        for _ in range(MAX_CACHE_FAILURES - 1):
            assert health.record_failure(0, CACHE_ERROR) is FailureOutcome.TRANSIENT
        assert not health.is_retired(0), "the success in between ended the streak"

        assert health.record_failure(0, CACHE_ERROR) is FailureOutcome.RETIRED
        assert health.is_retired(0)

    def test_streaks_are_counted_per_device(self):
        health = TritonKernelHealth()
        for device in [0, 1] * (MAX_CACHE_FAILURES - 1):
            health.record_failure(device, CACHE_ERROR)
        assert not health.is_retired(0) and not health.is_retired(1)


class TestKernelErrors:
    def test_any_other_error_retires_the_device_at_once_and_is_reported_once(self):
        health = TritonKernelHealth()
        assert health.record_failure(0, RuntimeError("no kernel image")) is FailureOutcome.RETIRED
        assert health.record_failure(0, RuntimeError("no kernel image")) is FailureOutcome.ALREADY_RETIRED
        assert health.record_failure(0, CACHE_ERROR) is FailureOutcome.ALREADY_RETIRED
        assert health.is_retired(0) and not health.is_retired(1)


def test_first_use_is_reported_once_per_device():
    health = TritonKernelHealth()
    assert [health.first_use(device) for device in (0, 0, 1, 0)] == [True, False, True, False]


def test_relative_error_separates_a_close_result_from_a_broken_one():
    expected = torch.randn(64, 32)
    assert relative_error(expected * 1.01, expected) == pytest.approx(0.01, rel=1e-4)
    assert relative_error(torch.zeros_like(expected), expected) == pytest.approx(1.0, rel=1e-4)
    assert math.isnan(relative_error(torch.full_like(expected, math.nan), expected))

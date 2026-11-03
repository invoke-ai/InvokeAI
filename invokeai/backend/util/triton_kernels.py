"""Failure policy shared by InvokeAI's optional Triton-backed kernels (SageAttention, int8 compute).

Both run a third-party or JIT-compiled kernel inside a forward pass that must not fail because of it, so both answer
the same questions the same way:

* A kernel that raised, or whose first result was wrong, is retired for its device until restart, with one warning.
  A second warning per call would flood the log for every layer of every step.
* Triton failing to write a compiled kernel to its on-disk cache says nothing about the kernel: that call falls back
  and the next tries again, unless it keeps failing (see ``MAX_CACHE_FAILURES``).
* Each kernel variant's first result is compared with the exact path it replaces (see ``relative_error``).

Out-of-memory errors are deliberately not classified here: SageAttention moves the rest of a generation to SDPA,
which needs less, while int8 compute lets them propagate because its fallback needs more. Each caller decides.
"""

import threading
from collections import Counter
from collections.abc import Hashable
from enum import Enum

import torch

# Triton failing to write a compiled kernel to its cache (on Windows, when two GPUs compile the same kernel at once)
# says nothing about the kernel. A device that fails this way this many times in a row is retired like any other, so
# a cache that can never be written does not recompile on every call.
MAX_CACHE_FAILURES = 3

# What a log line tells the user to install when Triton is missing.
TRITON_REQUIREMENT = "Triton, which on Windows is the `triton-windows` package"


def is_missing_triton(error: ModuleNotFoundError) -> bool:
    """Whether an import failed because Triton itself is not installed, rather than the package that needs it."""
    return (error.name or "").split(".")[0] == "triton"


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 distance of ``actual`` from ``expected``, computed in float32 and read back to the host.

    Reading it back synchronizes the device, which is the point for a first-call check: a kernel launch error that
    would otherwise surface in some later, unrelated operation is raised here, inside the caller's ``try``. A result
    that is not finite (uninitialized memory, an overflowed epilogue) comes back as NaN or inf.
    """
    difference = torch.linalg.vector_norm(actual - expected, dtype=torch.float32)
    reference = torch.linalg.vector_norm(expected, dtype=torch.float32).clamp_min(1e-6)
    return (difference / reference).item()


class FailureOutcome(Enum):
    """What :meth:`TritonKernelHealth.record_failure` decided about a failed kernel call."""

    # A Triton cache failure: fall back for this call, try the kernel again on the next.
    TRANSIENT = "transient"
    # The device was retired by this call; the caller warns once.
    RETIRED = "retired"
    # Another call retired the device first (concurrent sessions); nothing more to say.
    ALREADY_RETIRED = "already_retired"


class TritonKernelHealth:
    """Per-device state of one optional kernel: retired devices, cache-failure streaks, validated variants, first use.

    Keys are whatever identifies a device to the caller (a CUDA index, a ``torch.device``); validation keys add what
    selects a different compiled kernel (dtype, head size). One instance per kernel, for the life of the process.
    Thread-safe: InvokeAI runs one generation session per GPU concurrently, and those threads share it.

    The read paths (``is_retired``, ``is_validated``, ``record_success``'s common case) take no lock: they run on every
    kernel call, and reading a set or dict another thread may be adding to is safe under the GIL.
    """

    def __init__(self, max_cache_failures: int = MAX_CACHE_FAILURES) -> None:
        self._max_cache_failures = max_cache_failures
        self._lock = threading.Lock()
        self._retired: set[Hashable] = set()
        self._cache_failures: Counter[Hashable] = Counter()
        self._validated: set[Hashable] = set()
        self._announced: set[Hashable] = set()

    def is_retired(self, device: Hashable) -> bool:
        return device in self._retired

    def record_failure(self, device: Hashable, error: Exception) -> FailureOutcome:
        """Account for a kernel call that raised. Callers handle out-of-memory errors before calling this."""
        with self._lock:
            if isinstance(error, OSError) and device not in self._retired:
                self._cache_failures[device] += 1
                if self._cache_failures[device] < self._max_cache_failures:
                    return FailureOutcome.TRANSIENT
            if device in self._retired:
                return FailureOutcome.ALREADY_RETIRED
            self._retired.add(device)
            return FailureOutcome.RETIRED

    def record_success(self, device: Hashable) -> None:
        """End a device's cache-failure streak: only failures in a row retire it."""
        if self._cache_failures.get(device):
            with self._lock:
                self._cache_failures.pop(device, None)

    def is_validated(self, variant: Hashable) -> bool:
        return variant in self._validated

    def mark_validated(self, variant: Hashable) -> None:
        with self._lock:
            self._validated.add(variant)

    def first_use(self, device: Hashable) -> bool:
        """True exactly once per device: the call that should announce the kernel is in use there."""
        if device in self._announced:
            return False
        with self._lock:
            if device in self._announced:
                return False
            self._announced.add(device)
            return True

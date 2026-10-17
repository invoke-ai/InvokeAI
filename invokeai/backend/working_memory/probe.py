"""The working-memory probe: what an operation really took on its device, recorded next to what was estimated for it.

Active only inside a `calibration_scope`, which the session runner opens around each node of a calibration queue item.
There, every `LoadedModel.model_on_device(working_memory=...)` on a GPU is a *window*: the model is locked with
calibration headroom (`calibration_reservation`, so that a too-low estimate cannot spill and could never be corrected),
and from the moment the lock has settled until just before the unlock, the probe measures the device. The window's
`MeasuredWindow` lands in the scope's `windows`; nothing is written anywhere while a window is open.

What a window measures, per counter (driver, torch reserved, torch allocated; see `counters`):

1. Open: synchronize, release the allocator's cached blocks, read the baseline, reset the peaks. Tensors allocated
   before the lock are part of the baseline: in use, as they are in the model cache's own budget.
2. A *co-lock* -- another model locked on the same thread and device while the window is open (a ControlNet, an audio
   VAE, the vocoder) -- moves weights, not working memory. The segment ends before its lock (R0, read after a release)
   and a new one starts after it settled (R2, after a release); R2 - R0 joins the window's offset, negative when the
   lock moved more off the device than on.
3. A *nested window* (an estimated `model_on_device` inside an open one, such as a VAE encode inside a transformer's
   window): its lock is a co-lock for the parent, the parent is frozen while it is open, and it measures alone. When
   it closes, the parent releases the child's cached blocks, resets the peaks and goes on; the child's lasting outputs
   stay in the parent's figures.
4. A `TorchDevice.empty_cache()` inside the window ends a segment: the driver is read before the release. A direct
   `torch.cuda.empty_cache()` (a library's, or cuDNN's and MIOpen's after an algorithm search) and an offload on the
   cache's background thread do not, so the driver figure can miss what they released; torch's peaks still bound it.
5. Close: synchronize and read the final segment. Per counter, the delta is the largest segment peak less the
   baseline and that segment's offset; `demand_bytes` is the largest delta of the counters that could be read.

A window is invalid (`valid=False`, with its reasons) when another thread locked a model on the same device's cache
while it was open or held one when it opened (Expand Prompt and Image to Prompt run next to sessions), when an
allocation failed or only succeeded after the allocator released its cache (a retry squeezes the peak: an untiled
3072px FLUX VAE decode on an RTX 4090 read 22.5 GiB at the card's limit, ~1280 bytes per pixel instead of ~2150), when
Windows still keeps more than `SPILL_TOLERANCE_BYTES` of the process in shared memory `SPILL_SETTLE_S` after close
(right after a decode the counter spikes transiently, 589 MiB, and is back at 9 MiB at the next read), when the
operation raised or was canceled, when the Krea-2 SDPA benchmark (which resets the peaks itself) is active, when XPU's
device-wide usage grew beyond this process's own, when a probe step failed (`probe_error`; a failed lock hook also voids
the scope's later windows, since its lock bookkeeping is suspect), and when no counter could be read. A model less than fully resident at open keeps the window valid but not `fit_eligible`: streamed
weights are working memory the estimate does not price.

Without a scope, the probe costs a context-variable read per estimated `model_on_device`, a read of `armed` per lock and
unlock, and a truth test of `devices.EMPTY_CACHE_OBSERVERS` per `empty_cache`.
"""

import contextvars
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, Literal, Optional

from pydantic import BaseModel, ConfigDict

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.util import devices
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.working_memory import counters
from invokeai.backend.working_memory.counters import (
    VERIFIED_METHODS,
    CounterReading,
    DeviceCounters,
    Method,
    backend_for,
)
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm

if TYPE_CHECKING:
    from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord

GIB = 2**30
MIB = 2**20

CALIBRATION_HEADROOM_MIN_BYTES: Final = 1 * GIB
CALIBRATION_HEADROOM_FRACTION: Final = 0.5

SPILL_TOLERANCE_BYTES: Final = 256 * MIB
"""Shared-memory bytes above the window's baseline that count as spill once they persist."""
SPILL_SETTLE_S: float = 0.5
"""How long a close waits before re-reading a raised shared-memory counter. Module-level so tests can shorten it."""
FOREIGN_DEVICE_TOLERANCE_BYTES: Final = 256 * MIB
"""Growth of a device-wide counter (XPU) beyond this process's own reserved growth that marks another process's load."""

Reason = Literal[
    "foreign_lock",
    "foreign_device_load",
    "oom",
    "alloc_retry",
    "spill",
    "exception",
    "canceled",
    "krea2_sdpa_benchmark",
    "counters_unavailable",
    "probe_error",
    "partial_residency",
]
"""Why a window is invalid, or -- ``partial_residency`` alone -- valid but not fit-eligible."""

_NOT_INVALIDATING: Final[frozenset[Reason]] = frozenset({"partial_residency"})

CounterName = Literal["driver", "reserved", "allocated"]
_COUNTERS: Final[tuple[CounterName, ...]] = ("driver", "reserved", "allocated")


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CounterDelta(_Record):
    """One counter over a window, in bytes."""

    baseline: int
    peak: int
    """The largest value any segment reached, offsets not applied."""
    offset: int
    """What co-locks moved onto the device by the end of the window; negative when they moved more off."""
    delta: int
    """The largest of each segment's peak less the baseline and that segment's offset: the operation's own share."""


class MeasuredWindow(_Record):
    """One measured `model_on_device` window: who ran it, what was estimated, and what the device saw."""

    run_id: str
    queue_item_id: int
    node_id: str
    node_type: str
    model_key: str
    device: str

    estimator: str
    version: int
    features: dict[str, FeatureValue]
    terms: tuple[WorkingMemoryTerm, ...]
    reservation_bytes: int
    source: str
    reservation_applied_bytes: int
    """What the lock reserved: the estimate plus calibration headroom."""

    driver: Optional[CounterDelta]
    reserved: Optional[CounterDelta]
    allocated: Optional[CounterDelta]
    demand_bytes: Optional[int]
    """The largest delta of the counters that could be read."""
    method: Method
    """Where the driver figures came from; ``torch-only`` when there were none for the whole window."""
    verified: bool
    """Whether that method was verified on this backend on real hardware (`counters.VERIFIED_METHODS`)."""

    resident_fraction: float
    """The share of the window's model's weights on the device when the window opened."""
    co_locked_models: list[str]
    """The other models this thread held on the device while the window was open."""
    segments: int
    duration_s: float

    valid: bool
    fit_eligible: bool
    reasons: list[Reason]


def calibration_reservation(reservation_bytes: int) -> int:
    """What a calibration lock reserves for an estimate of `reservation_bytes`: half as much again, at least 1 GiB more."""
    return reservation_bytes + max(
        CALIBRATION_HEADROOM_MIN_BYTES, int(CALIBRATION_HEADROOM_FRACTION * reservation_bytes)
    )


# All window state, and the registry of open windows, under one lock. Re-entrant because a window's own release runs
# `TorchDevice.empty_cache`, which notifies the probe on the same thread. Lock order: the model cache's lock before
# this one, never the other way round -- `TorchDevice.empty_cache` runs inside cache methods, so nothing here calls into
# a cache while holding it.
_state_lock = threading.RLock()
_open_windows: list["_Window"] = []
_armed_scopes = 0

armed: bool = False
"""True while any thread has a calibration scope open. The model loader checks it before telling the probe about a lock
or an unlock, so without a scope that costs one attribute read."""

_boundary = threading.local()


def _release() -> None:
    """Return the allocator's cached blocks to the driver as a boundary of this thread's windows, which handle it
    themselves; other threads' windows see it like any other release."""
    _boundary.active = True
    try:
        TorchDevice.empty_cache(force=True)
    finally:
        _boundary.active = False


def _value(reading: CounterReading, name: CounterName) -> Optional[int]:
    return reading.driver if name == "driver" else reading.reserved if name == "reserved" else reading.allocated


def _peak(reading: CounterReading, name: CounterName) -> Optional[int]:
    if name == "driver":
        return reading.peak_driver if reading.peak_driver is not None else reading.driver
    return reading.peak_reserved if name == "reserved" else reading.peak_allocated


def _rose(before: Optional[int], after: Optional[int]) -> bool:
    return before is not None and after is not None and after > before


def _krea2_sdpa_benchmark_active() -> bool:
    # Imported here: the Krea-2 attention module pulls in diffusers, which a lock must not.
    from invokeai.backend.krea2.attention import KREA2_SDPA_BACKEND_ENV_VAR

    # Any non-empty value runs the benchmark (`_Krea2StepBenchmark.create`), or fails the node if it is invalid.
    return bool(os.environ.get(KREA2_SDPA_BACKEND_ENV_VAR, "").strip())


class _Window:
    """An open window's measurement. Mutated under `_state_lock` only."""

    def __init__(
        self,
        scope: "ProbeScope",
        cache: object,
        record: "CacheRecord",
        estimate: WorkingMemoryEstimate,
        reservation_applied_bytes: int,
    ) -> None:
        self.scope = scope
        self.cache = cache
        self.record = record
        self.estimate = estimate
        self.reservation_applied_bytes = reservation_applied_bytes
        self.device = record.cached_model.compute_device
        self.thread = threading.get_ident()
        self.parent: Optional[_Window] = None
        self.counters: Optional[DeviceCounters] = None
        self.state: Literal["measuring", "co_lock", "suspended"] = "measuring"
        self.first: Optional[CounterReading] = None
        self.baseline: dict[CounterName, Optional[int]] = {}
        self.offset: dict[CounterName, int] = dict.fromkeys(_COUNTERS, 0)
        self.peak: dict[CounterName, int] = {}
        self.delta: dict[CounterName, int] = {}
        self.unavailable: set[CounterName] = set()
        self.reasons: list[Reason] = []
        self.segments = 0
        self.pending_reset = False
        self.co_lock_record: Optional["CacheRecord"] = None
        self.co_lock_start: Optional[CounterReading] = None
        self.co_locked: list[str] = []
        self.resident_fraction = 1.0
        self.opened_at = 0.0
        self.recorded = False

    def add_reason(self, reason: Reason) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def fail(self) -> None:
        """A probe step on this window failed: its figures can no longer be trusted. Mark it, and leave it in a state
        the rest of the window can run in."""
        self.add_reason("probe_error")
        self.pending_reset = False
        if self.state == "co_lock":
            self.state = "measuring"
            self.co_lock_record = self.co_lock_start = None

    def start(self, reading: CounterReading) -> None:
        self.first = reading
        self.opened_at = time.perf_counter()
        for name in _COUNTERS:
            self.baseline[name] = _value(reading, name)
            if self.baseline[name] is None:
                self.unavailable.add(name)

    def fold(self, reading: CounterReading, ends_segment: bool = True) -> None:
        """Take `reading` as the end of the current segment, or with `ends_segment=False` as a sample within it."""
        for name in _COUNTERS:
            if name in self.unavailable:
                continue
            value, base = _peak(reading, name), self.baseline[name]
            if value is None or base is None:
                self.unavailable.add(name)
                continue
            self.peak[name] = max(self.peak.get(name, value), value)
            share = value - base - self.offset[name]
            self.delta[name] = max(self.delta.get(name, share), share)
        if ends_segment:
            self.segments += 1

    def begin_co_lock(self, record: "CacheRecord") -> None:
        assert self.counters is not None
        self.counters.synchronize()
        self.fold(self.counters.read())
        _release()
        self.co_lock_start = self.counters.read()
        self.co_lock_record = record
        self.state = "co_lock"

    def end_co_lock(self) -> None:
        assert self.counters is not None and self.co_lock_start is not None and self.co_lock_record is not None
        self.counters.synchronize()
        _release()
        settled = self.counters.read()
        for name in _COUNTERS:
            before, after = _value(self.co_lock_start, name), _value(settled, name)
            if before is None or after is None:
                self.unavailable.add(name)
            else:
                self.offset[name] += after - before
        if self.co_lock_record.key not in self.co_locked and self.co_lock_record.key != self.record.key:
            self.co_locked.append(self.co_lock_record.key)
        self.co_lock_record = self.co_lock_start = None
        self.counters.reset_peaks()
        self.state = "measuring"

    def counter(self, name: CounterName) -> Optional[CounterDelta]:
        base = self.baseline.get(name)
        if name in self.unavailable or base is None or name not in self.delta:
            return None
        return CounterDelta(baseline=base, peak=self.peak[name], offset=self.offset[name], delta=self.delta[name])

    def method(self) -> Method:
        if self.first is None:
            return "torch-only"
        # A driver figure that dropped out mid-window leaves the window to torch's counters.
        if self.first.driver is not None and "driver" in self.unavailable:
            return "torch-only"
        return self.first.method

    def measured(self, duration_s: float) -> MeasuredWindow:
        counter_deltas = {name: self.counter(name) for name in _COUNTERS}
        deltas = [delta.delta for delta in counter_deltas.values() if delta is not None]
        if not deltas:
            self.add_reason("counters_unavailable")
        valid = not any(reason not in _NOT_INVALIDATING for reason in self.reasons)
        method = self.method()
        backend = backend_for(self.device)
        estimate, scope = self.estimate, self.scope
        return MeasuredWindow(
            run_id=scope.run_id,
            queue_item_id=scope.queue_item_id,
            node_id=scope.node_id,
            node_type=scope.node_type,
            model_key=self.record.key,
            device=str(self.device),
            estimator=estimate.estimator,
            version=estimate.version,
            features=dict(estimate.features),
            terms=estimate.terms,
            reservation_bytes=estimate.reservation_bytes,
            source=estimate.source,
            reservation_applied_bytes=self.reservation_applied_bytes,
            driver=counter_deltas["driver"],
            reserved=counter_deltas["reserved"],
            allocated=counter_deltas["allocated"],
            demand_bytes=max(deltas) if deltas else None,
            method=method,
            verified=backend is not None and (method, backend) in VERIFIED_METHODS,
            resident_fraction=self.resident_fraction,
            co_locked_models=list(self.co_locked),
            segments=self.segments,
            duration_s=duration_s,
            valid=valid,
            fit_eligible=valid and "partial_residency" not in self.reasons,
            reasons=list(self.reasons),
        )

    def close(self, error: Optional[BaseException]) -> None:
        """End the window on its own thread, `error` being what the operation raised, and record it in its scope."""
        try:
            self._close(error)
        except Exception:
            InvokeAILogger.get_logger(__name__).warning(
                f"Working-memory probe failed to close the window of {self.record.key}", exc_info=True
            )
            try:
                with _state_lock:
                    self.add_reason("probe_error")
                    self._unregister()
                    parent = self.parent
                    if parent is not None and parent.state == "suspended":
                        # Its peaks were not reset after this window, so what it measures from here is suspect.
                        parent.fail()
                        parent.state = "measuring"
                if not self.recorded:
                    self.scope.windows.append(self.measured(time.perf_counter() - self.opened_at))
                    self.recorded = True
            except Exception:
                InvokeAILogger.get_logger(__name__).debug("Working-memory probe could not record", exc_info=True)

    def _unregister(self) -> None:
        if self in _open_windows:
            _open_windows.remove(self)
        if self in self.scope._stack:
            self.scope._stack.remove(self)
        if self.counters is not None:
            self.counters.close()

    def _close(self, error: Optional[BaseException]) -> None:
        if error is not None:
            self.add_reason("canceled" if isinstance(error, CanceledException) else "exception")
        final: Optional[CounterReading] = None
        with _state_lock:
            if self.counters is not None and self.first is not None:
                self.counters.synchronize()
                final = self.counters.read()
                self.fold(final)
                first = self.first
                if _rose(first.num_ooms, final.num_ooms):
                    self.add_reason("oom")
                if _rose(first.num_alloc_retries, final.num_alloc_retries):
                    self.add_reason("alloc_retry")
                if (
                    first.device_used is not None
                    and final.device_used is not None
                    and first.reserved is not None
                    and final.reserved is not None
                    and (final.device_used - first.device_used) - (final.reserved - first.reserved)
                    > FOREIGN_DEVICE_TOLERANCE_BYTES
                ):
                    self.add_reason("foreign_device_load")
            if self in _open_windows:
                _open_windows.remove(self)
            if self in self.scope._stack:
                self.scope._stack.remove(self)
        duration_s = time.perf_counter() - self.opened_at

        if final is not None and self.first is not None and self.counters is not None:
            baseline_paged = self.first.paged
            if (
                baseline_paged is not None
                and final.paged is not None
                and final.paged - baseline_paged > SPILL_TOLERANCE_BYTES
            ):
                # Shared usage spikes for a moment after large transfers; only what is still there later is spill.
                time.sleep(SPILL_SETTLE_S)
                settled = self.counters.read().paged
                if settled is None or settled - baseline_paged > SPILL_TOLERANCE_BYTES:
                    self.add_reason("spill")

        with _state_lock:
            parent = self.parent
            if parent is not None and parent in _open_windows and parent.counters is not None:
                _release()
                parent.counters.reset_peaks()
                parent.state = "measuring"
            if self.counters is not None:
                self.counters.close()
        record = self.measured(duration_s)
        self.scope.windows.append(record)
        self.recorded = True
        InvokeAILogger.get_logger(__name__).debug(
            f"Measured {record.estimator} on {record.device} for {record.model_key}: "
            f"demand {(record.demand_bytes or 0) / MIB:.0f} MB against {record.reservation_bytes / MIB:.0f} MB "
            f"estimated ({record.method}{'' if record.valid else ', invalid: ' + ', '.join(record.reasons)})"
        )


class ProbeScope:
    """One node of a calibration queue item, on the thread that runs it. `windows` collects that node's measurements
    in the order they closed; the caller persists them after the node."""

    def __init__(self, *, run_id: str, queue_item_id: int, node_id: str, node_type: str) -> None:
        self.run_id = run_id
        self.queue_item_id = queue_item_id
        self.node_id = node_id
        self.node_type = node_type
        self.windows: list[MeasuredWindow] = []
        # This thread's open windows, innermost last.
        self._stack: list[_Window] = []
        # The locks this thread holds, per (cache, model key).
        self._held: dict[tuple[int, str], int] = {}
        # Set when a probe hook failed on this thread: `_held` may be off, so no later window can trust its check.
        self._tainted = False

    def _held_on(self, cache: object) -> int:
        return sum(count for (cache_id, _key), count in self._held.items() if cache_id == id(cache))

    def open_window(
        self, cache: object, record: "CacheRecord", estimate: WorkingMemoryEstimate, reservation_applied_bytes: int
    ) -> Optional[_Window]:
        """Start measuring the operation `record`'s model is about to run, once its lock has settled.

        `cache` is the model cache the record is locked in. Never raises: a probe failure leaves the operation alone
        and records the window as invalid, or not at all.
        """
        window = _Window(self, cache, record, estimate, reservation_applied_bytes)
        # Registered before the cache is asked: from here on, a lock another thread takes on this cache voids the
        # window (`lock_started`, `lock_settled`), and one it already holds shows in `locks_held()`. Its counters are
        # not open yet, so releases pass it by.
        with _state_lock:
            _open_windows.append(window)
        try:
            if self._tainted:
                window.add_reason("probe_error")
            cached_model = record.cached_model
            total = cached_model.total_bytes()
            resident = cached_model.cur_vram_bytes()
            window.resident_fraction = resident / total if total > 0 else 1.0
            if resident < total:
                window.add_reason("partial_residency")
            window.co_locked = sorted(
                key
                for (cache_id, key), count in self._held.items()
                if cache_id == id(cache) and count > 0 and key != record.key
            )
            # Asked outside the probe's lock (see `_state_lock`). Every lock this thread took since the scope opened
            # is in `_held`, so any lock beyond those belongs to another thread.
            if cache.locks_held() > self._held_on(cache):  # type: ignore[attr-defined]
                window.add_reason("foreign_lock")
            if window.device.type == "cuda" and _krea2_sdpa_benchmark_active():
                window.add_reason("krea2_sdpa_benchmark")
        except Exception:
            InvokeAILogger.get_logger(__name__).debug("Working-memory probe could not inspect the lock", exc_info=True)
            window.add_reason("probe_error")

        try:
            window.counters = counters.DeviceCounters(window.device)
            with _state_lock:
                parent = self._stack[-1] if self._stack else None
                window.parent = parent
                window.counters.synchronize()
                if parent is not None and parent.counters is not None:
                    if parent.state == "measuring":
                        parent.fold(parent.counters.read())
                    parent.state = "suspended"
                _release()
                window.start(window.counters.read())
                window.counters.reset_peaks()
                self._stack.append(window)
            return window
        except Exception:
            InvokeAILogger.get_logger(__name__).warning(
                f"Working-memory probe could not open a window for {record.key}", exc_info=True
            )
            with _state_lock:
                window._unregister()
                parent = window.parent
                if parent is not None and parent.state == "suspended":
                    # Folded and released for a window that never opened.
                    parent.fail()
                    parent.state = "measuring"
            return None

    def _abandon(self) -> None:
        """Drop the windows still open when the scope ends, so no sampler outlives it."""
        with _state_lock:
            for window in list(self._stack):
                window._unregister()


_current_scope: contextvars.ContextVar[Optional[ProbeScope]] = contextvars.ContextVar(
    "working_memory_probe_scope", default=None
)


def current_scope() -> Optional[ProbeScope]:
    """The calibration scope open on this thread, if any."""
    return _current_scope.get()


class _EmptyCacheObserver:
    """Turns `TorchDevice.empty_cache` into segment boundaries of the open windows."""

    def before_empty_cache(self) -> None:
        try:
            me = threading.get_ident()
            own_boundary = getattr(_boundary, "active", False)
            with _state_lock:
                for window in _open_windows:
                    if window.counters is None or window.state != "measuring":
                        continue
                    if window.thread != me:
                        # Another thread's release (process-global on CUDA) takes this window's cached blocks too:
                        # read its driver before they go, but leave its peaks alone mid-operation.
                        window.fold(window.counters.read(), ends_segment=False)
                    elif not own_boundary:
                        window.fold(window.counters.read())
                        window.pending_reset = True
        except Exception:
            _all_failed("follow a release")

    def after_empty_cache(self) -> None:
        try:
            with _state_lock:
                for window in _open_windows:
                    if window.pending_reset and window.counters is not None:
                        window.counters.reset_peaks()
                        window.pending_reset = False
        except Exception:
            _all_failed("follow a release")


_observer = _EmptyCacheObserver()


def _arm() -> None:
    global armed, _armed_scopes
    with _state_lock:
        _armed_scopes += 1
        if _armed_scopes == 1:
            devices.EMPTY_CACHE_OBSERVERS.append(_observer)
            armed = True


def _disarm() -> None:
    global armed, _armed_scopes
    with _state_lock:
        _armed_scopes -= 1
        if _armed_scopes == 0:
            armed = False
            if _observer in devices.EMPTY_CACHE_OBSERVERS:
                devices.EMPTY_CACHE_OBSERVERS.remove(_observer)


@contextmanager
def calibration_scope(*, run_id: str, queue_item_id: int, node_id: str, node_type: str) -> Iterator[ProbeScope]:
    """Measure the windows of one node on this thread. Scopes do not nest."""
    if _current_scope.get() is not None:
        raise RuntimeError("A calibration scope is already open on this thread; scopes do not nest.")
    scope = ProbeScope(run_id=run_id, queue_item_id=queue_item_id, node_id=node_id, node_type=node_type)
    token = _current_scope.set(scope)
    _arm()
    try:
        yield scope
    finally:
        scope._abandon()
        _disarm()
        _current_scope.reset(token)


def _all_failed(what: str) -> None:
    """A release boundary failed part-way: any open window may have been folded without its reset."""
    InvokeAILogger.get_logger(__name__).debug(f"Working-memory probe failed to {what}", exc_info=True)
    try:
        with _state_lock:
            for window in _open_windows:
                window.fail()
    except Exception:
        pass


def _hook_failed(scope: Optional[ProbeScope], cache: object, what: str) -> None:
    """A lock hook failed part-way. On a scope's own thread every window of the scope is suspect, and so is its lock
    bookkeeping; on another thread, the windows on that cache which the hook was there to void."""
    InvokeAILogger.get_logger(__name__).debug(f"Working-memory probe failed to {what}", exc_info=True)
    try:
        with _state_lock:
            if scope is not None:
                scope._tainted = True
                affected = list(scope._stack)
            else:
                affected = [window for window in _open_windows if window.cache is cache]
            for window in affected:
                window.fail()
    except Exception:
        pass


def _invalidate_foreign(cache: object) -> None:
    me = threading.get_ident()
    for window in _open_windows:
        if window.cache is cache and window.thread != me:
            window.add_reason("foreign_lock")


def lock_started(cache: object, record: "CacheRecord") -> None:
    """A lock of `record` on its compute device begins on this thread. Called by the model loader while `armed`."""
    scope = _current_scope.get()
    try:
        with _state_lock:
            _invalidate_foreign(cache)
            if scope is None or not scope._stack:
                return
            window = scope._stack[-1]
            if (
                window.cache is cache
                and window.state == "measuring"
                and record.cached_model.compute_device.type != "cpu"
            ):
                window.begin_co_lock(record)
    except Exception:
        _hook_failed(scope, cache, "follow a lock")


def lock_settled(cache: object, record: "CacheRecord", locked: bool) -> None:
    """That lock has settled, its required tensors repaired, or it failed (`locked=False`, nothing stays locked)."""
    scope = _current_scope.get()
    try:
        with _state_lock:
            _invalidate_foreign(cache)
            if scope is None:
                return
            if locked:
                key = (id(cache), record.key)
                scope._held[key] = scope._held.get(key, 0) + 1
            if scope._stack:
                window = scope._stack[-1]
                if window.state == "co_lock" and window.co_lock_record is record:
                    window.end_co_lock()
    except Exception:
        _hook_failed(scope, cache, "follow a lock")


def locked_in_ram(cache: object, record: "CacheRecord") -> None:
    """`record` was pinned in RAM on this thread: no device memory moved, but the lock counts for `locks_held`."""
    scope = _current_scope.get()
    try:
        with _state_lock:
            _invalidate_foreign(cache)
            if scope is not None:
                key = (id(cache), record.key)
                scope._held[key] = scope._held.get(key, 0) + 1
    except Exception:
        _hook_failed(scope, cache, "follow a lock")


def unlocked(cache: object, record: "CacheRecord") -> None:
    """`record` was unlocked on this thread."""
    scope = _current_scope.get()
    try:
        with _state_lock:
            _invalidate_foreign(cache)
            if scope is not None:
                key = (id(cache), record.key)
                remaining = scope._held.get(key, 0) - 1
                if remaining > 0:
                    scope._held[key] = remaining
                else:
                    scope._held.pop(key, None)
    except Exception:
        _hook_failed(scope, cache, "follow an unlock")

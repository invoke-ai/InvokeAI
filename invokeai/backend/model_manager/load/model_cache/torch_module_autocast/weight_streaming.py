"""Async weight streaming for partially loaded models, opt-in via ``async_weight_streaming``.

A partially loaded model keeps part of its weights in RAM, and the custom modules copy each of them to the compute
device on every forward (`cast_to_device`). Without this feature that copy is a synchronous ``t.to(device)`` from
pageable memory: the CPU waits for the GPU to drain before every copy, and the GPU idles while the copy runs. With it,
the copy runs on a side stream from page-locked RAM and overlaps with the compute stream. Outputs are bit-identical:
the same bytes arrive before the same kernels read them.

Two parts:

- `PinnedStorageRegistry` page-locks CPU storages in place (``cudaHostRegister``, no second copy), shared by holders.
  `CachedModelWithPartialLoad` keeps the RAM-copy weights its streaming modules stream registered.
- `stream_to_device` issues the asynchronous copy of a registered tensor, or of a small one through a pinned staging
  buffer. Everything else returns None, and the caller copies synchronously as before.

Runs on CUDA devices: NVIDIA, and AMD through ROCm, where the HIP runtime provides the same calls
(measured on an RX 9060 XT with ROCm 10: registration works, outputs bit-identical).
"""

import ctypes
import fnmatch
import logging
import mmap
import os
import queue
import threading
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

import psutil
import torch

from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

# Bytes of streamed weights the compute stream has not reached yet. The CPU runs ahead of the GPU; without a bound it
# issues every copy of a forward at once and holds all streamed weights in VRAM at the same time (measured: +1.9 GiB
# for 1.69 GiB streamed), which defeats partial loading. Both paths hold the tensor being consumed; this one also holds
# up to the budget of copies in flight, and the model cache reserves that much VRAM (or the model's streamable bytes,
# if fewer) for a lock that leaves streamed weights in RAM (`CachedModelWithPartialLoad.weight_streaming_reserve_bytes`).
#
# The copies are allocated on the side stream, so their freed blocks are cached for it: the compute stream cannot take
# them directly. The allocator frees them on an out-of-memory retry and `empty_cache()` returns them, so the cache's
# reclaimable-VRAM credit (`ModelCache._get_reclaimable_allocator_bytes`) is right to count them, at the cost of a
# synchronization when it is drawn on. Between locks they hold about the reserve the next streaming lock plans for.
STREAM_BUDGET_BYTES = 384 * 2**20
# Tensors up to this size that are not registered (registration failed, or the pinned-RAM cap is reached) are copied
# into a pinned staging buffer first. A single synchronous copy among the asynchronous ones would stall the CPU until
# the GPU has drained, losing the overlap; for small tensors the extra CPU copy is cheap. Only while something is
# registered at all: where every large weight copies synchronously anyway, staging the small ones gains nothing.
STAGE_MAX_BYTES = 2**20
# Page-locked RAM cannot be swapped out. ComfyUI caps it at 40 % of physical RAM on Windows; storages past the cap stay
# pageable and stream synchronously.
PINNED_RAM_CAP_BYTES = int(0.4 * psutil.virtual_memory().total)

_CUDA_HOST_REGISTER_PORTABLE = 1  # Registered memory is pinned for every GPU, so shared weights serve all devices.
_PAGE_SIZE = mmap.PAGESIZE

# Opt-in like fp8/int8 compute. Tests override it process-wide without a configured app.
_override: bool | None = None


def set_async_weight_streaming_enabled(enabled: bool | None) -> None:
    """Override the configured setting process-wide. Pass ``None`` to revert to the config value."""
    global _override
    _override = enabled


def is_async_weight_streaming_enabled() -> bool:
    if _override is not None:
        return _override
    try:
        from invokeai.app.services.config.config_default import get_config

        return bool(get_config().async_weight_streaming)
    except Exception:
        # Backend code may run outside a configured app (scripts, tests). Default to the synchronous path.
        return False


_logged: set[str] = set()
_logged_lock = threading.Lock()


def _log_once(key: str, level: int, message: str) -> None:
    with _logged_lock:
        if key in _logged:
            return
        _logged.add(key)
    logger.log(level, message)


def is_streamable_tensor(t: torch.Tensor) -> bool:
    """Plain tensors only. Subclasses (GGMLTensor, SDNQTensor, bitsandbytes parameters) define their own transfers, and
    some have no storage of their own to page-lock."""
    return type(t) is torch.Tensor or type(t) is torch.nn.Parameter


def can_stream_weights_to(device: torch.device, has_ram_copy: bool) -> bool:
    """Whether the CPU-resident weights of a model computing on ``device`` stream asynchronously.

    When the feature is on but cannot work for the model, the log says why, once.
    """
    if not is_async_weight_streaming_enabled():
        return False
    if device.type != "cuda":
        _log_once(
            f"device:{device}",
            logging.INFO,
            f"async_weight_streaming is enabled, but not used on {device}: it needs an NVIDIA or AMD (ROCm) GPU. "
            "Weights in RAM are copied there synchronously.",
        )
        return False
    if not has_ram_copy:
        _log_once(
            "ram_copy",
            logging.INFO,
            "async_weight_streaming is enabled, but not used: it needs keep_ram_copy_of_weights: true. Weights in RAM "
            "are copied synchronously.",
        )
        return False
    return True


class _Runtime:
    """``cudaHostRegister``, ``cudaHostUnregister`` and ``cudaGetLastError`` (the ``hip`` equivalents on ROCm) of one
    runtime library, called through ctypes.

    Registration needs a way to clear the error a refused call leaves: the runtime keeps it as the thread's last
    error, and torch raised it from the next allocation that reached cudaMalloc, unrelated to the registration
    (measured on an RTX 4090 and an RX 9060 XT; a kernel launch did not clear it). ``torch.cuda.cudart()`` has no
    cudaGetLastError. Taking all three calls from the same library puts every error the registry causes into the error
    state it clears, whichever runtime instance torch itself uses, so no refused call can stay pending for torch.
    """

    def __init__(self, library: ctypes.CDLL, prefix: str) -> None:
        self._register = getattr(library, f"{prefix}HostRegister")
        self._register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self._register.restype = ctypes.c_int
        self._unregister = getattr(library, f"{prefix}HostUnregister")
        self._unregister.argtypes = [ctypes.c_void_p]
        self._unregister.restype = ctypes.c_int
        self._get_last_error = getattr(library, f"{prefix}GetLastError")
        self._get_last_error.argtypes = []
        self._get_last_error.restype = ctypes.c_int

    def register(self, ptr: int, nbytes: int) -> int:
        """Page-lock [ptr, ptr + nbytes) for every GPU. Returns the error code; a refusal is cleared at once."""
        error = self._register(ptr, nbytes, _CUDA_HOST_REGISTER_PORTABLE)
        if error != 0:
            self._get_last_error()
        return error

    def unregister(self, ptr: int) -> int:
        error = self._unregister(ptr)
        if error != 0:
            self._get_last_error()
        return error


def _find_runtime() -> _Runtime | None:
    """The runtime library torch uses, among those mapped into the process; None if there is none.

    Another wheel can bring a private copy of the runtime. Registering through it would use that copy's current device
    (0 by default), possibly creating a context on a GPU this worker does not use. So torch's own copy comes first: in
    torch's lib directory (Windows, ROCm wheels) or in the ``nvidia`` runtime wheel torch depends on (Linux), with the
    soname of torch's CUDA major version. The self-test then checks that torch sees the registrations.
    """
    hip = torch.version.hip is not None
    patterns = ("amdhip64*.dll", "libamdhip64.so*") if hip else ("cudart64_*.dll", "libcudart.so*")
    torch_lib = os.path.normcase(os.path.join(os.path.dirname(torch.__file__), "lib"))
    soname = None if hip or torch.version.cuda is None else f"libcudart.so.{torch.version.cuda.split('.')[0]}"
    try:
        paths = {m.path for m in psutil.Process().memory_maps()}
    except Exception:
        return None
    matching = [p for p in paths if any(fnmatch.fnmatch(os.path.basename(p).lower(), pat) for pat in patterns)]

    def preference(path: str) -> tuple[bool, bool, str]:
        directory = os.path.normcase(os.path.dirname(path))
        torchs = directory == torch_lib or f"{os.sep}nvidia{os.sep}" in directory
        return (not torchs, soname is not None and os.path.basename(path) != soname, path)

    matching.sort(key=preference)
    for path in matching:
        try:
            return _Runtime(ctypes.CDLL(path), "hip" if hip else "cuda")
        except (OSError, AttributeError):
            continue
    return None


# Chosen by the self-test (`_registration_self_test`); every registration goes through it.
_runtime: _Runtime | None = None
# Self-test buffers that could not be unregistered; never freed (see `_registration_self_test`).
_self_test_stranded: list[torch.Tensor] = []


def _fault_in(storage: torch.UntypedStorage) -> None:
    """Read a byte of every page of ``storage``, so the file behind it is read before the registration lock is taken.

    Weights from safetensors' ``load_file`` stay mapped from the file until first read, and registering them reads
    them in. Measured for a 3.89 GiB file in the page cache on an RTX 4090: 1.96 s to register without this, 0.59 s
    for this plus 1.43 s to register. The rest is the driver making the copy-on-write pages private as it locks them,
    which only a write would do earlier; the weights may be read-only, so `update` takes the lock per storage instead.
    """
    data = torch.empty(0, dtype=torch.uint8).set_(storage)
    data[::_PAGE_SIZE].sum()
    data[-1:].sum()


_StorageKey = tuple[int, int]


def _registration_self_test(device: torch.device | None = None) -> str | None:
    """Check what the registry relies on, once per process on a scratch buffer; None if it holds, else why not.

    Runs on ``device``, the compute device of the model that asks, with that device current: the runtime's calls act on
    the current device, which is GPU 0 on a thread that never chose one, and the self-test must not create a context
    on a GPU the process does not use, nor wait for its queued work. Without ``device``, the thread's current one.

    Storages are registered byte-exact, not page-rounded: safetensors' ``load_file`` returns tensors back to back
    inside one file mapping, so neighbours share pages. A copy that starts inside a registration and runs past its end
    fails ("invalid argument"), so a page-rounded registration would break the copies of the neighbour that starts in
    its last page, and two page-rounded registrations could not hold neighbours at all. Measured on Windows with an
    RTX 4090 (CUDA 13) and an RX 9060 XT (ROCm 10): registrations end at their exact byte, two may share a page, and
    memory past the end in the same page copies as pageable. A driver that rounds to pages keeps the synchronous copy
    instead of failing copies; its refusal is cleared by `_Runtime`.
    """
    if device is None:
        return _self_test_on_current_device(torch.device("cuda"))
    with torch.cuda.device(device):
        return _self_test_on_current_device(device)


def _self_test_on_current_device(device: torch.device) -> str | None:
    global _runtime
    runtime = _find_runtime()
    if runtime is None:
        # Nothing has been called that could fail.
        return "the GPU runtime library was not found"
    page = _PAGE_SIZE
    buffer = (torch.arange(8 * page, dtype=torch.int32) % 251).to(torch.uint8)
    origin = buffer.data_ptr()
    first = (origin // page + 1) * page
    boundary, last = first + page + 100, first + 3 * page

    def view(lo: int, hi: int) -> torch.Tensor:
        return buffer[lo - origin : hi - origin]

    def copies_intact(t: torch.Tensor) -> bool:
        return torch.equal(t.to(device, non_blocking=True).cpu(), t)

    exact_ranges = "this driver does not page-lock memory by exact ranges, so neighbouring weights could not be copied"
    # Torch must see what the runtime registers: a copy of the runtime that torch does not use (ROCm has no
    # driver/runtime split) would leave every copy pageable. `is_pinned` asks about a tensor's storage, so this
    # registers one storage whole.
    whole = torch.empty(2 * page, dtype=torch.uint8)
    # Every address registered and not yet unregistered: the `finally` unregisters what is left.
    registered: list[int] = []

    def unregister_last() -> bool:
        if runtime.unregister(registered[-1]) != 0:
            return False
        registered.pop()
        return True

    try:
        if runtime.register(whole.data_ptr(), whole.nbytes) != 0:
            return "page-locking a test buffer failed"
        registered.append(whole.data_ptr())
        visible = whole.is_pinned()
        if not unregister_last():
            return "unpinning a test buffer failed"
        if not visible:
            return "the GPU runtime library found is not the one torch uses"
        for lo, hi in ((first, boundary), (boundary, last)):
            if runtime.register(lo, hi - lo) != 0:
                return exact_ranges
            registered.append(lo)
        if not all(copies_intact(t) for t in (view(first + 50, boundary), view(boundary, last))):
            return exact_ranges
        if not unregister_last():
            return exact_ranges
        # Now pageable, starting in the last page of a registration.
        if not copies_intact(view(boundary, last)):
            return exact_ranges
        # A driver that cannot unregister would strand every model's weights it pinned.
        if not unregister_last():
            return "unpinning a test buffer failed"
        _runtime = runtime
        return None
    except RuntimeError as error:
        return f"its self-test failed ({error})"
    finally:
        # A registration that outlived its buffer would cover whatever the allocator puts there next, and copies from
        # that would fail. Buffers that cannot be unregistered are kept for the process lifetime instead.
        if [lo for lo in registered if runtime.unregister(lo) != 0]:
            _self_test_stranded.extend((buffer, whole))


class PinHolder:
    """The storages one holder (a cached model wrapper) holds, and which of them its model streams in its current
    lock. Mutated only under the registration lock."""

    __slots__ = ("keys", "wanted", "active")

    def __init__(self) -> None:
        self.keys: set[_StorageKey] = set()
        # What the model streams in its current lock (or streamed in its last one); counts only while `active`.
        self.wanted: set[_StorageKey] = set()
        self.active = False


@dataclass(eq=False)
class _StorageEntry:
    key: _StorageKey
    # Held so the memory cannot be freed while registered; freeing registered memory is undefined behaviour.
    storage: torch.UntypedStorage
    # False when the storage was already page-locked (torch's pinned allocator): it is never unregistered here.
    registered_here: bool
    holders: set[PinHolder] = field(default_factory=set)
    # Active holders that stream it in their current lock. At 0 a storage registered here is reclaimable.
    wanted_by: int = 0


def _storage_key(storage: torch.UntypedStorage) -> _StorageKey:
    return storage.data_ptr(), storage.nbytes()


def storage_keys(storages: Iterable[torch.UntypedStorage]) -> frozenset[_StorageKey]:
    """What `PinnedStorageRegistry.lock` takes: the keys of the storages (empty ones are never registered)."""
    return frozenset(_storage_key(s) for s in storages if s.nbytes() > 0)


class PinnedStorageRegistry:
    """Process-wide in-place page-locking of CPU storages, shared by holders.

    Storages are keyed by address and size, so tied weights and the canonical CPU tensors several per-device wrappers
    share (`SharedCpuWeightsStore`) are registered once and held once per holder. Each storage is registered by its
    exact byte range (see `_registration_self_test`), so every copy from it lies inside its registration.

    A holder keeps what it registered across loads and unloads, so models that take turns in VRAM do not pay for
    registration on every lock: about 0.03 s/GiB for memory already private, measured on Windows (RTX 4090) and Linux
    (RTX PRO 6000), and 0.3 to 0.4 s/GiB the first time for weights still mapped from their file, whose pages the
    driver makes private. `update` marks what a
    lock streams, `idle` marks the end of the model's last lock. A storage that no active holder streams is
    reclaimable: when a registration would pass `PINNED_RAM_CAP_BYTES`, reclaimable storages are unregistered, the
    longest unused first, and only as many as that registration needs.

    Three locks. The lookup lock only guards which storages are registered; the streamer holds it from its check to the
    enqueued copy. The registration lock serializes the slow driver calls. The small lock of `_unregistering` and
    `_locked` (a condition) guards the exclusion between locks and unregistration, below, and is never held across a
    driver call.

    Unregistration removes the entries under the lookup lock first, so no copy can start from them, then waits for the
    streamers' side streams and unregisters. Copies that the registry does not see are different: a partial load's
    `.to(device)`, the repair of required weights, the synchronous fallback of `cast_to_device` and the unpatching of a
    LoRA all copy from a RAM-copy storage on the caller's own stream, and unregistering during such a DMA is undefined.
    So a cached model locks the storages of its whole RAM copy (`lock`) from the start of its cache lock to the end of
    it, even before it holds a registration of its own (another device's wrapper may have registered the shared
    weights). A storage that is locked is not unregistered: reclaiming skips it, and a release leaves it registered,
    pending, until the last unlock. A lock that finds a storage being unregistered waits for that to finish: the copies
    that follow are then plain pageable ones. The waiter holds its cache lock, and the unregistering thread needs no
    cache lock, so the wait cannot deadlock.
    """

    def __init__(self) -> None:
        self._registration_lock = threading.Lock()
        self._lookup_lock = threading.Lock()
        # Posted releases (True) and idle marks (False). A finalizer runs in GC context, possibly on a thread that holds
        # one of the locks, so it only enqueues (as SharedCpuWeightsStore.release_deferred does); an idle mark comes
        # from under the cache's lock, which must not wait for a registration. Every registry call applies the queue
        # first, in order, and so does the streamer between them (`drain_deferred`).
        self._deferred: queue.SimpleQueue[tuple[PinHolder, bool]] = queue.SimpleQueue()
        self._entries: dict[_StorageKey, _StorageEntry] = {}
        # Reclaimable entries (registered here, `wanted_by == 0`), the longest unused first, and their bytes.
        self._reclaimable: dict[_StorageKey, None] = {}
        self._reclaimable_bytes = 0
        self._pinned_bytes = 0
        # Decided by the first registration (`_registration_self_test`).
        self._driver_ok: bool | None = None
        # Storages whose unregistration failed, kept alive for the process lifetime rather than freed while possibly
        # still registered. The cap keeps counting them.
        self._stranded: list[_StorageEntry] = []
        self._cond = threading.Condition()
        # Storages locks hold (count), and storages being unregistered right now. Guarded by `_cond`.
        self._locked: dict[_StorageKey, int] = {}
        self._unregistering: set[_StorageKey] = set()
        # Entries that were dropped while locked, still registered and counted, unregistered once unlocked. A storage
        # registered again meanwhile takes its entry back. Mutated under the registration lock.
        self._pending: dict[_StorageKey, _StorageEntry] = {}

    def update(
        self,
        holder: PinHolder,
        storages: Iterable[torch.UntypedStorage],
        device: torch.device | None = None,
        register_new: bool = True,
    ) -> None:
        """Mark ``holder`` active and streaming exactly ``storages``, and register those it does not hold yet, as far
        as they fit under the cap. What it holds beyond them stays registered, reclaimable. Never raises a driver
        error. ``device`` is where the model computes: registration and the first call's self-test run there.
        ``register_new=False``: the lock did not plan the VRAM for the copies that would follow, so the holder
        registers and joins nothing new; it streams only what it holds already."""
        candidates = {_storage_key(s): s for s in storages if s.nbytes() > 0}
        missing: dict[_StorageKey, torch.UntypedStorage] = {}
        with self._registration_lock:
            self._drain_deferred_locked()
            self._set_wanted_locked(holder, set(candidates), active=True)
            # Joining a storage another holder registered costs nothing, but it starts streaming from it: not in a
            # lock that planned no VRAM for copies in flight (another device may have registered it since).
            for key in candidates.keys() - holder.keys if register_new else ():
                if key in self._entries:
                    self._join_locked(holder, key)
            # The rest is registered below only as far as it can fit, so what the cap refuses is not read in on every
            # lock.
            room = PINNED_RAM_CAP_BYTES - self._pinned_bytes + self._reclaimable_bytes
            for key, storage in candidates.items():
                if key in holder.keys or not register_new:
                    continue
                if storage.is_pinned():  # Torch's pinned allocator: joining costs no room.
                    missing[key] = storage
                elif key[1] <= room:
                    room -= key[1]
                    missing[key] = storage
                else:
                    self._log_cap()
        if self._driver_ok is False:
            return  # The self-test failed: nothing registers.
        for storage in missing.values():
            _fault_in(storage)
        # One storage per lock hold: a release (another device's offload or eviction, under its cache lock) waits
        # for at most one registration and the reclaim it needs, not for a whole model's.
        for key, storage in missing.items():
            with self._registration_lock:
                if key in holder.keys or not holder.active or key not in holder.wanted:
                    continue  # Released or marked idle meanwhile.
                try:
                    if key not in self._entries and not self._register_locked(key, storage, device):
                        continue
                    self._join_locked(holder, key)
                except BaseException:
                    # An interrupt between tracking the registration and the join would leave an entry no holder
                    # holds, which nothing reclaims. Best effort, like the rest of the interrupt handling here: Python
                    # delivers it between any two statements.
                    entry = self._entries.get(key)
                    if entry is not None and not entry.holders:
                        self._drop_locked([entry])
                    raise

    def activate(self, holder: PinHolder) -> bool:
        """Mark ``holder`` active again, streaming what it streamed in its last `update`, for a lock that moved
        nothing. False if some of that is not registered (reclaimed, or refused at the cap): `update` it then."""
        with self._registration_lock:
            self._drain_deferred_locked()
            self._set_wanted_locked(holder, holder.wanted, active=True)
            return holder.wanted <= holder.keys

    def idle(self, holder: PinHolder) -> None:
        """Mark the end of the last lock of ``holder``'s model: what it holds stays registered but becomes reclaimable.
        Takes no lock; the next registry call applies it, before it looks for room."""
        self._deferred.put((holder, False))

    def release(self, holder: PinHolder) -> bool:
        """Release everything ``holder`` holds; whether it held anything. Best effort: never raises on a failing
        device."""
        with self._registration_lock:
            self._drain_deferred_locked()
            held = bool(holder.keys)
            self._release_holder_locked(holder)
            return held

    def release_deferred(self, holder: PinHolder) -> None:
        """Post a release without taking a lock. The only entry point that is safe from GC context."""
        self._deferred.put((holder, True))

    def drain_deferred(self) -> None:
        """Apply posted releases and idle marks, unless another thread is registering right now (it applies them)."""
        if (self._deferred.empty() and not self._pending) or not self._registration_lock.acquire(blocking=False):
            return
        try:
            self._drain_deferred_locked()
        finally:
            self._registration_lock.release()

    def is_registered(self, storage: torch.UntypedStorage) -> bool:
        return _storage_key(storage) in self._entries

    def holds_registrations(self) -> bool:
        return bool(self._entries)

    def can_register(self, device: torch.device | None = None) -> bool:
        """Whether this driver passed the self-test (run on the first call, on ``device``). Needs a CUDA device."""
        if self._driver_ok is None:
            with self._registration_lock:
                self._decide_driver_locked(device)
        return bool(self._driver_ok)

    def has_room_for_any(self, storages: Iterable[torch.UntypedStorage]) -> bool:
        """Whether any of ``storages`` is registered or could be: the model that streams them has something to stream.
        Advisory and lock-free: the caller holds the cache's lock, which must not wait behind a registration. The
        caller plans its VRAM by the answer and tells `update` (``register_new``), so a change after it does not
        start streaming that nobody reserved room for."""
        room = PINNED_RAM_CAP_BYTES - self._pinned_bytes + self._reclaimable_bytes
        for storage in storages:
            # Cheapest first: `is_pinned` asks the driver, which costs milliseconds over a whole model.
            if storage.nbytes() <= room or _storage_key(storage) in self._entries or storage.is_pinned():
                return True
        return False

    def lock(self, keys: frozenset[_StorageKey]) -> None:
        """Keep ``keys`` registered, or unregistered, as they are now until `unlock`: a cached model's cache lock
        copies from them on its own streams (see the class docstring). Waits while any is being unregistered."""
        with self._cond:
            while not self._unregistering.isdisjoint(keys):
                self._cond.wait()
            for key in keys:
                self._locked[key] = self._locked.get(key, 0) + 1

    def unlock(self, keys: frozenset[_StorageKey]) -> None:
        """Release a `lock`. A storage whose release was deferred is unregistered by the next registry call."""
        with self._cond:
            for key in keys:
                count = self._locked[key] - 1
                if count:
                    self._locked[key] = count
                else:
                    del self._locked[key]

    @contextmanager
    def lookup(self, storage: torch.UntypedStorage) -> Iterator[bool]:
        """Whether ``storage`` is registered, held stable for the block: a copy enqueued inside cannot race its
        unregistration."""
        with self._lookup_lock:
            yield _storage_key(storage) in self._entries

    def _drain_deferred_locked(self) -> None:
        while True:
            try:
                holder, release = self._deferred.get_nowait()
            except queue.Empty:
                self._unregister_pending_locked()
                return
            if release:
                self._release_holder_locked(holder)
            else:
                self._set_wanted_locked(holder, holder.wanted, active=False)

    def _unregister_pending_locked(self) -> None:
        if self._pending:
            self._unregister_locked(list(self._pending.values()))

    def _release_holder_locked(self, holder: PinHolder) -> None:
        self._set_wanted_locked(holder, set(), active=False)
        unheld: list[_StorageEntry] = []
        for key in holder.keys:
            entry = self._entries[key]
            entry.holders.discard(holder)
            if not entry.holders:
                unheld.append(entry)
        holder.keys.clear()
        self._drop_locked(unheld)

    def _join_locked(self, holder: PinHolder, key: _StorageKey) -> None:
        entry = self._entries[key]
        # The holder's side first: an interrupt in between leaves an entry without the holder, which a release drops,
        # not one that no release can reach.
        holder.keys.add(key)
        entry.holders.add(holder)
        if holder.active and key in holder.wanted:
            self._want_locked(entry)

    def _set_wanted_locked(self, holder: PinHolder, wanted: set[_StorageKey], active: bool) -> None:
        before = holder.wanted & holder.keys if holder.active else set()
        holder.wanted, holder.active = wanted, active
        after = wanted & holder.keys if active else set()
        for key in after - before:
            self._want_locked(self._entries[key])
        for key in before - after:
            self._unwant_locked(self._entries[key])

    def _want_locked(self, entry: _StorageEntry) -> None:
        entry.wanted_by += 1
        if entry.wanted_by == 1 and entry.key in self._reclaimable:
            del self._reclaimable[entry.key]
            self._reclaimable_bytes -= entry.key[1]

    def _unwant_locked(self, entry: _StorageEntry) -> None:
        entry.wanted_by -= 1
        if entry.wanted_by == 0 and entry.registered_here:
            self._reclaimable[entry.key] = None
            self._reclaimable_bytes += entry.key[1]

    def _make_room_locked(self, nbytes: int) -> bool:
        """Unregister reclaimable storages, the longest unused first, until ``nbytes`` more fit under the cap; nothing
        if even all of them would not make enough room."""
        shortfall = self._pinned_bytes + nbytes - PINNED_RAM_CAP_BYTES
        if shortfall <= 0:
            return True
        # Storages a cache lock holds are not reclaimed, however long they have been unused.
        with self._cond:
            candidates = [key for key in self._reclaimable if key not in self._locked]
        if shortfall > sum(key[1] for key in candidates):
            return False
        victims: list[_StorageEntry] = []
        for key in candidates:
            victims.append(self._entries[key])
            shortfall -= key[1]
            if shortfall <= 0:
                break
        for entry in victims:
            for holder in entry.holders:
                holder.keys.discard(entry.key)
            entry.holders.clear()
        self._drop_locked(victims)
        # A storage that could not be unregistered stays counted (stranded).
        return self._pinned_bytes + nbytes <= PINNED_RAM_CAP_BYTES

    def _log_cap(self) -> None:
        _log_once(
            "cap",
            logging.INFO,
            f"async_weight_streaming: the pinned-RAM cap of {PINNED_RAM_CAP_BYTES / 2**30:.1f} GiB is reached; "
            "further weights in RAM are copied synchronously.",
        )

    def _decide_driver_locked(self, device: torch.device | None) -> None:
        if self._driver_ok is not None:
            return
        reason = _registration_self_test(device)
        self._driver_ok = reason is None
        if reason is not None:
            _log_once(
                "driver",
                logging.WARNING,
                f"async_weight_streaming: {reason}; weights in RAM are copied synchronously.",
            )

    def _register_locked(self, key: _StorageKey, storage: torch.UntypedStorage, device: torch.device | None) -> bool:
        """Register ``storage`` and track it in `_entries`; whether it is tracked now."""
        self._decide_driver_locked(device)
        if not self._driver_ok:
            return False
        pending = self._pending.pop(key, None)
        if pending is not None:  # Dropped while a lock held it, so still registered.
            with self._lookup_lock:
                self._entries[key] = pending
            return True
        if storage.is_pinned():
            with self._lookup_lock:
                self._entries[key] = _StorageEntry(key, storage, registered_here=False)
            return True
        ptr, nbytes = key
        if not self._make_room_locked(nbytes):
            self._log_cap()
            return False
        assert _runtime is not None  # The self-test passed.
        entry = _StorageEntry(key, storage, registered_here=True)
        # Counted before the call: an interrupt that arrives once the range is page-locked must find it counted.
        self._pinned_bytes += nbytes
        try:
            with torch.cuda.device(device) if device is not None else nullcontext():
                error = _runtime.register(ptr, nbytes)
        except BaseException:
            # The call may or may not have registered the range before the interrupt. Unregistering one that is not
            # registered is refused, which is harmless; nothing has copied from it yet, so no copy needs waiting for.
            try:
                _runtime.unregister(ptr)
            finally:
                self._pinned_bytes -= nbytes
            raise
        if error != 0:
            self._pinned_bytes -= nbytes
            _log_once(
                f"register:{error}",
                logging.WARNING,
                f"async_weight_streaming: page-locking {nbytes / 2**20:.1f} MiB of weights failed (CUDA error "
                f"{error}); weights that cannot be page-locked are copied synchronously.",
            )
            return False
        try:
            with self._lookup_lock:
                self._entries[key] = entry
        except BaseException:
            # The range is page-locked: an interrupt here must not leave it untracked and counted forever.
            with self._lookup_lock:
                self._entries.pop(key, None)
            self._unregister_locked([entry])
            raise
        return True

    def _drop_locked(self, entries: list[_StorageEntry]) -> None:
        """Remove entries no holder holds any more, and unregister those registered here."""
        if not entries:
            return
        unregister = [entry for entry in entries if entry.registered_here]
        try:
            with self._lookup_lock:
                for entry in entries:
                    del self._entries[entry.key]
            for entry in entries:
                if entry.key in self._reclaimable:
                    del self._reclaimable[entry.key]
                    self._reclaimable_bytes -= entry.key[1]
        except BaseException:
            # Best effort: what left `_entries` and would be forgotten is kept registered for the process lifetime.
            self._stranded.extend(entry for entry in unregister if entry.key not in self._entries)
            raise
        if unregister:
            self._unregister_locked(unregister)

    def _unregister_locked(self, entries: list[_StorageEntry]) -> None:
        """Unregister ``entries``, which are out of `_entries`, except those a cache lock still holds: they stay
        registered, pending, until the lock ends.

        Best effort: eviction runs this, also from paths that must not raise (a dying worker's last frame). What cannot
        be unregistered is stranded. An interrupt here strands what is left as well before it propagates: dropping
        them would free memory that may still be registered.
        """
        with self._cond:
            now: list[_StorageEntry] = []
            for entry in entries:
                if entry.key in self._locked:
                    self._pending[entry.key] = entry
                else:
                    self._pending.pop(entry.key, None)
                    self._unregistering.add(entry.key)
                    now.append(entry)
        if not now:
            return
        remaining = list(now)
        try:
            try:
                # Copies the streamers issued before the entries were removed may still be reading them. Everything
                # else that copies from these storages holds a lock on them, which kept them out of `now`.
                _synchronize_issued_copies()
            except Exception as error:
                _log_once("sync", logging.WARNING, f"async_weight_streaming: waiting for copies failed ({error}).")
                self._stranded.extend(remaining)
                return
            assert _runtime is not None  # Only storages registered here are unregistered.
            while remaining:
                entry = remaining[-1]
                try:
                    error = _runtime.unregister(entry.key[0])
                except Exception:
                    error = -1
                remaining.pop()
                if error != 0:
                    _log_once(
                        f"unregister:{error}", logging.WARNING, f"async_weight_streaming: unpinning failed ({error})."
                    )
                    self._stranded.append(entry)
                    continue
                self._pinned_bytes -= entry.key[1]
        except BaseException:
            self._stranded.extend(remaining)
            raise
        finally:
            with self._cond:
                self._unregistering.difference_update(entry.key for entry in now)
                self._cond.notify_all()


# Process-global: shared CPU weights of several devices are one set of storages.
PINNED_STORAGES = PinnedStorageRegistry()


@dataclass
class StreamingStats:
    """Counters of one device's streamer. Which path a copy took has no other observable: the output is identical."""

    streamed_tensors: int = 0
    staged_tensors: int = 0
    bounded_waits: int = 0
    max_inflight_bytes: int = 0


class _DeviceStreamer:
    """Issues asynchronous weight copies to one CUDA device and bounds how many bytes are in flight."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._side = torch.cuda.Stream(device)
        self._lock = threading.Lock()
        # Events recorded on the compute stream after each copy, oldest first, with the bytes they release.
        self._inflight: deque[tuple[torch.cuda.Event, int]] = deque()
        self._inflight_bytes = 0
        self._announced = False
        self._used = False  # Copies allocated since the last `release_side_pool`.
        self.stats = StreamingStats()

    def synchronize_copies(self) -> None:
        self._side.synchronize()

    def take_used(self) -> bool:
        with self._lock:
            used, self._used = self._used, False
            return used

    def copy(self, t: torch.Tensor) -> torch.Tensor | None:
        PINNED_STORAGES.drain_deferred()
        storage = t.untyped_storage()
        nbytes = t.nbytes
        if not PINNED_STORAGES.is_registered(storage) and (
            nbytes > STAGE_MAX_BYTES or not PINNED_STORAGES.holds_registrations()
        ):
            return None
        with self._lock:
            self._wait_for_room(nbytes)
            compute = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self._side):
                # Allocated before the lookup lock, which covers only the check and the enqueue.
                dst = torch.empty(t.shape, dtype=t.dtype, device=self.device)
                self._used = True
                with PINNED_STORAGES.lookup(storage) as registered:
                    if registered:
                        dst.copy_(t, non_blocking=True)
                if registered:
                    self.stats.streamed_tensors += 1
                    if not self._announced:
                        self._announced = True
                        logger.info(f"async_weight_streaming: weights in RAM stream to {self.device} asynchronously.")
                else:
                    if nbytes > STAGE_MAX_BYTES:
                        return None
                    try:
                        staged = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                    except RuntimeError as error:
                        _log_once("stage", logging.WARNING, f"async_weight_streaming: staging failed ({error}).")
                        return None
                    staged.copy_(t)
                    # The staging block returns to torch's pinned pool once the copy reading it has finished.
                    dst.copy_(staged, non_blocking=True)
                    self.stats.staged_tensors += 1
            # Flush the queued work to the GPU now. On Windows (WDDM), the driver batches submissions; a later host
            # wait on an event whose work is still unsubmitted serialized copy and compute (measured on an RTX 4090
            # with HAGS: 125 ms instead of 68 ms for overlapped copy + GEMM). A query forces the submission: 70 ms.
            # ROCm on an RX 9060 XT did not need it; it is harmless there.
            self._side.query()
            compute.wait_stream(self._side)
            # The allocator hands this block out again only after the compute stream has consumed it, however long
            # the returned tensor lives. A fixed ring of buffers would be overwritten silently in that case.
            dst.record_stream(compute)
            event = torch.cuda.Event()
            event.record(compute)
            compute.query()  # Flush, as above.
            self._inflight.append((event, nbytes))
            self._inflight_bytes += nbytes
            self.stats.max_inflight_bytes = max(self.stats.max_inflight_bytes, self._inflight_bytes)
            return dst

    def _wait_for_room(self, nbytes: int) -> None:
        inflight = self._inflight
        while inflight and inflight[0][0].query():
            self._inflight_bytes -= inflight.popleft()[1]
        # A tensor larger than the budget proceeds once nothing else is in flight.
        while inflight and self._inflight_bytes + nbytes > STREAM_BUDGET_BYTES:
            event, done_bytes = inflight.popleft()
            event.synchronize()
            self._inflight_bytes -= done_bytes
            self.stats.bounded_waits += 1


# One streamer per device, created on first use.
_streamers: dict[torch.device, _DeviceStreamer] = {}
_streamers_lock = threading.Lock()


def _synchronize_issued_copies() -> None:
    for streamer in list(_streamers.values()):
        streamer.synchronize_copies()


def _streamer_for(device: torch.device) -> _DeviceStreamer:
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    streamer = _streamers.get(device)
    if streamer is None:
        with _streamers_lock:
            streamer = _streamers.get(device)
            if streamer is None:
                streamer = _DeviceStreamer(device)
                _streamers[device] = streamer
    return streamer


def release_side_pool(device: torch.device) -> None:
    """Return the blocks the side stream's copies left cached, when a lock that streamed to ``device`` has ended.

    Freed copies stay cached for the side stream, and the compute stream cannot reuse them: measured end to end
    (Z-Image and ERNIE-Image on an RTX 4090 at max_cache_vram_gb 8), the process peak rose by 846 MiB in the VAE decode
    after a streamed denoise. ``empty_cache`` releases the unused blocks of every stream, so it runs only after a lock
    that allocated copies, and through `TorchDevice.empty_cache`, which defers while another GPU is mid-session.
    """
    if device.type != "cuda":
        return
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    streamer = _streamers.get(device)
    if streamer is None or not streamer.take_used():
        return
    try:
        # Freed copies become free blocks once their copy and the compute stream's reads are done; empty_cache waits
        # for the latter itself.
        streamer.synchronize_copies()
        TorchDevice.empty_cache()
    except Exception as error:
        # Runs as a lock ends; the memory is advisory, the unlock is not.
        _log_once("release", logging.WARNING, f"async_weight_streaming: returning streamed copies failed ({error}).")


def _is_capturing(device: torch.device) -> bool:
    if device.index == torch.cuda.current_device():
        return torch.cuda.is_current_stream_capturing()
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def stream_to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    """Copy CPU tensor ``t`` to CUDA ``device`` asynchronously, or return None for the caller's synchronous copy."""
    if not is_async_weight_streaming_enabled() or not is_streamable_tensor(t):
        return None
    if not t.is_contiguous() or (t.requires_grad and torch.is_grad_enabled()):
        return None
    streamer = _streamer_for(device)
    if _is_capturing(streamer.device):
        return None
    return streamer.copy(t)


def streaming_stats(device: torch.device) -> StreamingStats | None:
    """A snapshot of the counters of ``device``'s streamer, or None if nothing streamed to it yet."""
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    streamer = _streamers.get(device)
    return replace(streamer.stats) if streamer is not None else None

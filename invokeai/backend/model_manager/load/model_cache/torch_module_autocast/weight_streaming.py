"""Async weight streaming for partially loaded models, opt-in via ``async_weight_streaming``.

A partially loaded model keeps part of its weights in RAM, and the custom modules copy each of them to the compute
device on every forward (`cast_to_device`). Without this feature that copy is a synchronous ``t.to(device)`` from
pageable memory: the CPU waits for the GPU to drain before every copy, and the GPU idles while the copy runs. With it,
the copy runs on a side stream from page-locked RAM and overlaps with the compute stream. Outputs are bit-identical:
the same bytes arrive before the same kernels read them.

Two parts:

- `PinnedStorageRegistry` page-locks CPU storages in place (``cudaHostRegister``, no second copy), refcounted per
  holder. `CachedModelWithPartialLoad` keeps the CPU-resident weights of its streaming modules registered.
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
from contextlib import contextmanager
from dataclasses import dataclass, replace
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


def _registration_self_test() -> str | None:
    """Check what the registry relies on, once per process on a scratch buffer; None if it holds, else why not.

    Storages are registered byte-exact, not page-rounded: safetensors' ``load_file`` returns tensors back to back
    inside one file mapping, so neighbours share pages. A copy that starts inside a registration and runs past its end
    fails ("invalid argument"), so a page-rounded registration would break the copies of the neighbour that starts in
    its last page, and two page-rounded registrations could not hold neighbours at all. Measured on Windows with an
    RTX 4090 (CUDA 13) and an RX 9060 XT (ROCm 10): registrations end at their exact byte, two may share a page, and
    memory past the end in the same page copies as pageable. A driver that rounds to pages keeps the synchronous copy
    instead of failing copies; its refusal is cleared by `_Runtime`.
    """
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
        return torch.equal(t.to("cuda", non_blocking=True).cpu(), t)

    exact_ranges = "this driver does not page-lock memory by exact ranges, so neighbouring weights could not be copied"
    registered: list[int] = []
    try:
        # Torch must see what the runtime registers: a copy of the runtime that torch does not use (ROCm has no
        # driver/runtime split) would leave every copy pageable. `is_pinned` asks about a tensor's storage, so this
        # registers one storage whole.
        whole = torch.empty(2 * page, dtype=torch.uint8)
        if runtime.register(whole.data_ptr(), whole.nbytes) != 0:
            return "page-locking a test buffer failed"
        visible = whole.is_pinned()
        runtime.unregister(whole.data_ptr())
        if not visible:
            return "the GPU runtime library found is not the one torch uses"
        for lo, hi in ((first, boundary), (boundary, last)):
            if runtime.register(lo, hi - lo) != 0:
                return exact_ranges
            registered.append(lo)
        if not all(copies_intact(t) for t in (view(first + 50, boundary), view(boundary, last))):
            return exact_ranges
        if runtime.unregister(registered.pop()) != 0:
            return exact_ranges
        # Now pageable, starting in the last page of a registration.
        if not copies_intact(view(boundary, last)):
            return exact_ranges
        _runtime = runtime
        return None
    except RuntimeError as error:
        return f"its self-test failed ({error})"
    finally:
        for lo in registered:
            runtime.unregister(lo)


@dataclass(eq=False)
class _StorageEntry:
    key: _StorageKey
    # Held so the memory cannot be freed while registered; freeing registered memory is undefined behaviour.
    storage: torch.UntypedStorage
    # False when the storage was already page-locked (torch's pinned allocator): it is never unregistered here.
    registered_here: bool
    refcount: int = 0  # Holders


class PinHolder:
    """The storages one holder (a cached model wrapper) has registered. Mutated only under the registration lock."""

    __slots__ = ("keys",)

    def __init__(self) -> None:
        self.keys: set[_StorageKey] = set()


def _storage_key(storage: torch.UntypedStorage) -> _StorageKey:
    return storage.data_ptr(), storage.nbytes()


class PinnedStorageRegistry:
    """Process-wide in-place page-locking of CPU storages, refcounted per holder.

    Storages are keyed by address and size, so tied weights and the canonical CPU tensors several per-device wrappers
    share (`SharedCpuWeightsStore`) are registered once and counted once per holder. Each storage is registered by its
    exact byte range (see `_registration_self_test`), so every copy from it lies inside its registration.

    Two locks. The lookup lock only guards which storages are registered; the streamer holds it from its check to the
    enqueued copy. The registration lock serializes the slow driver calls. Unregistration removes the entries under the
    lookup lock first, so no copy can start from them, then waits for the copies already issued and unregisters.
    """

    def __init__(self) -> None:
        self._registration_lock = threading.Lock()
        self._lookup_lock = threading.Lock()
        # Releases posted by wrapper finalizers. A finalizer runs in GC context, possibly on a thread that holds one of
        # the locks, so it only enqueues (as SharedCpuWeightsStore.release_deferred does). The next update or release
        # applies the queue, and so does the streamer between them (`drain_deferred`).
        self._deferred: queue.SimpleQueue[PinHolder] = queue.SimpleQueue()
        self._entries: dict[_StorageKey, _StorageEntry] = {}
        self._pinned_bytes = 0
        # Decided by the first registration (`_registration_self_test`).
        self._driver_ok: bool | None = None
        # Storages whose unregistration failed, kept alive for the process lifetime rather than freed while possibly
        # still registered. The cap keeps counting them.
        self._stranded: list[_StorageEntry] = []

    def update(self, holder: PinHolder, storages: Iterable[torch.UntypedStorage]) -> None:
        """Make ``holder`` hold exactly ``storages``, as far as they can be registered. Never raises a driver error."""
        desired = {_storage_key(s): s for s in storages if s.nbytes() > 0}
        if self._driver_ok is not False:
            for key, storage in desired.items():
                if key not in self._entries:
                    _fault_in(storage)
        with self._registration_lock:
            self._drain_deferred_locked()
            self._release_keys_locked(holder, holder.keys - desired.keys())
        # One storage per lock hold: a release (another device's offload or eviction, under its cache lock) waits
        # for at most one registration, not for a whole model's.
        for key, storage in desired.items():
            with self._registration_lock:
                if key not in holder.keys and self._acquire_locked(key, storage):
                    holder.keys.add(key)

    def release(self, holder: PinHolder) -> None:
        """Release everything ``holder`` holds. Best effort: never raises on a failing device."""
        with self._registration_lock:
            self._drain_deferred_locked()
            self._release_keys_locked(holder, set(holder.keys))

    def release_deferred(self, holder: PinHolder) -> None:
        """Post a release without taking a lock. The only entry point that is safe from GC context."""
        self._deferred.put(holder)

    def drain_deferred(self) -> None:
        """Apply posted releases, unless another thread is registering right now (it applies them itself)."""
        if self._deferred.empty() or not self._registration_lock.acquire(blocking=False):
            return
        try:
            self._drain_deferred_locked()
        finally:
            self._registration_lock.release()

    def is_registered(self, storage: torch.UntypedStorage) -> bool:
        return _storage_key(storage) in self._entries

    def holds_registrations(self) -> bool:
        return bool(self._entries)

    def can_register(self) -> bool:
        """Whether this driver passed the self-test (run on the first call). Needs a CUDA device."""
        if self._driver_ok is None:
            with self._registration_lock:
                self._decide_driver_locked()
        return bool(self._driver_ok)

    @contextmanager
    def lookup(self, storage: torch.UntypedStorage) -> Iterator[bool]:
        """Whether ``storage`` is registered, held stable for the block: a copy enqueued inside cannot race its
        unregistration."""
        with self._lookup_lock:
            yield _storage_key(storage) in self._entries

    def _drain_deferred_locked(self) -> None:
        while True:
            try:
                holder = self._deferred.get_nowait()
            except queue.Empty:
                return
            self._release_keys_locked(holder, set(holder.keys))

    def _acquire_locked(self, key: _StorageKey, storage: torch.UntypedStorage) -> bool:
        entry = self._entries.get(key)
        if entry is None:
            entry = self._register_locked(key, storage)
            if entry is None:
                return False
            with self._lookup_lock:
                self._entries[key] = entry
        entry.refcount += 1
        return True

    def _decide_driver_locked(self) -> None:
        if self._driver_ok is not None:
            return
        reason = _registration_self_test()
        self._driver_ok = reason is None
        if reason is not None:
            _log_once(
                "driver",
                logging.WARNING,
                f"async_weight_streaming: {reason}; weights in RAM are copied synchronously.",
            )

    def _register_locked(self, key: _StorageKey, storage: torch.UntypedStorage) -> _StorageEntry | None:
        self._decide_driver_locked()
        if not self._driver_ok:
            return None
        if storage.is_pinned():
            return _StorageEntry(key, storage, registered_here=False)
        ptr, nbytes = key
        if self._pinned_bytes + nbytes > PINNED_RAM_CAP_BYTES:
            _log_once(
                "cap",
                logging.INFO,
                f"async_weight_streaming: the pinned-RAM cap of {PINNED_RAM_CAP_BYTES / 2**30:.1f} GiB is reached; "
                "further weights in RAM are copied synchronously.",
            )
            return None
        assert _runtime is not None  # The self-test passed.
        error = _runtime.register(ptr, nbytes)
        if error != 0:
            _log_once(
                f"register:{error}",
                logging.WARNING,
                f"async_weight_streaming: page-locking {nbytes / 2**20:.1f} MiB of weights failed (CUDA error "
                f"{error}); weights that cannot be page-locked are copied synchronously.",
            )
            return None
        self._pinned_bytes += nbytes
        return _StorageEntry(key, storage, registered_here=True)

    def _release_keys_locked(self, holder: PinHolder, keys: set[_StorageKey]) -> None:
        released: list[_StorageEntry] = []
        for key in keys:
            holder.keys.discard(key)
            entry = self._entries[key]
            entry.refcount -= 1
            if entry.refcount == 0:
                released.append(entry)
        if not released:
            return
        with self._lookup_lock:
            for entry in released:
                del self._entries[entry.key]
        unregister = [entry for entry in released if entry.registered_here]
        if unregister:
            self._unregister_locked(unregister)

    def _unregister_locked(self, entries: list[_StorageEntry]) -> None:
        # Best effort: eviction runs this, also from paths that must not raise (a dying worker's last frame). What
        # cannot be unregistered is stranded. These entries are already out of `_entries`, so an interrupt here strands
        # what is left as well before it propagates: dropping them would free memory that may still be registered.
        remaining = list(entries)
        try:
            try:
                # Copies issued before the entries were removed may still be reading them.
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

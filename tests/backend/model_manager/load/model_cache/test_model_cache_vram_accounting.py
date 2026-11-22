"""Tests for the VRAM budget arithmetic behind lock()/partial loads.

`_get_vram_available` must count every byte this process can actually obtain: driver-free memory
PLUS the torch caching allocator's reserved-but-unused blocks (the allocator reuses those
directly, and `empty_cache()` returns whole unoccupied segments to the driver). Budgeting on
driver-free alone under-reported by whatever earlier stages freed without an `empty_cache()`
— observed in the wild as a fully-evictable multi-GB reserve pushing a 20 GB transformer down
to 0% VRAM residency while the allocator happily reused the "missing" memory for activations.
"""

import logging
import re
import sys
import threading
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache import model_cache as model_cache_module
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.util import devices as devices_module
from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
from invokeai.backend.util.devices import EmptyCacheOutcome, TorchDevice

GB = 1024**3
MB = 1024**2

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")


class BigModule(torch.nn.Module):
    """A module whose single parameter is `mb` MiB of fp32."""

    def __init__(self, mb: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(mb * MB // 4, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_cache(logger: logging.Logger | MagicMock | None = None) -> ModelCache:
    if logger is None:
        logger = MagicMock()
        logger.getEffectiveLevel.return_value = logging.INFO
    return ModelCache(
        execution_device_working_mem_gb=0.1,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cuda:0",
        storage_device="cpu",
        logger=logger,
        shared_cpu_weights=None,
    )


@requires_cuda
def test_lock_offloads_unlocked_models_under_working_memory_pressure():
    """A big working-memory reservation must evict resident-but-unlocked models so the locked
    model still loads fully (the freed bytes become budget via the allocator-reserve credit and
    the post-offload empty_cache)."""
    torch.cuda.empty_cache()
    cache = _make_cache()
    cache.put("A", BigModule(512))
    rec_a = cache.get("A")
    cache.lock(rec_a, None)
    cache.unlock(rec_a)
    assert rec_a.cached_model.cur_vram_bytes() >= 512 * MB

    free, _total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    # Without offloading A there is only 128 MB of budget — far less than B needs.
    working = free - 128 * MB

    cache.put("B", BigModule(256))
    rec_b = cache.get("B")
    cache.lock(rec_b, working)
    try:
        assert rec_a.cached_model.cur_vram_bytes() == 0, "unlocked resident model was not offloaded"
        assert rec_b.cached_model.cur_vram_bytes() == rec_b.cached_model.total_bytes()
    finally:
        cache.unlock(rec_b)


@requires_cuda
def test_get_vram_available_credits_reserved_but_free_allocator_blocks():
    """Freed-but-not-empty_cache'd allocator blocks are reclaimable and must count as available."""
    torch.cuda.empty_cache()
    cache = _make_cache()

    # Simulate a previous pipeline stage's freed activations: 1 GiB allocated then dropped, with
    # no empty_cache — the bytes stay in the allocator's reserve, invisible to mem_get_info.
    junk = torch.empty(1 * GB, dtype=torch.uint8, device="cuda:0")
    del junk

    free, _total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    working = free - 128 * MB
    available = cache._get_vram_available(working)

    # Driver-free alone would report ~128 MB; the credited reserve must dominate. The margin
    # tolerates concurrent allocations by other processes on a shared dev GPU.
    assert available >= 900 * MB, f"reserved-but-free blocks not credited (available={available / MB:.0f}MB)"

    torch.cuda.empty_cache()


@requires_cuda
def test_negative_budget_warns_and_names_locked_residents():
    """When the budget stays short after offloading, the first-pass warning must name what is
    still occupying the device — locked entries especially, since the offload cannot touch them."""
    torch.cuda.empty_cache()
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    cache = _make_cache(logger)

    cache.put("stuck", BigModule(256))
    rec_stuck = cache.get("stuck")
    cache.lock(rec_stuck, None)  # deliberately left locked

    free, _total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    impossible_working = free + 10 * GB

    cache.put("victim", BigModule(64))
    rec_victim = cache.get("victim")
    cache.lock(rec_victim, impossible_working)
    try:
        # ModelCache wraps its logger in a PrefixedLoggerAdapter, so adapter.warning() reaches
        # the underlying (mock) logger as .log(WARNING, msg).
        warnings = [str(call.args[0]) for call in logger.warning.call_args_list]
        warnings += [
            str(call.args[1])
            for call in logger.log.call_args_list
            if call.args and call.args[0] == logging.WARNING and len(call.args) > 1
        ]
        budget_warnings = [message for message in warnings if "VRAM budget for 'victim' is short by" in message]

        assert budget_warnings, f"no budget-short warning emitted; warnings: {warnings}"
        assert "stuck=" in budget_warnings[0]
        assert "[locked]" in budget_warnings[0]
    finally:
        cache.unlock(rec_victim)
        cache.unlock(rec_stuck)
        torch.cuda.empty_cache()


@requires_cuda
def test_reclaimable_credit_withheld_under_expandable_segments(monkeypatch: pytest.MonkeyPatch):
    """Under expandable-segments mode the (reserved - allocated) figure counts intra-segment
    holes that empty_cache cannot reclaim and a large allocation cannot use — the credit must be
    withheld entirely (the env parse is authoritative: the mode is fixed before torch import)."""
    cache = _make_cache()

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert cache._get_reclaimable_allocator_bytes() == 0

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "expandable_segments: true")
    assert cache._get_reclaimable_allocator_bytes() == 0

    monkeypatch.delenv("PYTORCH_HIP_ALLOC_CONF")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    # With no allocator config the credit path is active again (>= 0 by construction).
    assert cache._get_reclaimable_allocator_bytes() >= 0


def test_physical_availability_shares_the_reclaimable_credit_policy(monkeypatch: pytest.MonkeyPatch):
    """`_get_physical_vram_available` (out-of-cache loads, `make_room_in_vram`) must budget from the
    same device measurement as `lock()`: ignoring the cache cap, but still crediting the allocator
    reserve only through `_get_reclaimable_allocator_bytes` — never the raw reserved-minus-allocated
    figure, which over-reports under expandable segments."""
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
        shared_cpu_weights=None,
        max_vram_cache_size_gb=1.0,
    )
    cache._execution_device = torch.device("cuda")  # policy only; every VRAM query below is patched out
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * GB, 24 * GB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 2 * GB)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 5 * GB)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {"inactive_split_bytes.all.current": GB})
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_HIP_ALLOC_CONF", raising=False)

    # The cap governs the cache's own budget (1 GB cap - 1 GB working - 2 GB in use)...
    assert cache._get_vram_available(None) == -2 * GB
    # ...but not an out-of-cache load: 4 GB free + (5 - 2 - 1 inactive-split) GB reclaimable - 1 GB working.
    assert cache._get_physical_vram_available() == 5 * GB

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert cache._get_physical_vram_available() == 3 * GB


def test_making_room_keeps_free_the_reserve_a_lock_with_that_working_memory_keeps(monkeypatch: pytest.MonkeyPatch):
    """A caller about to lock a model with `working_mem_bytes` asks `make_room_in_vram` with the same figure: the
    reserve is that request, but never less than `device_working_mem_gb`, exactly as the lock's."""
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
        shared_cpu_weights=None,
    )
    cache._execution_device = torch.device("cuda")  # policy only; every VRAM query below is patched out
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * GB, 24 * GB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 2 * GB)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 5 * GB)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {"inactive_split_bytes.all.current": GB})
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_HIP_ALLOC_CONF", raising=False)

    # 4 GB free + 2 GB reclaimable, less the reserve.
    assert cache.make_room_in_vram(GB) == 5 * GB
    assert cache.make_room_in_vram(GB, working_mem_bytes=GB // 2) == 5 * GB
    assert cache.make_room_in_vram(GB, working_mem_bytes=5 * GB // 2) == 7 * GB // 2


def test_the_windows_video_memory_budget_caps_the_measured_free_vram(monkeypatch: pytest.MonkeyPatch):
    """On Windows ROCm, torch reports the device total minus this process's usage. Windows pages allocations into
    shared system memory once the process passes its video-memory budget, so the budget's headroom is what the cache
    may plan with -- not the larger figure torch reports."""
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
        shared_cpu_weights=None,
    )
    cache._execution_device = torch.device("cuda")  # policy only; every VRAM query below is patched out
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (12 * GB, 16 * GB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 2 * GB)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2 * GB)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {"inactive_split_bytes.all.current": 0})
    monkeypatch.setattr("invokeai.backend.util.devices.video_memory_budget", lambda device: 11 * GB)

    # 11 GB budget - 4 GB live = 7 GB headroom, + 2 GB allocated - 1 GB working - 2 GB in use; not torch's 12 GB.
    assert cache._get_vram_available(None) == 6 * GB


@pytest.mark.parametrize("cache_cap_gb", [None, 12.0])
@pytest.mark.parametrize("reserve_vram_gb", [0.0, 1.5])
def test_reserve_vram_reduces_available_memory(
    monkeypatch: pytest.MonkeyPatch, cache_cap_gb: float | None, reserve_vram_gb: float
):
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        max_vram_cache_size_gb=cache_cap_gb,
        reserve_vram_gb=reserve_vram_gb,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
        shared_cpu_weights=None,
    )
    cache._execution_device = torch.device("cuda")  # Avoid requiring a physical GPU.
    monkeypatch.setattr(TorchDevice, "cuda_mem_get_info", classmethod(lambda cls, device: (8 * GB, 16 * GB)))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 2 * GB)
    monkeypatch.setattr(cache, "_get_reclaimable_allocator_bytes", lambda: 0)

    reserved_bytes = int(reserve_vram_gb * GB)
    expected_cache = (9 if cache_cap_gb is not None else 7) * GB - reserved_bytes
    assert cache._get_vram_available(None) == expected_cache
    assert cache._get_physical_vram_available() == 7 * GB - reserved_bytes


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
@pytest.mark.parametrize("outcome", list(EmptyCacheOutcome), ids=lambda outcome: outcome.name.lower())
def test_offloading_under_expandable_segments_stops_once_enough_is_free(
    monkeypatch: pytest.MonkeyPatch, outcome: EmptyCacheOutcome, device_type: str
):
    """Under expandable segments the driver sees an offloaded model's pages only after empty_cache(), and no
    allocator credit stands in for them. Re-measuring without it saw no progress and unloaded every unlocked model.

    `empty_cache` is peer-aware: while another generation device is mid-session it defers instead of releasing, so
    the measurement cannot see the offload at all and the freed bytes are credited to it directly. A release the
    backend refuses leaves them with torch just the same, which reuses them, so they are credited too."""
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
        shared_cpu_weights=None,
    )
    cache._execution_device = torch.device(device_type)  # policy only; every VRAM query below is patched out
    device = {"free": 0, "allocated": 12 * GB, "unmappable": 0}
    backend = getattr(torch, device_type)

    def move_to_ram(entry, bytes_to_free):
        device["allocated"] -= 4 * GB
        device["unmappable"] += 4 * GB
        return 4 * GB

    def empty_cache(force: bool = False) -> EmptyCacheOutcome:
        if outcome is EmptyCacheOutcome.DEFERRED:  # nothing is released, and the driver keeps reporting the old figure
            TorchDevice._empty_cache_deferred.set()
        elif outcome is EmptyCacheOutcome.RAN:
            device["free"] += device["unmappable"]
            device["unmappable"] = 0
        return outcome

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(TorchDevice, "_empty_cache_deferred", threading.Event())
    # A discrete XPU: the driver's free memory plus what torch allocated, as for a CUDA device.
    monkeypatch.setattr(
        TorchDevice, "memory_available_to_process", classmethod(lambda cls, d: device["free"] + device["allocated"])
    )
    monkeypatch.setattr(model_cache_module, "device_has_dedicated_vram", lambda d: True)
    monkeypatch.setattr(backend, "memory_allocated", lambda d: device["allocated"])
    monkeypatch.setattr(backend, "memory_reserved", lambda d: device["allocated"] + device["unmappable"])
    monkeypatch.setattr(TorchDevice, "empty_cache", empty_cache)
    monkeypatch.setattr(cache, "_move_model_to_ram", MagicMock(side_effect=move_to_ram))
    for key in "ABC":
        entry = MagicMock(key=key, is_locked=False)
        entry.cached_model.total_bytes.return_value = 4 * GB
        cache._cached_models[key] = entry

    # 2 GB needed + 1 GB working memory: one 4 GB model is enough.
    assert cache._offload_unlocked_models(2 * GB) == 4 * GB
    assert cache._move_model_to_ram.call_count == 1


class TestReserveRelease:
    """When a lock hands torch's cached blocks back to the driver (`_release_allocator_blocks_for_reserve`).

    The reserve a lock budgets must be free for the driver too: cuDNN loads a convolution engine's kernel module into
    driver memory on first use, and on Windows a driver allocation that does not fit is retried indefinitely instead of
    failing. The driver's and torch's figures are faked, so this runs without a GPU.
    """

    @pytest.fixture(autouse=True)
    def _windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")

    @staticmethod
    def _cache(device: str) -> ModelCache:
        """A cache with the 0.1 GB default reserve, built for the CPU: a CUDA cache reads the card's properties."""
        cache = ModelCache(
            execution_device_working_mem_gb=0.1,
            enable_partial_loading=True,
            keep_ram_copy_of_weights=True,
            execution_device="cpu",
            storage_device="cpu",
            logger=MagicMock(),
            shared_cpu_weights=None,
        )
        cache._execution_device = torch.device(device)
        return cache

    def _release(self, monkeypatch, *, driver_free_mb: int, releasable_mb: int, working_mb, device="cuda") -> list:
        cache = self._cache(device)
        calls: list = []
        monkeypatch.setattr(
            TorchDevice, "cuda_mem_get_info", classmethod(lambda cls, d: (driver_free_mb * MB, 24 * GB))
        )
        monkeypatch.setattr(cache, "_get_reclaimable_allocator_bytes", lambda: releasable_mb * MB)
        monkeypatch.setattr(model_cache_module, "expandable_segments_enabled", lambda: False)
        monkeypatch.setattr(
            TorchDevice, "empty_cache", classmethod(lambda cls, force=False: calls.append(force) or True)
        )
        cache._release_allocator_blocks_for_reserve(None if working_mb is None else working_mb * MB)
        return calls

    def test_a_short_reserve_with_releasable_blocks_is_released_past_a_busy_peer(self, monkeypatch):
        assert self._release(monkeypatch, driver_free_mb=2048, releasable_mb=3072, working_mb=4096) == [True]

    def test_a_reserve_that_is_driver_free_costs_nothing(self, monkeypatch):
        assert self._release(monkeypatch, driver_free_mb=4096, releasable_mb=3072, working_mb=4096) == []

    def test_slivers_left_by_a_partial_load_are_not_worth_a_sync(self, monkeypatch):
        assert self._release(monkeypatch, driver_free_mb=2048, releasable_mb=64, working_mb=4096) == []

    @pytest.mark.parametrize(("driver_free_mb", "expected"), [(80, [True]), (120, [])])
    def test_no_request_means_the_default_reserve(self, monkeypatch, driver_free_mb, expected):
        assert (
            self._release(monkeypatch, driver_free_mb=driver_free_mb, releasable_mb=1024, working_mb=None) == expected
        )

    @pytest.mark.parametrize("device", ["xpu", "mps"])
    def test_other_devices_are_left_alone(self, monkeypatch, device):
        assert self._release(monkeypatch, driver_free_mb=0, releasable_mb=4096, working_mb=4096, device=device) == []

    def test_off_windows_the_release_waits_for_a_busy_peer(self, monkeypatch):
        """Where a driver allocation that does not fit fails instead of hanging, a forced release would only bring
        back the cross-GPU empty_cache convoy."""
        monkeypatch.setattr(sys, "platform", "linux")
        assert self._release(monkeypatch, driver_free_mb=2048, releasable_mb=3072, working_mb=4096) == [False]

    def test_under_expandable_segments_the_unused_reserved_pages_count(self, monkeypatch):
        """The allocator's reclaimable figure is 0 there, but empty_cache() unmaps freed pages inside segments
        (the default on ROCm under Windows)."""
        cache = self._cache("cuda")
        calls: list = []
        monkeypatch.setattr(TorchDevice, "cuda_mem_get_info", classmethod(lambda cls, d: (2 * GB, 24 * GB)))
        monkeypatch.setattr(model_cache_module, "expandable_segments_enabled", lambda: True)
        monkeypatch.setattr(cache, "_get_reclaimable_allocator_bytes", lambda: 0)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 9 * GB)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 6 * GB)
        monkeypatch.setattr(
            TorchDevice, "empty_cache", classmethod(lambda cls, force=False: calls.append(force) or True)
        )

        cache._release_allocator_blocks_for_reserve(4 * GB)

        assert calls == [True]


class _Pool:
    """A GPU's driver and torch's caching allocator, as far as the cache's budget reads them.

    `fragmented` bytes of the unused reserve sit in chunks that hold live allocations: a trim cannot return them.
    cudaMallocAsync reports no inactive splits; the native allocator reports the fragmented bytes as such.

    Releases go through the real `TorchDevice.empty_cache` against a faked backend, so its deferral reads the
    generation device pool. `busy="peer"`: another GPU (cuda:1) is mid-session while this thread runs its session on
    the cache's device (cuda:0). `busy="own"`: a single-GPU install whose one GPU, the cache's, is mid-session while
    this thread runs none (an API request). Either way an unforced release defers. `off_queue` (with `busy="peer"`):
    this thread holds cuda:0 as an off-queue borrow (image embedding, Expand Prompt) rather than as a session.
    `refuse`: the backend raises on every release, as a broken or exhausted device context does.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        backend: str,
        reserved: int,
        allocated: int,
        fragmented: int,
        busy: str | None = None,
        off_queue: bool = False,
        refuse: bool = False,
    ):
        self.total = 24 * GB
        self.reserved = reserved
        self.allocated = allocated
        self.fragmented = fragmented
        self.refuse = refuse
        self.trims: list[bool] = []  # `force` of each empty_cache() that was asked for
        stats = {} if backend == "cudaMallocAsync" else {"inactive_split_bytes.all.current": fragmented}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: backend)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: self.reserved)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: self.allocated)
        monkeypatch.setattr(torch.cuda, "memory_stats", lambda device=None: stats)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (self.driver_free, self.total))
        monkeypatch.setattr(model_cache_module, "expandable_segments_enabled", lambda: False)

        monkeypatch.setattr(torch.cuda, "empty_cache", self._release)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices_module, "_xpu_is_available", lambda: False)
        monkeypatch.setattr(devices_module, "EMPTY_CACHE_OBSERVERS", [])
        monkeypatch.setattr(TorchDevice, "_empty_cache_deferred", threading.Event())
        empty_cache = TorchDevice.empty_cache

        def recorded_empty_cache(cls, force: bool = False) -> EmptyCacheOutcome:
            self.trims.append(force)
            return empty_cache(force=force)

        monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(recorded_empty_cache))

        devices = ["cuda:0", "cuda:1"] if busy == "peer" else ["cuda:0"]
        device_locks = {device: threading.Lock() for device in devices}
        if busy is not None:
            device_locks["cuda:1" if busy == "peer" else "cuda:0"].acquire()
        if off_queue:
            device_locks["cuda:0"].acquire()
        monkeypatch.setattr(GENERATION_DEVICE_POOL, "_device_locks", device_locks)
        monkeypatch.setattr(GENERATION_DEVICE_POOL, "_order", devices)
        monkeypatch.setattr(GENERATION_DEVICE_POOL, "_lent", {"cuda:0"} if off_queue else set())
        monkeypatch.setattr(GENERATION_DEVICE_POOL, "_lent_off_queue", {"cuda:0"} if off_queue else set())
        session = threading.local()
        if busy != "own":
            session.device = torch.device("cuda:0")
        monkeypatch.setattr(TorchDevice, "_session_device", session)

    @property
    def driver_free(self) -> int:
        return self.total - self.reserved

    def _release(self) -> None:
        if self.refuse:
            raise torch.OutOfMemoryError("CUDA out of memory while trimming the pool")
        self.reserved -= max(0, self.reserved - self.allocated - self.fragmented)


class _Resident:
    """An unlocked cache entry as the offload reads it: its size and how much of it is still on the device."""

    is_locked = False

    def __init__(self, key: str, size: int):
        self.key = key
        self.size = size
        self.on_device = size
        self.cached_model = self

    def total_bytes(self) -> int:
        return self.size

    def cur_vram_bytes(self) -> int:
        return self.on_device


def _pool_cache(logger: logging.Logger | None = None, **kwargs) -> ModelCache:
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger or MagicMock(),
        shared_cpu_weights=None,
        **kwargs,
    )
    cache._execution_device = torch.device("cuda:0")  # policy only; every VRAM query is faked by `_Pool`
    return cache


def _resident_models(
    monkeypatch: pytest.MonkeyPatch, cache: ModelCache, pool: _Pool, sizes: dict[str, int], fragmenting=()
) -> dict[str, _Resident]:
    """Unlocked entries, smallest first, whose offload returns their weights to the pool; those named in `fragmenting`
    leave them in chunks that other live allocations share."""
    models = {key: _Resident(key, size) for key, size in sizes.items()}

    def move_to_ram(entry, bytes_to_free, keep_required_weights_in_vram=None):
        model = models.get(entry.key)
        if model is None:  # the model being locked: it has no weights on the device
            return 0
        freed, model.on_device = model.on_device, 0
        pool.allocated -= freed
        if entry.key in fragmenting:
            pool.fragmented += freed
        return freed

    monkeypatch.setattr(cache, "_move_model_to_ram", move_to_ram)
    cache._cached_models.update(models)  # type: ignore[arg-type]
    return models


def _megabytes(caplog: pytest.LogCaptureFixture, marker: str) -> list[list[int]]:
    """The MB figures of each log message that contains `marker`."""
    return [
        [int(n) for n in re.findall(r"(\d+)MB", record.getMessage())]
        for record in caplog.records
        if marker in record.getMessage()
    ]


def test_an_async_pool_gets_no_credit_for_its_unused_bytes_and_a_native_one_keeps_its_credit(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without inactive splits to tell, a fragmented cudaMallocAsync pool's whole reserved-minus-allocated gap would
    count as releasable, but a trim returns only the chunks without a live allocation."""
    _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=12 * GB, fragmented=9 * GB)
    assert _pool_cache()._get_reclaimable_allocator_bytes() == 0

    _Pool(monkeypatch, backend="native", reserved=22 * GB, allocated=12 * GB, fragmented=4 * GB)
    assert _pool_cache()._get_reclaimable_allocator_bytes() == 6 * GB


def test_a_release_logs_what_the_driver_got_back_and_offloads_weights_when_the_pool_kept_the_rest(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=12 * GB, fragmented=9 * GB)
    cache = _pool_cache(logging.getLogger("pool_release"))
    models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

    with caplog.at_level(logging.DEBUG, logger="pool_release"):
        cache._release_allocator_blocks_for_reserve(5 * GB)  # 2 GB driver-free; 10 GB of the pool's gap look free

    # 3 GB driver-free after the trim, 2 GB short of the reserve: the unlocked weights go back to the driver.
    assert models["idle"].on_device == 0
    assert pool.driver_free == 7 * GB
    # Returned, believed releasable, driver-free after the trim, reserve -- then offloaded, driver-free after that.
    assert _megabytes(caplog, "Returned") == [[1024, 10240, 3072, 5120]]
    assert _megabytes(caplog, "short of the working-memory reserve") == [[3072, 5120, 4096, 7168]]


def test_the_offload_after_a_release_measures_the_driver_not_the_cache_cap(monkeypatch: pytest.MonkeyPatch):
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=12 * GB, fragmented=9 * GB)
    cache = _pool_cache(max_vram_cache_size_gb=20.0)  # the capped budget: 20 - 5 working - 12 in use, with room left
    models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

    cache._release_allocator_blocks_for_reserve(5 * GB)

    assert models["idle"].on_device == 0
    assert pool.driver_free == 7 * GB


@pytest.mark.parametrize(
    ("busy", "allocated", "fragmented", "models", "on_device_after", "driver_free_after", "trims"),
    [
        (None, 12 * GB, 9 * GB, {"idle": 4 * GB}, [0], 7 * GB, [False, False]),
        ("peer", 12 * GB, 9 * GB, {"idle": 4 * GB}, [0], 7 * GB, [False, False, True]),
        ("peer", 20 * GB, 0, {"a": GB, "b": GB, "c": GB}, [0, 0, 0], 7 * GB, [False, False, False, False, True]),
        ("peer", 12 * GB, 9 * GB, {}, [], 3 * GB, [False, True]),
        ("own", 12 * GB, 9 * GB, {"idle": 4 * GB}, [4 * GB], 3 * GB, [True]),
        ("own", 20 * GB, 0, {"a": GB, "b": GB, "c": GB}, [GB, GB, GB], 4 * GB, [True]),
    ],
    ids=[
        "alone",
        "fragmented-pool",
        "unfragmented-pool",
        "nothing-to-offload",
        "outside-a-session",
        "outside-a-session-unfragmented-pool",
    ],
)
def test_on_windows_the_reserve_release_of_an_async_pool_runs_at_most_one_trim_past_a_busy_device(
    monkeypatch: pytest.MonkeyPatch,
    busy: str | None,
    allocated: int,
    fragmented: int,
    models: dict[str, int],
    on_device_after: list[int],
    driver_free_after: int,
    trims: list[bool],
):
    """2 GB driver-free against a 5 GB reserve, the pool 22 GB reserved. Alone, the trim and then the offload run
    unforced. With a peer mid-session the unlocked weights go first, their trims deferred, and the one trim past the
    peer returns them with the pool's free chunks: the reserve is met even when 9 GB of the pool's 10 GB unused sit in
    chunks with live allocations, at the price of offloading all three 1 GB models where the pool's free 2 GB would
    have made one enough. With nothing to offload that trim goes to the pool. A thread outside any session gets the one
    forced trim on the pool and no offload."""
    monkeypatch.setattr(sys, "platform", "win32")
    pool = _Pool(
        monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=allocated, fragmented=fragmented, busy=busy
    )
    cache = _pool_cache()
    resident = _resident_models(monkeypatch, cache, pool, models)

    cache._release_allocator_blocks_for_reserve(5 * GB)

    assert [model.on_device for model in resident.values()] == on_device_after
    assert pool.driver_free == driver_free_after
    assert pool.trims == trims


@pytest.mark.parametrize("busy", ["peer", "own"], ids=["after-the-offload", "outside-a-session"])
def test_on_windows_a_reserve_trim_the_backend_refuses_raises(monkeypatch: pytest.MonkeyPatch, busy: str):
    """The node cannot run as if the release had returned anything: a short reserve hangs on Windows."""
    monkeypatch.setattr(sys, "platform", "win32")
    pool = _Pool(
        monkeypatch,
        backend="cudaMallocAsync",
        reserved=22 * GB,
        allocated=12 * GB,
        fragmented=9 * GB,
        busy=busy,
        refuse=True,
    )
    cache = _pool_cache()
    _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

    with pytest.raises(torch.OutOfMemoryError, match="while trimming the pool"):
        cache._release_allocator_blocks_for_reserve(5 * GB)

    assert pool.trims[-1] is True


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_a_reserve_release_the_backend_refuses_offloads_nothing_and_raises_on_windows(
    monkeypatch: pytest.MonkeyPatch, platform: str
):
    """A refusing context returns nothing to the driver, and offloading would only add refused trims. On Windows the
    short reserve would hang the node, so the release is forced and the refusal raises."""
    monkeypatch.setattr(sys, "platform", platform)
    pool = _Pool(
        monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=12 * GB, fragmented=9 * GB, refuse=True
    )
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

    if platform == "win32":
        with pytest.raises(torch.OutOfMemoryError, match="while trimming the pool"):
            cache._release_allocator_blocks_for_reserve(5 * GB)
    else:
        cache._release_allocator_blocks_for_reserve(5 * GB)

    assert models["idle"].on_device == 4 * GB
    assert pool.trims == ([False, True] if platform == "win32" else [False])


def test_on_windows_a_reserve_still_short_after_refused_offload_trims_ends_in_the_forced_trim(
    monkeypatch: pytest.MonkeyPatch,
):
    """A busy peer defers the release, so the weights go first; by their trim the peer is idle and the context refuses.
    That earns the offload only refused credit, which never makes a trim past the peer worth running, but the reserve
    is still short: the release ends in its one forced trim, which raises."""
    monkeypatch.setattr(sys, "platform", "win32")
    pool = _Pool(
        monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=12 * GB, fragmented=9 * GB, busy="peer"
    )
    cache = _pool_cache()
    _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})
    move_to_ram = cache._move_model_to_ram

    def peer_idles_and_the_context_breaks(*args, **kwargs):
        GENERATION_DEVICE_POOL._device_locks["cuda:1"].release()
        pool.refuse = True
        return move_to_ram(*args, **kwargs)

    monkeypatch.setattr(cache, "_move_model_to_ram", peer_idles_and_the_context_breaks)

    with pytest.raises(torch.OutOfMemoryError, match="while trimming the pool"):
        cache._release_allocator_blocks_for_reserve(5 * GB)

    assert pool.trims == [False, False, True]


def test_off_windows_a_release_deferred_by_a_busy_peer_changes_nothing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(
        monkeypatch,
        backend="cudaMallocAsync",
        reserved=22 * GB,
        allocated=12 * GB,
        fragmented=9 * GB,
        busy="peer",
    )
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

    cache._release_allocator_blocks_for_reserve(5 * GB)

    assert pool.trims == [False]
    assert models["idle"].on_device == 4 * GB


def test_a_native_allocator_that_is_still_short_after_the_trim_offloads_nothing(monkeypatch: pytest.MonkeyPatch):
    pool = _Pool(monkeypatch, backend="native", reserved=22 * GB, allocated=12 * GB, fragmented=8 * GB)
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

    cache._release_allocator_blocks_for_reserve(5 * GB)

    assert pool.driver_free == 4 * GB
    assert models["idle"].on_device == 4 * GB


def test_a_trim_deferred_by_a_busy_peer_credits_the_offloaded_weights_once(monkeypatch: pytest.MonkeyPatch):
    """The driver cannot see an offload before the trim runs, and an async pool credits nothing for it, so the loop's
    own credit for the deferred trim is the only one: 1 GB available, 7 GB needed, three 4 GB models."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=22 * GB, fragmented=0, busy="peer")
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"a": 4 * GB, "b": 4 * GB, "c": 4 * GB})
    assert cache._get_vram_available(None) == 1 * GB

    cache._offload_unlocked_models(7 * GB)

    assert [m.on_device for m in models.values()] == [0, 0, 4 * GB]


def test_a_trim_after_each_offload_shows_what_the_driver_got_back(monkeypatch: pytest.MonkeyPatch):
    """The first model's weights sat in chunks that live allocations share: its trim returned nothing, so the second
    has to go too."""
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=12 * GB, allocated=12 * GB, fragmented=0)
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"a": 4 * GB, "b": 5 * GB}, fragmenting={"a"})

    cache._offload_unlocked_models(14 * GB)  # 11 GB available

    assert [m.on_device for m in models.values()] == [0, 0]
    assert pool.trims == [False, False]
    assert pool.driver_free == 17 * GB


@pytest.mark.parametrize(
    ("reserved", "fragmented", "on_device_after", "trims"),
    [
        (22 * GB, 0, {"a": 2 * GB, "b": 4 * GB}, [False]),
        (22 * GB, 9 * GB, {"a": 0, "b": 0}, [False, False, False]),
        (16 * GB, 0, {"a": 0, "b": 4 * GB}, [False, False]),
    ],
    ids=["pool-covers-it", "pool-is-fragmented", "pool-covers-part-of-it"],
)
def test_the_pool_is_trimmed_once_before_a_model_is_offloaded_for_a_shortfall_it_may_cover(
    monkeypatch: pytest.MonkeyPatch, reserved: int, fragmented: int, on_device_after: dict[str, int], trims: list[bool]
):
    """Smallest first: `a` is offloaded only for what the trimmed pool did not cover, `b` only after that. Each
    offload is followed by a trim of its own."""
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=reserved, allocated=12 * GB, fragmented=fragmented)
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"a": 2 * GB, "b": 4 * GB})
    required = cache._get_vram_available(None) + 5 * GB

    cache._offload_unlocked_models(required)

    assert {key: model.on_device for key, model in models.items()} == on_device_after
    assert pool.trims == trims


@pytest.mark.parametrize(("unused", "trims"), [(255 * MB, []), (256 * MB, [False])], ids=["below-minimum", "minimum"])
def test_the_pool_is_not_trimmed_before_an_offload_for_fewer_unused_bytes_than_the_minimum(
    monkeypatch: pytest.MonkeyPatch, unused: int, trims: list[bool]
):
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=12 * GB + unused, allocated=12 * GB, fragmented=0)
    cache = _pool_cache()

    cache._offload_unlocked_models(cache._get_vram_available(None) + GB)

    assert pool.trims == trims


@pytest.mark.parametrize(
    ("busy", "reserved", "fragmented", "shortfall", "idle_on_device_after", "trims"),
    [
        ("peer", 22 * GB, 0, 5 * GB, 4 * GB, [False, True]),
        ("peer", 14 * GB, 0, 5 * GB, 0, [False, False, True]),
        ("peer", 22 * GB, 4 * GB, 8 * GB, 0, [False, True, False]),
        ("own", 22 * GB, 0, 5 * GB, 0, [False, False]),
        ("peer", 22 * GB, 0, 5 * GB, None, [False]),
    ],
    ids=[
        "pool-may-cover-it",
        "pool-cannot-cover-it",
        "pool-covers-part-of-it",
        "outside-a-session",
        "nothing-to-offload",
    ],
)
def test_the_one_trim_past_a_busy_peer_goes_to_the_pre_trim_when_the_pool_alone_may_cover_the_shortfall(
    monkeypatch: pytest.MonkeyPatch,
    busy: str,
    reserved: int,
    fragmented: int,
    shortfall: int,
    idle_on_device_after: int | None,
    trims: list[bool],
):
    """12 GB allocated and a 4 GB unlocked model. The pre-trim defers for the busy peer. When the pool's unused bytes
    may cover the shortfall, the trim past the peer goes to them and nothing is offloaded; when they cannot, it waits
    for the offloaded weights, which it returns along with them. Either way it is the only one: the offload after a
    pre-trim that returned too little (4 GB of the 10 GB sit in chunks with live allocations) stays deferred. A thread
    outside any session forces none, since the busy GPU may be its own, and without a model to offload no stall would
    be paid anyway, so none is forced for the pool alone."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(
        monkeypatch, backend="cudaMallocAsync", reserved=reserved, allocated=12 * GB, fragmented=fragmented, busy=busy
    )
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {} if idle_on_device_after is None else {"idle": 4 * GB})

    cache._offload_unlocked_models(cache._get_vram_available(None) + shortfall)

    if idle_on_device_after is not None:
        assert models["idle"].on_device == idle_on_device_after
    assert pool.trims == trims


def test_a_request_that_spent_its_trim_past_a_busy_peer_does_not_force_the_pre_trim(monkeypatch: pytest.MonkeyPatch):
    """The pool may cover the shortfall and a model could be offloaded, but the ledger handed in has no trim left."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=12 * GB, fragmented=0, busy="peer")
    cache = _pool_cache()
    _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

    cache._offload_unlocked_models(
        cache._get_vram_available(None) + 5 * GB, ledger=model_cache_module.OffloadLedger(peer_trim_left=False)
    )

    assert True not in pool.trims


def test_a_trim_past_a_busy_peer_that_returns_less_than_believed_is_followed_up_from_what_the_driver_reports(
    monkeypatch: pytest.MonkeyPatch,
):
    """1 GB available, 3 GB needed. The deferred trim's credit for `a` meets the shortfall, but `a`'s weights sit in
    chunks with live allocations and the trim past the peer returns none of them: the offload goes on from the driver's
    figure and takes `b` as well, instead of leaving the load to find the shortfall."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=22 * GB, fragmented=0, busy="peer")
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"a": 2 * GB, "b": 4 * GB}, fragmenting={"a"})

    cache._offload_unlocked_models(3 * GB)

    assert [m.on_device for m in models.values()] == [0, 0]
    assert pool.trims == [False, True, False]


def test_a_capped_budget_is_not_trimmed_for_and_does_not_count_deferred_trims(monkeypatch: pytest.MonkeyPatch):
    """The cap's measurement is the allocated bytes: no trim shows up in it, and an offload shows up at once, so a
    deferred trim must not credit the offloaded bytes a second time. 14 GB cap - 1 GB working - 12 GB allocated =
    1 GB; 7 GB needed: two 4 GB models."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=12 * GB, fragmented=0, busy="peer")
    cache = _pool_cache(max_vram_cache_size_gb=14.0)
    models = _resident_models(monkeypatch, cache, pool, {"a": 4 * GB, "b": 4 * GB, "c": 4 * GB})
    assert cache._get_vram_available(None) == 1 * GB

    cache._offload_unlocked_models(7 * GB)

    assert [m.on_device for m in models.values()] == [0, 0, 4 * GB]
    assert pool.trims == [False, False]  # one per offload; none before the first


def _paced_victim(monkeypatch: pytest.MonkeyPatch, cache: ModelCache, pool: _Pool, total: int):
    """A cached model of `total` bytes whose moves to the device are faked against `pool`; returns it and a reader of
    how much of it is on the device. As under cudaMallocAsync, its allocations reuse the pool's unused bytes (the
    fragmented ones first) before taking new chunks from the driver."""
    cache.put("victim", torch.nn.Linear(256, 256, bias=False), execution_device=torch.device("cuda"))
    victim = cache.get("victim")
    loaded = 0
    monkeypatch.setattr(victim.cached_model, "total_bytes", lambda: total)
    monkeypatch.setattr(victim.cached_model, "cur_vram_bytes", lambda: loaded)

    def move_to_vram(entry, vram_available, max_move_bytes=None):
        nonlocal loaded
        moved = min(total - loaded, max(0, vram_available), max_move_bytes or total)
        loaded += moved
        reused = min(moved, pool.reserved - pool.allocated)
        pool.fragmented -= min(reused, pool.fragmented)
        pool.allocated += moved
        pool.reserved += moved - reused
        return moved, loaded < total and moved == max_move_bytes

    monkeypatch.setattr(cache, "_move_model_to_vram", move_to_vram)
    return victim, lambda: loaded


def _lock_in_passes(cache: ModelCache, victim, working_mem_bytes: int) -> int:
    """Lock `victim` as a loaded model does (`_lock_paced`: a first pass through `lock`, then `continue_lock` in
    1 GB passes until the move settles). Returns the number of passes."""
    passes = 0
    lock, continue_lock = cache.lock, cache.continue_lock

    def counted(method):
        def run(*args, **kwargs):
            nonlocal passes
            passes += 1
            return method(*args, **kwargs)

        return run

    cache.lock, cache.continue_lock = counted(lock), counted(continue_lock)
    try:
        LoadedModelWithoutConfig(cache_record=victim, cache=cache)._lock_paced(working_mem_bytes)
    finally:
        del cache.lock, cache.continue_lock
    return passes


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_a_paced_lock_is_not_left_at_minimum_weights_by_trims_a_busy_peer_deferred(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, platform: str
):
    """2 GB driver-free, 1 GB working memory, a 3 GB model to stream in 1 GB passes, a 4 GB unlocked model to give up.
    With a peer mid-session the offload's own trim is deferred and an async pool is credited nothing, so every pass
    would find no room: the offload trims once past the peer instead, and the model loads in full."""
    monkeypatch.setattr(sys, "platform", platform)
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=22 * GB, fragmented=0, busy="peer")
    cache = _pool_cache(logging.getLogger("pool_busy_peer"))
    try:
        victim, loaded = _paced_victim(monkeypatch, cache, pool, 3 * GB)
        models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

        with caplog.at_level(logging.WARNING, logger="pool_busy_peer"):
            passes = _lock_in_passes(cache, victim, GB)
        cache.unlock(victim)

        assert models["idle"].on_device == 0
        assert (passes, loaded()) == (3, 3 * GB)
        assert pool.trims == [False, True]  # the offload's deferred trim, then one past the peer
        assert "short by" not in caplog.text
    finally:
        cache._cached_models.pop("idle", None)
        cache._execution_device = torch.device("cpu")
        cache.shutdown()


def test_a_paced_lock_runs_at_most_one_trim_past_a_busy_peer(monkeypatch: pytest.MonkeyPatch):
    """Each trim past the peer waits out its current kernel while holding the cache. Three unlocked 2 GB models whose
    weights sit in chunks with live allocations, so no trim returns them: every pass of the 3 GB stream finds itself
    short again, and each would have offloaded the next model and forced another trim."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=22 * GB, fragmented=0, busy="peer")
    cache = _pool_cache(logging.getLogger("pool_one_peer_trim"))
    try:
        victim, _ = _paced_victim(monkeypatch, cache, pool, 3 * GB)

        models = _resident_models(
            monkeypatch, cache, pool, {"a": 2 * GB, "b": 2 * GB, "c": 2 * GB}, fragmenting={"a", "b", "c"}
        )

        _lock_in_passes(cache, victim, GB)
        cache.unlock(victim)

        assert pool.trims.count(True) == 1
        # `a` for the shortfall, `b` for what the trim did not return; the deferred `b` covers the later passes.
        assert [model.on_device for model in models.values()] == [0, 0, 2 * GB]
    finally:
        for key in "abc":
            cache._cached_models.pop(key, None)
        cache._execution_device = torch.device("cpu")
        cache.shutdown()


def test_a_lock_from_outside_any_session_does_not_trim_past_the_session_on_its_gpu(monkeypatch: pytest.MonkeyPatch):
    """A single-GPU install with its session mid-step, and a lock from a thread that runs none (an API request, the
    image index worker): the GPU its trims defer for is the session's own, so none is forced. The model streams at what
    the driver reports, its allocations reusing the pool bytes of the weights whose trim stays deferred."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=22 * GB, fragmented=0, busy="own")
    cache = _pool_cache(logging.getLogger("pool_sessionless"))
    try:
        victim, _ = _paced_victim(monkeypatch, cache, pool, 3 * GB)
        models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

        _lock_in_passes(cache, victim, GB)
        cache.unlock(victim)

        assert models["idle"].on_device == 0
        assert True not in pool.trims
    finally:
        cache._cached_models.pop("idle", None)
        cache._execution_device = torch.device("cpu")
        cache.shutdown()


def test_a_trim_the_backend_refuses_is_credited_but_forces_nothing(monkeypatch: pytest.MonkeyPatch):
    """A broken or exhausted device context refuses the offload's trim. Torch still reuses the offloaded bytes, so the
    request credits them, and 1 GB available for 3 GB needed costs one 4 GB model, not both (for that call only: no
    release is pending for them that a later pass could count on). No trim is forced for them: it would only
    fail again."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=22 * GB, fragmented=0, refuse=True)
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"a": 4 * GB, "b": 4 * GB})

    cache.make_room_in_vram(3 * GB)

    assert [model.on_device for model in models.values()] == [0, 4 * GB]
    assert pool.trims == [False]


@pytest.mark.parametrize("entry_point", ["lock", "make_room_in_vram"])
def test_a_trim_past_a_busy_peer_that_the_backend_refuses_does_not_fail_the_offload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, entry_point: str
):
    """The busy peer defers the offload's trim, the trim past it then fails in the backend: the offload asked for the
    release, so it completes, and the load measures what the driver reports."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(
        monkeypatch,
        backend="cudaMallocAsync",
        reserved=22 * GB,
        allocated=22 * GB,
        fragmented=0,
        busy="peer",
        refuse=True,
    )
    cache = _pool_cache(logging.getLogger("pool_refused"))
    try:
        victim, _ = _paced_victim(monkeypatch, cache, pool, 3 * GB)
        models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

        with caplog.at_level(logging.WARNING, logger="pool_refused"):
            if entry_point == "lock":
                _lock_in_passes(cache, victim, GB)
                cache.unlock(victim)
            else:
                cache.make_room_in_vram(3 * GB)

        assert models["idle"].on_device == 0
        assert pool.trims.count(True) == 1
        assert "Could not return the offloaded weights to the driver: CUDA out of memory" in caplog.text
    finally:
        cache._cached_models.pop("idle", None)
        cache._execution_device = torch.device("cpu")
        cache.shutdown()


@pytest.mark.parametrize(
    ("busy", "offloaded"),
    [("own", 8 * GB), ("peer", 16 * GB)],
    ids=["outside-a-session", "trim-spent"],
)
def test_a_paced_lock_offloads_for_its_shortfall_once_while_the_release_stays_deferred(
    monkeypatch: pytest.MonkeyPatch, busy: str, offloaded: int
):
    """3 GB available after 1 GB of working memory, a 10 GB model streamed in 1 GB passes, ten unlocked 2 GB models,
    the first four in chunks with live allocations. Their trims defer and the driver never sees them, but the stream's
    allocations reuse them: later passes must count them instead of offloading the shortfall again. Outside a session
    nothing is forced, so the 7 GB shortfall costs four models. With a peer mid-session the one trim past it returns
    nothing of those four, and four more go for the same 7 GB."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=20 * GB, allocated=20 * GB, fragmented=0, busy=busy)
    cache = _pool_cache(logging.getLogger("pool_credit_across_passes"))
    keys = [f"m{index}" for index in range(10)]
    try:
        victim, _ = _paced_victim(monkeypatch, cache, pool, 10 * GB)
        models = _resident_models(monkeypatch, cache, pool, dict.fromkeys(keys, 2 * GB), fragmenting=set(keys[:4]))

        _lock_in_passes(cache, victim, GB)
        cache.unlock(victim)

        assert sum(2 * GB - model.on_device for model in models.values()) == offloaded
        assert pool.trims.count(True) == (1 if busy == "peer" else 0)
    finally:
        for key in keys:
            cache._cached_models.pop(key, None)
        cache._execution_device = torch.device("cpu")
        cache.shutdown()


@pytest.mark.parametrize("returned_by", ["a-flush", "the-load"])
def test_deferred_weights_count_only_while_the_pool_still_holds_them_for_the_request(
    monkeypatch: pytest.MonkeyPatch, returned_by: str
):
    """A lock offloads `a` while the session on the GPU defers its trim. Then either a step-boundary flush returns it,
    and the session frees 4 GB of activations into the pool, or the lock's own load takes those 4 GB from the pool.
    Either way the pool no longer holds `a`'s weights for the lock: counting them again would leave the next pass
    4 GB short with `b` still on the device."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=22 * GB, fragmented=0, busy="own")
    cache = _pool_cache()
    models = _resident_models(monkeypatch, cache, pool, {"a": 4 * GB, "b": 4 * GB})
    ledger = model_cache_module.OffloadLedger()

    cache._offload_unlocked_models(4 * GB, ledger=ledger)  # 1 GB available: `a` goes, its trim defers
    assert [model.on_device for model in models.values()] == [0, 4 * GB]

    if returned_by == "a-flush":
        GENERATION_DEVICE_POOL._device_locks["cuda:0"].release()  # the session reaches a step boundary
        TorchDevice.flush_deferred_empty_cache()
        pool.reserved += 4 * GB  # the session's activations, freed into the pool
    else:
        pool.allocated += 4 * GB  # the load's weights, allocated from the pool's unused bytes

    cache._offload_unlocked_models(4 * GB, pretrim=False, ledger=ledger)

    assert [model.on_device for model in models.values()] == [0, 0]


def test_an_off_queue_borrow_does_not_trim_past_the_session_on_another_gpu(monkeypatch: pytest.MonkeyPatch):
    """Image embedding or Expand Prompt borrowed cuda:0 while a session runs on cuda:1: background work, which must
    not stall that session, so the deferred trim stays deferred."""
    monkeypatch.setattr(sys, "platform", "linux")
    pool = _Pool(
        monkeypatch,
        backend="cudaMallocAsync",
        reserved=22 * GB,
        allocated=22 * GB,
        fragmented=0,
        busy="peer",
        off_queue=True,
    )
    cache = _pool_cache(logging.getLogger("pool_off_queue"))
    try:
        victim, _ = _paced_victim(monkeypatch, cache, pool, 3 * GB)
        models = _resident_models(monkeypatch, cache, pool, {"idle": 4 * GB})

        _lock_in_passes(cache, victim, GB)
        cache.unlock(victim)

        assert models["idle"].on_device == 0
        assert pool.trims and True not in pool.trims
    finally:
        cache._cached_models.pop("idle", None)
        cache._execution_device = torch.device("cpu")
        cache.shutdown()


def test_a_paced_lock_trims_the_pool_on_its_first_pass_only(monkeypatch: pytest.MonkeyPatch):
    """A 16 GB model streams in 1 GB passes against a pool whose unused 7 GB cannot be trimmed: the first pass finds
    that out, the others must not pay for a synchronizing trim each."""
    monkeypatch.setattr(sys, "platform", "win32")
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=10 * GB, allocated=2 * GB, fragmented=7 * GB)
    cache = _pool_cache(logging.getLogger("pool_paced"))
    try:
        victim, _ = _paced_victim(monkeypatch, cache, pool, 16 * GB)

        assert _lock_in_passes(cache, victim, GB) > 4
        cache.unlock(victim)

        # The first pass's trim only: the stream's allocations reuse the pool, so the settled pass finds the reserve
        # free and releases nothing.
        assert pool.trims == [False]
    finally:
        cache._execution_device = torch.device("cpu")
        cache.shutdown()


def test_a_lock_that_stays_short_after_offloading_still_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    pool = _Pool(monkeypatch, backend="cudaMallocAsync", reserved=22 * GB, allocated=12 * GB, fragmented=9 * GB)
    cache = _pool_cache(logging.getLogger("pool_short"))
    try:
        cache.put("victim", torch.nn.Linear(256, 256, bias=False), execution_device=torch.device("cuda"))
        victim = cache.get("victim")
        _resident_models(monkeypatch, cache, pool, {"other": 4 * GB})
        monkeypatch.setattr(cache, "_move_model_to_vram", lambda entry, available, max_move_bytes=None: (0, False))

        with caplog.at_level(logging.WARNING, logger="pool_short"):
            cache.lock(victim, 20 * GB)
        cache.unlock(victim)

        # 7 GB driver-free once the other model is offloaded and the pool trimmed, against 20 GB working memory.
        assert pool.driver_free == 7 * GB
        assert "VRAM budget for 'victim' is short by 13312MB even after offloading" in caplog.text
    finally:
        cache._cached_models.pop("other", None)  # a stand-in entry without the lifecycle fields shutdown() reads
        cache._execution_device = torch.device("cpu")
        cache.shutdown()


@pytest.mark.slow
@requires_cuda
def test_lock_hands_cached_blocks_back_when_the_reserve_is_not_driver_free():
    """On a real card: torch holds most of the free memory in cached blocks, then a lock asks for a reserve that only
    those blocks can cover; afterwards torch holds nothing worth releasing. Needs a quiet GPU (it fills the card to
    within 1 GiB), hence `slow`.
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    # Slack in segments that other live tensors keep, which no release can return.
    baseline_unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    free, _total = torch.cuda.mem_get_info(device)
    if free < 4 * GB:
        pytest.skip("needs at least 4 GiB of free VRAM")
    cache = _make_cache()
    cache.put("A", BigModule(64))
    record = cache.get("A")
    chunk = 256 * MB
    blocks = [torch.empty(chunk, dtype=torch.uint8, device=device) for _ in range(int((free - GB) // chunk))]
    del blocks
    driver_free = torch.cuda.mem_get_info(device)[0]
    held_unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)

    cache.lock(record, driver_free + held_unused // 2)
    try:
        # Process-local, so another program on the card cannot decide the outcome. Read from the allocator itself: the
        # cache's reclaimable figure is 0 under expandable segments whether or not anything was released.
        unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        assert unused - baseline_unused < 256 * MB
    finally:
        cache.unlock(record)
        torch.cuda.empty_cache()


def _calibrating_cache(device: str, cap_gb: float | None) -> ModelCache:
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        max_vram_cache_size_gb=cap_gb,
        reserve_vram_gb=0.5,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
        shared_cpu_weights=None,
    )
    cache._execution_device = torch.device(device)  # policy only; every memory query below is patched out
    return cache


@pytest.mark.parametrize("cap_gb", [None, 8.0], ids=["uncapped", "capped"])
def test_a_calibration_on_a_shared_memory_device_plans_with_what_its_lock_budgets(
    monkeypatch: pytest.MonkeyPatch, cap_gb: float | None
):
    """On MPS the calibration's pre-lock offload moves nothing, so what is live stays live: the capacity it plans with
    is the lock's budget with the working memory given back. 10 GB available and 2 GB the driver holds, 3 GB live:
    9 GB, or 5 GB under an 8 GB cap; less the 0.5 GB reserve."""
    monkeypatch.setattr(model_cache_module.psutil, "virtual_memory", lambda: MagicMock(available=10 * GB))
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 2 * GB)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 3 * GB)
    cache = _calibrating_cache("mps", cap_gb)
    working = 4 * GB

    assert cache.calibration_capacity() == (8.5 if cap_gb is None else 4.5) * GB
    assert cache.calibration_capacity() == cache._get_vram_available(working) + working


def test_a_calibration_on_a_dedicated_card_counts_the_models_its_offload_moves_out(monkeypatch: pytest.MonkeyPatch):
    """6 GB free and 2 GB the resident models hold: the offload moves them out before the lock, so 8 GB, less the
    reserve."""
    monkeypatch.setattr(TorchDevice, "cuda_mem_get_info", classmethod(lambda cls, device: (6 * GB, 24 * GB)))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 2 * GB)

    assert _calibrating_cache("cuda", None).calibration_capacity() == 7.5 * GB

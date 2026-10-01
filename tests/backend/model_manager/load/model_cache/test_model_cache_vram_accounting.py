"""Tests for the VRAM budget arithmetic behind lock()/partial loads.

`_get_vram_available` must count every byte this process can actually obtain: driver-free memory
PLUS the torch caching allocator's reserved-but-unused blocks (the allocator reuses those
directly, and `empty_cache()` returns whole unoccupied segments to the driver). Budgeting on
driver-free alone under-reported by whatever earlier stages freed without an `empty_cache()`
— observed in the wild as a fully-evictable multi-GB reserve pushing a 20 GB transformer down
to 0% VRAM residency while the allocator happily reused the "missing" memory for activations.
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.util.devices import TorchDevice

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


@pytest.mark.parametrize("peer_busy", [False, True], ids=["alone", "peer-device-busy"])
def test_offloading_under_expandable_segments_stops_once_enough_is_free(monkeypatch: pytest.MonkeyPatch, peer_busy):
    """Under expandable segments the driver sees an offloaded model's pages only after empty_cache(), and no
    allocator credit stands in for them. Re-measuring without it saw no progress and unloaded every unlocked model.

    `empty_cache` is peer-aware: while another generation device is mid-session it defers instead of releasing, so
    the measurement cannot see the offload at all and the freed bytes are credited to it directly."""
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
    device = {"free": 0, "allocated": 12 * GB, "unmappable": 0}

    def move_to_ram(entry, bytes_to_free):
        device["allocated"] -= 4 * GB
        device["unmappable"] += 4 * GB
        return 4 * GB

    def empty_cache() -> bool:
        if peer_busy:  # deferred: nothing is released, and the driver keeps reporting the old figure
            return False
        device["free"] += device["unmappable"]
        device["unmappable"] = 0
        return True

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (device["free"], 16 * GB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: device["allocated"])
    monkeypatch.setattr(TorchDevice, "empty_cache", empty_cache)
    monkeypatch.setattr(cache, "_move_model_to_ram", MagicMock(side_effect=move_to_ram))
    for key in "ABC":
        entry = MagicMock(key=key, is_locked=False)
        entry.cached_model.total_bytes.return_value = 4 * GB
        cache._cached_models[key] = entry

    # 2 GB needed + 1 GB working memory: one 4 GB model is enough.
    assert cache._offload_unlocked_models(2 * GB) == 4 * GB
    assert cache._move_model_to_ram.call_count == 1

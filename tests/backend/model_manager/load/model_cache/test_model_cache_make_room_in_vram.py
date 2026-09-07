"""Tests for `ModelCache.make_room_in_vram` - the entry point for loads that put a model on the GPU *outside*
the cache (e.g. a BitsAndBytes-quantized text encoder that cannot be moved between devices).

Such a load never passes through `lock()`, which is where the cache normally makes room for the model being
locked, so without an explicit request it only gets whatever VRAM the resident models happened to leave free.
Issue #9147: the quantized Qwen2.5-VL encoder was planned onto the CPU by `device_map="auto"` because the cached
transformer and VAE still filled the card, and BitsAndBytes int8 refused to run that way.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule

MB = 2**20


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


@pytest.fixture
def cache(mock_logger):
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=mock_logger,
    )
    yield cache
    cache.shutdown()


class _FakeVram:
    """Simulates the execution device's free VRAM as the cache offloads models.

    `ModelCache._get_vram_available` needs a real accelerator, so the CPU-only tests below replace it with this
    bookkeeping: `available` grows by whatever `_move_model_to_ram` reports freed.
    """

    def __init__(self, available: int):
        self.available = available
        self.working_mem_seen: list[int | None] = []
        self.moved: list[tuple[str, int]] = []

    def get_vram_available(self, working_mem_bytes):
        self.working_mem_seen.append(working_mem_bytes)
        return self.available

    def move_model_to_ram(self, cache_entry, vram_bytes_to_free, keep_required_weights_in_vram=None):
        freed = cache_entry.cached_model.total_bytes()
        self.moved.append((cache_entry.key, vram_bytes_to_free))
        self.available += freed
        return freed


def _put(cache: ModelCache, key: str, size_bytes: int) -> None:
    """Cache a module whose weights occupy exactly `size_bytes` (a bare tensor is sized at 0 by the cache)."""
    assert size_bytes % 4 == 0
    module = torch.nn.Linear(size_bytes // 4, 1, bias=False)  # fp32: 4 bytes per weight
    assert sum(p.numel() * p.element_size() for p in module.parameters()) == size_bytes
    cache.put(key, module)


@pytest.fixture
def gpu_accounting_cache(mock_logger):
    """A cache whose execution device is a GPU as far as the policy is concerned, without touching a real one.

    `_get_vram_available` needs an accelerator; the tests replace it (and the VRAM moves) with `_FakeVram`. Note that
    `_FakeVram` reports freed memory immediately, which is the loop's *contract*; on real hardware the driver only
    sees it after the trailing `empty_cache()`, so the loop there tends to offload every unlocked model.
    """
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=mock_logger,
    )
    cache._execution_device = torch.device("cuda")  # policy only; every VRAM touch is patched out below
    yield cache
    cache._execution_device = torch.device("cpu")
    cache.shutdown()


def test_cpu_execution_device_is_a_no_op(cache: ModelCache):
    """A CPU-only install has no VRAM to make room in. `_get_vram_available` raises for a cpu device, and the
    quantized encoder path calls this on every run once anything is cached, so it must short-circuit."""
    _put(cache, "resident", 40 * MB)

    assert cache.make_room_in_vram(30 * MB) == 0
    assert "resident" in cache._cached_models


def test_offload_runs_under_the_cache_lock(gpu_accounting_cache: ModelCache):
    """Out-of-cache callers race the session workers' own lock()/unlock(); the offload must own the cache lock."""
    cache = gpu_accounting_cache
    _put(cache, "resident", 40 * MB)
    owned: list[bool] = []

    def offload(vram_bytes_required, working_mem_bytes=None):
        owned.append(cache._lock._is_owned())
        return 0

    with patch.object(cache, "_offload_unlocked_models", side_effect=offload):
        cache.make_room_in_vram(30 * MB)

    assert owned == [True]


def test_offloads_unlocked_models_until_the_request_is_satisfied(gpu_accounting_cache: ModelCache):
    """Models are offloaded smallest-first, and the loop stops once the availability check reports enough free
    VRAM - the larger model that is not needed stays resident."""
    cache = gpu_accounting_cache
    _put(cache, "small", 10 * MB)
    _put(cache, "medium", 20 * MB)
    _put(cache, "large", 40 * MB)
    vram = _FakeVram(available=5 * MB)

    with (
        patch.object(cache, "_get_vram_available", side_effect=vram.get_vram_available),
        patch.object(cache, "_move_model_to_ram", side_effect=vram.move_model_to_ram),
    ):
        freed = cache.make_room_in_vram(30 * MB)

    assert [key for key, _ in vram.moved] == ["small", "medium"]
    assert freed == 30 * MB
    assert vram.available == 35 * MB


def test_locked_models_are_never_offloaded(gpu_accounting_cache: ModelCache):
    cache = gpu_accounting_cache
    """A locked model is in use by another invocation; it must be skipped even when the request cannot otherwise
    be satisfied."""
    _put(cache, "in_use", 40 * MB)
    _put(cache, "idle", 10 * MB)
    cache._cached_models["in_use"].lock()
    vram = _FakeVram(available=0)

    with (
        patch.object(cache, "_get_vram_available", side_effect=vram.get_vram_available),
        patch.object(cache, "_move_model_to_ram", side_effect=vram.move_model_to_ram),
    ):
        freed = cache.make_room_in_vram(100 * MB)

    assert [key for key, _ in vram.moved] == ["idle"]
    assert freed == 10 * MB


def test_no_op_when_enough_vram_is_already_free(gpu_accounting_cache: ModelCache):
    cache = gpu_accounting_cache
    _put(cache, "resident", 40 * MB)
    vram = _FakeVram(available=50 * MB)

    with (
        patch.object(cache, "_get_vram_available", side_effect=vram.get_vram_available),
        patch.object(cache, "_move_model_to_ram", side_effect=vram.move_model_to_ram),
    ):
        freed = cache.make_room_in_vram(30 * MB)

    assert vram.moved == []
    assert freed == 0


def test_working_memory_is_forwarded_to_the_availability_check(gpu_accounting_cache: ModelCache):
    cache = gpu_accounting_cache
    """The caller's working memory must reach `_get_vram_available`, where it is floored at the configured default
    exactly as in `lock()`; otherwise the encoder's activations would have to fit in whatever is left over."""
    _put(cache, "resident", 40 * MB)
    vram = _FakeVram(available=0)

    with (
        patch.object(cache, "_get_vram_available", side_effect=vram.get_vram_available),
        patch.object(cache, "_move_model_to_ram", side_effect=vram.move_model_to_ram),
    ):
        cache.make_room_in_vram(10 * MB, working_mem_bytes=7 * MB)

    assert vram.working_mem_seen and all(w == 7 * MB for w in vram.working_mem_seen)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize("partial", [False, True])
def test_gpu_make_room_in_vram_actually_moves_weights_off_the_device(partial: bool):
    """Real-accelerator check: after `make_room_in_vram`, the unlocked models' weights are on the CPU and the
    locked model's weights are still on the GPU, with both cached-model flavours."""
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    cache = ModelCache(
        execution_device_working_mem_gb=0.0,
        enable_partial_loading=partial,
        keep_ram_copy_of_weights=True,
        execution_device="cuda",
        storage_device="cpu",
        logger=logger,
    )
    try:
        idle, in_use = DummyModule(), DummyModule()
        cache.put("idle", idle)
        cache.put("in_use", in_use)
        for key in ("idle", "in_use"):
            cache._cached_models[key].cached_model.full_load_to_vram()
        cache._cached_models["in_use"].lock()
        assert all(p.device.type == "cuda" for p in idle.parameters())

        # Ask for more than the whole card so every unlocked model has to go.
        _, total = torch.cuda.mem_get_info()
        freed = cache.make_room_in_vram(2 * total)

        assert freed == cache._cached_models["idle"].cached_model.total_bytes()
        assert all(p.device.type == "cpu" for p in idle.parameters())
        assert all(p.device.type == "cuda" for p in in_use.parameters())
        # Same policy as lock(): the entry is offloaded, not evicted, so the next use re-streams weights.
        assert "idle" in cache._cached_models
    finally:
        cache.shutdown()

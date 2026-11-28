"""Tests for make_room()'s eviction guards."""

import logging
import threading
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache import model_cache as model_cache_module
from invokeai.backend.model_manager.load.model_cache.model_cache import GB, ModelCache
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast import weight_streaming
from invokeai.backend.util.devices import EmptyCacheOutcome, TorchDevice
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule

CUDA0 = torch.device("cuda", 0)


def _make_cache() -> ModelCache:
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger,
    )


def test_held_record_survives_sibling_admissions_until_it_locks():
    """Multi-model invocations load their whole set before locking any of it: an earlier
    sibling's wrapper must survive later puts (whose sweep clears the unowned admission grace)
    and the make_rooms they trigger, for as long as the wrapper is alive and un-entered."""
    cache = _make_cache()
    cache.put("first", DummyModule())
    record, claim = cache.get_with_first_use_claim("first")
    handle = LoadedModelWithoutConfig(cache_record=record, cache=cache, first_use_claim=claim)

    # Two more siblings admitted — each put() runs the stale-grace sweep and a make_room.
    cache.put("second", DummyModule())
    cache.put("third", DummyModule())
    assert record.first_use_holds > 0, "a sibling's put() stripped a live wrapper's first-use hold"

    cache.make_room(10**15)
    assert "first" in cache._cached_models

    # The handle finally locks: hold released, ordinary cache content again.
    with handle:
        pass
    assert record.first_use_holds == 0
    cache.make_room(10**15)
    assert "first" not in cache._cached_models


def test_make_room_reports_what_it_cleared():
    """The clear-model-cache route sums each device cache's result, so make_room must report
    the entries it dropped and the bytes they held."""
    cache = _make_cache()
    cache.put("fresh", DummyModule())
    record = cache.get("fresh")
    size = record.cached_model.total_bytes()

    result = cache.make_room(10**15)
    assert "fresh" not in cache._cached_models
    assert result.models_cleared == 1
    assert result.bytes_freed == size


def test_make_room_does_not_infer_host_headroom_from_idle_eviction(monkeypatch: pytest.MonkeyPatch):
    cache = _make_cache()
    try:
        cache._ram_cache_size_bytes = 64 * GB
        cache.put("idle", DummyModule())
        entry = cache._cached_models["idle"]
        entry.cached_model.total_bytes = lambda: 20 * GB  # type: ignore[method-assign]
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 12 * GB)

        result = cache.make_room(25 * GB)

        assert not result.ram_headroom_satisfied
        assert result.models_cleared == 1
        assert "idle" not in cache._cached_models
    finally:
        cache.shutdown()


def test_make_room_allows_target_larger_than_cache_cap_when_host_fits(monkeypatch: pytest.MonkeyPatch):
    cache = _make_cache()
    try:
        cache._ram_cache_size_bytes = 4 * GB
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 32 * GB)

        result = cache.make_room(6 * GB)

        assert result.ram_headroom_satisfied
    finally:
        cache.shutdown()


def test_make_room_separates_cache_capacity_from_additional_host_bytes(monkeypatch: pytest.MonkeyPatch):
    cache = _make_cache()
    try:
        cache._ram_cache_size_bytes = 4 * GB
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 2 * GB)

        result = cache.make_room(6 * GB, host_bytes_needed=1 * GB)

        assert result.ram_headroom_satisfied

        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 512 * 1024**2)
        result = cache.make_room(6 * GB, host_bytes_needed=1 * GB)

        assert not result.ram_headroom_satisfied
    finally:
        cache.shutdown()


def test_make_room_accounts_for_pending_weight_pin_reservation(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    cache = _make_cache()
    storage = torch.empty(1 * 2**20, dtype=torch.uint8).untyped_storage()
    entered, proceed = threading.Event(), threading.Event()
    monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 2 * GB)
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: 2 * GB)

    def fault_in(_storage: torch.UntypedStorage) -> None:
        entered.set()
        assert proceed.wait(timeout=30)

    monkeypatch.setattr(weight_streaming, "_fault_in", fault_in)
    holder = weight_streaming.PinHolder(CUDA0)
    thread = threading.Thread(target=registry.update, args=(holder, [storage]))
    thread.start()
    assert entered.wait(timeout=30)

    try:
        result = cache.make_room(1, host_bytes_needed=2 * GB)
        assert not result.ram_headroom_satisfied
    finally:
        proceed.set()
        thread.join(timeout=30)
        registry.release(holder)
        cache.shutdown()


def test_make_room_samples_host_headroom_atomically_with_pin_reservation(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    registry, _ = fake_streaming
    cache = _make_cache()
    storage = torch.empty(1 * 2**20, dtype=torch.uint8).untyped_storage()
    fault_entered, fault_proceed, update_done = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: 2 * GB)

    def fault_in(_storage: torch.UntypedStorage) -> None:
        fault_entered.set()
        assert fault_proceed.wait(timeout=30)

    monkeypatch.setattr(weight_streaming, "_fault_in", fault_in)
    holder = weight_streaming.PinHolder(CUDA0)

    def update() -> None:
        try:
            registry.update(holder, [storage])
        finally:
            update_done.set()

    thread = threading.Thread(target=update)
    thread.start()
    assert fault_entered.wait(timeout=30)

    def host_available() -> int:
        acquired = registry._registration_lock.acquire(blocking=False)
        if acquired:
            registry._registration_lock.release()
            fault_proceed.set()
            assert update_done.wait(timeout=30)
        return 2 * GB

    monkeypatch.setattr(model_cache_module, "_host_memory_available", host_available)
    try:
        result = cache.make_room(1, host_bytes_needed=2 * GB)
        assert not result.ram_headroom_satisfied
    finally:
        fault_proceed.set()
        thread.join(timeout=30)
        assert not thread.is_alive()
        registry.release(holder)
        cache.shutdown()


def test_make_room_retries_host_admission_after_releasing_allocator_blocks(monkeypatch: pytest.MonkeyPatch):
    cache = _make_cache()
    try:
        cache._execution_device = torch.device("cuda")
        monkeypatch.setattr(model_cache_module, "device_is_integrated", lambda _device: True)
        host_available = [0]
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: host_available[0])
        monkeypatch.setattr(cache, "_get_reclaimable_allocator_bytes", lambda: 2 * GB)
        monkeypatch.setattr(TorchDevice, "cuda_mem_get_info", classmethod(lambda _cls, _device: (8 * GB, 16 * GB)))

        def empty_cache() -> EmptyCacheOutcome:
            host_available[0] = 2 * GB
            return EmptyCacheOutcome.RAN

        monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(lambda _cls, **_kwargs: empty_cache()))

        result = cache.make_room(1, host_bytes_needed=1 * GB)

        assert result.ram_headroom_satisfied
    finally:
        cache.shutdown()


def test_make_room_still_refuses_when_releasing_allocator_blocks_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
):
    cache = _make_cache()
    try:
        cache._execution_device = torch.device("cuda")
        monkeypatch.setattr(model_cache_module, "device_is_integrated", lambda _device: True)
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 0)
        monkeypatch.setattr(cache, "_get_reclaimable_allocator_bytes", lambda: 2 * GB)
        monkeypatch.setattr(TorchDevice, "cuda_mem_get_info", classmethod(lambda _cls, _device: (8 * GB, 16 * GB)))
        empty_cache = MagicMock(return_value=EmptyCacheOutcome.RAN)
        monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(lambda _cls, **_kwargs: empty_cache()))

        result = cache.make_room(1, host_bytes_needed=1 * GB)

        assert not result.ram_headroom_satisfied
        empty_cache.assert_called_once()
    finally:
        cache.shutdown()


def test_make_room_does_not_flush_allocator_for_cache_only_shortage(monkeypatch: pytest.MonkeyPatch):
    cache = _make_cache()
    try:
        cache._execution_device = torch.device("cuda")
        monkeypatch.setattr(cache, "_get_ram_available", lambda: 0)
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 2 * GB)
        monkeypatch.setattr(cache, "_get_reclaimable_allocator_bytes", lambda: 2 * GB)
        monkeypatch.setattr(TorchDevice, "cuda_mem_get_info", classmethod(lambda _cls, _device: (8 * GB, 16 * GB)))
        empty_cache = MagicMock(return_value=EmptyCacheOutcome.RAN)
        monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(lambda _cls, **_kwargs: empty_cache()))

        result = cache.make_room(2 * GB, host_bytes_needed=1 * GB)

        assert result.ram_headroom_satisfied
        empty_cache.assert_not_called()
    finally:
        cache.shutdown()


def test_make_room_refuses_when_locked_models_leave_host_headroom_short(
    monkeypatch: pytest.MonkeyPatch,
):
    cache = _make_cache()
    try:
        cache.put("live", DummyModule())
        record = cache.get("live")
        cache.lock(record, None)
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 12 * GB)

        result = cache.make_room(25 * GB)

        assert not result.ram_headroom_satisfied
        assert "live" in cache._cached_models
    finally:
        cache.unlock(record)
        cache.shutdown()


def test_make_room_reports_host_headroom_when_locked_cache_capacity_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    cache = _make_cache()
    record = None
    try:
        cache._ram_cache_size_bytes = 1 * GB
        cache.put("live", DummyModule())
        record = cache.get("live")
        cache.lock(record, None)
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 32 * GB)

        result = cache.make_room(25 * GB)

        assert result.ram_headroom_satisfied
        assert "live" in cache._cached_models
    finally:
        if record is not None:
            cache.unlock(record)
        cache.shutdown()


def test_make_room_does_not_credit_dedicated_gpu_residency_as_host_ram(
    monkeypatch: pytest.MonkeyPatch,
):
    cache = _make_cache()
    try:
        cache._ram_cache_size_bytes = 64 * GB
        cache.put("gpu", DummyModule())
        entry = cache._cached_models["gpu"]
        entry.cached_model.total_bytes = lambda: 20 * GB  # type: ignore[method-assign]
        entry.cached_model.cur_vram_bytes = lambda: 20 * GB  # type: ignore[method-assign]
        entry.cached_model.get_cpu_state_dict = lambda: None  # type: ignore[method-assign]
        entry.cached_model._compute_device = torch.device("cuda")
        monkeypatch.setattr(model_cache_module, "device_has_dedicated_vram", lambda _device: True)
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 12 * GB)

        result = cache.make_room(25 * GB)

        assert not result.ram_headroom_satisfied
        assert "gpu" not in cache._cached_models
    finally:
        cache.shutdown()


def test_windows_integrated_cuda_eviction_does_not_credit_gpu_only_bytes_as_host_ram(
    monkeypatch: pytest.MonkeyPatch,
):
    cache = _make_cache()
    try:
        cache._ram_cache_size_bytes = 64 * GB
        cache.put("gpu", DummyModule())
        entry = cache._cached_models["gpu"]
        entry.cached_model.total_bytes = lambda: 20 * GB  # type: ignore[method-assign]
        entry.cached_model.cur_vram_bytes = lambda: 20 * GB  # type: ignore[method-assign]
        entry.cached_model.get_cpu_state_dict = lambda: None  # type: ignore[method-assign]
        entry.cached_model._compute_device = torch.device("cuda")
        monkeypatch.setattr(model_cache_module.sys, "platform", "win32")
        monkeypatch.setattr(model_cache_module, "_is_integrated_cuda", lambda _device: True)
        monkeypatch.setattr(model_cache_module, "device_has_dedicated_vram", lambda _device: False)
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 12 * GB)

        result = cache.make_room(25 * GB)

        assert not result.ram_headroom_satisfied
        assert "gpu" not in cache._cached_models
    finally:
        cache.shutdown()


def test_make_room_asks_idle_peer_cache_to_release_host_ram(monkeypatch: pytest.MonkeyPatch):
    budget = RamBudget(max_bytes=100 * GB, shared_store=None)
    owner = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
        shared_cpu_weights=None,
        ram_budget=budget,
    )
    peer = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
        shared_cpu_weights=None,
        ram_budget=budget,
    )
    for cache in (owner, peer):
        cache._logger.logger.getEffectiveLevel.return_value = logging.INFO
    try:
        peer.put("idle", DummyModule())
        record = peer.get("idle")
        peer.lock(record, None)
        peer.unlock(record)
        entry = peer._cached_models["idle"]
        entry.cached_model.total_bytes = lambda: 20 * GB  # type: ignore[method-assign]
        ref = budget._find_cache_ref(peer)
        assert ref is not None
        budget._non_shared_bytes = 20 * GB
        budget._non_shared_bytes_by_cache[ref] = 20 * GB
        monkeypatch.setattr(
            model_cache_module,
            "_host_memory_available",
            lambda: 12 * GB if budget.total_in_use() else 32 * GB,
        )

        result = owner.make_room(25 * GB)

        assert result.ram_headroom_satisfied
        assert "idle" not in peer._cached_models
    finally:
        owner.shutdown()
        peer.shutdown()


def test_make_room_does_not_credit_integrated_cuda_eviction_when_host_sample_stays_low(
    monkeypatch: pytest.MonkeyPatch,
):
    cache = _make_cache()
    try:
        cache._ram_cache_size_bytes = 64 * GB
        cache.put("gpu", DummyModule())
        entry = cache._cached_models["gpu"]
        entry.cached_model.total_bytes = lambda: 20 * GB  # type: ignore[method-assign]
        entry.cached_model.cur_vram_bytes = lambda: 20 * GB  # type: ignore[method-assign]
        entry.cached_model.get_cpu_state_dict = lambda: None  # type: ignore[method-assign]
        entry.cached_model._compute_device = torch.device("cuda")
        monkeypatch.setattr(model_cache_module, "_is_integrated_cuda", lambda _device: True)
        monkeypatch.setattr(model_cache_module, "device_has_dedicated_vram", lambda _device: False)
        monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: 12 * GB)
        monkeypatch.setattr(
            model_cache_module.TorchDevice,
            "empty_cache",
            lambda: model_cache_module.EmptyCacheOutcome.DEFERRED,
        )

        result = cache.make_room(25 * GB)

        assert not result.ram_headroom_satisfied
        assert "gpu" not in cache._cached_models
    finally:
        cache.shutdown()

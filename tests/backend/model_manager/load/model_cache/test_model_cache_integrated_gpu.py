"""ModelCache behaviour on an Intel integrated GPU, whose "VRAM" is system RAM.

Two things follow from that topology and are asserted here: a RAM copy of the weights is not
kept (it would double every model's footprint against the same DRAM), and partial loading stays
on so device residency remains bounded. Both are driven through `put()` with the Level Zero
integrated-GPU probe stubbed, so no Intel hardware is required.
"""

import gc
import logging
import threading
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache import model_cache as model_cache_module
from invokeai.backend.model_manager.load.model_cache.cache_stats import CacheStats
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import GB, ModelCache, integrated_cuda_admission_lock
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.util.devices import EmptyCacheOutcome, TorchDevice
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


def _make_cache(
    device: str,
    logger: MagicMock,
    enable_partial_loading: bool = False,
    keep_ram_copy_of_weights: bool = True,
) -> ModelCache:
    # ModelCache.__init__ sizes the RAM cache from the device's total VRAM (heuristic 2), which on
    # an xpu execution device reads torch.xpu.get_device_properties() -- an AssertionError on the
    # CPU-only torch builds CI runs. Stub a fixed total so construction works everywhere; the
    # tests patch the integrated-GPU probe around put() themselves.
    with patch.object(
        torch.xpu,
        "get_device_properties",
        create=True,
        return_value=SimpleNamespace(total_memory=8 * 2**30),
    ):
        return ModelCache(
            execution_device_working_mem_gb=1.0,
            enable_partial_loading=enable_partial_loading,
            keep_ram_copy_of_weights=keep_ram_copy_of_weights,
            execution_device=device,
            storage_device="cpu",
            logger=logger,
        )


@contextmanager
def _patch_integrated(is_integrated: bool | None):
    """Stub the Level Zero probe. None models a device whose type cannot be determined."""
    with (
        patch("invokeai.backend.util.devices.xpu_device_is_integrated", return_value=is_integrated),
        patch(
            "invokeai.backend.model_manager.load.model_cache.model_cache.xpu_device_is_integrated",
            return_value=is_integrated,
        ),
    ):
        yield


@contextmanager
def _patch_integrated_cuda():
    with (
        patch("invokeai.backend.util.devices.device_is_integrated", return_value=True),
        patch("invokeai.backend.model_manager.load.model_cache.model_cache.device_is_integrated", return_value=True),
        patch("torch.version.hip", None),
        patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=8 * 2**30)),
        patch.object(TorchDevice, "cuda_mem_get_info", classmethod(lambda cls, device: (7 * 2**30, 8 * 2**30))),
    ):
        yield


def _make_cuda_cache(
    logger: MagicMock,
    enable_partial_loading: bool = False,
    keep_ram_copy: bool = True,
    max_vram_cache_size_gb: float | None = None,
    ram_budget: RamBudget | None = None,
    shared_cpu_weights: SharedCpuWeightsStore | None = None,
) -> ModelCache:
    with _patch_integrated_cuda():
        return ModelCache(
            execution_device_working_mem_gb=1.0,
            enable_partial_loading=enable_partial_loading,
            keep_ram_copy_of_weights=keep_ram_copy,
            max_vram_cache_size_gb=max_vram_cache_size_gb,
            execution_device="cuda:0",
            storage_device="cpu",
            logger=logger,
            shared_cpu_weights=shared_cpu_weights,
            ram_budget=ram_budget,
        )


def _logged_messages(logger: MagicMock) -> list[str]:
    """Every string the cache logged.

    ModelCache wraps the logger in a PrefixedLoggerAdapter, and LoggerAdapter funnels .warning()
    through the underlying logger's .log(level, msg), so asserting on .warning() alone silently
    matches nothing.
    """
    messages: list[str] = []
    for method in (logger.log, logger.warning, logger.info, logger.debug):
        for call in method.call_args_list:
            messages.extend(arg for arg in call.args if isinstance(arg, str))
    return messages


def test_integrated_gpu_drops_the_ram_copy(mock_logger: MagicMock):
    """The RAM copy is what makes full loading cost 2x on unified memory."""
    cache = _make_cache("xpu:0", mock_logger)
    try:
        with _patch_integrated(True):
            cache.put("m", DummyModule())
        assert cache._cached_models["m"].cached_model.get_cpu_state_dict() is None
    finally:
        cache.shutdown()


@pytest.mark.parametrize("is_integrated", [False, None], ids=["discrete", "type-unknown"])
def test_non_integrated_xpu_keeps_the_ram_copy(is_integrated: bool | None, mock_logger: MagicMock):
    """A discrete card has its own VRAM, and an undeterminable one keeps the existing behaviour."""
    cache = _make_cache("xpu:0", mock_logger)
    try:
        with _patch_integrated(is_integrated):
            cache.put("m", DummyModule())
        assert cache._cached_models["m"].cached_model.get_cpu_state_dict() is not None
    finally:
        cache.shutdown()


def test_cpu_execution_device_is_untouched(mock_logger: MagicMock):
    """CPU and MPS share memory too, but their behaviour here is deliberately out of scope."""
    cache = _make_cache("cpu", mock_logger)
    try:
        cache.put("m", DummyModule())
        assert cache._cached_models["m"].cached_model.get_cpu_state_dict() is not None
    finally:
        cache.shutdown()


def test_partial_loading_is_honoured_on_an_integrated_gpu():
    """Streaming is the only bounded loading path there: it respects vram_available, whereas the
    full-load path ignores it and overshooting is an uncatchable OOM-kill on Linux."""
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    cache = _make_cache("xpu:0", logger, enable_partial_loading=True)
    try:
        with _patch_integrated(True):
            cache.put("m", DummyModule())
        assert isinstance(cache._cached_models["m"].cached_model, CachedModelWithPartialLoad)
        # and it still streams without a RAM copy
        assert cache._cached_models["m"].cached_model.get_cpu_state_dict() is None
    finally:
        cache.shutdown()


@pytest.mark.parametrize("keep_ram_copy", [True, False])
def test_integrated_cuda_preserves_explicit_ram_copy_setting(keep_ram_copy: bool, mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=keep_ram_copy)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
        assert (cache._cached_models["m"].cached_model.get_cpu_state_dict() is not None) is keep_ram_copy
    finally:
        cache.shutdown()


def test_partial_loading_is_honoured_on_integrated_cuda(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, enable_partial_loading=True)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
        assert isinstance(cache._cached_models["m"].cached_model, CachedModelWithPartialLoad)
    finally:
        cache.shutdown()


def test_integrated_cuda_partial_load_refuses_when_required_weights_do_not_fit_and_can_retry(
    mock_logger: MagicMock,
):
    """A minimum-weight shortfall raises before transfer and leaves the cache entry reusable."""
    cache = _make_cuda_cache(mock_logger, enable_partial_loading=True)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            cache._offload_unlocked_models = MagicMock(return_value=0)  # type: ignore[method-assign]
            cache._move_model_to_ram = MagicMock(return_value=0)  # type: ignore[method-assign]
            cache._get_vram_available = MagicMock(return_value=0)  # type: ignore[method-assign]
            move = MagicMock()
            cache._move_model_to_vram = move  # type: ignore[method-assign]

            with pytest.raises(torch.OutOfMemoryError, match="minimum required weights"):
                cache.lock(entry, None)

            assert move.call_count == 0
            assert cache._cached_models["m"] is entry
            assert not entry.is_locked

            move.return_value = (0, False)
            cache._get_vram_available.return_value = entry.cached_model.total_bytes() + 2 * 2**20
            cache._release_allocator_blocks_for_reserve = MagicMock()  # type: ignore[method-assign]
            assert cache.lock(entry, None)
            cache.unlock(entry)
            assert move.call_count == 1
    finally:
        cache.shutdown()


def test_integrated_cuda_partial_relock_with_no_required_weights_tolerates_negative_availability(
    mock_logger: MagicMock,
):
    cache = _make_cuda_cache(mock_logger, enable_partial_loading=True)
    try:
        with _patch_integrated_cuda():
            model = torch.nn.Sequential(torch.nn.Linear(2, 2))
            apply_custom_layers_to_model(model)
            cache.put("m", model)
            entry = _entry(cache, "m")
            cache._offload_unlocked_models = MagicMock(return_value=0)  # type: ignore[method-assign]
            cache._move_model_to_ram = MagicMock(return_value=0)  # type: ignore[method-assign]
            cache._get_vram_available = MagicMock(return_value=-1)  # type: ignore[method-assign]
            cache._move_model_to_vram = MagicMock(return_value=(0, False))  # type: ignore[method-assign]
            cache._release_allocator_blocks_for_reserve = MagicMock()  # type: ignore[method-assign]

            assert cache.lock(entry, None)
            cache.unlock(entry)
    finally:
        cache.shutdown()


def test_integrated_cuda_relock_checks_host_headroom_before_copying_gpu_only_model(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._is_in_vram = True
            cache._offload_unlocked_models = MagicMock(return_value=0)  # type: ignore[method-assign]
            cache._get_vram_available = MagicMock(return_value=-1)  # type: ignore[method-assign]
            cache._get_physical_vram_available = MagicMock(return_value=2**30)  # type: ignore[method-assign]
            full_unload = MagicMock(side_effect=AssertionError("copied before host admission"))
            with (
                patch.object(entry.cached_model, "full_unload_from_vram", full_unload),
                patch.object(model_cache_module, "_host_memory_available", return_value=0),
            ):
                with pytest.raises(torch.OutOfMemoryError, match="host memory"):
                    cache.lock(entry, None)

            assert cache._cached_models["m"] is entry
            assert entry.is_locked is False
            full_unload.assert_not_called()
    finally:
        cache.shutdown()


def test_integrated_cuda_partial_relock_unloads_only_the_shortfall_before_copying(mock_logger: MagicMock):
    """Partial loading can recover a small shortage without copying the full resident model to the host."""
    cache = _make_cuda_cache(mock_logger, enable_partial_loading=True, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            total_bytes = entry.cached_model.total_bytes()
            entry.cached_model._cur_vram_bytes = total_bytes
            cache._offload_unlocked_models = MagicMock(return_value=0)  # type: ignore[method-assign]
            cache._get_vram_available = MagicMock(side_effect=[-1, -1, 2**30, 2**30, 2**30])  # type: ignore[method-assign]
            cache._get_physical_vram_available = MagicMock(return_value=1)  # type: ignore[method-assign]
            cache._move_model_to_ram = MagicMock(return_value=1)  # type: ignore[method-assign]
            cache._move_model_to_vram = MagicMock(return_value=(0, False))  # type: ignore[method-assign]
            cache._release_allocator_blocks_for_reserve = MagicMock()  # type: ignore[method-assign]

            assert cache.lock(entry, None)
            cache._move_model_to_ram.assert_called_once_with(entry, 1)
            cache.unlock(entry)
    finally:
        cache.shutdown()


def test_integrated_cuda_public_partial_unload_refuses_before_cpu_transfer(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, enable_partial_loading=True, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            model = entry.cached_model.model
            cpu_state_dict = model.state_dict()
            model.state_dict = lambda: {key: SimpleNamespace(device=torch.device("cuda")) for key in cpu_state_dict}  # type: ignore[method-assign]
            entry.cached_model._cur_vram_bytes = entry.cached_model.total_bytes()
            cache._get_physical_vram_available = MagicMock(return_value=2**30)  # type: ignore[method-assign]
            with patch.object(model_cache_module, "_host_memory_available", return_value=0):
                with pytest.raises(torch.OutOfMemoryError, match="host memory"):
                    cache.unload_model_from_vram(entry, entry.cached_model.total_bytes())

            assert "m" in cache._cached_models
            assert not entry.detached
    finally:
        cache.shutdown()


def test_integrated_cuda_relock_with_cpu_shadow_does_not_require_new_host_memory(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=True)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._is_in_vram = True
            cache._offload_unlocked_models = MagicMock(return_value=0)  # type: ignore[method-assign]
            cache._get_vram_available = MagicMock(return_value=-1)  # type: ignore[method-assign]
            cache._get_physical_vram_available = MagicMock(return_value=0)  # type: ignore[method-assign]
            cache._move_model_to_ram = MagicMock(return_value=entry.cached_model.total_bytes())  # type: ignore[method-assign]
            cache._move_model_to_vram = MagicMock(return_value=(0, False))  # type: ignore[method-assign]
            cache._release_allocator_blocks_for_reserve = MagicMock()  # type: ignore[method-assign]

            assert cache.lock(entry, None)
            cache._move_model_to_ram.assert_called_once()
            cache._move_model_to_vram.assert_called_once()
            cache.unlock(entry)
    finally:
        cache.shutdown()


def test_partial_loading_still_off_on_shared_memory_devices_we_do_not_own(mock_logger: MagicMock):
    """CPU and MPS keep their existing behaviour; this PR does not change other platforms."""
    cache = _make_cache("cpu", mock_logger, enable_partial_loading=True)
    try:
        cache.put("m", DummyModule())
        assert isinstance(cache._cached_models["m"].cached_model, CachedModelOnlyFullLoad)
    finally:
        cache.shutdown()


def test_full_load_without_a_ram_copy_moves_weights_instead_of_copying():
    """The point of dropping the RAM copy: the load must not materialise a second set of tensors.

    Asserted at the wrapper level, on CPU, since it is a property of CachedModelOnlyFullLoad
    rather than of device selection.
    """
    device = torch.device("cpu")
    kept = CachedModelOnlyFullLoad(DummyModule(), device, total_bytes=1, keep_ram_copy=True)
    dropped = CachedModelOnlyFullLoad(DummyModule(), device, total_bytes=1, keep_ram_copy=False)

    assert kept.get_cpu_state_dict() is not None
    assert dropped.get_cpu_state_dict() is None

    before = {id(p) for p in dropped.model.parameters()}
    dropped.full_load_to_vram()
    assert {id(p) for p in dropped.model.parameters()} == before


def test_native_integrated_cuda_ram_capacity_uses_shared_pool(mock_logger: MagicMock):
    with patch("invokeai.backend.model_manager.load.model_cache.model_cache.psutil.virtual_memory") as virtual_memory:
        virtual_memory.return_value.total = 128 * 2**30
        cache = _make_cuda_cache(mock_logger)
    try:
        assert cache.local_ram_cache_size_bytes == 7 * 2**30
    finally:
        cache.shutdown()


def test_native_integrated_cuda_ram_capacity_uses_cuda_allocation_total(mock_logger: MagicMock):
    """Spark's CUDA total includes the shared allocation pool; the device property can expose only the carveout."""
    with (
        patch("invokeai.backend.util.devices.device_is_integrated", return_value=True),
        patch("invokeai.backend.model_manager.load.model_cache.model_cache.device_is_integrated", return_value=True),
        patch("torch.version.hip", None),
        patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=8 * GB)),
        patch.object(TorchDevice, "cuda_mem_get_info", classmethod(lambda cls, device: (20 * GB, 24 * GB))),
    ):
        cache = ModelCache(
            execution_device_working_mem_gb=1.0,
            enable_partial_loading=False,
            keep_ram_copy_of_weights=True,
            execution_device="cuda:0",
            storage_device="cpu",
            logger=mock_logger,
        )
    try:
        assert cache.local_ram_cache_size_bytes == 23 * GB
    finally:
        cache.shutdown()


def test_native_integrated_cuda_ram_capacity_honors_vram_cap(mock_logger: MagicMock):
    with patch("invokeai.backend.model_manager.load.model_cache.model_cache.psutil.virtual_memory") as virtual_memory:
        virtual_memory.return_value.total = 128 * 2**30
        cache = _make_cuda_cache(mock_logger, max_vram_cache_size_gb=4.0)
    try:
        assert cache.local_ram_cache_size_bytes == 3 * 2**30
    finally:
        cache.shutdown()


def _entry(cache: ModelCache, key: str):
    return cache._cached_models[key]


def test_oversized_model_on_an_integrated_gpu_raises_instead_of_being_oom_killed(mock_logger: MagicMock):
    """A shared-memory device cannot survive overshooting: the kernel SIGKILLs the process, so
    there is no exception to catch and nothing actionable in the log. Refuse up front instead."""
    cache = _make_cache("xpu:0", mock_logger)
    try:
        with _patch_integrated(True):
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            with pytest.raises(torch.OutOfMemoryError, match="shares memory with the CPU"):
                cache._move_model_to_vram(entry, vram_available=0)
    finally:
        cache.shutdown()


def test_relocking_a_resident_model_on_an_integrated_gpu_is_not_refused(mock_logger: MagicMock):
    """The guard must compare the bytes still to be moved, not the model's total.

    A resident model's weights occupy the same DRAM that vram_available is read from, so once
    loaded, any model bigger than what is left will always have total > available. lock() runs
    on every use, and full_load_to_vram() is a no-op for a resident model -- comparing the total
    would refuse the re-lock and evict a healthy model on every other generation.
    """
    cache = _make_cache("xpu:0", mock_logger)
    try:
        with _patch_integrated(True):
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            # Present the model as already resident without needing Intel hardware.
            entry.cached_model._is_in_vram = True
            assert entry.cached_model.cur_vram_bytes() == entry.cached_model.total_bytes()
            # Free memory is now smaller than the model (it is the model); the re-lock must pass.
            assert cache._move_model_to_vram(entry, vram_available=0) == (0, False)
            assert "m" in cache._cached_models
    finally:
        cache.shutdown()


def test_oversized_model_on_integrated_cuda_raises_before_full_load(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            with pytest.raises(torch.OutOfMemoryError, match="shares memory with the CPU"):
                cache._move_model_to_vram(_entry(cache, "m"), vram_available=0)
    finally:
        cache.shutdown()


def test_relocking_a_resident_integrated_cuda_model_is_not_refused(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._is_in_vram = True
            assert cache._move_model_to_vram(entry, vram_available=0) == (0, False)
            assert "m" in cache._cached_models
    finally:
        cache.shutdown()


@pytest.mark.parametrize("offload", ["except", "unlocked"])
@pytest.mark.parametrize("keep_ram_copy", [True, False], ids=["retained-copy", "no-copy"])
def test_integrated_cuda_offload_only_releases_a_retained_device_copy(
    offload: str, keep_ram_copy: bool, mock_logger: MagicMock
):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=keep_ram_copy)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            size = entry.cached_model.total_bytes()
            entry.cached_model.cur_vram_bytes = lambda: size  # type: ignore[method-assign]
            with (
                patch.object(cache, "_move_model_to_ram", return_value=size) as move,
                patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN),
            ):
                freed = (
                    cache.offload_models_from_vram_except(set())
                    if offload == "except"
                    else cache.offload_unlocked_models_from_vram()
                )
                assert freed == (size if keep_ram_copy else 0)
        if keep_ram_copy:
            move.assert_called_once()
        else:
            move.assert_not_called()
    finally:
        cache.shutdown()


def test_integrated_cuda_keeps_idle_resident_model_without_pressure(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.awaiting_first_use = False
            entry.cached_model._is_in_vram = True
            with patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN):
                cache._offload_unlocked_models(0, vram_available_fn=lambda: 0)
        assert "m" in cache._cached_models
    finally:
        cache.shutdown()


def test_integrated_cuda_pressure_drops_idle_model(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.awaiting_first_use = False
            entry.cached_model._is_in_vram = True
            total_bytes = entry.cached_model.total_bytes()
            with patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN):
                cache._offload_unlocked_models(total_bytes, vram_available_fn=lambda: 0)
        assert "m" not in cache._cached_models
    finally:
        cache.shutdown()


def test_integrated_cuda_pressure_drops_idle_model_without_copying_to_cpu(mock_logger: MagicMock):
    """An unowned UMA entry must be dropped directly, without materialising a CPU copy first."""
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.awaiting_first_use = False
            entry.cached_model._is_in_vram = True
            total_bytes = entry.cached_model.total_bytes()
            with (
                patch.object(
                    cache, "_move_model_to_ram", side_effect=AssertionError("unexpected CPU materialization")
                ) as move,
                patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN),
            ):
                assert cache._offload_unlocked_models(total_bytes, vram_available_fn=lambda: 0) == total_bytes
            move.assert_not_called()
        assert "m" not in cache._cached_models
    finally:
        cache.shutdown()


def test_integrated_cuda_eviction_unloads_model_held_by_stale_handle(mock_logger: MagicMock):
    budget = RamBudget(max_bytes=2**30, shared_store=None)
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False, ram_budget=budget)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            entry.cached_model._is_in_vram = True
            model_bytes = entry.cached_model.total_bytes()
            with (
                patch.object(cache, "_move_model_to_ram") as move,
                patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN),
            ):
                cache._offload_unlocked_models(model_bytes, vram_available_fn=lambda: 0)
        assert "m" not in cache._cached_models
        move.assert_not_called()
        assert loaded.resident_weight_bytes == model_bytes
        assert cache._get_ram_in_use() == entry.cached_model.total_bytes()
        assert budget.total_in_use() == entry.cached_model.total_bytes()
        assert entry.detached
        released = threading.Event()
        release = cache._release_detached_model_handle

        def release_and_signal(cache_entry):
            release(cache_entry)
            released.set()

        with patch.object(cache, "_release_detached_model_handle", side_effect=release_and_signal):
            del loaded, entry
            gc.collect()
            assert released.wait(timeout=10)
        assert budget.total_in_use() == 0
    finally:
        cache.shutdown()


def test_integrated_cuda_pressure_offloads_live_handle_with_cpu_shadow(mock_logger: MagicMock):
    """A live handle may remain usable while its retained CPU shadow makes GPU eviction safe."""
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=True)
    try:
        with _patch_integrated_cuda():
            cache.put("m1", DummyModule())
            cache.put("m2", DummyModule())
            entry = _entry(cache, "m1")
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            size = entry.cached_model.total_bytes()
            entry.cached_model.cur_vram_bytes = lambda: size  # type: ignore[method-assign]
            second = _entry(cache, "m2")
            second.awaiting_first_use = False
            second.cached_model._is_in_vram = True
            second.cached_model.cur_vram_bytes = lambda: size  # type: ignore[method-assign]
            available = 0

            def move_to_ram(_entry: object, _bytes: int) -> int:
                nonlocal available
                available += size
                return size

            with (
                patch.object(cache, "_move_model_to_ram", side_effect=move_to_ram) as move,
                patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN),
            ):
                assert cache._offload_unlocked_models(size, vram_available_fn=lambda: available) == size
        assert "m1" in cache._cached_models
        assert "m2" in cache._cached_models
        assert not entry.detached
        move.assert_called_once_with(entry, size)
        assert loaded.resident_weight_bytes == size
    finally:
        cache.shutdown()


def test_integrated_cuda_attached_handle_work_does_not_pin_evicted_model(mock_logger: MagicMock):
    """A finalizer queued before WeakSet removal must not keep an evicted attached record alive."""
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    queued: list[object] = []
    try:
        with _patch_integrated_cuda(), patch.object(cache, "_dispatch_deferred", side_effect=queued.append):
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            model_ref = weakref.ref(entry.cached_model)
            cache._release_loaded_model_handle(entry)
            entry.live_handles.clear()
            cache._delete_cache_entry(entry)
            del loaded, entry
            gc.collect()
        assert queued
        assert model_ref() is None
    finally:
        cache.shutdown()


def test_integrated_cuda_eviction_rechecks_handles_after_publishing_detached_record(mock_logger: MagicMock):
    """A last handle disappearing during detachment must take the ordinary release path."""
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)

    class LastHandleRace:
        def __init__(self) -> None:
            self.calls = 0

        def __bool__(self) -> bool:
            self.calls += 1
            return self.calls == 1

    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.live_handles = LastHandleRace()  # type: ignore[assignment]
            cache._delete_cache_entry(entry)
        assert "m" not in cache._cached_models
        assert not entry.detached
        assert cache._detached_handle_records == []
    finally:
        cache.shutdown()


def test_integrated_cuda_detached_handle_release_recovers_after_worker_death(mock_logger: MagicMock):
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=2**30, shared_store=store)
    cache = _make_cuda_cache(mock_logger, ram_budget=budget, shared_cpu_weights=store)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._cpu_state_dict = None
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            cache._offload_unlocked_models(entry.cached_model.total_bytes(), vram_available_fn=lambda: 0)
        assert entry.detached
        assert store.refcount("m") == 1
        worker = cache._deferred_work_thread
        assert worker is not None
        cache._deferred_work_queue.put(model_cache_module._DEFERRED_STOP)
        worker.join(timeout=10)
        assert not worker.is_alive()

        released = threading.Event()
        release = cache._release_detached_model_handle

        def release_and_signal(cache_entry):
            release(cache_entry)
            released.set()

        with patch.object(cache, "_release_detached_model_handle", side_effect=release_and_signal):
            del loaded, entry
            gc.collect()
            assert released.wait(timeout=10)
        assert store.refcount("m") == 0
        assert budget.total_in_use() == 0
    finally:
        cache.shutdown()


def test_integrated_cuda_detached_release_retries_after_shutdown_worker_start_failure(
    mock_logger: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    """A failed cleanup-thread start must not strand detached accounting permanently."""
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            cache._offload_unlocked_models(entry.cached_model.total_bytes(), vram_available_fn=lambda: 0)
        assert entry.detached

        real_start = threading.Thread.start

        worker = cache._deferred_work_thread
        assert worker is not None
        cache._deferred_work_queue.put(model_cache_module._DEFERRED_STOP)
        worker.join(timeout=10)
        assert not worker.is_alive()

        def fail_shutdown_worker(thread: threading.Thread) -> None:
            if thread.name == "model-cache-shutdown-deferred-work":
                raise RuntimeError("forced cleanup worker start failure")
            real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", fail_shutdown_worker)
        del loaded
        gc.collect()
        assert cache._shutdown_deferred_work_thread is None
        assert cache._detached_handle_records == [entry]
        assert not cache._shutdown_event.is_set()

        monkeypatch.undo()
        released = threading.Event()
        release = cache._release_detached_model_handle

        def release_and_signal(cache_entry):
            release(cache_entry)
            released.set()

        with patch.object(cache, "_release_detached_model_handle", side_effect=release_and_signal):
            # Timeout activity is a normal cache activity and must retry startup while live.
            cache._on_timeout()
            assert released.wait(timeout=10)
        assert cache._detached_handle_records == []
        assert entry.cached_model.uses_shared_weights is False
    finally:
        cache.shutdown()


def test_integrated_cuda_detached_release_survives_start_lock_contention_and_worker_death(
    mock_logger: MagicMock,
):
    """Startup contention must not route detached work to a worker that can die before enqueue."""
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            cache._offload_unlocked_models(entry.cached_model.total_bytes(), vram_available_fn=lambda: 0)
        assert entry.detached
        entry.live_handles.clear()

        worker = cache._deferred_work_thread
        assert worker is not None
        released = threading.Event()
        release = cache._release_detached_model_handle

        def release_and_signal(cache_entry):
            release(cache_entry)
            released.set()

        with patch.object(cache, "_release_detached_model_handle", side_effect=release_and_signal):
            # Nonblocking startup reports contention while the normal worker is still alive. The
            # normal worker then dies before any activity can retry the dedicated worker.
            with cache._shutdown_deferred_worker_start_lock:
                cache._dispatch_deferred(model_cache_module._DetachedHandleRelease(cache, weakref.ref(entry)))
                cache._deferred_work_queue.put(model_cache_module._DEFERRED_STOP)
                worker.join(timeout=10)
                assert not worker.is_alive()
            assert cache._shutdown_deferred_work_thread is None

            cache._record_activity()
            assert released.wait(timeout=10)
        assert cache._detached_handle_records == []
    finally:
        cache.shutdown()


def test_integrated_cuda_detached_release_uses_dedicated_worker(mock_logger: MagicMock):
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=2**30, shared_store=store)
    cache = _make_cuda_cache(mock_logger, ram_budget=budget, shared_cpu_weights=store)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._cpu_state_dict = None
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            cache._offload_unlocked_models(entry.cached_model.total_bytes(), vram_available_fn=lambda: 0)
        entry.live_handles.clear()
        assert entry.detached
        assert store.refcount("m") == 1

        handler_entered = threading.Event()
        allow_release = threading.Event()
        released = threading.Event()
        handler_thread_name: list[str] = []
        release = cache._release_detached_model_handle

        def release_and_signal(cache_entry):
            handler_thread_name.append(threading.current_thread().name)
            handler_entered.set()
            assert allow_release.wait(timeout=10)
            release(cache_entry)
            released.set()

        with patch.object(cache, "_release_detached_model_handle", side_effect=release_and_signal):
            cache._dispatch_deferred(model_cache_module._DetachedHandleRelease(cache, weakref.ref(entry)))
            worker = cache._deferred_work_thread
            assert worker is not None
            assert handler_entered.wait(timeout=10)
            cache._deferred_work_queue.put(model_cache_module._DEFERRED_STOP)
            worker.join(timeout=10)
            assert not worker.is_alive()
            assert not released.is_set()
            assert handler_thread_name == ["model-cache-shutdown-deferred-work"]
            allow_release.set()
            assert released.wait(timeout=10)
        assert store.refcount("m") == 0
        assert budget.total_in_use() == 0
    finally:
        cache.shutdown()


def test_integrated_cuda_detached_release_keeps_cache_alive_until_accounting_cleanup(mock_logger: MagicMock):
    """A queued detached release must retain its cache until budget accounting is released."""
    budget = RamBudget(max_bytes=2**30, shared_store=None)
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False, ram_budget=budget)
    queue = model_cache_module.queue.SimpleQueue()
    item_ready = threading.Event()
    allow_get = threading.Event()

    class BlockingQueue:
        def put(self, item):
            queue.put(item)

        def get(self):
            item_ready.set()
            assert allow_get.wait(timeout=10)
            return queue.get()

    worker = threading.Thread(
        target=model_cache_module._run_shutdown_deferred_work,
        args=(weakref.ref(cache), BlockingQueue()),
        daemon=True,
    )
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            cache._offload_unlocked_models(entry.cached_model.total_bytes(), vram_available_fn=lambda: 0)
        assert budget.total_in_use() == entry.cached_model.total_bytes()
        queue.put(model_cache_module._DetachedHandleRelease(cache, weakref.ref(entry)))
        worker.start()
        assert item_ready.wait(timeout=10)
        cache_ref = weakref.ref(cache)
        del loaded, entry, cache
        gc.collect()
        assert cache_ref() is not None
        allow_get.set()
        queue.put(model_cache_module._SHUTDOWN_DEFERRED_STOP)
        worker.join(timeout=10)
        assert not worker.is_alive()
        gc.collect()
        assert budget.total_in_use() == 0
    finally:
        allow_get.set()
        worker.join(timeout=10)


def test_shutdown_deferred_worker_start_is_serialized(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger)
    barrier = threading.Barrier(3)
    starters: list[threading.Thread] = []
    real_start = threading.Thread.start
    shutdown_starts: list[threading.Thread] = []

    def record_start(thread: threading.Thread) -> None:
        if thread.name == "model-cache-shutdown-deferred-work":
            shutdown_starts.append(thread)
        real_start(thread)

    def start_worker() -> None:
        barrier.wait()
        cache._ensure_shutdown_deferred_worker(blocking=False)

    try:
        with patch.object(threading.Thread, "start", new=record_start):
            starters = [threading.Thread(target=start_worker) for _ in range(2)]
            for starter in starters:
                starter.start()
            barrier.wait()
            for starter in starters:
                starter.join(timeout=10)
        assert all(not starter.is_alive() for starter in starters)
        assert len(shutdown_starts) == 1
    finally:
        cache.shutdown()


def test_integrated_cuda_handle_finalizer_does_not_reenter_shared_store_lock(mock_logger: MagicMock):
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=2**30, shared_store=store)
    cache = _make_cuda_cache(mock_logger, ram_budget=budget, shared_cpu_weights=store)
    gc.freeze()
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._cpu_state_dict = None
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            cache._offload_unlocked_models(entry.cached_model.total_bytes(), vram_available_fn=lambda: 0)
            assert entry.detached

        released = threading.Event()
        release = cache._release_detached_model_handle

        def release_and_signal(cache_entry):
            release(cache_entry)
            released.set()

        holder = [loaded]
        del loaded
        lock_held = threading.Event()
        finished = threading.Event()

        def collect_inside_store_lock():
            with store._lock:
                lock_held.set()
                holder.clear()
                gc.collect()
                finished.set()

        with patch.object(cache, "_release_detached_model_handle", side_effect=release_and_signal):
            collector = threading.Thread(target=collect_inside_store_lock)
            collector.start()
            assert lock_held.wait(timeout=2)
            if not finished.wait(timeout=2):
                # The pre-fix callback deadlocks the collector while it owns store._lock. Release
                # it from this test thread so the red run can unwind instead of hanging pytest.
                store._lock.release()
                collector.join(timeout=2)
                pytest.fail("handle finalizer tried to reacquire the store lock")
            collector.join(timeout=2)
            assert released.wait(timeout=10)
        assert store.total_bytes_in_use() == 0
        assert budget.total_in_use() == 0
    finally:
        try:
            cache.shutdown()
        finally:
            gc.unfreeze()


def test_integrated_cuda_handle_finalizer_runs_after_weakset_removal(mock_logger: MagicMock):
    """Inline deferred handling must see the dying handle already removed from the WeakSet."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=2**30, shared_store=store)
    cache = _make_cuda_cache(mock_logger, ram_budget=budget, shared_cpu_weights=store)
    try:
        cache.stats = CacheStats()
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._cpu_state_dict = None
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            cache._offload_unlocked_models(entry.cached_model.total_bytes(), vram_available_fn=lambda: 0)
            assert entry.detached
            assert store.refcount("m") == 1
            assert budget.total_in_use() == store.total_bytes_in_use()

        def dispatch_inline(work):
            cache._release_detached_model_handle(work.cache_entry_ref())

        release = cache._release_detached_model_handle

        def release_after_weakset_removal(cache_entry):
            assert len(cache_entry.live_handles) == 0
            return release(cache_entry)

        with (
            patch.object(cache, "_dispatch_deferred", side_effect=dispatch_inline),
            patch.object(cache, "_release_detached_model_handle", side_effect=release_after_weakset_removal),
        ):
            del loaded
            gc.collect()

        assert store.refcount("m") == 0
        assert budget.total_in_use() == 0
        assert cache.stats.cache_used == 0
        assert cache.stats.in_cache == 0
        cache._release_detached_model_handle(entry)
        assert store.refcount("m") == 0
        assert budget.total_in_use() == 0
    finally:
        cache.shutdown()


def test_integrated_cuda_detached_handle_release_refreshes_stats(mock_logger: MagicMock):
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=2**30, shared_store=store)
    cache = _make_cuda_cache(mock_logger, ram_budget=budget, shared_cpu_weights=store)
    try:
        cache.stats = CacheStats()
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._cpu_state_dict = None
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            cache._offload_unlocked_models(entry.cached_model.total_bytes(), vram_available_fn=lambda: 0)

        assert cache.stats.cache_used > 0
        assert cache.stats.in_cache == 0
        cleared = cache.stats.cleared
        cache.shutdown()
        released = threading.Event()
        release = cache._release_detached_model_handle

        def release_and_signal(cache_entry):
            release(cache_entry)
            released.set()

        with patch.object(cache, "_release_detached_model_handle", side_effect=release_and_signal):
            del loaded
            gc.collect()
            assert released.wait(timeout=10)
        assert store.total_bytes_in_use() == 0
        assert budget.total_in_use() == 0
        assert cache.stats.cache_used == 0
        assert cache.stats.in_cache == 0
        assert cache.stats.cleared == cleared
    finally:
        cache.shutdown()


def test_integrated_cuda_cpu_only_entry_keeps_budget_while_handle_lives(mock_logger: MagicMock):
    budget = RamBudget(max_bytes=2**30, shared_store=None)
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False, ram_budget=budget)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model._compute_device = torch.device("cpu")
            loaded = LoadedModelWithoutConfig(entry, cache)
            loaded._end_first_use_window()
            entry.awaiting_first_use = False
            model_bytes = entry.cached_model.total_bytes()
            cache._offload_unlocked_models(model_bytes, vram_available_fn=lambda: 0)
        assert "m" not in cache._cached_models
        assert budget.total_in_use() == model_bytes
        released = threading.Event()
        release = cache._release_detached_model_handle

        def release_and_signal(cache_entry):
            release(cache_entry)
            released.set()

        with patch.object(cache, "_release_detached_model_handle", side_effect=release_and_signal):
            del loaded, entry
            gc.collect()
            assert released.wait(timeout=10)
        assert budget.total_in_use() == 0
    finally:
        cache.shutdown()


def test_integrated_cuda_eviction_credits_actual_partial_residency(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, enable_partial_loading=True, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.awaiting_first_use = False
            total_bytes = entry.cached_model.total_bytes()
            resident_bytes = total_bytes // 2
            entry.cached_model.cur_vram_bytes = lambda: resident_bytes  # type: ignore[method-assign]
            with (
                patch.object(cache, "_move_model_to_ram", return_value=resident_bytes),
                patch.object(cache, "_notify_models_cleared") as notify,
                patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN),
            ):
                assert cache._offload_unlocked_models(total_bytes, vram_available_fn=lambda: 0) == resident_bytes
        notify.assert_called_once_with(
            models_cleared=1,
            bytes_requested=total_bytes,
            bytes_freed=resident_bytes,
            cumulative_stats=True,
        )
    finally:
        cache.shutdown()


def test_integrated_cuda_public_offload_waits_for_admission(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger)
    entered = threading.Event()
    finished = threading.Event()
    try:
        with _patch_integrated_cuda(), integrated_cuda_admission_lock(torch.device("cuda:0")):
            cache._offload_entries_from_vram = MagicMock(side_effect=lambda _select: entered.set())  # type: ignore[method-assign]
            thread = threading.Thread(target=lambda: (cache.offload_models_from_vram_except(set()), finished.set()))
            thread.start()
            assert not entered.wait(timeout=0.1)
        assert entered.wait(timeout=2)
        assert finished.wait(timeout=2)
        thread.join(timeout=2)
    finally:
        cache.shutdown()


def test_integrated_cuda_pressure_preserves_locked_and_first_use_entries(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("locked", DummyModule())
            cache.put("held", DummyModule())
            locked = _entry(cache, "locked")
            held = _entry(cache, "held")
            locked.cached_model._is_in_vram = True
            held.cached_model._is_in_vram = True
            cache.lock_in_ram(locked)
            held.awaiting_first_use = True
            with patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN):
                cache._offload_unlocked_models(2**30, vram_available_fn=lambda: 0)
        assert set(cache._cached_models) == {"locked", "held"}
    finally:
        cache.shutdown()


def test_integrated_cuda_pressure_batches_cleanup_and_drops_cpu_resident_entries(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("cuda", DummyModule())
            cache.put("cpu", DummyModule())
            cuda_entry = _entry(cache, "cuda")
            cpu_entry = _entry(cache, "cpu")
            cuda_entry.awaiting_first_use = False
            cpu_entry.awaiting_first_use = False
            cuda_entry.cached_model._is_in_vram = True
            assert cpu_entry.cached_model.cur_vram_bytes() == 0
            bytes_requested = cuda_entry.cached_model.total_bytes() + cpu_entry.cached_model.total_bytes()
            with (
                patch(
                    "invokeai.backend.model_manager.load.model_cache.model_cache.expandable_segments_enabled",
                    return_value=False,
                ),
                patch.object(cache, "_notify_models_cleared") as notify,
                patch("invokeai.backend.model_manager.load.model_cache.model_cache.gc.collect") as collect,
                patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN) as empty_cache,
            ):
                cache._offload_unlocked_models(bytes_requested, vram_available_fn=lambda: 0)
        assert not cache._cached_models
        notify.assert_called_once_with(
            models_cleared=2,
            bytes_requested=bytes_requested,
            bytes_freed=cuda_entry.cached_model.total_bytes(),
            cumulative_stats=True,
        )
        collect.assert_called_once()
        empty_cache.assert_called_once()
    finally:
        cache.shutdown()


def test_integrated_cuda_reuses_non_expandable_allocator_credit_after_eviction(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with (
            _patch_integrated_cuda(),
            patch(
                "invokeai.backend.model_manager.load.model_cache.model_cache.expandable_segments_enabled",
                return_value=False,
            ),
            patch.object(TorchDevice, "memory_available_to_process", return_value=8 * 2**30),
            patch.object(cache, "_get_vram_in_use", return_value=0),
            patch.object(cache, "_get_reclaimable_allocator_bytes", return_value=2 * 2**30),
        ):
            assert cache._get_vram_available(None) == 9 * 2**30
    finally:
        cache.shutdown()


def test_integrated_cuda_calibration_capacity_matches_uncapped_availability(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with (
            _patch_integrated_cuda(),
            patch(
                "invokeai.backend.model_manager.load.model_cache.model_cache.expandable_segments_enabled",
                return_value=False,
            ),
            patch.object(TorchDevice, "memory_available_to_process", return_value=8 * 2**30),
            patch.object(cache, "_get_vram_in_use", return_value=0),
            patch.object(cache, "_get_reclaimable_allocator_bytes", return_value=2 * 2**30),
        ):
            assert cache.calibration_capacity() == cache._get_vram_available(None) + 1 * 2**30
    finally:
        cache.shutdown()


@pytest.mark.parametrize(
    ("cap_gb", "expected_available_gb"),
    [(12.0, 9.0), (6.0, 5.0)],
    ids=["high-cap-uses-credit", "low-cap-remains-hard-limit"],
)
def test_integrated_cuda_capped_availability_uses_credit_but_honors_cap(
    cap_gb: float, expected_available_gb: float, mock_logger: MagicMock
):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False, max_vram_cache_size_gb=cap_gb)
    try:
        with (
            _patch_integrated_cuda(),
            patch(
                "invokeai.backend.model_manager.load.model_cache.model_cache.expandable_segments_enabled",
                return_value=False,
            ),
            patch.object(TorchDevice, "memory_available_to_process", return_value=8 * 2**30),
            patch.object(cache, "_get_vram_in_use", return_value=0),
            patch.object(cache, "_get_reclaimable_allocator_bytes", return_value=2 * 2**30),
        ):
            assert cache._get_vram_available(None) == int(expected_available_gb * 2**30)
            assert cache.calibration_capacity() == int(min(10.0, cap_gb) * 2**30)
    finally:
        cache.shutdown()


def test_integrated_cuda_stops_after_first_eviction_when_allocator_reuses_freed_capacity(
    mock_logger: MagicMock,
):
    cache = _make_cuda_cache(mock_logger, keep_ram_copy=False)
    try:
        with _patch_integrated_cuda():
            cache.put("first", DummyModule())
            cache.put("second", DummyModule())
            first = _entry(cache, "first")
            second = _entry(cache, "second")
            first.awaiting_first_use = False
            second.awaiting_first_use = False
            first.cached_model._is_in_vram = True
            second.cached_model._is_in_vram = True
            model_bytes = first.cached_model.total_bytes()
            with (
                patch(
                    "invokeai.backend.model_manager.load.model_cache.model_cache.expandable_segments_enabled",
                    return_value=False,
                ),
                patch.object(cache, "_get_vram_available", side_effect=[0, model_bytes]),
                patch.object(TorchDevice, "empty_cache", return_value=EmptyCacheOutcome.RAN),
            ):
                cache._offload_unlocked_models(model_bytes)
        assert set(cache._cached_models) == {"second"}
    finally:
        cache.shutdown()


def test_discrete_cuda_pressure_moves_model_to_ram(mock_logger: MagicMock):
    cache = _make_cuda_cache(mock_logger)
    try:
        with (
            patch("invokeai.backend.util.devices.device_is_integrated", return_value=False),
            patch(
                "invokeai.backend.model_manager.load.model_cache.model_cache.device_is_integrated",
                return_value=False,
            ),
            patch("torch.version.hip", None),
        ):
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.awaiting_first_use = False
            entry.cached_model._is_in_vram = True
            entry.cached_model.full_unload_from_vram = MagicMock(return_value=1)
            cache._offload_unlocked_models(1, vram_available_fn=lambda: 0)
        assert "m" in cache._cached_models
        entry.cached_model.full_unload_from_vram.assert_called_once()
    finally:
        cache.shutdown()


def test_oversized_model_on_a_discrete_card_still_tries(mock_logger: MagicMock):
    """A device with its own VRAM fails with a catchable allocator error, so the try-anyway
    behaviour is left exactly as it was."""
    cache = _make_cache("xpu:0", mock_logger)
    try:
        with _patch_integrated(False):
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model.full_load_to_vram = MagicMock(return_value=123)
            assert cache._move_model_to_vram(entry, vram_available=0) == (123, False)
    finally:
        cache.shutdown()


def test_cpu_execution_device_is_not_guarded(mock_logger: MagicMock):
    """CPU has no device memory to exhaust separately from RAM; loading there is a no-op move."""
    cache = _make_cache("cpu", mock_logger)
    try:
        cache.put("m", DummyModule())
        entry = _entry(cache, "m")
        entry.cached_model.full_load_to_vram = MagicMock(return_value=7)
        assert cache._move_model_to_vram(entry, vram_available=0) == (7, False)
    finally:
        cache.shutdown()


def test_mps_is_left_alone(mock_logger: MagicMock):
    """MPS shares memory the same way, but changing Mac behaviour is out of scope for this work.

    Pinned as a test because the natural predicate to reach for -- "does this device have
    dedicated VRAM" -- is False for MPS too, so the guard silently covers Mac unless it is
    scoped deliberately.
    """
    cache = _make_cache("cpu", mock_logger)
    try:
        cache.put("m", DummyModule())
        entry = _entry(cache, "m")
        # Present the cached model as MPS-resident without needing a Mac.
        entry.cached_model._compute_device = torch.device("mps")
        entry.cached_model.full_load_to_vram = MagicMock(return_value=42)
        assert cache._move_model_to_vram(entry, vram_available=0) == (42, False)
    finally:
        cache.shutdown()


def test_overridden_keep_ram_copy_is_reported_once_per_device(mock_logger: MagicMock):
    """Overriding a setting the user turned on is exactly the silent behaviour worth avoiding."""
    cache = _make_cache("xpu:0", mock_logger)
    try:
        with _patch_integrated(True):
            cache.put("a", DummyModule())
            cache.put("b", DummyModule())
        notices = [m for m in _logged_messages(mock_logger) if "keep_ram_copy_of_weights" in m]
        assert len(notices) == 1
        assert "xpu:0" in notices[0]
    finally:
        cache.shutdown()


def test_no_keep_ram_copy_notice_when_the_user_already_disabled_it(mock_logger: MagicMock):
    """Nothing was overridden, so there is nothing to report."""
    cache = _make_cache("xpu:0", mock_logger, keep_ram_copy_of_weights=False)
    try:
        with _patch_integrated(True):
            cache.put("m", DummyModule())
        assert not [m for m in _logged_messages(mock_logger) if "keep_ram_copy_of_weights" in m]
    finally:
        cache.shutdown()


def test_no_keep_ram_copy_notice_on_a_discrete_card(mock_logger: MagicMock):
    cache = _make_cache("xpu:0", mock_logger)
    try:
        with _patch_integrated(False):
            cache.put("m", DummyModule())
        assert not [m for m in _logged_messages(mock_logger) if "keep_ram_copy_of_weights" in m]
    finally:
        cache.shutdown()

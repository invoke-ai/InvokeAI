"""ModelCache behaviour on an Intel integrated GPU, whose "VRAM" is system RAM.

Two things follow from that topology and are asserted here: a RAM copy of the weights is not
kept (it would double every model's footprint against the same DRAM), and partial loading stays
on so device residency remains bounded. Both are driven through `put()` with the Level Zero
integrated-GPU probe stubbed, so no Intel hardware is required.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


def _make_cache(device: str, logger: MagicMock, enable_partial_loading: bool = False) -> ModelCache:
    return ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=enable_partial_loading,
        keep_ram_copy_of_weights=True,
        execution_device=device,
        storage_device="cpu",
        logger=logger,
    )


def _patch_integrated(is_integrated: bool | None):
    """Stub the Level Zero probe. None models a device whose type cannot be determined."""
    return patch(
        "invokeai.backend.model_manager.load.model_cache.model_cache.xpu_device_is_integrated",
        return_value=is_integrated,
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


def test_oversized_model_on_a_discrete_card_still_tries(mock_logger: MagicMock):
    """A device with its own VRAM fails with a catchable allocator error, so the try-anyway
    behaviour is left exactly as it was."""
    cache = _make_cache("xpu:0", mock_logger)
    try:
        with _patch_integrated(False):
            cache.put("m", DummyModule())
            entry = _entry(cache, "m")
            entry.cached_model.full_load_to_vram = MagicMock(return_value=123)
            assert cache._move_model_to_vram(entry, vram_available=0) == 123
    finally:
        cache.shutdown()


def test_cpu_execution_device_is_not_guarded(mock_logger: MagicMock):
    """CPU has no device memory to exhaust separately from RAM; loading there is a no-op move."""
    cache = _make_cache("cpu", mock_logger)
    try:
        cache.put("m", DummyModule())
        entry = _entry(cache, "m")
        entry.cached_model.full_load_to_vram = MagicMock(return_value=7)
        assert cache._move_model_to_vram(entry, vram_available=0) == 7
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
        assert cache._move_model_to_vram(entry, vram_available=0) == 42
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
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=False,
        execution_device="xpu:0",
        storage_device="cpu",
        logger=mock_logger,
    )
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

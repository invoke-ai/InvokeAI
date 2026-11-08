"""`LoadedModel.weight_bytes` and `.resident_weight_bytes` against a real cache.

A caller that holds two large models at once has to bound its own residency, and to size that
request it needs both numbers: how big the model is, and how much of it is currently on the compute
device. `Ideogram4DenoiseInvocation` is the first such caller; `wan_denoise.py` does the same thing
by reaching through `_cache_record.cached_model` to `total_bytes()` / `cur_vram_bytes()`.

These cells exist because that reach-through is what the properties wrap. Their node-level tests
use a hand-written `LoadedModel` fake with its own attributes, so renaming either cache method
would leave those green and fail at runtime — after both multi-GiB branches are already locked. The
two cached-model kinds answer differently and are covered separately: a CPU execution device yields
`CachedModelOnlyFullLoad`, which is all-or-nothing, while a CUDA one yields
`CachedModelWithPartialLoad`, where residency is the point.
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache

MB = 1024**2

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")


def _module_bytes(module: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))


def _cache(execution_device: str) -> ModelCache:
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return ModelCache(
        execution_device_working_mem_gb=0.1,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device=execution_device,
        storage_device="cpu",
        logger=logger,
    )


def _wrap(cache: ModelCache, module: torch.nn.Module) -> LoadedModelWithoutConfig:
    cache.put("model", module)
    return LoadedModelWithoutConfig(cache.get("model"), cache)


def test_weight_bytes_reports_the_models_own_size() -> None:
    """The size is what the caller compares against the device's capacity, so it must be the
    model's bytes and not the cache's or the record's."""
    cache = _cache("cpu")
    try:
        module = torch.nn.Linear(256, 256)
        assert _wrap(cache, module).weight_bytes == _module_bytes(module)
    finally:
        cache.shutdown()


def test_an_unloaded_model_reports_no_residency() -> None:
    """A model the cache holds only in RAM has nothing on the compute device. A caller sizing an
    unload from this would otherwise ask the cache to free bytes that are not there."""
    cache = _cache("cpu")
    try:
        loaded = _wrap(cache, torch.nn.Linear(256, 256))
        assert loaded.resident_weight_bytes == 0
        assert loaded.weight_bytes > 0
    finally:
        cache.shutdown()


@requires_cuda
def test_residency_follows_an_unload_while_the_size_does_not() -> None:
    """The two numbers move independently, which is the whole reason both exist.

    A caller reading size where it means residency would keep asking for an unload that already
    happened; reading residency where it means size would under-read a model that is only partly
    loaded and conclude the device has room it does not have.
    """
    cache = _cache("cuda:0")
    try:
        module = torch.nn.Linear(2048, 2048)
        loaded = _wrap(cache, module)
        size = loaded.weight_bytes
        assert size == _module_bytes(module)

        with loaded.model_on_device():
            assert loaded.resident_weight_bytes > 0
            resident_before = loaded.resident_weight_bytes

            loaded.unload_from_vram(resident_before // 2)

            assert loaded.resident_weight_bytes < resident_before
            assert loaded.weight_bytes == size
    finally:
        cache.shutdown()

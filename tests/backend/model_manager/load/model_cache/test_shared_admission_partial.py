import logging

import torch

from invokeai.backend.model_manager.load.model_cache import model_cache as cache_module
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore


def test_partial_load_adoption_keeps_peer_without_extra_budget(monkeypatch):
    # Select the partial-loading boundary while keeping all tensors and operations on CPU.
    monkeypatch.setattr(cache_module, "device_has_dedicated_vram", lambda _: True)
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=4096 + 100, shared_store=store)
    caches = [
        ModelCache(
            execution_device_working_mem_gb=1,
            enable_partial_loading=True,
            keep_ram_copy_of_weights=True,
            execution_device="cpu",
            storage_device="cpu",
            logger=logging.getLogger(__name__),
            shared_cpu_weights=store,
            ram_budget=budget,
        )
        for _ in range(2)
    ]
    try:
        caches[0].put("m", torch.nn.Linear(32, 32, bias=False), prefetch=True)
        canonical = store.peek("m")
        assert canonical is not None
        adopted = torch.nn.Linear(32, 32, bias=False)
        adopted.load_state_dict(canonical, assign=True)
        caches[1].put("m", adopted, prefetch=True)
        assert "m" in caches[0]._cached_models
        assert isinstance(caches[1].get("m").cached_model, CachedModelWithPartialLoad)
        assert store.refcount("m") == 2
        assert budget.total_in_use() == 4096
    finally:
        for cache in caches:
            cache.shutdown()
    assert budget.total_in_use() == 0

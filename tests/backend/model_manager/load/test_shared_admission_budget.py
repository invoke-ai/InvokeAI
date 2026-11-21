"""Exercise shared-weight admission through real loaders, caches and the global budget."""

import logging
from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 16)


def test_adoption_keeps_peer_cached_and_avoids_reload(tmp_path):
    store = SharedCpuWeightsStore()
    weight_bytes = 64 * 16 * 4
    budget = RamBudget(max_bytes=weight_bytes + 100, shared_store=store)
    logger = logging.getLogger(__name__)
    loaders = []
    disk_loads = 0
    weights_path = tmp_path / "weights.pt"
    torch.save(TinyModel().state_dict(), weights_path)

    def disk_load(*args):
        nonlocal disk_loads
        disk_loads += 1
        model = TinyModel()
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        return model

    for _ in range(2):
        cache = ModelCache(
            execution_device_working_mem_gb=1,
            enable_partial_loading=False,
            keep_ram_copy_of_weights=True,
            execution_device="cpu",
            storage_device="cpu",
            logger=logger,
            shared_cpu_weights=store,
            ram_budget=budget,
        )
        loader = object.__new__(ModelLoader)
        loader._ram_cache = cache
        loader._logger = logger
        loader._app_config = SimpleNamespace(log_memory_usage=False)
        loader.config_check = None
        loader._get_model_path = lambda config: tmp_path
        loader._report_inert_fp8_request = lambda *args: None
        loader._get_execution_device = lambda *args: torch.device("cpu")
        loader.get_size_fs = lambda *args: weight_bytes
        loader._load_model = disk_load
        loaders.append(loader)

    config = SimpleNamespace(base="any", type="test", name="tiny", key="tiny", path="unused")
    try:
        first, claim = loaders[0]._load_and_cache(config)
        claim.release()
        loaders[0]._ram_cache.lock(first, None)
        loaders[0]._ram_cache.unlock(first)
        second, claim = loaders[1]._load_and_cache(config)
        claim.release()
        loaders[1]._ram_cache.lock(second, None)
        loaders[1]._ram_cache.unlock(second)
        peer_still_cached = "tiny" in loaders[0]._ram_cache._cached_models
        first_again, claim = loaders[0]._load_and_cache(config)
        claim.release()
        assert disk_loads == 1
        assert peer_still_cached
        assert first_again is first
        assert store.refcount("tiny") == 2
        assert budget.total_in_use() == weight_bytes
        assert second.cached_model.total_bytes() == weight_bytes
    finally:
        for loader in loaders:
            loader._ram_cache.shutdown()
    assert store.total_bytes_in_use() == 0
    assert budget.total_in_use() == 0


@pytest.mark.parametrize("case", ["independent", "clone", "private_buffer", "extra_state"])
def test_nonshared_admission_still_reserves_ram(case, monkeypatch):
    store = SharedCpuWeightsStore()
    size = 64 * 16 * 4
    budget = RamBudget(max_bytes=size + 100, shared_store=store)
    caches = [
        ModelCache(
            execution_device_working_mem_gb=1,
            enable_partial_loading=False,
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
        caches[0].put("tiny", TinyModel())
        record = caches[0].get("tiny")
        caches[0].lock(record, None)
        caches[0].unlock(record)
        canonical = store.peek("tiny")
        assert canonical is not None
        model = TinyModel()
        key = "other" if case == "independent" else "tiny"
        if case == "private_buffer":
            model.load_state_dict(canonical, assign=True)
            model.register_buffer("private", torch.ones(64), persistent=False)
        if case == "extra_state":
            model.load_state_dict(canonical, assign=True)
            original_state_dict = model.state_dict
            first_call = True

            def state_dict_with_extra_once(*args, **kwargs):
                nonlocal first_call
                state = original_state_dict(*args, **kwargs)
                if first_call:
                    first_call = False
                    state["_extra_state"] = {"version": 1}
                return state

            monkeypatch.setattr(model, "state_dict", state_dict_with_extra_once)
        caches[1].put(key, model)
        assert "tiny" not in caches[0]._cached_models
    finally:
        for cache in caches:
            cache.shutdown()
    assert store.total_bytes_in_use() == 0

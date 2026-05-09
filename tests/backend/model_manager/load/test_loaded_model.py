import pytest
import torch

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)


class ModelWithRequiredScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.scale = torch.nn.Parameter(torch.ones(4))


class FakeCache:
    def __init__(self):
        self.lock_calls = 0
        self.unlock_calls = 0

    def lock(self, cache_record: CacheRecord, working_mem_bytes: int | None) -> None:
        del cache_record, working_mem_bytes
        self.lock_calls += 1

    def unlock(self, cache_record: CacheRecord) -> None:
        del cache_record
        self.unlock_calls += 1


def test_model_on_device_repairs_required_tensors_for_partial_models():
    model = ModelWithRequiredScale()
    apply_custom_layers_to_model(model, device_autocasting_enabled=True)
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
    loaded_model = LoadedModelWithoutConfig(
        cache_record=CacheRecord(key="test", cached_model=cached_model), cache=FakeCache()
    )

    with loaded_model.model_on_device():
        assert model.scale.device.type == "meta"
        assert all(param.device.type == "cpu" for param in model.linear.parameters())


def test_model_on_device_leaves_full_load_models_unchanged():
    model = torch.nn.Linear(4, 4)
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device("meta"), total_bytes=1, keep_ram_copy=False
    )
    loaded_model = LoadedModelWithoutConfig(
        cache_record=CacheRecord(key="test", cached_model=cached_model), cache=FakeCache()
    )

    with loaded_model.model_on_device() as (_, returned_model):
        assert returned_model is model
        assert all(param.device.type == "cpu" for param in model.parameters())


def test_enter_unlocks_if_repair_raises():
    class BrokenCachedModel(CachedModelWithPartialLoad):
        def repair_required_tensors_on_compute_device(self) -> int:
            raise RuntimeError("repair failed")

    model = ModelWithRequiredScale()
    apply_custom_layers_to_model(model, device_autocasting_enabled=True)
    cached_model = BrokenCachedModel(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
    fake_cache = FakeCache()
    loaded_model = LoadedModelWithoutConfig(
        cache_record=CacheRecord(key="test", cached_model=cached_model), cache=fake_cache
    )

    with pytest.raises(RuntimeError, match="repair failed"):
        loaded_model.__enter__()

    assert fake_cache.lock_calls == 1
    assert fake_cache.unlock_calls == 1

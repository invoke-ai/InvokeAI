import logging

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
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate, WorkingMemoryTerm


class ModelWithRequiredScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.scale = torch.nn.Parameter(torch.ones(4))


class FakeCache:
    def __init__(self):
        self.lock_calls = 0
        self.unlock_calls = 0
        self.reserved: list[int | None] = []

    def lock(self, cache_record: CacheRecord, working_mem_bytes: int | None, max_move_bytes: int | None = None) -> bool:
        del cache_record, max_move_bytes
        self.lock_calls += 1
        self.reserved.append(working_mem_bytes)
        return True

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


def _full_load_model(cache: FakeCache) -> LoadedModelWithoutConfig:
    cached_model = CachedModelOnlyFullLoad(
        model=torch.nn.Linear(4, 4), compute_device=torch.device("meta"), total_bytes=1, keep_ram_copy=False
    )
    return LoadedModelWithoutConfig(cache_record=CacheRecord(key="vae-key", cached_model=cached_model), cache=cache)


_ESTIMATE = WorkingMemoryEstimate(
    estimator="vae.flux2",
    version=1,
    features={"operation": "decode"},
    terms=(WorkingMemoryTerm.fitted("convolutions", "k_decode", 3 * 2**29 + 0.25, {"k_decode": 2.0}),),
    source="shipped:conv_backend=cudnn",
)


def test_a_working_memory_estimate_reserves_its_bytes_and_logs_where_they_came_from(caplog):
    cache = FakeCache()
    logger = "invokeai.backend.model_manager.load.load_base"
    # Created before the level is lowered: creating it sets the configured level, which would undo that.
    InvokeAILogger.get_logger(logger)

    with (
        caplog.at_level(logging.DEBUG, logger=logger),
        _full_load_model(cache).model_on_device(working_memory=_ESTIMATE),
    ):
        pass

    assert cache.reserved == [3 * 2**30]
    assert cache.unlock_calls == 1
    (record,) = [r for r in caplog.records if r.name == logger]
    assert "vae-key" in record.message
    assert "vae.flux2 reserves 3072 MB (shipped:conv_backend=cudnn)" in record.message


def test_an_estimate_and_a_byte_count_together_are_refused_before_locking():
    cache = FakeCache()

    with pytest.raises(ValueError, match="not both"):
        with _full_load_model(cache).model_on_device(1024, working_memory=_ESTIMATE):
            pass

    assert cache.lock_calls == 0


def test_an_estimate_passed_as_a_byte_count_says_where_it_goes():
    """The positional slot is `working_mem_bytes`; an estimate there would fail deep inside the cache instead."""
    cache = FakeCache()

    with pytest.raises(TypeError, match="working_memory="):
        with _full_load_model(cache).model_on_device(_ESTIMATE):
            pass

    assert cache.lock_calls == 0


def test_supports_partial_loading_reflects_cached_model_type():
    partial_model = CachedModelWithPartialLoad(
        model=torch.nn.Linear(4, 4), compute_device=torch.device("meta"), keep_ram_copy=False
    )
    full_model = CachedModelOnlyFullLoad(
        model=torch.nn.Linear(4, 4), compute_device=torch.device("meta"), total_bytes=1, keep_ram_copy=False
    )

    loaded_partial_model = LoadedModelWithoutConfig(CacheRecord(key="partial", cached_model=partial_model), cache=None)
    assert loaded_partial_model.supports_partial_loading
    assert not LoadedModelWithoutConfig(
        CacheRecord(key="full", cached_model=full_model), cache=None
    ).supports_partial_loading


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

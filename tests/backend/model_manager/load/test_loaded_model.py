import logging
import threading
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache import model_cache as model_cache_module
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK, integrated_cuda_admission_lock
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


def test_model_lock_pin_waits_for_cold_load_writer():
    writer_holding = threading.Event()
    release_writer = threading.Event()

    def writer() -> None:
        with MODEL_LOAD_LOCK.write_lock():
            writer_holding.set()
            release_writer.wait(timeout=10)

    class CacheThatStartsWriter(FakeCache):
        def lock(self, cache_record, working_mem_bytes, max_move_bytes=None):
            del cache_record, working_mem_bytes, max_move_bytes
            self.lock_calls += 1
            self.writer_thread = threading.Thread(target=writer)
            self.writer_thread.start()
            return True

    cached_model = CachedModelWithPartialLoad(
        model=torch.nn.Linear(4, 4), compute_device=torch.device("cpu"), keep_ram_copy=False
    )
    record = CacheRecord(key="partial", cached_model=cached_model)
    cache = CacheThatStartsWriter()
    loaded_model = LoadedModelWithoutConfig(record, cache=cache)
    pin_started = threading.Event()
    with (
        patch.object(loaded_model, "repair_required_tensors_on_device", side_effect=writer_holding.wait),
        patch.object(cached_model, "pin_streamed_weights", side_effect=pin_started.set),
    ):
        pin_thread = threading.Thread(target=loaded_model._lock_on_device, args=(None,))
        pin_thread.start()
        try:
            assert writer_holding.wait(timeout=10)
            assert not pin_started.wait(timeout=0.1), "pin registration raced a cold-load writer"
        finally:
            release_writer.set()
            pin_thread.join(timeout=10)
    cache.writer_thread.join(timeout=10)
    assert not pin_thread.is_alive() and not cache.writer_thread.is_alive()
    assert pin_started.is_set()
    loaded_model._unlock()


def test_pin_waits_for_integrated_cuda_transfer_admission(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_cache_module, "_is_integrated_cuda", lambda _device: True)
    cached_model = CachedModelWithPartialLoad(
        model=torch.nn.Linear(4, 4), compute_device=torch.device("cuda"), keep_ram_copy=False
    )
    record = CacheRecord(key="partial", cached_model=cached_model)
    record.lock()
    loaded_model = LoadedModelWithoutConfig(record, cache=None)
    transfer_holding = threading.Event()
    release_transfer = threading.Event()
    pin_started = threading.Event()

    def transfer() -> None:
        with integrated_cuda_admission_lock(torch.device("cuda")):
            transfer_holding.set()
            release_transfer.wait(timeout=10)

    transfer_thread = threading.Thread(target=transfer)
    transfer_thread.start()
    assert transfer_holding.wait(timeout=10)
    with patch.object(cached_model, "pin_streamed_weights", side_effect=pin_started.set):
        pin_thread = threading.Thread(target=loaded_model._pin_streamed_weights_of_a_trimmed_lock, args=(1,))
        pin_thread.start()
        try:
            assert not pin_started.wait(timeout=0.1), "pin registration raced an integrated transfer"
        finally:
            release_transfer.set()
            pin_thread.join(timeout=10)
    transfer_thread.join(timeout=10)
    assert not pin_thread.is_alive() and not transfer_thread.is_alive()
    assert pin_started.is_set()


@pytest.mark.parametrize("first_operation", ["pin", "transfer"])
def test_discrete_pin_and_integrated_transfer_resample_shared_host_headroom(
    first_operation: str, monkeypatch: pytest.MonkeyPatch
):
    """A discrete-device pin and integrated offload cannot spend the same host headroom concurrently."""
    monkeypatch.setattr(model_cache_module, "_is_integrated_cuda", lambda device: device.index == 0)
    cached_model = CachedModelWithPartialLoad(
        model=torch.nn.Linear(4, 4), compute_device=torch.device("cuda:1"), keep_ram_copy=False
    )
    record = CacheRecord(key="discrete", cached_model=cached_model)
    record.lock()
    loaded_model = LoadedModelWithoutConfig(record, cache=None)
    available = [10]
    samples: list[tuple[str, int]] = []
    first_started = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_sampled = threading.Event()
    second_refused = threading.Event()
    errors: list[BaseException] = []

    def spend_headroom(operation: str) -> None:
        observed = available[0]
        samples.append((operation, observed))
        if operation != first_operation:
            second_sampled.set()
        if operation == first_operation:
            first_started.set()
            assert release_first.wait(timeout=5), "first host allocation was not released"
        if observed < 7:
            second_refused.set()
            return
        available[0] -= 7

    def pin(is_second: bool = False) -> None:
        try:
            if is_second:
                second_attempted.set()
            loaded_model._pin_streamed_weights_of_a_trimmed_lock(1)
        except BaseException as error:
            errors.append(error)

    def integrated_transfer(is_second: bool = False) -> None:
        try:
            if is_second:
                second_attempted.set()
            with MODEL_LOAD_LOCK.read_lock(), integrated_cuda_admission_lock(torch.device("cuda:0")):
                spend_headroom("transfer")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(cached_model, "pin_streamed_weights", lambda: spend_headroom("pin"))
    first = threading.Thread(target=pin if first_operation == "pin" else integrated_transfer)
    second = threading.Thread(
        target=integrated_transfer if first_operation == "pin" else pin,
        kwargs={"is_second": True},
    )
    try:
        first.start()
        assert first_started.wait(timeout=5)
        second.start()
        assert second_attempted.wait(timeout=1)
        assert not second_sampled.wait(timeout=0.1), "second host allocation sampled before the first completed"
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
    finally:
        release_first.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert len(samples) == 2
    assert samples[0][0] == first_operation and samples[0][1] == 10
    assert samples[1][0] != first_operation and samples[1][1] == 3
    assert second_refused.is_set()
    record.unlock()


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

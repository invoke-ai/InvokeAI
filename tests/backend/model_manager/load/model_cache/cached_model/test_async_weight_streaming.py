"""Async weight streaming (`async_weight_streaming`) of partially loaded models.

Forwards run on the GPU only (bf16, or fp32 for the GGUF layer). Bit-identity against the synchronous path is the
correctness gate: a copy that is read before it lands, or a block reused too early, shows up as different output, not
as an error. Which path a copy took has no other observable than the streamer's counters.
"""

import logging
import mmap
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from invokeai.app.services.config import config_default
from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast import weight_streaming
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.weight_streaming import (
    set_async_weight_streaming_enabled,
    stream_to_device,
    streaming_stats,
)
from tests.backend.quantization.gguf.test_ggml_tensor import quantize_tensor

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")

DEVICE = torch.device("cuda")
DIM, HIDDEN = 512, 2048  # bf16 weights of 2 MiB: above the staging limit, so they stream only when registered.
_LOGGER = weight_streaming.logger.name


class _Mlp(torch.nn.Module):
    def __init__(self, blocks: int = 6):
        super().__init__()
        # A root buffer sits in no autocast module: the cache keeps it on the device, and repairs it there.
        self.register_buffer("scale", torch.ones(DIM))
        layers: list[torch.nn.Module] = []
        for _ in range(blocks):
            layers += [torch.nn.Linear(DIM, HIDDEN), torch.nn.GELU(), torch.nn.Linear(HIDDEN, DIM)]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) * self.scale


def _build_model() -> _Mlp:
    torch.manual_seed(0)
    model = _Mlp().to(torch.bfloat16)
    apply_custom_layers_to_model(model)
    return model


def _input(dim: int = DIM, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    return torch.randn(64, dim, device=DEVICE, dtype=dtype, generator=generator)


def _settle(cached: CachedModelWithPartialLoad, vram_bytes: int) -> None:
    """What a lock does: load, then pin outside the cache's locks (`LoadedModelWithoutConfig._lock_on_device`)."""
    cached.partial_load_to_vram(vram_bytes)
    cached.pin_streamed_weights()


def _half_loaded(model: torch.nn.Module, **kwargs) -> CachedModelWithPartialLoad:
    cached = CachedModelWithPartialLoad(model=model, compute_device=DEVICE, keep_ram_copy=True, **kwargs)
    _settle(cached, cached.total_bytes() // 2)
    assert 0 < cached.cur_vram_bytes() < cached.total_bytes()
    return cached


@torch.no_grad()
def _forward(model: torch.nn.Module, *, streaming: bool, x: torch.Tensor | None = None) -> torch.Tensor:
    set_async_weight_streaming_enabled(streaming)
    try:
        out = model(_input() if x is None else x)
        torch.cuda.synchronize()
        return out
    finally:
        set_async_weight_streaming_enabled(True)


def _cpu_storages(model: torch.nn.Module) -> list[torch.UntypedStorage]:
    return [p.untyped_storage() for p in model.parameters() if p.device.type == "cpu"]


def _stats() -> weight_streaming.StreamingStats:
    stats = streaming_stats(DEVICE)
    return stats if stats is not None else weight_streaming.StreamingStats()


def _messages(caplog: pytest.LogCaptureFixture, text: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER and text in r.getMessage()]


def _mock_logger() -> MagicMock:
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


def _cache_with_room_for_half_of(model: torch.nn.Module) -> ModelCache:
    """A cache on cuda:0 whose VRAM cap leaves room for half of ``model``. The cache counts every allocation on the
    device as in use, so the cap sits above what is allocated already."""
    total_bytes = sum(p.nbytes for p in model.parameters()) + sum(b.nbytes for b in model.buffers())
    return ModelCache(
        execution_device_working_mem_gb=0.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        max_vram_cache_size_gb=(torch.cuda.memory_allocated(DEVICE) + total_bytes / 2) / 2**30,
        execution_device="cuda:0",
        storage_device="cpu",
        logger=_mock_logger(),
        shared_cpu_weights=None,
    )


class _Streaming:
    def __init__(self, registry: weight_streaming.PinnedStorageRegistry, holders: list[weight_streaming.PinHolder]):
        self.registry = registry
        self.holders = holders

    def holds_anything(self) -> bool:
        self.registry.drain_deferred()
        return any(holder.keys for holder in self.holders)


@pytest.fixture
def streaming(monkeypatch: pytest.MonkeyPatch):
    """Feature on, with a fresh registry, streamers and log-once state. Every holder is recorded, so teardown can
    check that nothing stays pinned and release what a failing test left, keeping later tests independent."""
    registry = weight_streaming.PinnedStorageRegistry()
    holders: list[weight_streaming.PinHolder] = []

    class RecordingHolder(weight_streaming.PinHolder):
        __slots__ = ()

        def __init__(self) -> None:
            super().__init__()
            holders.append(self)

    monkeypatch.setattr(weight_streaming, "PINNED_STORAGES", registry)
    monkeypatch.setattr(weight_streaming, "PinHolder", RecordingHolder)
    monkeypatch.setattr(weight_streaming, "_streamers", {})
    monkeypatch.setattr(weight_streaming, "_logged", set())
    monkeypatch.setattr(weight_streaming, "_runtime", None)
    set_async_weight_streaming_enabled(True)
    state = _Streaming(registry, holders)
    yield state
    set_async_weight_streaming_enabled(None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    leaked = state.holds_anything()
    for holder in holders:
        registry.release(holder)
    assert not leaked, "a test left storages registered"
    assert not registry.holds_registrations(), "a storage stayed registered after every holder released it"
    # Counters a later registration decides by; a leak here would refuse or over-reclaim.
    assert registry._pinned_bytes == 0 and registry._reclaimable_bytes == 0 and not registry._reclaimable


def test_a_device_without_streaming_pins_nothing_and_says_so_once(streaming, caplog: pytest.LogCaptureFixture):
    """Runs without a GPU: the meta device stands in for MPS, XPU and the CPU."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    model = _build_model()
    cached = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=True)

    _settle(cached, cached.total_bytes() // 2)
    _settle(cached, cached.total_bytes())

    assert not streaming.holders
    assert cached.weight_streaming_reserve_bytes() == 0
    assert len(_messages(caplog, "not used on meta")) == 1


@requires_cuda
def test_off_by_default_pins_and_streams_nothing(streaming, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config_default, "get_config", lambda: config_default.InvokeAIAppConfig())
    set_async_weight_streaming_enabled(None)
    model = _build_model()

    cached = _half_loaded(model)
    with torch.no_grad():
        model(_input())

    assert not streaming.holders
    assert cached.weight_streaming_reserve_bytes() == 0
    assert streaming_stats(DEVICE) is None


@requires_cuda
def test_streamed_forward_is_bit_identical_to_the_synchronous_path(streaming, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    model = _build_model()
    cached = _half_loaded(model)
    assert all(s.is_pinned() for s in _cpu_storages(model))

    reference = _forward(model, streaming=False)
    streamed = _forward(model, streaming=True)
    streamed_again = _forward(model, streaming=True)

    assert _stats().streamed_tensors > 0
    assert torch.equal(streamed, reference)
    assert torch.equal(streamed_again, reference)
    assert len(_messages(caplog, "stream to cuda:")) == 1
    cached.release_pinned_weights()


@requires_cuda
def test_bytes_in_flight_stay_within_the_budget_while_the_gpu_lags(streaming, monkeypatch: pytest.MonkeyPatch):
    budget = 3 * 2**20
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", budget)
    model = _build_model()
    cached = _half_loaded(model)
    largest = max(p.nbytes for p in model.parameters() if p.device.type == "cpu")
    reference = _forward(model, streaming=False)

    with torch.no_grad():
        # Hold the compute stream back so the CPU runs ahead: without the bound every copy would be issued at once.
        torch.cuda._sleep(20_000_000)
        streamed = model(_input())
        torch.cuda.synchronize()

    stats = _stats()
    assert stats.bounded_waits > 0
    assert stats.max_inflight_bytes <= budget + largest
    assert torch.equal(streamed, reference)
    cached.release_pinned_weights()


def _counting_registrations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Runs the self-test, then records the size of every registration the driver gets."""
    assert weight_streaming.PINNED_STORAGES.can_register()
    runtime = weight_streaming._runtime
    assert runtime is not None
    sizes: list[int] = []
    register = runtime.register

    def counting(ptr: int, nbytes: int) -> int:
        sizes.append(nbytes)
        return register(ptr, nbytes)

    monkeypatch.setattr(runtime, "register", counting)
    return sizes


@requires_cuda
def test_pins_outlast_unloads_and_are_released_with_the_model(streaming):
    model = _build_model()
    cached = _half_loaded(model)
    offloaded = _cpu_storages(model)
    # A second lock, which brings back only the root buffer, keeps the same pins: one release still unpins everything.
    cached.partial_unload_from_vram(1)  # The root buffer, which the next load brings back first.
    _settle(cached, model.scale.nbytes)
    assert [s.data_ptr() for s in _cpu_storages(model)] == [s.data_ptr() for s in offloaded]
    assert all(s.is_pinned() for s in offloaded)

    cached.full_unload_from_vram()
    assert all(s.is_pinned() for s in offloaded), "a model moved to RAM keeps its pins for its next lock"
    cached.full_load_to_vram()
    cached.pin_streamed_weights()
    assert all(s.is_pinned() for s in offloaded), "and so does a fully resident one, for its next unload"

    cached.partial_unload_from_vram(cached.total_bytes() // 2)
    _settle(cached, 0)
    assert streaming.holds_anything()
    set_async_weight_streaming_enabled(False)
    _settle(cached, 0)
    assert not streaming.holds_anything(), "switching the setting off releases the pins on the next lock"
    set_async_weight_streaming_enabled(True)

    _settle(cached, 0)
    assert streaming.holds_anything(), "switching it back on pins again"
    cached.release_pinned_weights()
    assert not any(s.is_pinned() for s in _cpu_storages(model))

    cached.partial_unload_from_vram(1)  # Stale pins, so the next lock reaches the retired check.
    holders = len(streaming.holders)
    _settle(cached, model.scale.nbytes)
    assert not streaming.holds_anything(), "a released model must not pin again"
    assert len(streaming.holders) == holders, "nor register anything only to release it"


@requires_cuda
def test_repairing_required_weights_pins_what_an_unload_left_in_ram(streaming):
    model = _build_model()
    cached = _half_loaded(model)
    assert model.layers[0].weight.device.type == "cuda"

    # The root buffer comes first in the state dict, so this offloads it and the first weight.
    cached.partial_unload_from_vram(model.scale.nbytes + 1)
    weight = model.layers[0].weight
    assert model.scale.device.type == "cpu" and weight.device.type == "cpu"
    assert not streaming.registry.is_registered(weight.untyped_storage())

    assert cached.repair_required_tensors_on_compute_device() == 1
    cached.pin_streamed_weights()

    assert streaming.registry.is_registered(weight.untyped_storage())
    cached.release_pinned_weights()


@requires_cuda
def test_a_dropped_wrapper_releases_its_pins(streaming):
    model = _build_model()
    cached = _half_loaded(model)
    offloaded = _cpu_storages(model)

    del cached

    assert not streaming.holds_anything()
    assert not any(s.is_pinned() for s in offloaded)


@requires_cuda
def test_eviction_from_the_model_cache_releases_the_pins(streaming):
    model = _build_model()
    cache = _cache_with_room_for_half_of(model)
    try:
        cache.put("m", model)
        record = cache.get("m")
        for _ in range(2):
            with LoadedModelWithoutConfig(record, cache):
                pass
        cached = record.cached_model
        assert isinstance(cached, CachedModelWithPartialLoad)
        assert 0 < cached.cur_vram_bytes() < cached.total_bytes()
        offloaded = _cpu_storages(model)
        assert all(s.is_pinned() for s in offloaded)

        assert cache.drop_model("m") == 1

        assert not streaming.holds_anything()
        assert not any(s.is_pinned() for s in offloaded)
        # Its next load locks the weights again: said once, so a RAM cache too small for a generation shows.
        logged = cache._logger.logger  # The mock behind the cache's prefixing adapter.
        hints = [c for c in logged.method_calls if "max_cache_ram_gb" in str(c)]
        assert len(hints) == 1
    finally:
        cache.shutdown()


@requires_cuda
def test_a_streaming_lock_reserves_vram_for_the_copies_in_flight(streaming, monkeypatch: pytest.MonkeyPatch):
    budget = 4 * 2**20
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", budget)

    def resident_bytes(enabled: bool) -> int:
        set_async_weight_streaming_enabled(enabled)
        model = _build_model()
        cache = _cache_with_room_for_half_of(model)
        try:
            cache.put("m", model)
            record = cache.get("m")
            cache.lock(record, None)
            resident = record.cached_model.cur_vram_bytes()
            cache.unlock(record)
            cache.drop_model("m")
            return resident
        finally:
            cache.shutdown()

    without_streaming = resident_bytes(False)
    with_streaming = resident_bytes(True)

    # Loading picks whole tensors, so the difference may miss the budget by the small ones (biases).
    assert without_streaming - with_streaming >= budget - 2**16


@requires_cuda
def test_a_tiny_pinned_ram_cap_streams_through_the_fallbacks_identically(streaming, monkeypatch: pytest.MonkeyPatch):
    # Room for one bias: the weights copy synchronously, the other biases through staging buffers.
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", HIDDEN * 2)
    model = _build_model()
    cached = _half_loaded(model)
    reference = _forward(model, streaming=False)

    streamed = _forward(model, streaming=True)

    stats = _stats()
    assert stats.streamed_tensors > 0
    assert stats.staged_tensors > 0, "small tensors past the cap stage through pinned buffers"
    assert torch.equal(streamed, reference)
    cached.release_pinned_weights()


@requires_cuda
def test_wrappers_sharing_cpu_weights_hold_one_registration_each(streaming):
    """The multi-GPU shape: two cached wrappers (one per device) aliasing the same canonical CPU tensors."""
    store = SharedCpuWeightsStore()
    model_a, model_b = _build_model(), _build_model()
    cached_a = _half_loaded(model_a, shared_store=store, cache_key="k")
    cached_b = _half_loaded(model_b, shared_store=store, cache_key="k")
    shared = _cpu_storages(model_b)
    assert {s.data_ptr() for s in shared} <= {s.data_ptr() for s in _cpu_storages(model_a)}
    reference = _forward(model_b, streaming=False)

    cached_a.release_pinned_weights()
    cached_a.release_shared_weights()

    assert all(s.is_pinned() for s in shared)
    assert torch.equal(_forward(model_b, streaming=True), reference)
    assert _stats().streamed_tensors > 0

    cached_b.release_pinned_weights()
    cached_b.release_shared_weights()
    assert not any(s.is_pinned() for s in shared)


def _pack_back_to_back(model: torch.nn.Module, tmp_path: Path) -> None:
    """Every tensor its own storage, laid end to end from an address that is not page-aligned."""
    state_dict = model.state_dict()
    offset = 2680
    buffer = np.empty(offset + sum(t.nbytes for t in state_dict.values()), dtype=np.uint8)
    packed = {}
    for key, tensor in state_dict.items():
        view = buffer[offset : offset + tensor.nbytes]
        view[:] = tensor.contiguous().view(torch.uint8).reshape(-1).numpy()
        packed[key] = torch.from_numpy(view).view(tensor.dtype).reshape(tensor.shape)
        offset += tensor.nbytes
    model.load_state_dict(packed, assign=True)


def _load_from_safetensors(model: torch.nn.Module, tmp_path: Path) -> None:
    """``load_file`` returns the tensors back to back inside one mapping of the file, as model loaders keep them."""
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    model.load_state_dict(load_file(str(path)), assign=True)


@requires_cuda
@pytest.mark.parametrize("layout", [_pack_back_to_back, _load_from_safetensors], ids=["adjacent", "safetensors"])
def test_storages_sharing_pages_all_stream(streaming, layout, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    model = _build_model()
    layout(model, tmp_path)
    cached = _half_loaded(model)
    offloaded = _cpu_storages(model)
    reference = _forward(model, streaming=False)

    streamed = _forward(model, streaming=True)

    stats = _stats()
    assert stats.streamed_tensors == len(offloaded)
    assert stats.staged_tensors == 0
    assert torch.equal(streamed, reference)
    assert not [r for r in caplog.records if r.name == _LOGGER and r.levelno >= logging.WARNING]
    cached.release_pinned_weights()
    assert not any(s.is_pinned() for s in offloaded)


class _TensorSubclass(torch.Tensor):
    pass


@requires_cuda
def test_ineligible_tensors_fall_back_to_the_synchronous_copy(streaming):
    weight = torch.randn(HIDDEN, DIM)
    holder = weight_streaming.PinHolder()
    streaming.registry.update(holder, [weight.untyped_storage()])
    assert weight.is_pinned()

    assert stream_to_device(weight.t(), DEVICE) is None, "non-contiguous"
    assert stream_to_device(weight.as_subclass(_TensorSubclass), DEVICE) is None, "subclass"
    assert stream_to_device(torch.nn.Parameter(weight), DEVICE) is None, "requires grad with grad enabled"
    set_async_weight_streaming_enabled(False)
    assert stream_to_device(weight, DEVICE) is None, "feature off"
    set_async_weight_streaming_enabled(True)

    streamed = stream_to_device(weight, DEVICE)
    assert streamed is not None
    assert torch.equal(streamed.cpu(), weight)

    graph = torch.cuda.CUDAGraph()
    counter = torch.zeros(1, device=DEVICE)
    with torch.cuda.graph(graph):
        captured = stream_to_device(weight, DEVICE)
        counter.add_(1)
    assert captured is None, "a copy cannot be enqueued into a graph capture"

    streaming.registry.release(holder)
    assert not weight.is_pinned()


@requires_cuda
def test_a_gguf_layer_loads_and_runs_with_streaming_on(streaming):
    """GGMLTensor has no storage of its own to page-lock; its layer keeps its own transfer."""
    torch.manual_seed(0)
    ggml_layer = torch.nn.Linear(256, 64)
    ggml_layer.weight = torch.nn.Parameter(quantize_tensor(ggml_layer.weight, gguf.GGMLQuantizationType.Q8_0))
    ggml_layer.bias = torch.nn.Parameter(quantize_tensor(ggml_layer.bias, gguf.GGMLQuantizationType.Q8_0))
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), ggml_layer)
    apply_custom_layers_to_model(model)
    cached = CachedModelWithPartialLoad(model=model, compute_device=DEVICE, keep_ram_copy=True)

    _settle(cached, 0)
    x = _input(64, torch.float32)
    reference = _forward(model, streaming=False, x=x)
    streamed = _forward(model, streaming=True, x=x)

    assert torch.equal(streamed, reference)
    assert _stats().streamed_tensors > 0, "the plain layer still streams"
    plain = model[0]
    assert cached.weight_streaming_reserve_bytes() == plain.weight.nbytes + plain.bias.nbytes
    cached.release_pinned_weights()


@requires_cuda
def test_without_a_ram_copy_nothing_is_pinned_and_the_log_says_why(streaming, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    model = _build_model()
    cached = CachedModelWithPartialLoad(model=model, compute_device=DEVICE, keep_ram_copy=False)

    _settle(cached, cached.total_bytes() // 2)
    _settle(cached, cached.total_bytes() // 2)
    reference = _forward(model, streaming=False)
    streamed = _forward(model, streaming=True)

    assert not streaming.holders
    assert cached.weight_streaming_reserve_bytes() == 0
    assert torch.equal(streamed, reference)
    stats = _stats()
    assert stats.streamed_tensors == 0 and stats.staged_tensors == 0, "staging alone buys nothing"
    assert len(_messages(caplog, "keep_ram_copy_of_weights")) == 1
    assert not _messages(caplog, "stream to")


@requires_cuda
def test_a_storage_the_driver_refuses_stays_pageable_and_still_copies(streaming, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    page = mmap.PAGESIZE
    buffer = (torch.arange(64 * page, dtype=torch.int32) % 251).to(torch.uint8)
    storage = buffer.untyped_storage()
    # Register an aligned range inside the storage behind the registry's back, so registering the storage fails.
    inner = (storage.data_ptr() // page + 8) * page
    cudart = torch.cuda.cudart()
    assert int(cudart.cudaHostRegister(inner, 8 * page, 1)) == 0
    other = torch.randn(HIDDEN, DIM)
    holder = weight_streaming.PinHolder()
    try:
        streaming.registry.update(holder, [storage, other.untyped_storage()])

        assert not streaming.registry.is_registered(storage)
        assert streaming.registry.is_registered(other.untyped_storage())
        assert len(_messages(caplog, "page-locking")) == 1
        # The failure's pending error would surface in the next allocation that reaches the driver.
        torch.cuda.empty_cache()
        torch.empty(48 * 2**20, dtype=torch.uint8, device=DEVICE).fill_(1)
        copied = stream_to_device(buffer, DEVICE)
        assert copied is not None and torch.equal(copied.cpu(), buffer)
        assert _stats().staged_tensors == 1
    finally:
        streaming.registry.release(holder)
        torch.cuda.synchronize()
        assert int(cudart.cudaHostUnregister(inner)) == 0


@requires_cuda
def test_a_driver_that_rounds_registrations_to_pages_keeps_the_synchronous_copy(
    streaming, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    monkeypatch.setattr(
        weight_streaming,
        "_registration_self_test",
        lambda: "this driver does not page-lock memory by exact ranges, so neighbouring weights could not be copied",
    )
    model = _build_model()
    cached = _half_loaded(model)
    reference = _forward(model, streaming=False)

    streamed = _forward(model, streaming=True)

    assert not streaming.holds_anything()
    assert cached.weight_streaming_reserve_bytes() == 0
    assert _stats().streamed_tensors == 0
    assert torch.equal(streamed, reference)
    assert len(_messages(caplog, "exact ranges")) == 1
    cached.release_pinned_weights()


@requires_cuda
def test_an_eviction_during_the_first_pin_leaves_nothing_registered(streaming, monkeypatch: pytest.MonkeyPatch):
    model = _build_model()
    cached = CachedModelWithPartialLoad(model=model, compute_device=DEVICE, keep_ram_copy=True)
    cached.partial_load_to_vram(cached.total_bytes() // 2)
    update = streaming.registry.update

    def evicted_while_registering(holder, storages):
        # The cache evicts the record (another thread, under its own lock) before this update registers.
        cached.release_pinned_weights()
        update(holder, storages)

    monkeypatch.setattr(streaming.registry, "update", evicted_while_registering)
    cached.pin_streamed_weights()

    assert not streaming.holds_anything()
    assert not any(s.is_pinned() for s in _cpu_storages(model))


@requires_cuda
def test_weights_an_unload_left_in_ram_stream_after_the_next_lock(streaming):
    model = _build_model()
    cached = _half_loaded(model)
    cached.partial_unload_from_vram(DIM * HIDDEN * 2, keep_required_weights_in_vram=True)
    offloaded = _cpu_storages(model)
    assert not all(streaming.registry.is_registered(s) for s in offloaded), "an unload does not pin"

    _settle(cached, 0)  # A lock that moves nothing.

    assert all(streaming.registry.is_registered(s) for s in offloaded)
    _forward(model, streaming=True)
    assert _stats().streamed_tensors == len(offloaded)
    cached.release_pinned_weights()


@requires_cuda
def test_models_taking_turns_in_vram_register_their_weights_once(streaming, monkeypatch: pytest.MonkeyPatch):
    """A text encoder and a transformer that do not fit together evict each other on every generation."""
    sizes = _counting_registrations(monkeypatch)
    turns = [
        (model, CachedModelWithPartialLoad(model=model, compute_device=DEVICE, keep_ram_copy=True))
        for model in (_build_model(), _build_model())
    ]
    streamed: list[torch.UntypedStorage] = []
    first_pins = 0

    for turn in range(3):
        for model, cached in turns:
            for _, other in turns:
                other.full_unload_from_vram()
            _settle(cached, cached.total_bytes() // 2)
            _forward(model, streaming=True)
            cached.mark_streamed_weights_idle()  # What the cache does when the lock ends.
            if turn == 0:
                streamed += _cpu_storages(model)
        if turn == 0:
            first_pins = len(sizes)

    assert first_pins > 0 and len(sizes) == first_pins
    assert all(s.is_pinned() for s in streamed)
    for _, cached in turns:
        cached.release_pinned_weights()


@requires_cuda
def test_an_idle_model_gives_a_streaming_one_only_the_room_it_lacks(streaming, monkeypatch: pytest.MonkeyPatch):
    sizes = _counting_registrations(monkeypatch)
    idle_model, busy_model = _build_model(), _build_model()
    cache = _cache_with_room_for_half_of(idle_model)
    try:
        cache.put("idle", idle_model)
        record = cache.get("idle")
        # The streaming reserve takes the half the cap leaves, so only the root buffer stays resident.
        with LoadedModelWithoutConfig(record, cache):
            resident = record.cached_model.cur_vram_bytes()
        # The end of that lock marked its weights idle.
        idle_streamed = _cpu_storages(idle_model)
        idle_bytes = sum(s.nbytes() for s in idle_streamed)
        largest = max(s.nbytes() for s in idle_streamed)
        assert all(s.is_pinned() for s in idle_streamed)
        busy = CachedModelWithPartialLoad(model=busy_model, compute_device=DEVICE, keep_ram_copy=True)
        busy.partial_load_to_vram(busy.total_bytes() // 2)
        busy_streamed = _cpu_storages(busy_model)
        busy_bytes = sum(s.nbytes() for s in busy_streamed)
        monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", idle_bytes + busy_bytes // 2)

        busy.pin_streamed_weights()

        assert all(s.is_pinned() for s in busy_streamed), "the streaming model gets the room it lacks"
        kept = sum(s.nbytes() for s in idle_streamed if s.is_pinned())
        assert idle_bytes - busy_bytes // 2 - largest <= kept <= idle_bytes - busy_bytes // 2, "and only that much"

        busy.mark_streamed_weights_idle()
        busy.full_unload_from_vram()  # Gives the cache back the VRAM it saw.
        registered_before = sum(sizes)
        with LoadedModelWithoutConfig(record, cache):
            assert record.cached_model.cur_vram_bytes() == resident
            assert all(s.is_pinned() for s in idle_streamed), "its next lock pins again"
        assert sum(sizes) - registered_before == idle_bytes - kept, "paying only for what it gave"
        busy.release_pinned_weights()
        cache.drop_model("idle")
    finally:
        cache.shutdown()


@requires_cuda
def test_only_the_last_unlock_makes_the_pins_reclaimable(streaming, monkeypatch: pytest.MonkeyPatch):
    """Two holders of one record at once, as a session and the image index worker."""
    model = _build_model()
    cache = _cache_with_room_for_half_of(model)
    other = weight_streaming.PinHolder()
    taker = torch.empty(2 * 2**20, dtype=torch.uint8)
    try:
        cache.put("m", model)
        record = cache.get("m")
        with LoadedModelWithoutConfig(record, cache):
            streamed = _cpu_storages(model)
            monkeypatch.setattr(
                weight_streaming, "PINNED_RAM_CAP_BYTES", sum(s.nbytes() for s in streamed) + taker.nbytes // 2
            )
            with LoadedModelWithoutConfig(record, cache):
                pass
            streaming.registry.update(other, [taker.untyped_storage()])
            assert all(s.is_pinned() for s in streamed) and not taker.is_pinned(), "the model is still locked"

        streaming.registry.update(other, [taker.untyped_storage()])

        assert taker.is_pinned(), "after the last unlock its pins make room"
        streaming.registry.release(other)
        cache.drop_model("m")
    finally:
        cache.shutdown()


@requires_cuda
def test_shared_weights_one_device_still_streams_stay_pinned_when_the_other_goes_idle(
    streaming, monkeypatch: pytest.MonkeyPatch
):
    store = SharedCpuWeightsStore()
    model_a, model_b = _build_model(), _build_model()
    cached_a = _half_loaded(model_a, shared_store=store, cache_key="k")
    cached_b = _half_loaded(model_b, shared_store=store, cache_key="k")
    shared = _cpu_storages(model_b)
    assert {s.data_ptr() for s in shared} == {s.data_ptr() for s in _cpu_storages(model_a)}
    other = weight_streaming.PinHolder()
    taker = torch.empty(2 * 2**20, dtype=torch.uint8)
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", sum(s.nbytes() for s in shared) + taker.nbytes // 2)

    cached_a.mark_streamed_weights_idle()
    streaming.registry.update(other, [taker.untyped_storage()])

    assert all(s.is_pinned() for s in shared) and not taker.is_pinned()
    streaming.registry.release(other)
    for cached in (cached_a, cached_b):
        cached.release_pinned_weights()
        cached.release_shared_weights()


@requires_cuda
def test_weights_a_lock_moves_to_ram_stream_for_the_rest_of_it(streaming, monkeypatch: pytest.MonkeyPatch):
    """A node that makes room within its lock, as Wan's second expert and Ideogram 4's second branch do."""
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", 2 * 2**20)  # Leaves weights resident.
    model = _build_model()
    cache = _cache_with_room_for_half_of(model)
    try:
        cache.put("m", model)
        loaded = LoadedModelWithoutConfig(cache.get("m"), cache)
        with loaded:
            resident = sum(p.nbytes for p in model.parameters() if p.device.type == "cuda")
            assert resident > 0

            loaded.unload_from_vram(resident, keep_required_weights_in_vram=True)

            assert all(p.device.type == "cpu" for p in model.parameters())
            assert all(s.is_pinned() for s in _cpu_storages(model))
        cache.drop_model("m")
    finally:
        cache.shutdown()


@requires_cuda
def test_a_model_locked_again_keeps_its_pins_when_another_reaches_the_cap(streaming, monkeypatch: pytest.MonkeyPatch):
    running, other = _build_model(), _build_model()
    cached_running = _half_loaded(running)
    cached_running.mark_streamed_weights_idle()
    _settle(cached_running, 0)  # Its next lock, which moves nothing.
    streamed = _cpu_storages(running)
    streamed_bytes = sum(s.nbytes() for s in streamed)
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", streamed_bytes + streamed_bytes // 2)

    cached_other = _half_loaded(other)

    assert all(s.is_pinned() for s in streamed)
    assert not all(s.is_pinned() for s in _cpu_storages(other)), "what does not fit copies synchronously"
    cached_running.release_pinned_weights()
    cached_other.release_pinned_weights()


@requires_cuda
def test_a_model_reuses_the_room_of_its_own_pins_that_moved_to_vram(streaming, monkeypatch: pytest.MonkeyPatch):
    model = _build_model()
    cached = CachedModelWithPartialLoad(model=model, compute_device=DEVICE, keep_ram_copy=True)
    cached.partial_load_to_vram(cached.total_bytes() // 2)
    first = _cpu_storages(model)
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", sum(s.nbytes() for s in first) * 2 // 3)
    cached.pin_streamed_weights()
    assert not all(s.is_pinned() for s in first), "the cap leaves part of it pageable"

    _settle(cached, cached.total_bytes() // 4)  # More of it in VRAM: a shorter tail streams.

    tail = _cpu_storages(model)
    assert sum(s.nbytes() for s in tail) <= weight_streaming.PINNED_RAM_CAP_BYTES, "the tail fits by itself"
    assert all(s.is_pinned() for s in tail), "the pins now in VRAM made room for it"
    cached.release_pinned_weights()


@requires_cuda
def test_only_weights_of_the_ram_copy_are_pinned(streaming):
    model = _build_model()
    cached = _half_loaded(model)
    layer = model.layers[-1]
    assert layer.weight.device.type == "cpu" and layer.weight.is_pinned()
    # A holder casting a layer in place, as the SD VAE nodes upcast the VAE.
    layer.weight.data = layer.weight.data.float()
    cast = layer.weight.untyped_storage()
    cached.partial_unload_from_vram(1)  # The root buffer: the next lock pins again.

    _settle(cached, model.scale.nbytes)

    assert not streaming.registry.is_registered(cast)
    assert all(streaming.registry.is_registered(s) for s in _cpu_storages(model) if s.data_ptr() != cast.data_ptr())
    cached.restore_counted_dtypes()
    cached.release_pinned_weights()


@requires_cuda
def test_a_lock_that_moves_nothing_pins_again_what_was_reclaimed(streaming, monkeypatch: pytest.MonkeyPatch):
    """A lock that moves nothing must still pin again what another model reclaimed since the last one."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.Linear(256, 64))
    apply_custom_layers_to_model(model)
    cached = CachedModelWithPartialLoad(model=model, compute_device=DEVICE, keep_ram_copy=True)
    _settle(cached, 0)
    storages = _cpu_storages(model)
    assert cached.cur_vram_bytes() == 0 and all(s.is_pinned() for s in storages)
    cached.mark_streamed_weights_idle()
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", 256 * 2**10)
    other = weight_streaming.PinHolder()
    taker = torch.empty(200 * 2**10, dtype=torch.uint8)
    streaming.registry.update(other, [taker.untyped_storage()])
    assert taker.is_pinned() and not all(s.is_pinned() for s in storages)
    streaming.registry.idle(other)

    _settle(cached, 0)
    _forward(model, streaming=True, x=torch.randn(4, 64, device=DEVICE))

    assert all(s.is_pinned() for s in storages)
    assert _stats().streamed_tensors == len(storages)
    streaming.registry.release(other)
    cached.release_pinned_weights()


class _PageRoundingRuntime:
    """Refuses the second registration of the self-test's neighbouring pair, as a driver that rounds registrations to
    pages would: by really registering over the range it still holds, so the refusal is a real error of the real
    runtime. Tracks what it holds, so the test can check that the self-test leaves nothing registered."""

    def __init__(self, real) -> None:
        self._real = real
        self.held: set[int] = set()

    def register(self, ptr: int, nbytes: int) -> int:
        if self.held:
            return self._real.register(next(iter(self.held)), nbytes)
        error = self._real.register(ptr, nbytes)
        if error == 0:
            self.held.add(ptr)
        return error

    def unregister(self, ptr: int) -> int:
        self.held.discard(ptr)
        return self._real.unregister(ptr)


@requires_cuda
@pytest.mark.parametrize("failure", ["page-rounding driver", "no runtime library found"])
def test_a_failing_self_test_leaves_no_error_for_the_next_allocation(
    streaming, monkeypatch: pytest.MonkeyPatch, failure: str
):
    """The self-test runs inside a model load; an error it left would fail that load's next allocation, and a
    registration it left would outlive its buffer and break copies from whatever the allocator puts there next."""
    runtime = None
    if failure == "page-rounding driver":
        runtime = _PageRoundingRuntime(weight_streaming._find_runtime())
        monkeypatch.setattr(weight_streaming, "_find_runtime", lambda: runtime)
    else:
        monkeypatch.setattr(weight_streaming, "_find_runtime", lambda: None)
    # Free room inside a live segment, as during a model load: allocations served from it never reach cudaMalloc, so
    # they cannot absorb a pending error. The first allocation below comes from there; the second needs a new segment,
    # and cudaMalloc is where a pending error would surface.
    torch.cuda.empty_cache()
    segment = torch.empty(160 * 2**20, dtype=torch.uint8, device=DEVICE)
    del segment
    live = torch.empty(16 * 2**20, dtype=torch.uint8, device=DEVICE)

    assert weight_streaming._registration_self_test() is not None
    assert weight_streaming._runtime is None
    assert runtime is None or not runtime.held

    torch.empty(64 * 2**20, dtype=torch.uint8, device=DEVICE).fill_(1)
    torch.empty(256 * 2**20, dtype=torch.uint8, device=DEVICE).fill_(1)
    del live


@requires_cuda
def test_the_end_of_a_streaming_lock_returns_the_copies_cached_for_the_side_stream(streaming):
    model = _build_model()
    cache = _cache_with_room_for_half_of(model)
    try:
        cache.put("m", model)
        loaded = LoadedModelWithoutConfig(cache.get("m"), cache)
        with loaded:
            _forward(model, streaming=True)
            assert _stats().streamed_tensors > 0
            torch.cuda.synchronize()
            cached_while_locked = torch.cuda.memory_reserved(DEVICE) - torch.cuda.memory_allocated(DEVICE)
        cached_after = torch.cuda.memory_reserved(DEVICE) - torch.cuda.memory_allocated(DEVICE)
        cache.drop_model("m")
    finally:
        cache.shutdown()

    # Blocks in segments that still hold live tensors stay; the copies' blocks go.
    assert cached_after < cached_while_locked


class _InvisibleRuntime:
    """A runtime copy torch does not use: its registrations succeed, but torch does not see the memory as pinned."""

    def register(self, ptr: int, nbytes: int) -> int:
        return 0

    def unregister(self, ptr: int) -> int:
        return 0


class _StuckRuntime:
    """Registrations it reports cannot be undone. Registers nothing for real, so no memory stays page-locked."""

    def register(self, ptr: int, nbytes: int) -> int:
        return 0

    def unregister(self, ptr: int) -> int:
        return 1


def test_a_self_test_buffer_that_cannot_be_unpinned_is_never_freed(monkeypatch: pytest.MonkeyPatch):
    stranded: list[torch.Tensor] = []
    monkeypatch.setattr(weight_streaming, "_self_test_stranded", stranded)
    monkeypatch.setattr(weight_streaming, "_find_runtime", _StuckRuntime)
    monkeypatch.setattr(weight_streaming, "_runtime", None)

    assert weight_streaming._registration_self_test() == "unpinning a test buffer failed"

    # The whole-storage buffer is the one left registered.
    assert any(t.nbytes == 2 * weight_streaming._PAGE_SIZE for t in stranded)


class _LastUnpinFailsRuntime:
    """Unregisters for real, but reports the self-test's last unregistration, on its success path, as failed."""

    def __init__(self, real) -> None:
        self._real = real
        self._unregistered = 0

    def register(self, ptr: int, nbytes: int) -> int:
        return self._real.register(ptr, nbytes)

    def unregister(self, ptr: int) -> int:
        self._unregistered += 1
        error = self._real.unregister(ptr)
        return 1 if self._unregistered == 3 else error


@requires_cuda
def test_a_self_test_whose_last_unpin_fails_does_not_pass(streaming, monkeypatch: pytest.MonkeyPatch):
    runtime = _LastUnpinFailsRuntime(weight_streaming._find_runtime())
    monkeypatch.setattr(weight_streaming, "_find_runtime", lambda: runtime)
    monkeypatch.setattr(weight_streaming, "_self_test_stranded", [])

    assert weight_streaming._registration_self_test() == "unpinning a test buffer failed"
    assert weight_streaming._runtime is None


@requires_cuda
def test_a_runtime_whose_registrations_torch_cannot_see_disables_registration(
    streaming, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(weight_streaming, "_find_runtime", _InvisibleRuntime)

    reason = weight_streaming._registration_self_test()

    assert reason is not None and "not the one torch uses" in reason
    assert weight_streaming._runtime is None


def test_the_runtime_torch_links_is_preferred_over_another_wheels_copy(monkeypatch: pytest.MonkeyPatch):
    """Linux layout: torch's runtime sits in the ``nvidia`` wheel, another package may bring its own copy."""
    site = os.path.join(os.sep, "venv", "site-packages")
    other = os.path.join(site, "aaa_other", "lib", "libcudart.so.12")
    torchs = os.path.join(site, "nvidia", "cu13", "lib", "libcudart.so.13")
    maps = [SimpleNamespace(path=other), SimpleNamespace(path=torchs)]
    monkeypatch.setattr(weight_streaming.psutil, "Process", lambda: SimpleNamespace(memory_maps=lambda: maps))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    opened: list[str] = []
    monkeypatch.setattr(weight_streaming.ctypes, "CDLL", lambda path: opened.append(path) or MagicMock())

    assert weight_streaming._find_runtime() is not None
    assert opened == [torchs]

"""The last `ModelCache.unlock` returns weights a holder cast in place to the dtypes the cache counts them in.

The SD VAE nodes cast the cached VAE to the dtype they run, fp16 to fp32 for an upcast, and the cast reaches the weights
on the CPU as well as those on the compute device. The cache counts every weight in the dtype it was cached with, so a
weight left in fp32 holds more than the cache charges: the next load moves it into VRAM past the room the cache made,
and the next decode's estimate, reading an fp32 VAE, reserves nothing for its upcast.

Partial loading runs on dedicated VRAM only. The meta device stands in for it here so the tests run on any machine: it
keeps every tensor's dtype and size but no data, so values are compared wherever a weight holds data.
"""

import logging
import threading
from unittest.mock import MagicMock, patch

import accelerate
import pytest
import torch

from invokeai.backend.model_manager.load.model_cache import model_cache as model_cache_module
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.util.calc_tensor_size import calc_tensor_size
from tests.backend.model_manager.load.model_cache.cached_model.utils import TiedWeightsModule

devices = pytest.mark.parametrize(
    "device",
    [
        "meta",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")),
    ],
)
keep_ram_copy = pytest.mark.parametrize("keep_ram_copy", [True, False])
partial_loading = pytest.mark.parametrize("partial_loading", [True, False], ids=["partial", "full"])


@pytest.fixture
def make_cache():
    caches: list[ModelCache] = []

    def make(keep_ram_copy: bool, enable_partial_loading: bool = True) -> ModelCache:
        logger = MagicMock()
        logger.getEffectiveLevel.return_value = logging.INFO
        cache = ModelCache(
            execution_device_working_mem_gb=1.0,
            enable_partial_loading=enable_partial_loading,
            keep_ram_copy_of_weights=keep_ram_copy,
            execution_device="cpu",
            storage_device="cpu",
            logger=logger,
        )
        caches.append(cache)
        return cache

    yield make
    for cache in caches:
        cache.shutdown()


def _put(cache: ModelCache, model: torch.nn.Module, partial_loading: bool):
    """Cache `model` and return its record: for partial loading on the meta stand-in for VRAM, else to run on the CPU."""
    if partial_loading:
        with patch.object(model_cache_module, "device_has_dedicated_vram", return_value=True):
            cache.put("vae", model, execution_device=torch.device("meta"))
    else:
        cache.put("vae", model)
    record = cache.get("vae")
    assert isinstance(record.cached_model, CachedModelWithPartialLoad if partial_loading else CachedModelOnlyFullLoad)
    return record


def _bytes_on(model: torch.nn.Module, device_type: str) -> int:
    """The bytes the model's weights hold on a device, each storage once however many names share it. Storages are told
    apart by identity: the meta device has no data pointers to compare."""
    storages = {
        t.untyped_storage()._cdata: calc_tensor_size(t)
        for t in model.state_dict().values()
        if t.device.type == device_type
    }
    return sum(storages.values())


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: t.clone() for key, t in model.state_dict().items() if t.device.type != "meta"}


def _is_tied(model: TiedWeightsModule) -> bool:
    return model.head.weight.untyped_storage()._cdata == model.embed.weight.untyped_storage()._cdata


def _assert_at_the_counted_state(
    model: TiedWeightsModule,
    cached_model: CachedModelWithPartialLoad | CachedModelOnlyFullLoad,
    device: str,
    snapshot: dict[str, torch.Tensor],
) -> None:
    """Every weight in the dtype the cache counts and at the value it was cached with, the tied matrix one tensor, the
    CPU weights the RAM copy when there is one, and the bytes on each device what the cache reports."""
    state_dict = model.state_dict()
    assert {t.dtype for t in state_dict.values()} == {torch.float16}
    for key, before in snapshot.items():
        if state_dict[key].device.type != "meta":
            assert torch.equal(state_dict[key].cpu(), before.cpu()), key
    assert _is_tied(model)
    cpu_state_dict = cached_model.get_cpu_state_dict()
    if cpu_state_dict is not None:
        for key, t in state_dict.items():
            if t.device.type == "cpu":
                assert t.data_ptr() == cpu_state_dict[key].data_ptr(), key
    on_device = _bytes_on(model, device) if device != "cpu" else 0
    assert on_device == cached_model.cur_vram_bytes()
    assert on_device + _bytes_on(model, "cpu") == cached_model.total_bytes()


@devices
@keep_ram_copy
def test_the_last_unlock_returns_a_partially_resident_model_to_its_counted_state(
    make_cache, device: str, keep_ram_copy: bool
):
    """The cast reaches the weights on both devices and unties the tied matrix, which `Module.to(dtype)` casts name by
    name: restored name by name, the matrix took twice its count in RAM and then in VRAM. Each weight is restored where
    it is, so the VRAM count the lock took stays true."""
    cache = make_cache(keep_ram_copy)
    model = TiedWeightsModule().to(torch.float16)
    with patch.object(model_cache_module, "device_has_dedicated_vram", return_value=True):
        cache.put("vae", model, execution_device=torch.device(device))
    record = cache.get("vae")
    cached_model = record.cached_model
    assert isinstance(cached_model, CachedModelWithPartialLoad)
    # The untied linear on the device, the tied matrix on the CPU.
    cached_model.partial_load_to_vram(calc_tensor_size(model.linear.weight) + calc_tensor_size(model.linear.bias))
    cached_model.cur_vram_bytes()  # Counted now, as `lock()` counts it before it loads.
    placement = {key: t.device.type for key, t in model.state_dict().items()}
    snapshot = _snapshot(model)
    cache.lock_in_ram(record)
    model.to(dtype=torch.float32)

    cache.unlock(record)

    assert {key: t.device.type for key, t in model.state_dict().items()} == placement
    _assert_at_the_counted_state(model, cached_model, device, snapshot)
    cached_model.partial_load_to_vram(cached_model.total_bytes())
    assert cached_model.cur_vram_bytes() == cached_model.total_bytes()
    _assert_at_the_counted_state(model, cached_model, device, snapshot)
    if device == "meta" and not keep_ram_copy:
        # Without a RAM copy an offload copies the weights off the device, and the meta device holds no data.
        return
    cached_model.partial_unload_from_vram(calc_tensor_size(model.linear.weight))
    assert model.embed.weight.device.type == "cpu" and model.linear.weight.device.type == device
    _assert_at_the_counted_state(model, cached_model, device, snapshot)


@devices
@keep_ram_copy
def test_the_last_unlock_returns_a_fully_loaded_model_to_its_counted_state(
    make_cache, device: str, keep_ram_copy: bool
):
    cache = make_cache(keep_ram_copy, enable_partial_loading=False)
    model = TiedWeightsModule().to(torch.float16)
    cache.put("vae", model, execution_device=torch.device(device))
    record = cache.get("vae")
    cached_model = record.cached_model
    assert isinstance(cached_model, CachedModelOnlyFullLoad)
    snapshot = _snapshot(model)
    cached_model.full_load_to_vram()
    cache.lock_in_ram(record)
    model.to(dtype=torch.float32)

    cache.unlock(record)

    _assert_at_the_counted_state(model, cached_model, device, snapshot)
    if device == "meta" and not keep_ram_copy:
        # Without a RAM copy an offload copies the weights off the device, and the meta device holds no data.
        return
    cached_model.full_unload_from_vram()
    _assert_at_the_counted_state(model, cached_model, device, snapshot)


@keep_ram_copy
def test_the_last_unlock_returns_a_cpu_only_model_to_its_counted_state(make_cache, keep_ram_copy: bool):
    """A model that runs on the CPU is never moved, so nothing but the unlock would ever put it back."""
    cache = make_cache(keep_ram_copy)
    model = TiedWeightsModule().to(torch.float16)
    record = _put(cache, model, partial_loading=False)
    snapshot = _snapshot(model)
    cache.lock(record, None)
    model.to(dtype=torch.float32)

    cache.unlock(record)

    _assert_at_the_counted_state(model, record.cached_model, "cpu", snapshot)


class _MixedDtypeModule(TiedWeightsModule):
    """Weights counted in two dtypes, and a persistent buffer."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(64))


@partial_loading
def test_every_weight_returns_to_its_own_counted_dtype_buffers_included(make_cache, partial_loading: bool):
    cache = make_cache(keep_ram_copy=False, enable_partial_loading=partial_loading)
    model = _MixedDtypeModule().to(torch.float16)
    model.linear.to(torch.float32)
    counted = {key: t.dtype for key, t in model.state_dict().items()}
    record = _put(cache, model, partial_loading)
    if partial_loading:
        record.cached_model.partial_load_to_vram(
            calc_tensor_size(model.linear.weight) + calc_tensor_size(model.linear.bias)
        )
    cache.lock_in_ram(record)
    model.to(dtype=torch.bfloat16)

    cache.unlock(record)

    assert {key: t.dtype for key, t in model.state_dict().items()} == counted


def test_a_model_cast_back_in_part_is_restored_in_full(make_cache):
    """The upcast encode casts the decoder's mid-block back to the VAE's dtype before it runs, so at the unlock some
    weights already hold their counted dtype and the rest do not."""
    cache = make_cache(keep_ram_copy=False)
    model = TiedWeightsModule().to(torch.float16)
    record = _put(cache, model, partial_loading=True)
    cache.lock_in_ram(record)
    model.to(dtype=torch.float32)
    model.linear.to(torch.float16)

    cache.unlock(record)

    assert {t.dtype for t in model.state_dict().values()} == {torch.float16}
    assert _is_tied(model)


def test_an_inner_unlock_keeps_the_cast_an_outer_holder_runs_with(make_cache):
    cache = make_cache(keep_ram_copy=True)
    model = TiedWeightsModule().to(torch.float16)
    record = _put(cache, model, partial_loading=False)
    cache.lock(record, None)
    model.to(dtype=torch.float32)
    cache.lock(record, None)

    cache.unlock(record)
    assert {p.dtype for p in model.parameters()} == {torch.float32}

    cache.unlock(record)
    assert {p.dtype for p in model.parameters()} == {torch.float16}


@partial_loading
@keep_ram_copy
def test_a_restore_during_a_model_construction_keeps_real_weights(
    make_cache, partial_loading: bool, keep_ram_copy: bool
):
    """A construction on another thread routes every `register_parameter` to the meta device while it runs
    (`accelerate.init_empty_weights` patches it process-wide), and an unlock does not wait for one."""
    cache = make_cache(keep_ram_copy, enable_partial_loading=partial_loading)
    model = TiedWeightsModule().to(torch.float16)
    record = _put(cache, model, partial_loading)
    cache.lock_in_ram(record)
    model.to(dtype=torch.float32)
    constructing, construction_done = threading.Event(), threading.Event()

    def construction() -> None:
        with accelerate.init_empty_weights():
            constructing.set()
            construction_done.wait(timeout=10)

    builder = threading.Thread(target=construction, daemon=True)
    builder.start()
    try:
        assert constructing.wait(timeout=10)
        cache.unlock(record)
    finally:
        construction_done.set()
        builder.join(timeout=10)

    assert {t.device.type for t in model.state_dict().values()} == {"cpu"}
    assert {t.dtype for t in model.state_dict().values()} == {torch.float16}
    assert _is_tied(model)


def test_a_restore_that_fails_drops_the_entry_and_lets_the_run_finish(make_cache, monkeypatch: pytest.MonkeyPatch):
    """Partly restored, the model no longer matches its counts, as after a failed move. The unlock runs in the `finally`
    of a run whose result is not saved yet, so it warns instead of raising."""
    cache = make_cache(keep_ram_copy=True)
    record = _put(cache, TiedWeightsModule().to(torch.float16), partial_loading=False)
    cache.lock(record, None)
    monkeypatch.setattr(record.cached_model, "restore_counted_dtypes", MagicMock(side_effect=RuntimeError("no room")))
    monkeypatch.setattr(cache, "_logger", MagicMock())

    cache.unlock(record)

    assert not record.is_locked
    cache._logger.warning.assert_called_once()
    with pytest.raises(IndexError):
        cache.get("vae")

"""The working-memory probe's windows, measured on a scripted device through the model loader's real lock path.

`FakeDevice` stands in for a GPU with a caching allocator: live tensors (allocated), the blocks the allocator holds
(reserved = allocated + cached), what the driver counts, peaks since the last reset, and a release that returns the
cached blocks when `TorchDevice.empty_cache` runs -- as a real device does. Each test scripts allocations through it and
reads the `MeasuredWindow` the probe recorded; expected values follow from the script alone. The probe reads it through
`FakeCounters`, or with `cuda_device` through the real `counters.DeviceCounters` and torch's CUDA functions. Windows
lock their models in `FakeCache`, or in `CacheOnFakeDevice`, the real model cache, where its own offloads and evictions
are under test.
"""

import gc
import logging
import sys
import threading
import weakref
from contextlib import contextmanager
from typing import Callable, Iterator, Optional
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.util import devices
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.working_memory import counters, measurement, probe
from invokeai.backend.working_memory.counters import CounterReading, Method
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.probe import MeasuredWindow, ProbeScope, calibration_scope

GIB = 2**30
MIB = 2**20


class FakeDevice:
    """A GPU as the probe reads it. `retained_pool` makes it behave like HIP under Windows: released blocks stay counted
    by the driver and serve the next growth of the allocator before the driver grows."""

    def __init__(self, *, method: Method = "windows-d3dkmt", driver: bool = True, retained_pool: bool = False):
        self.method: Method = method
        self.has_driver = driver
        self.retained_pool = retained_pool
        self.readable = True
        self.allocated = 0
        self.cached = 0
        self.retained = 0
        self.outside = 0
        """What the driver counts beyond the allocator's blocks: library workspaces, kernel modules."""
        self.peak_reserved = 0
        self.peak_allocated = 0
        self.num_ooms = 0
        self.num_alloc_retries = 0
        self.paged: list[int] = [0]
        self.device_used_by_others: Optional[int] = None
        self.releases = 0
        self.fail_on_release: Optional[int] = None
        self.fail_next_read = False

    @property
    def reserved(self) -> int:
        return self.allocated + self.cached

    def alloc(self, n: int, *, reserve: Optional[int] = None) -> None:
        """A live tensor of `n` bytes, from cached blocks first. `reserve` overrides how much the allocator grows."""
        from_cache = min(self.cached, n)
        growth = n - from_cache if reserve is None else reserve
        self.cached -= from_cache
        self.retained -= min(self.retained, growth)
        self.allocated += n
        # A reserve override (cudaMallocAsync) grows the pool by less than the tensor; cached absorbs the difference.
        self.cached += growth - (n - from_cache)
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)

    def free(self, n: int) -> None:
        self.allocated -= n
        self.cached += n

    def reset_peaks(self) -> None:
        self.peak_reserved = self.reserved
        self.peak_allocated = self.allocated

    def memory_stats(self) -> dict[str, int]:
        """What `torch.cuda.memory_stats` reports for this device."""
        return {
            "reserved_bytes.all.current": self.reserved,
            "allocated_bytes.all.current": self.allocated,
            "reserved_bytes.all.peak": self.peak_reserved,
            "allocated_bytes.all.peak": self.peak_allocated,
            "num_ooms": self.num_ooms,
            "num_alloc_retries": self.num_alloc_retries,
        }

    def reading(self) -> CounterReading:
        if self.fail_next_read:
            self.fail_next_read = False
            raise RuntimeError("the driver did not answer")
        if not self.readable:
            return CounterReading(None, None, None, None, None, None, None, None, None, None, "torch-only")
        paged = self.paged.pop(0) if len(self.paged) > 1 else self.paged[0]
        return CounterReading(
            reserved=self.reserved,
            allocated=self.allocated,
            peak_reserved=self.peak_reserved,
            peak_allocated=self.peak_allocated,
            num_ooms=self.num_ooms,
            num_alloc_retries=self.num_alloc_retries,
            driver=self.reserved + self.retained + self.outside if self.has_driver else None,
            peak_driver=None,
            paged=paged,
            device_used=None if self.device_used_by_others is None else self.reserved + self.device_used_by_others,
            method=self.method if self.has_driver or self.device_used_by_others is not None else "torch-only",
        )

    # The caching allocator's side of `TorchDevice.empty_cache`.
    def before_empty_cache(self) -> None:
        pass

    def after_empty_cache(self) -> None:
        self.releases += 1
        if self.releases == self.fail_on_release:
            raise RuntimeError("the device context is sick")
        if self.retained_pool:
            self.retained += self.cached
        self.cached = 0


class FakeCounters:
    """`counters.DeviceCounters` over a `FakeDevice`."""

    def __init__(self, device: FakeDevice):
        self.device = device
        self.closed = False

    def read(self) -> CounterReading:
        return self.device.reading()

    def reset_peaks(self) -> None:
        self.device.reset_peaks()

    def synchronize(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeCachedModel:
    """A model's weights as the cache sees them: a size, how much of it is on the device, and where it computes."""

    def __init__(self, total_bytes: int, device: str):
        self.total = total_bytes
        self.resident = 0
        self.compute_device = torch.device(device)
        self.model = object()

    def total_bytes(self) -> int:
        return self.total

    def cur_vram_bytes(self) -> int:
        return self.resident

    def get_cpu_state_dict(self) -> None:
        return None


class FakeCache:
    """A model cache on a `FakeDevice`: a lock puts the model's weights there, through a staging buffer of
    `staging_bytes`, after offloading the models named in `offloads`; `partial` caps a model's residency."""

    def __init__(self, device: FakeDevice, staging_bytes: int = 0):
        self.device = device
        self.staging_bytes = staging_bytes
        self.reserved: list[Optional[int]] = []
        self.offloads: dict[str, list[CacheRecord]] = {}
        self.partial: dict[str, int] = {}
        self.records: dict[str, CacheRecord] = {}
        self.in_use_at_lock: list[int] = []
        """What the device had allocated when each lock began."""
        self._locks: dict[str, int] = {}

    def offload_unlocked_models_from_vram(self, keep_cache_keys=()) -> int:
        """`ModelCache.offload_unlocked_models_from_vram`: every unlocked record but the kept ones leaves whole."""
        freed = 0
        for key, record in self.records.items():
            if key not in keep_cache_keys and not self._locks.get(key) and record.cached_model.resident:
                self.device.free(record.cached_model.resident)
                freed += record.cached_model.resident
                record.cached_model.resident = 0
        return freed

    def lock(self, record: CacheRecord, working_mem_bytes: Optional[int], max_move_bytes: Optional[int] = None) -> bool:
        self.reserved.append(working_mem_bytes)
        self.in_use_at_lock.append(self.device.allocated)
        for victim in self.offloads.get(record.key, []):
            self.device.free(victim.cached_model.resident)
            victim.cached_model.resident = 0
        model = record.cached_model
        if model.compute_device.type != "cpu":
            target = self.partial.get(record.key, model.total)
            if model.resident < target:
                self.device.alloc(self.staging_bytes)
                self.device.alloc(target - model.resident)
                self.device.free(self.staging_bytes)
                model.resident = target
        self._locks[record.key] = self._locks.get(record.key, 0) + 1
        return True

    def lock_in_ram(self, record: CacheRecord) -> None:
        self._locks[record.key] = self._locks.get(record.key, 0) + 1

    def unlock(self, record: CacheRecord) -> None:
        self._locks[record.key] -= 1

    def locks_held(self) -> int:
        return sum(self._locks.values())


class CacheOnFakeDevice:
    """A real `ModelCache`, whose offload and eviction paths run as they are, standing in for a GPU's on a `FakeDevice`.
    It runs on the CPU, with its VRAM budget capped and its VRAM in use read from the fake device, so that it never
    asks a GPU; its entries' weights are on the fake device until they move to RAM or their model dies. Its lock --
    replaced but for `lock_dropped` -- offloads the entries `offloads` names for the locked model through the cache,
    then pins the record and puts its weights on the device."""

    def __init__(self, device: FakeDevice):
        logger = MagicMock()
        logger.getEffectiveLevel.return_value = logging.INFO
        self.device = device
        self.cache = ModelCache(
            execution_device_working_mem_gb=1.0,
            enable_partial_loading=False,
            keep_ram_copy_of_weights=True,
            max_vram_cache_size_gb=8.0,
            execution_device="cpu",
            storage_device="cpu",
            logger=logger,
        )
        self.cache.lock = self._lock  # type: ignore[method-assign]
        self.cache._get_vram_in_use = lambda: device.allocated  # type: ignore[method-assign]
        self.offloads: dict[str, list[str]] = {}
        self._on_device: dict[str, weakref.finalize] = {}

    def admit(self, key: str, on_device: bool = True) -> int:
        """Admit an unlocked model of 2 MiB that computes on the device, its weights there or in RAM; its size."""
        self.cache.put(key, torch.nn.Linear(1024, 512, bias=False))
        cached_model = self.cache._cached_models[key].cached_model
        cached_model._compute_device = torch.device("meta")
        if on_device:
            cached_model._is_in_vram = True
            self._weights_on_device(self.cache._cached_models[key])
        return cached_model.total_bytes()

    def lock_dropped(self, record: CacheRecord) -> None:
        """Lock and unlock a record the cache dropped, through the cache's own lock, which puts its weights on the
        device (#7513)."""
        ModelCache.lock(self.cache, record, None)
        self._weights_on_device(record)
        self.cache.unlock(record)

    def _weights_on_device(self, record: CacheRecord) -> None:
        weights = record.cached_model.total_bytes()
        self.device.alloc(weights)
        self._on_device[record.key] = weakref.finalize(record.cached_model.model, self.device.free, weights)

    def model(self, key: str, total_bytes: int = 1 * GIB) -> LoadedModelWithoutConfig:
        """A model a window can lock on this cache, not one of its entries."""
        return LoadedModelWithoutConfig(
            CacheRecord(key=key, cached_model=FakeCachedModel(total_bytes, "meta")), cache=self.cache
        )

    def offload(self, key: str) -> None:
        """Move the entry to RAM, as a node's `context.models.offload_from_vram` does."""
        freed = self.cache.offload_model_from_vram(key)
        self._on_device.pop(key).detach()
        self.device.free(freed)

    def drop(self, key: str) -> None:
        """Drop the entry, as a changed model setting does; its weights go when its model's last reference does."""
        assert self.cache.drop_model(key) == 1

    def _lock(
        self, record: CacheRecord, working_mem_bytes: Optional[int], max_move_bytes: Optional[int] = None
    ) -> bool:
        for key in self.offloads.get(record.key, []):
            self.offload(key)
        model = record.cached_model
        self.device.alloc(model.total - model.resident)
        model.resident = model.total
        with self.cache._lock:
            self.cache._pin(record)
        return True


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDevice]:
    fake = FakeDevice()
    opened: list[FakeCounters] = []

    def device_counters(torch_device: torch.device) -> FakeCounters:
        opened.append(FakeCounters(fake))
        return opened[-1]

    # The fake allocator observes releases ahead of the probe, which registers when a scope opens.
    monkeypatch.setattr(devices, "EMPTY_CACHE_OBSERVERS", [fake])
    monkeypatch.setattr(counters, "DeviceCounters", device_counters)
    monkeypatch.setattr(measurement, "SPILL_SETTLE_S", 0.0)
    yield fake
    assert all(c.closed for c in opened), "every window's counters are closed, so no sampler outlives it"
    assert probe.armed is False
    assert probe.locking == [], "every lock leaves `locking` once it has settled or failed"


@pytest.fixture
def model_cache(device: FakeDevice) -> Iterator[CacheOnFakeDevice]:
    # The cache's drops and offloads run a full `gc.collect()`; frozen, the worker's earlier heap is not walked.
    gc.freeze()
    cache = CacheOnFakeDevice(device)
    try:
        yield cache
    finally:
        cache.cache.shutdown()
        gc.unfreeze()


_DEVICE_COUNTERS = counters.DeviceCounters


@pytest.fixture
def cuda_device(device: FakeDevice, monkeypatch: pytest.MonkeyPatch) -> FakeDevice:
    """`device` as cuda:0 behind the real `counters.DeviceCounters`: only torch's allocator statistics, peak reset and
    synchronization, and the driver's figure, are the fake's."""
    monkeypatch.setattr(counters, "DeviceCounters", _DEVICE_COUNTERS)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device: device.memory_stats())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: device.reset_peaks())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(
        counters, "_driver_reading", lambda _device: (device.reserved + device.retained, 0, None, "windows-d3dkmt")
    )
    return device


def _estimate(reservation: int = 2 * GIB) -> WorkingMemoryEstimate:
    return WorkingMemoryEstimate(
        estimator="vae.test",
        version=3,
        features={"operation": "decode", "height": 1024, "width": 1024, "fp32": False, "tile": None},
        terms=(WorkingMemoryTerm.fitted("convolutions", "k_decode", 1024 * 1024, {"k_decode": reservation / 2**20}),),
        source="shipped:conv_backend=cudnn",
    )


def _model(cache: FakeCache, key: str, total_bytes: int = 1 * GIB, device: str = "meta") -> LoadedModelWithoutConfig:
    record = CacheRecord(key=key, cached_model=FakeCachedModel(total_bytes, device))
    cache.records[key] = record
    return LoadedModelWithoutConfig(record, cache=cache)


def _on_another_thread(model: LoadedModelWithoutConfig) -> None:
    """Lock and unlock `model` on another thread, as Expand Prompt does next to a session."""

    def use() -> None:
        with model.model_on_device():
            pass

    thread = threading.Thread(target=use)
    thread.start()
    thread.join()


@contextmanager
def _scope() -> Iterator[ProbeScope]:
    with calibration_scope(run_id="run-1", queue_item_id=7, node_id="l2i-3", node_type="l2i") as scope:
        yield scope


def test_a_window_measures_what_the_operation_took_beyond_what_was_in_use(device: FakeDevice):
    cache = FakeCache(device)
    vae = _model(cache, "vae")
    device.alloc(512 * MIB)  # latents allocated before the lock: in use, not the operation's
    device.alloc(2 * GIB)
    device.free(2 * GIB)  # an earlier node's cached blocks, released when the window opens

    with _scope() as scope:
        with vae.model_on_device(working_memory=_estimate()):
            device.alloc(3 * GIB)
            device.free(3 * GIB)

    (window,) = scope.windows
    assert window.demand_bytes == 3 * GIB
    assert window.reserved is not None and window.reserved.delta == 3 * GIB
    assert window.allocated is not None and window.allocated.delta == 3 * GIB
    assert window.driver is not None and window.driver.delta == 3 * GIB
    assert window.reserved.baseline == 512 * MIB + 1 * GIB, "the latents and the VAE's weights, nothing cached"
    assert (window.valid, window.fit_eligible, window.reasons) == (True, True, [])
    assert (window.run_id, window.queue_item_id, window.node_id, window.node_type) == ("run-1", 7, "l2i-3", "l2i")
    assert (window.model_key, window.device, window.estimator, window.version) == ("vae", "meta", "vae.test", 3)
    assert window.reservation_bytes == 2 * GIB and window.source == "shipped:conv_backend=cudnn"
    assert window.resident_fraction == 1.0 and window.co_locked_models == []
    assert MeasuredWindow.model_validate_json(window.model_dump_json()) == window


@pytest.mark.parametrize(("reservation", "applied"), [(1 * GIB, 2 * GIB), (4 * GIB, 6 * GIB)])
def test_a_calibration_lock_reserves_headroom_beyond_the_estimate(device: FakeDevice, reservation: int, applied: int):
    """At least 1 GiB, or half the estimate: an estimate that is too low must not spill, or it could never be fixed."""
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate(reservation)):
            pass

    assert cache.reserved == [applied]
    (window,) = scope.windows
    assert (window.reservation_bytes, window.reservation_applied_bytes) == (reservation, applied)


def test_without_a_scope_the_lock_is_unchanged_and_no_counter_is_read(monkeypatch: pytest.MonkeyPatch):
    def refuse(_device: torch.device) -> None:
        raise AssertionError("a counter was read without a calibration scope")

    monkeypatch.setattr(counters, "DeviceCounters", refuse)
    monkeypatch.setattr(counters, "read_counters", refuse)
    cache = FakeCache(FakeDevice())
    vae = _model(cache, "vae")

    with vae.model_on_device(working_memory=_estimate(2 * GIB)):
        pass
    with vae.model_on_device(512 * MIB):
        pass
    with vae:
        pass

    assert cache.reserved == [2 * GIB, 512 * MIB, None]
    assert cache.locks_held() == 0


def test_a_byte_count_or_a_cpu_model_never_opens_a_window(device: FakeDevice):
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(512 * MIB):
            pass
        with _model(cache, "cpu-vae", device="cpu").model_on_device(working_memory=_estimate(2 * GIB)):
            pass

    assert scope.windows == []
    assert cache.reserved == [512 * MIB, 2 * GIB], "no headroom where nothing is measured"


def test_a_co_lock_counts_its_weights_as_offset_and_its_lock_outside_the_peak(device: FakeDevice):
    """A ControlNet locked inside the window moves 2 GiB of weights through a 3 GiB staging buffer; neither is the
    operation's working memory."""
    cache = FakeCache(device, staging_bytes=3 * GIB)
    transformer = _model(cache, "transformer")
    controlnet = _model(cache, "controlnet", total_bytes=2 * GIB)

    with _scope() as scope:
        with transformer.model_on_device(working_memory=_estimate()):
            device.alloc(1 * GIB)  # lives across the co-lock
            with controlnet:
                device.alloc(1536 * MIB)
                device.free(1536 * MIB)

    (window,) = scope.windows
    assert window.demand_bytes == 1 * GIB + 1536 * MIB
    assert window.reserved is not None and window.reserved.offset == 2 * GIB
    assert window.co_locked_models == ["controlnet"]
    assert window.valid


def _resident(cache: FakeCache, key: str, total_bytes: int) -> LoadedModelWithoutConfig:
    """A model a previous node left on the device, unlocked."""
    model = _model(cache, key, total_bytes=total_bytes)
    with model:
        pass
    return model


def test_a_calibration_lock_moves_every_other_unlocked_model_out_first(device: FakeDevice):
    """Left to the lock, a UNet would be split by the predicted shortfall, and a split that freed a little less than
    predicted ran the VAE partly resident. Moved out whole first, nothing is split."""
    cache = FakeCache(device)
    unet = _resident(cache, "sdxl:unet", 5 * GIB)
    text_encoder = _resident(cache, "sdxl:text_encoder_2", 1 * GIB)
    vae = _model(cache, "sdxl:vae", total_bytes=160 * MIB)

    with _scope() as scope:
        with vae.model_on_device(working_memory=_estimate()):
            pass

    assert cache.in_use_at_lock[-1] == 0, "both moved out before the VAE's lock began"
    assert unet._cache_record.cached_model.resident == text_encoder._cache_record.cached_model.resident == 0
    (window,) = scope.windows
    assert window.resident_fraction == 1.0 and window.fit_eligible


def test_outside_a_calibration_scope_a_lock_moves_nothing_else_out(device: FakeDevice):
    cache = FakeCache(device)
    unet = _resident(cache, "sdxl:unet", 5 * GIB)
    vae = _model(cache, "sdxl:vae", total_bytes=160 * MIB)

    with vae.model_on_device(working_memory=_estimate()):
        pass

    assert cache.in_use_at_lock[-1] == 5 * GIB
    assert unet._cache_record.cached_model.resident == 5 * GIB


def test_a_model_the_window_holds_locked_stays_on_the_device_for_a_nested_calibration_lock(device: FakeDevice):
    """A transformer whose own window is open locks a VAE for a reference image: the transformer, a co-lock of the
    VAE's window, stays; the text encoder the node locked and released inside the window goes."""
    cache = FakeCache(device)
    text_encoder = _model(cache, "flux:text_encoder", total_bytes=2 * GIB)
    transformer = _model(cache, "flux:transformer", total_bytes=4 * GIB)
    vae = _model(cache, "flux:vae", total_bytes=160 * MIB)

    with _scope() as scope:
        with transformer.model_on_device(working_memory=_estimate(8 * GIB)):
            with text_encoder:
                pass
            with vae.model_on_device(working_memory=_estimate(2 * GIB)):
                assert cache.in_use_at_lock[-1] == 4 * GIB, "the transformer only"
                assert text_encoder._cache_record.cached_model.resident == 0
            assert transformer._cache_record.cached_model.resident == 4 * GIB

    child, parent = scope.windows
    assert child.co_locked_models == ["flux:transformer"] and child.valid and parent.valid


def test_a_co_lock_that_offloads_more_than_it_loads_takes_its_offload_off_the_offset(device: FakeDevice):
    """A text encoder locked inside the window stays resident after its unlock (+1 GiB); the adapter's co-lock then
    moves it out to load its own 512 MiB, a co-lock offset of -512 MiB."""
    cache = FakeCache(device)
    text_encoder = _model(cache, "text-encoder", total_bytes=1 * GIB)
    vae = _model(cache, "vae")
    adapter = _model(cache, "adapter", total_bytes=512 * MIB)
    cache.offloads["adapter"] = [text_encoder._cache_record]

    with _scope() as scope:
        with vae.model_on_device(working_memory=_estimate()):
            with text_encoder:
                pass
            with adapter:
                pass
            device.alloc(2 * GIB)
            device.free(2 * GIB)

    (window,) = scope.windows
    assert window.reserved is not None and window.reserved.offset == 1 * GIB - 512 * MIB
    assert window.demand_bytes == 2 * GIB


def test_a_co_lock_that_offloads_more_than_it_loads_has_a_negative_offset(
    device: FakeDevice, model_cache: CacheOnFakeDevice
):
    """The cache's own offload, inside the lock of this thread's co-lock: what it moves is offset, not a void."""
    text_encoder = model_cache.admit("text-encoder")  # resident and unlocked: part of the baseline
    vae = model_cache.model("vae")
    adapter = model_cache.model("adapter", total_bytes=text_encoder // 4)
    model_cache.offloads["adapter"] = ["text-encoder"]

    with _scope() as scope:
        with vae.model_on_device(working_memory=_estimate()):
            with adapter:
                pass
            device.alloc(2 * GIB)
            device.free(2 * GIB)

    (window,) = scope.windows
    assert window.reserved is not None and window.reserved.offset == text_encoder // 4 - text_encoder
    assert window.demand_bytes == 2 * GIB
    assert (window.valid, window.reasons) == (True, [])


@pytest.mark.parametrize("evict", ["drop", "offload"])
def test_weights_another_thread_takes_off_the_device_void_the_window(
    device: FakeDevice, model_cache: CacheOnFakeDevice, evict: str
):
    """While the window is open, another thread drops or offloads a model the window does not use (Clear Model Cache,
    a changed model setting, a load making room), and only then does the operation allocate. The counters fell by what
    left, so the window would credit the operation with less than it took."""
    text_encoder = model_cache.admit("text-encoder")
    opened, evicted = threading.Event(), threading.Event()

    def clear_model_cache() -> None:
        if opened.wait(timeout=30):
            getattr(model_cache, evict)("text-encoder")
            evicted.set()

    thread = threading.Thread(target=clear_model_cache)
    thread.start()
    try:
        with _scope() as scope:
            with model_cache.model("vae").model_on_device(working_memory=_estimate()):
                opened.set()
                assert evicted.wait(timeout=30), "the other thread took the weights off the device"
                device.alloc(3 * text_encoder // 2)
                device.free(3 * text_encoder // 2)
    finally:
        opened.set()
        thread.join()

    (window,) = scope.windows
    assert window.demand_bytes == text_encoder // 2, "what the counters show: the allocation less what left"
    assert (window.valid, window.fit_eligible, window.reasons) == (False, False, ["weights_left"])


def test_weights_another_thread_moves_void_a_window_that_closes_before_the_move_returns(
    device: FakeDevice, model_cache: CacheOnFakeDevice
):
    """The probe hears of an offload before the weights move: the window allocates and closes while the move is still
    under way on the other thread."""
    text_encoder = model_cache.admit("text-encoder")
    cached_model = model_cache.cache._cached_models["text-encoder"].cached_model
    opened, moved, closed = threading.Event(), threading.Event(), threading.Event()
    unload = cached_model.full_unload_from_vram

    def unload_while_the_window_closes() -> int:
        freed = unload()
        model_cache._on_device.pop("text-encoder").detach()
        device.free(freed)
        moved.set()
        closed.wait(timeout=30)
        return freed

    cached_model.full_unload_from_vram = unload_while_the_window_closes
    unlock = model_cache.cache.unlock

    def closed_then_unlock(record: CacheRecord) -> None:
        closed.set()  # the window has closed; its unlock waits for the cache's lock, which the move holds
        unlock(record)

    model_cache.cache.unlock = closed_then_unlock  # type: ignore[method-assign]

    def offload() -> None:
        if opened.wait(timeout=30):
            model_cache.cache.offload_model_from_vram("text-encoder")

    thread = threading.Thread(target=offload)
    thread.start()
    try:
        with _scope() as scope:
            with model_cache.model("vae").model_on_device(working_memory=_estimate()):
                opened.set()
                assert moved.wait(timeout=30), "the other thread moved the weights"
                device.alloc(3 * text_encoder // 2)
                device.free(3 * text_encoder // 2)
    finally:
        opened.set()
        closed.set()
        thread.join()

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["weights_left"])


@pytest.mark.parametrize("evict", ["drop", "offload"])
def test_weights_this_thread_takes_off_the_device_outside_a_lock_void_the_window(
    device: FakeDevice, model_cache: CacheOnFakeDevice, evict: str
):
    """A model loaded mid-operation makes room in RAM by dropping one that was on the device, or a node offloads one
    it is done with: no co-lock offsets it."""
    text_encoder = model_cache.admit("text-encoder")

    with _scope() as scope:
        with model_cache.model("vae").model_on_device(working_memory=_estimate()):
            getattr(model_cache, evict)("text-encoder")
            device.alloc(3 * text_encoder // 2)
            device.free(3 * text_encoder // 2)

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["weights_left"])


def test_a_child_windows_co_lock_offloading_weights_voids_its_parent(
    device: FakeDevice, model_cache: CacheOnFakeDevice
):
    """The child's offset takes what its co-lock moves; the suspended parent offsets nothing, so it is void."""
    text_encoder = model_cache.admit("text-encoder")
    adapter = model_cache.model("adapter", total_bytes=text_encoder // 4)
    model_cache.offloads["adapter"] = ["text-encoder"]

    with _scope() as scope:
        with model_cache.model("transformer", total_bytes=4 * GIB).model_on_device(working_memory=_estimate(8 * GIB)):
            with model_cache.model("vae", total_bytes=GIB // 4).model_on_device(working_memory=_estimate()):
                with adapter:
                    pass
            device.alloc(3 * text_encoder // 2)
            device.free(3 * text_encoder // 2)

    child, parent = scope.windows
    assert child.reserved is not None and child.reserved.offset == text_encoder // 4 - text_encoder
    assert (child.valid, child.reasons) == (True, [])
    assert (parent.valid, parent.reasons) == (False, ["weights_left"])


@pytest.mark.parametrize("let_go", ["before-the-window-opened", "while-the-window-measures", "after-the-window-closed"])
def test_a_dropped_model_voids_the_window_its_weights_die_in(
    device: FakeDevice, model_cache: CacheOnFakeDevice, let_go: str
):
    """The cache drops a model a node still holds, which keeps its weights on the device until the node lets go -- as
    a reference cycle keeps them until a garbage collection. Only a window open when they leave is void."""
    text_encoder = model_cache.admit("text-encoder")
    held = [model_cache.cache._cached_models["text-encoder"]]

    with _scope() as scope:
        model_cache.drop("text-encoder")
        if let_go == "before-the-window-opened":
            held.clear()
        with model_cache.model("vae").model_on_device(working_memory=_estimate()):
            if let_go == "while-the-window-measures":
                held.clear()
            device.alloc(3 * text_encoder // 2)
            device.free(3 * text_encoder // 2)

    (window,) = scope.windows
    if let_go == "while-the-window-measures":
        assert window.demand_bytes == text_encoder // 2
        assert (window.valid, window.reasons) == (False, ["weights_left"])
    else:
        assert window.demand_bytes == 3 * text_encoder // 2
        assert (window.valid, window.reasons) == (True, [])


def test_a_dropped_record_a_lock_put_on_the_device_voids_the_window_its_weights_die_in(
    device: FakeDevice, model_cache: CacheOnFakeDevice
):
    """Dropped while its weights were in RAM, the record had nothing to watch; a handle then locked it onto the device,
    and no later drop reports it. Its weights die with the handle while a window measures."""
    text_encoder = model_cache.admit("text-encoder", on_device=False)
    held = [model_cache.cache._cached_models["text-encoder"]]

    with _scope() as scope:
        model_cache.drop("text-encoder")
        model_cache.lock_dropped(held[0])
        with model_cache.model("vae").model_on_device(working_memory=_estimate()):
            held.clear()
            device.alloc(3 * text_encoder // 2)
            device.free(3 * text_encoder // 2)

    (window,) = scope.windows
    assert window.demand_bytes == text_encoder // 2
    assert (window.valid, window.reasons) == (False, ["weights_left"])


def test_a_model_without_weights_on_the_device_leaving_the_cache_leaves_the_window_valid(
    device: FakeDevice, model_cache: CacheOnFakeDevice
):
    """A LoRA dropped from RAM to make room for a load: nothing leaves the device."""
    model_cache.admit("lora", on_device=False)

    with _scope() as scope:
        with model_cache.model("vae").model_on_device(working_memory=_estimate()):
            model_cache.drop("lora")
            device.alloc(1 * GIB)
            device.free(1 * GIB)

    (window,) = scope.windows
    assert (window.valid, window.reasons, window.demand_bytes) == (True, [], 1 * GIB)


def test_a_cpu_only_model_leaving_the_device_cache_leaves_the_window_valid(
    device: FakeDevice, model_cache: CacheOnFakeDevice
):
    """A `cpu_only` text encoder in the GPU's cache computes on the CPU, its "VRAM" RAM: offloaded or dropped, nothing
    leaves the device."""
    model_cache.cache.put("t5", torch.nn.Linear(1024, 512, bias=False))
    cached_model = model_cache.cache._cached_models["t5"].cached_model
    assert cached_model.compute_device.type == "cpu"
    cached_model._is_in_vram = True

    with _scope() as scope:
        with model_cache.model("vae").model_on_device(working_memory=_estimate()):
            model_cache.cache.offload_model_from_vram("t5")
            cached_model._is_in_vram = True
            model_cache.drop("t5")
            device.alloc(1 * GIB)
            device.free(1 * GIB)

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (True, [])


def test_a_nested_window_measures_alone_and_its_outputs_stay_in_the_parent(device: FakeDevice):
    cache = FakeCache(device)
    transformer = _model(cache, "transformer", total_bytes=4 * GIB)
    vae = _model(cache, "vae", total_bytes=1 * GIB)

    with _scope() as scope:
        with transformer.model_on_device(working_memory=_estimate(8 * GIB)):
            device.alloc(1 * GIB)
            with vae.model_on_device(working_memory=_estimate(2 * GIB)):
                device.alloc(2 * GIB)
                device.free(2 * GIB)
                TorchDevice.empty_cache()  # a boundary of the child alone: the parent is frozen
                device.alloc(1 * GIB)
                device.free(1 * GIB)  # still cached when the child closes; the parent releases it
                device.alloc(256 * MIB)  # the encoded reference latents, kept
            device.alloc(512 * MIB)
            device.free(512 * MIB)

    child, parent = scope.windows
    assert (child.model_key, parent.model_key) == ("vae", "transformer")
    assert child.demand_bytes == 2 * GIB and child.segments == 2
    assert child.co_locked_models == ["transformer"]
    assert parent.demand_bytes == 1 * GIB + 256 * MIB + 512 * MIB, "not the child's 2 GiB transient"
    assert parent.reserved is not None and parent.reserved.offset == 1 * GIB, "the VAE's weights"
    assert parent.co_locked_models == ["vae"]
    assert child.valid and parent.valid


def test_an_empty_cache_inside_a_window_ends_a_segment_so_the_driver_peak_is_kept(device: FakeDevice):
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            device.alloc(3 * GIB)
            device.free(3 * GIB)
            TorchDevice.empty_cache()
            device.alloc(1 * GIB)
            device.free(1 * GIB)

    (window,) = scope.windows
    assert window.driver is not None and window.driver.delta == 3 * GIB
    assert window.reserved is not None and window.reserved.delta == 3 * GIB
    assert window.segments == 2


def test_an_allocated_peak_above_the_reserved_peak_decides_the_demand(device: FakeDevice):
    """cudaMallocAsync: `max_memory_reserved` measured 128 MiB under an allocated peak of 136 MiB."""
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            device.alloc(136 * MIB, reserve=128 * MIB)
            device.free(136 * MIB)

    (window,) = scope.windows
    assert window.reserved is not None and window.reserved.delta == 128 * MIB
    assert window.demand_bytes == 136 * MIB


def test_a_driver_that_keeps_a_retained_pool_reads_below_reserved_and_reserved_decides(device: FakeDevice):
    """HIP under Windows keeps released memory counted and hands it to the next allocation."""
    device.retained_pool = True
    device.alloc(3 * GIB)
    device.free(3 * GIB)
    TorchDevice.empty_cache()  # 3 GiB retained; loading the VAE's weights takes 1 GiB of it
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            device.alloc(6 * GIB)
            device.free(6 * GIB)

    (window,) = scope.windows
    assert window.driver is not None and window.driver.delta == 4 * GIB
    assert window.demand_bytes == 6 * GIB


def test_memory_outside_torchs_allocator_decides_the_demand_through_the_drivers_figure(device: FakeDevice):
    """On HIP the first operation of a process maps ~300 MiB beyond the reserved peak, and library workspaces and
    kernel modules are the driver's alone everywhere: torch's counters never see them."""
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            device.alloc(1 * GIB)
            device.free(1 * GIB)
            device.outside += 300 * MIB

    (window,) = scope.windows
    assert window.reserved is not None and window.reserved.delta == 1 * GIB
    assert window.driver is not None and window.driver.delta == 1 * GIB + 300 * MIB
    assert window.demand_bytes == 1 * GIB + 300 * MIB


def test_without_a_driver_reading_the_window_rests_on_torch_and_is_unverified(device: FakeDevice):
    device.has_driver = False
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae", device="cuda:0").model_on_device(working_memory=_estimate()):
            device.alloc(1 * GIB)
            device.free(1 * GIB)

    (window,) = scope.windows
    assert window.driver is None and window.demand_bytes == 1 * GIB
    assert (window.method, window.verified, window.valid) == ("torch-only", False, True)


def test_a_driver_reading_by_a_verified_method_is_marked_verified(device: FakeDevice):
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae", device="cuda:0").model_on_device(working_memory=_estimate()):
            pass

    (window,) = scope.windows
    assert (window.method, window.verified) == ("windows-d3dkmt", True)


def test_a_lock_by_another_thread_on_the_same_cache_voids_the_window(device: FakeDevice):
    cache = FakeCache(device)
    vae = _model(cache, "vae")
    prompt_expander = _model(cache, "llm")
    elsewhere = _model(FakeCache(device), "other-gpu-model")

    with _scope() as scope:
        with vae.model_on_device(working_memory=_estimate()):
            _on_another_thread(elsewhere)
        with vae.model_on_device(working_memory=_estimate()):
            _on_another_thread(prompt_expander)

    other_device, same_device = scope.windows
    assert other_device.valid, "another device's cache does not share this one"
    assert (same_device.valid, same_device.reasons) == (False, ["foreign_lock"])


def test_a_lock_another_thread_holds_when_the_window_opens_voids_it(device: FakeDevice):
    cache = FakeCache(device)
    vae = _model(cache, "vae")
    prompt_expander = _model(cache, "llm")
    holding, done = threading.Event(), threading.Event()

    def hold() -> None:
        with prompt_expander.model_on_device():
            holding.set()
            done.wait()

    thread = threading.Thread(target=hold)
    thread.start()
    holding.wait()
    try:
        with _scope() as scope:
            with vae.model_on_device(working_memory=_estimate()):
                pass
    finally:
        done.set()
        thread.join()

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["foreign_lock"])


def test_a_lock_another_thread_takes_while_the_window_opens_voids_it(
    device: FakeDevice, monkeypatch: pytest.MonkeyPatch
):
    """Between the cache's lock count and the baseline lie a synchronization and a release; a lock another thread
    takes and settles there must not go unseen."""
    cache = FakeCache(device)
    vae = _model(cache, "vae")
    prompt_expander = _model(cache, "llm")
    device_counters = counters.DeviceCounters

    def opening(torch_device: torch.device) -> FakeCounters:
        _on_another_thread(prompt_expander)
        return device_counters(torch_device)

    monkeypatch.setattr(counters, "DeviceCounters", opening)

    with _scope() as scope:
        with vae.model_on_device(working_memory=_estimate()):
            pass

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["foreign_lock"])


class GatedCache(FakeCache):
    """A cache whose lock of the model `gated` holds still twice: before it moves anything, as a lock that waits for
    the cache's lock or for MODEL_LOAD_LOCK does, and once it moved the weights, before it settles, as a paced lock does
    between passes."""

    def __init__(self, device: FakeDevice, gated: str):
        super().__init__(device)
        self.gated = gated
        self.waiting, self.go, self.moved, self.settle = (threading.Event() for _ in range(4))

    def lock(self, record: CacheRecord, working_mem_bytes: Optional[int], max_move_bytes: Optional[int] = None) -> bool:
        if record.key != self.gated:
            return super().lock(record, working_mem_bytes, max_move_bytes)
        self.waiting.set()
        if not self.go.wait(timeout=30):
            raise TimeoutError("the test never let the lock go on")
        settled = super().lock(record, working_mem_bytes, max_move_bytes)
        self.moved.set()
        if not self.settle.wait(timeout=30):
            raise TimeoutError("the test never let the lock settle")
        return settled


@pytest.mark.parametrize("lock_read_armed", ["before-the-scope-armed", "before-the-window-opened"])
def test_a_lock_another_thread_has_in_progress_when_the_window_opens_voids_it(device: FakeDevice, lock_read_armed: str):
    """Read before the scope armed, `armed` says nothing is measured, and the lock never reports. Read after it, the lock
    reports its start while no window is registered and its end once the window has closed. Either way its weights
    move while the window measures."""
    cache = GatedCache(device, gated="llm")
    vae = _model(cache, "vae")
    llm = _model(cache, "llm", total_bytes=2 * GIB)

    def expand_prompt() -> None:
        with llm.model_on_device():
            pass

    thread = threading.Thread(target=expand_prompt)

    def lock_until_it_waits() -> None:
        thread.start()
        assert cache.waiting.wait(timeout=30), "the lock read `armed` and waits for the cache"

    try:
        if lock_read_armed == "before-the-scope-armed":
            lock_until_it_waits()
        with _scope() as scope:
            if lock_read_armed == "before-the-window-opened":
                lock_until_it_waits()
            with vae.model_on_device(working_memory=_estimate()):
                cache.go.set()
                assert cache.moved.wait(timeout=30), "the weights moved onto the device inside the window"
            cache.settle.set()
            thread.join(timeout=30)
            assert not thread.is_alive()
    finally:
        cache.go.set()
        cache.settle.set()
        if thread.ident is not None:
            thread.join()

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["foreign_lock"])
    assert cache.locks_held() == 0


class GatedAppend(list):
    """`probe.locking`, whose append on the thread named `gated` waits until the test lets it go on."""

    def __init__(self, gated: str):
        super().__init__()
        self.gated = gated
        self.waiting, self.go = threading.Event(), threading.Event()

    def append(self, cache: object) -> None:
        if threading.current_thread().name == self.gated:
            self.waiting.set()
            if not self.go.wait(timeout=30):
                raise TimeoutError("the test never let the lock go on")
        super().append(cache)


def test_a_lock_enters_locking_before_it_reads_armed(device: FakeDevice, monkeypatch: pytest.MonkeyPatch):
    """The other way round, a lock could read `armed` before the scope armed and enter `locking` only after the window
    opened: neither would see the other, and its weights would move while the window measures."""
    locking = GatedAppend(gated="expand-prompt")
    monkeypatch.setattr(probe, "locking", locking)
    cache = FakeCache(device)
    vae = _model(cache, "vae")
    llm = _model(cache, "llm", total_bytes=2 * GIB)
    locked, release = threading.Event(), threading.Event()

    def expand_prompt() -> None:
        with llm.model_on_device():
            locked.set()
            release.wait(timeout=30)

    thread = threading.Thread(target=expand_prompt, name="expand-prompt")
    try:
        thread.start()
        assert locking.waiting.wait(timeout=30), "the lock is entering `locking`"
        with _scope() as scope:
            with vae.model_on_device(working_memory=_estimate()):
                locking.go.set()
                assert locked.wait(timeout=30), "the weights moved onto the device inside the window"
            # Held until the window has closed, so that its unlock tells the window nothing.
            release.set()
            thread.join(timeout=30)
            assert not thread.is_alive()
    finally:
        locking.go.set()
        release.set()
        thread.join()

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["foreign_lock"])


def test_a_model_this_thread_pins_in_ram_is_its_own_lock(device: FakeDevice):
    """LoRAs are pinned in RAM before the transformer locks; that pin belongs to this thread, not to another."""
    cache = FakeCache(device)
    lora = _model(cache, "lora")

    with _scope() as scope:
        with lora.model_in_ram():
            with _model(cache, "transformer").model_on_device(working_memory=_estimate()):
                pass

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (True, [])
    assert window.co_locked_models == ["lora"]


def test_after_an_unlock_a_lock_another_thread_holds_still_voids_the_next_window(device: FakeDevice):
    cache = FakeCache(device)
    vae = _model(cache, "vae")
    prompt_expander = _model(cache, "llm")
    holding, done = threading.Event(), threading.Event()

    def hold() -> None:
        with prompt_expander.model_on_device():
            holding.set()
            done.wait()

    thread = threading.Thread(target=hold)
    with _scope() as scope:
        with vae.model_on_device(working_memory=_estimate()):
            pass
        thread.start()
        holding.wait()
        try:
            with vae.model_on_device(working_memory=_estimate()):
                pass
        finally:
            done.set()
            thread.join()

    first, second = scope.windows
    assert first.valid
    assert (second.valid, second.reasons) == (False, ["foreign_lock"])


@pytest.mark.parametrize("release", [1, 2], ids=["co-lock-start", "co-lock-settled"])
def test_a_release_that_fails_around_a_co_lock_voids_the_window_and_the_rest_of_the_scope(
    device: FakeDevice, release: int
):
    cache = FakeCache(device)
    transformer = _model(cache, "transformer")

    with _scope() as scope:
        with transformer.model_on_device(working_memory=_estimate()):
            device.fail_on_release = device.releases + release
            with _model(cache, "controlnet"):
                pass
            device.alloc(1 * GIB)
            device.free(1 * GIB)
        with transformer.model_on_device(working_memory=_estimate()):
            pass

    failed, later = scope.windows
    assert (failed.valid, failed.reasons) == (False, ["probe_error"])
    assert (later.valid, later.reasons) == (False, ["probe_error"]), "the scope's lock bookkeeping is suspect"
    assert cache.locks_held() == 0


def test_a_segment_boundary_that_fails_voids_the_window(device: FakeDevice):
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            device.fail_next_read = True
            TorchDevice.empty_cache()

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["probe_error"])


def _works(times: int, then: Callable[[], None]) -> Callable[..., None]:
    """A torch call that does `then` `times` times and raises from then on, as on a device context gone bad."""
    left = [times]

    def call(*_args: object) -> None:
        if left[0] == 0:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        left[0] -= 1
        then()

    return call


_UNOPENED = ["probe_error", "counters_unavailable"]
"""A window that could not open: the operation ran unmeasured."""


@pytest.mark.parametrize(
    ("function", "works", "reasons"),
    [
        pytest.param("synchronize", 0, _UNOPENED, id="synchronize-at-open"),
        pytest.param("reset_peak_memory_stats", 0, _UNOPENED, id="reset-at-open"),
        pytest.param("reset_peak_memory_stats", 1, ["probe_error"], id="reset-after-a-release"),
        pytest.param("synchronize", 1, ["probe_error"], id="synchronize-at-close"),
    ],
)
def test_a_window_whose_peaks_cannot_be_reset_or_device_synchronized_is_invalid(
    cuda_device: FakeDevice, monkeypatch: pytest.MonkeyPatch, function: str, works: int, reasons: list[str]
):
    """Peaks not reset carry what came before into the window; unsynchronized work lands in the wrong segment."""
    then = cuda_device.reset_peaks if function == "reset_peak_memory_stats" else (lambda: None)
    monkeypatch.setattr(torch.cuda, function, _works(works, then))
    cache = FakeCache(cuda_device)

    with _scope() as scope:
        with _model(cache, "vae", device="cuda:0").model_on_device(working_memory=_estimate()):
            cuda_device.alloc(2 * GIB)
            cuda_device.free(2 * GIB)
            TorchDevice.empty_cache()
            cuda_device.alloc(1 * GIB)
            cuda_device.free(1 * GIB)

    (window,) = scope.windows
    assert (window.valid, window.fit_eligible, window.reasons) == (False, False, reasons)
    assert cache.locks_held() == 0


def test_a_child_window_that_fails_to_close_is_recorded_invalid_and_the_parent_resumes(device: FakeDevice):
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "transformer").model_on_device(working_memory=_estimate()):
            with _model(cache, "vae").model_on_device(working_memory=_estimate()):
                device.fail_next_read = True
            device.alloc(1 * GIB)
            device.free(1 * GIB)

    child, parent = scope.windows
    assert (child.model_key, child.valid) == ("vae", False)
    assert child.reasons == ["probe_error", "counters_unavailable"], "its final segment was never read"
    assert (parent.model_key, parent.valid, parent.reasons) == ("transformer", False, ["probe_error"])
    assert parent.segments == 3, "before the child's lock, up to the child, and after it again"
    assert cache.locks_held() == 0


@pytest.mark.parametrize(("counter", "reason"), [("num_ooms", "oom"), ("num_alloc_retries", "alloc_retry")])
def test_an_allocator_that_failed_or_retried_voids_the_window(device: FakeDevice, counter: str, reason: str):
    """A retry releases the cache and squeezes the peak below what the operation takes with room to spare."""
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            setattr(device, counter, getattr(device, counter) + 1)

    (window,) = scope.windows
    assert (window.valid, window.fit_eligible, window.reasons) == (False, False, [reason])


@pytest.mark.parametrize(("paged_at_close", "valid"), [([600 * MIB, 9 * MIB], True), ([600 * MIB, 600 * MIB], False)])
def test_only_spill_that_persists_after_close_voids_the_window(device: FakeDevice, paged_at_close, valid: bool):
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            device.paged = list(paged_at_close)

    (window,) = scope.windows
    assert window.valid is valid
    assert window.reasons == ([] if valid else ["spill"])


def test_a_device_wide_counter_growing_beyond_this_process_voids_the_window(device: FakeDevice):
    """XPU: Sysman counts the whole device, so growth beyond this process's own is another process's load."""
    device.method, device.has_driver, device.device_used_by_others = "xpu-sysman", False, 0
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            device.alloc(1 * GIB)
            device.device_used_by_others = 2 * GIB

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["foreign_device_load"])
    assert window.demand_bytes == 1 * GIB, "the device-wide counter is a guard, never demand"


@pytest.mark.parametrize(
    ("error", "reason"), [(RuntimeError("decode failed"), "exception"), (CanceledException(), "canceled")]
)
def test_a_window_closed_by_an_error_is_invalid_and_the_error_propagates(
    device: FakeDevice, error: BaseException, reason: str
):
    cache = FakeCache(device)

    with _scope() as scope:
        with pytest.raises(type(error)):
            with _model(cache, "vae").model_on_device(working_memory=_estimate()):
                device.alloc(1 * GIB)
                raise error

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, [reason])
    assert cache.locks_held() == 0


def test_a_partly_resident_model_is_measured_but_not_fit_eligible(device: FakeDevice):
    """Streamed weights are working memory the estimate does not price."""
    cache = FakeCache(device)
    cache.partial["transformer"] = 3 * GIB

    with _scope() as scope:
        with _model(cache, "transformer", total_bytes=4 * GIB).model_on_device(working_memory=_estimate()):
            pass

    (window,) = scope.windows
    assert (window.valid, window.fit_eligible, window.reasons) == (True, False, ["partial_residency"])
    assert window.resident_fraction == 0.75


def test_the_krea2_sdpa_benchmark_voids_the_window(device: FakeDevice, monkeypatch: pytest.MonkeyPatch):
    """It resets the device's peak statistics itself, under the window's feet."""
    monkeypatch.setenv("INVOKE_KREA2_SDPA_BACKEND", "cudnn")
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae", device="cuda:0").model_on_device(working_memory=_estimate()):
            pass

    (window,) = scope.windows
    assert (window.valid, window.reasons) == (False, ["krea2_sdpa_benchmark"])


def test_a_device_without_readable_counters_gives_an_invalid_window_and_no_demand(device: FakeDevice):
    device.readable = False
    cache = FakeCache(device)

    with _scope() as scope:
        with _model(cache, "vae").model_on_device(working_memory=_estimate()):
            pass

    (window,) = scope.windows
    assert (window.valid, window.demand_bytes, window.reasons) == (False, None, ["counters_unavailable"])


def test_calibration_scopes_do_not_nest(device: FakeDevice):
    with _scope():
        with pytest.raises(RuntimeError, match="do not nest"):
            with _scope():
                pass


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA or ROCm GPU")


@pytest.mark.slow
@requires_gpu
def test_on_hardware_a_vae_decode_window_is_valid_and_covers_the_reserved_growth():
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
    from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_sd15_sdxl

    torch.manual_seed(0)
    gpu = torch.device("cuda", 0)
    vae = AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=1,
        latent_channels=4,
    ).to(torch.float16)
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device=gpu,
        storage_device="cpu",
        shared_cpu_weights=None,
    )
    cache.put("vae", vae)
    loaded = LoadedModelWithoutConfig(cache.get("vae"), cache)
    latents = torch.randn(1, 4, 96, 96, dtype=torch.float16)
    estimate = estimate_vae_working_memory_sd15_sdxl(
        operation="decode", image_tensor=latents, vae=vae, tile_size=None, fp32=False, device=gpu
    )

    with calibration_scope(run_id="hw", queue_item_id=1, node_id="l2i", node_type="l2i") as scope:
        with loaded.model_on_device(working_memory=estimate) as (_, model):
            torch.cuda.synchronize(gpu)
            reserved_at_open = torch.cuda.memory_reserved(gpu)
            with torch.inference_mode():
                model.decode(latents.to(gpu))
            torch.cuda.synchronize(gpu)
            reserved_growth = torch.cuda.max_memory_reserved(gpu) - reserved_at_open

    (window,) = scope.windows
    assert window.valid, window.reasons
    assert reserved_growth > 0
    assert window.demand_bytes is not None and window.demand_bytes >= reserved_growth
    assert window.reserved is not None and window.reserved.delta == reserved_growth
    if sys.platform == "win32":
        assert (window.method, window.verified) == ("windows-d3dkmt", True)
        assert window.driver is not None
    elif sys.platform.startswith("linux") and torch.version.hip is None:
        if window.method == "torch-only":
            assert (window.driver, window.verified) == (None, False)
        else:
            assert (window.method, window.verified) == ("nvml", True)
            assert window.driver is not None

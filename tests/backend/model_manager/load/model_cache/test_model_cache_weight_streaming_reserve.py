"""The cache's VRAM bookkeeping for `async_weight_streaming`, on the CPU.

A streaming lock keeps `weight_streaming_reserve_bytes()` free for the copies in flight. The reserve arithmetic lives in
`ModelCache._load_locked_model` and `hold_model_residency`, so these tests give the cache a fake GPU's free memory and a
runtime that registers nothing for real. The registry itself is tested in `test_pinned_storage_registry.py`, the
streaming on a GPU in `cached_model/test_async_weight_streaming.py`.
"""

import logging
import threading
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast import weight_streaming
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)

MB = 2**20
RESERVE = 5 * MB
WEIGHT_MB = 8  # Per layer, in fp32.
LAYERS = 4


class _Layers(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(WEIGHT_MB * MB // 4, 1, bias=False) for _ in range(LAYERS))


def _model() -> _Layers:
    model = _Layers()
    apply_custom_layers_to_model(model)
    return model


class _FakeVram:
    """The device's free memory as the cache sees it. Moving weights to RAM grows it by what was asked."""

    def __init__(self, available: int) -> None:
        self.available = available
        self.loads: list[int] = []  # The budget of every move to VRAM.
        self.unloads: list[int] = []  # The bytes every move to RAM was asked for.

    def attach(self, cache: ModelCache) -> None:
        cache._get_vram_available = lambda working_mem_bytes=None, honor_cap=True: self.available  # type: ignore
        cache._offload_unlocked_models = lambda *args, **kwargs: 0  # type: ignore
        cache._move_model_to_vram = self._move_to_vram  # type: ignore
        cache._move_model_to_ram = self._move_to_ram  # type: ignore
        cache._release_allocator_blocks_for_reserve = lambda working_mem_bytes: None  # type: ignore

    def _move_to_vram(self, cache_entry, vram_available, max_move_bytes=None):
        self.loads.append(vram_available)
        return 0, False

    def _move_to_ram(self, cache_entry, vram_bytes_to_free, keep_required_weights_in_vram=None):
        self.unloads.append(vram_bytes_to_free)
        self.available += vram_bytes_to_free
        return vram_bytes_to_free


@pytest.fixture
def cache():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    cache = ModelCache(
        execution_device_working_mem_gb=0.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger,
    )
    cache._execution_device = torch.device("cuda")  # Policy only: every device touch is replaced below.
    yield cache
    cache._execution_device = torch.device("cpu")
    cache.shutdown()


@pytest.fixture
def reserve(monkeypatch: pytest.MonkeyPatch) -> int:
    """Streaming is possible and wants `RESERVE` for a model that streams."""
    monkeypatch.setattr(CachedModelWithPartialLoad, "weight_streaming_reserve_bytes", lambda self: RESERVE)
    return RESERVE


def _record(cache: ModelCache):
    cache.put("m", _model())
    record = cache.get("m")
    assert isinstance(record.cached_model, CachedModelWithPartialLoad)
    # On the CPU nothing is on the "cuda" compute device: the model needs all of its bytes.
    assert record.cached_model.cur_vram_bytes() == 0
    return record


def test_a_model_that_does_not_fit_loads_the_reserve_less(cache: ModelCache, reserve: int):
    record = _record(cache)
    total = record.cached_model.total_bytes()
    vram = _FakeVram(total // 2)
    vram.attach(cache)

    cache._load_locked_model(record)

    assert vram.loads == [total // 2 - reserve + MB]
    assert vram.unloads == []


def test_a_model_that_fits_fully_loads_without_the_reserve(cache: ModelCache, reserve: int):
    record = _record(cache)
    total = record.cached_model.total_bytes()
    vram = _FakeVram(total + 2 * MB)
    vram.attach(cache)

    cache._load_locked_model(record)

    # It streams nothing, so nothing is kept free for copies: a fully loaded model is much faster.
    assert vram.loads == [total + 2 * MB + MB]


def test_the_reserve_stays_out_of_the_budget_after_the_own_model_is_unloaded(cache: ModelCache, reserve: int):
    record = _record(cache)
    short = 3 * MB
    vram = _FakeVram(reserve - short)  # The reserve alone leaves the budget `short` below zero.
    vram.attach(cache)

    cache._load_locked_model(record)

    # The model's own weights are trimmed by what is missing. Of the memory that frees, the reserve stays free, so the
    # budget that follows is empty; without the reserve in it, the model would load `reserve` more.
    assert vram.unloads == [short]
    assert vram.loads == [MB]


def _held(cache: ModelCache, monkeypatch: pytest.MonkeyPatch, resident: int):
    record = _record(cache)
    monkeypatch.setattr(record.cached_model, "cur_vram_bytes", lambda: resident)
    vram = _FakeVram(0)
    vram.attach(cache)
    record.lock()
    return record, vram


@pytest.mark.parametrize(
    ("resident_after_lock", "moved"),
    [
        # Fully resident: the copies' reserve comes on top of what is above the limit.
        (32 * MB, 32 * MB + RESERVE - 20 * MB),
        # The lock settled above the limit, but under it by less than the reserve it kept free: the trim is the rest.
        (20 * MB, RESERVE),
        (20 * MB - RESERVE // 2, RESERVE // 2),
    ],
    ids=["fully-resident", "at-the-limit", "just-under-the-limit"],
)
def test_holding_residency_leaves_room_for_the_copies_within_the_limit(
    cache: ModelCache, reserve: int, monkeypatch: pytest.MonkeyPatch, resident_after_lock: int, moved: int
):
    record, vram = _held(cache, monkeypatch, resident_after_lock)

    assert cache.hold_model_residency(record, 20 * MB) == moved
    assert vram.unloads == [moved]


def test_holding_residency_leaves_a_model_alone_that_is_under_the_limit_by_its_reserve(
    cache: ModelCache, reserve: int, monkeypatch: pytest.MonkeyPatch
):
    record, vram = _held(cache, monkeypatch, 20 * MB - RESERVE)

    assert cache.hold_model_residency(record, 20 * MB) == 0
    assert vram.unloads == []


def test_holding_residency_without_streaming_counts_resident_bytes_only(
    cache: ModelCache, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(CachedModelWithPartialLoad, "weight_streaming_reserve_bytes", lambda self: 0)
    record, vram = _held(cache, monkeypatch, 32 * MB)

    assert cache.hold_model_residency(record, 20 * MB) == 12 * MB
    assert vram.unloads == [12 * MB]


def test_holding_a_limit_above_the_model_asks_for_no_reserve(cache: ModelCache, monkeypatch: pytest.MonkeyPatch):
    asked: list[bool] = []
    monkeypatch.setattr(
        CachedModelWithPartialLoad, "weight_streaming_reserve_bytes", lambda self: asked.append(True) or RESERVE
    )
    record, vram = _held(cache, monkeypatch, 32 * MB)

    assert cache.hold_model_residency(record, record.cached_model.total_bytes()) == 0
    assert not asked and vram.unloads == []


def test_a_plain_trim_frees_exactly_what_was_asked(cache: ModelCache, reserve: int, monkeypatch: pytest.MonkeyPatch):
    record, vram = _held(cache, monkeypatch, 32 * MB)

    cache.unload_model_from_vram(record, 10 * MB, keep_required_weights_in_vram=True)

    assert vram.unloads == [10 * MB]


def _cached(model: torch.nn.Module) -> CachedModelWithPartialLoad:
    return CachedModelWithPartialLoad(model=model, compute_device=torch.device("cuda"), keep_ram_copy=True)


def test_the_reserve_is_planned_while_something_can_stream(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", RESERVE)
    cached = _cached(_model())
    try:
        assert cached.weight_streaming_reserve_bytes() == RESERVE
    finally:
        cached.release_pinned_weights()


def test_no_reserve_is_planned_while_the_pinned_ram_cap_is_held_by_a_running_model(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    registry, _ = fake_streaming
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", RESERVE)
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", 2 * WEIGHT_MB * MB)
    running, waiting = _cached(_model()), _cached(_model())
    try:
        running.pin_streamed_weights()  # Takes the cap: two of its four weights fit.
        assert registry._pinned_bytes == 2 * WEIGHT_MB * MB
        assert waiting.weight_streaming_reserve_bytes() == 0

        # Once the running model's lock ends, its pins are reclaimable and the waiting model may stream again.
        running.mark_streamed_weights_idle()
        registry.drain_deferred()
        assert waiting.weight_streaming_reserve_bytes() == RESERVE
    finally:
        running.release_pinned_weights()
        waiting.release_pinned_weights()


def test_a_model_whose_weights_are_registered_keeps_its_reserve_at_a_full_cap(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    registry, _ = fake_streaming
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", RESERVE)
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", 2 * WEIGHT_MB * MB)
    cached = _cached(_model())
    try:
        cached.pin_streamed_weights()
        assert registry._pinned_bytes == 2 * WEIGHT_MB * MB
        assert cached.weight_streaming_reserve_bytes() == RESERVE
    finally:
        cached.release_pinned_weights()


def test_a_lock_that_planned_no_reserve_registers_nothing_new_even_if_room_appears(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    """The VRAM was not kept free for copies in flight, so streaming must not start in this lock."""
    registry, runtime = fake_streaming
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", RESERVE)
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", 2 * WEIGHT_MB * MB)
    running, waiting = _cached(_model()), _cached(_model())
    try:
        running.pin_streamed_weights()
        waiting.begin_streaming_lock()
        assert waiting.weight_streaming_reserve_bytes() == 0  # Planned while the cap was full ...
        running.mark_streamed_weights_idle()  # ... and room appears during the move that follows.

        waiting.pin_streamed_weights()

        assert not any(registry.is_registered(s) for s in waiting._streamable_ram_storages())
        waiting.end_streaming_lock()
        # The next lock plans again, and streams.
        waiting.begin_streaming_lock()
        assert waiting.weight_streaming_reserve_bytes() == RESERVE
        waiting.pin_streamed_weights()
        assert any(registry.is_registered(s) for s in waiting._streamable_ram_storages())
        waiting.end_streaming_lock()
    finally:
        running.release_pinned_weights()
        waiting.release_pinned_weights()


def test_a_cache_lock_keeps_the_ram_copy_registered_through_an_eviction(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    """The partial load, the repair, the fallback copies and the LoRA unpatch of a lock copy from the RAM copy on
    their own streams. Another device's eviction of the shared weights must not unpin them meanwhile."""
    registry, runtime = fake_streaming
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", RESERVE)
    cached = _cached(_model())
    cached.begin_streaming_lock()
    cached.weight_streaming_reserve_bytes()
    cached.pin_streamed_weights()
    registered = set(runtime.registered)
    assert len(registered) == LAYERS

    cached.release_pinned_weights()  # What eviction does.

    assert runtime.registered == registered, "still locked: unregistering would race the copies of the lock"
    cached.end_streaming_lock()

    assert runtime.registered == set() and registry._pinned_bytes == 0, "the last unlock unpins what was left pending"


def test_a_lock_plans_streaming_afresh_and_does_not_inherit_the_last_ones_plan(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    """The second lock finds the model fitting, so it asks for no reserve, and an in-lock trim then must not stream."""
    registry, runtime = fake_streaming
    monkeypatch.setattr(weight_streaming, "STREAM_BUDGET_BYTES", RESERVE)
    cached = _cached(_model())
    try:
        cached.begin_streaming_lock()
        assert cached.weight_streaming_reserve_bytes() == RESERVE
        cached.end_streaming_lock()
        cached.release_pinned_weights()
        cached._pinning_retired = False

        cached.begin_streaming_lock()  # Nothing asks for a reserve this time.
        cached.pin_streamed_weights()

        assert runtime.registered == set()
        cached.end_streaming_lock()
    finally:
        cached.release_pinned_weights()


def test_a_wrapper_that_never_streamed_locks_the_storages_another_wrapper_registered(fake_streaming):
    """Shared weights: the second device has no holder yet, but it copies from what the first registered."""
    registry, runtime = fake_streaming
    cached = _cached(_model())
    holder = weight_streaming.PinHolder(torch.device("cuda", 0))
    registry.update(holder, cached._streamable_ram_storages())
    cached.begin_streaming_lock()

    registry.release(holder)

    assert len(runtime.registered) == LAYERS
    cached.end_streaming_lock()
    registry.drain_deferred()
    assert runtime.registered == set()


def test_nothing_is_locked_while_streaming_is_off(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    monkeypatch.setattr(weight_streaming, "can_stream_weights_to", lambda device, has_ram_copy: False)
    cached = _cached(_model())

    cached.begin_streaming_lock()

    assert not registry._locked
    cached.end_streaming_lock()


def test_a_cache_pin_locks_the_model_for_as_long_as_any_lock_lasts(cache: ModelCache, fake_streaming):
    registry, _ = fake_streaming
    record = _record(cache)

    cache._pin(record)
    cache._pin(record)
    assert registry._locked
    cache._unpin(record)
    assert registry._locked, "one lock is still held"
    cache._unpin(record)

    assert not registry._locked


class _WatchedLock:
    """A lock that says when a thread had to wait for it."""

    def __init__(self, contended: threading.Event) -> None:
        self._lock = threading.Lock()
        self._contended = contended

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            self._contended.set()
            self._lock.acquire()

    def __exit__(self, *exc) -> None:
        self._lock.release()


def test_two_threads_pinning_a_model_for_the_first_time_create_one_holder(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    registry, runtime = fake_streaming
    cached = _cached(_model())
    decided, first_in_factory, proceed = threading.Event(), threading.Event(), threading.Event()
    created: list[weight_streaming.PinHolder] = []

    class _Holder(weight_streaming.PinHolder):
        __slots__ = ()

        def __init__(self, device: torch.device) -> None:
            super().__init__(device)
            created.append(self)
            if len(created) == 1:
                first_in_factory.set()
                assert proceed.wait(timeout=30)
            else:
                decided.set()  # A second holder: nothing kept the threads apart.

    monkeypatch.setattr(weight_streaming, "PinHolder", _Holder)
    monkeypatch.setattr(cached, "_pin_lock", _WatchedLock(decided))
    threads = [threading.Thread(target=cached.pin_streamed_weights) for _ in range(2)]
    threads[0].start()
    assert first_in_factory.wait(timeout=30)
    threads[1].start()
    assert decided.wait(timeout=30), "the second thread neither waited for the pin lock nor made a holder"
    proceed.set()
    for thread in threads:
        thread.join(timeout=60)

    try:
        assert not any(thread.is_alive() for thread in threads)
        assert len(created) == 1
        assert len(runtime.registered) == LAYERS
    finally:
        cached.release_pinned_weights()
    assert runtime.registered == set() and registry._pinned_bytes == 0

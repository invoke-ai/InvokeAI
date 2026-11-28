"""`PinnedStorageRegistry`, on the CPU, against a runtime that registers nothing for real (`fake_streaming`).

What it guards is the exclusion between cache locks and unregistration: a cache lock copies from the storages of its
model's RAM copy on its own streams, where the registry cannot see the copies, so those storages must neither be
unregistered under it nor be unregistered while it takes hold. The copies and their bit-identity on a GPU are checked
in `cached_model/test_async_weight_streaming.py`.
"""

import threading

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast import weight_streaming
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.weight_streaming import (
    PinHolder,
    storage_keys,
)

MB = 2**20
CUDA0 = torch.device("cuda", 0)


def _storage(nbytes: int = MB) -> torch.UntypedStorage:
    return torch.empty(nbytes, dtype=torch.uint8).untyped_storage()


class _SignalingCondition(threading.Condition):
    """Says when a thread is about to wait: the lock is held until `wait` releases it, so the signal cannot be early."""

    def __init__(self, about_to_wait: threading.Event) -> None:
        super().__init__()
        self._about_to_wait = about_to_wait

    def wait(self, timeout: float | None = None) -> bool:
        self._about_to_wait.set()
        return super().wait(timeout)


def test_a_release_during_a_lock_keeps_the_storage_registered_until_the_lock_ends(fake_streaming):
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])
    registry.lock(keys)

    registry.release(holder)

    assert runtime.registered, "a copy under the lock may still read it"
    assert registry._pinned_bytes == MB and not registry.holds_registrations()
    registry.unlock(keys)
    registry.drain_deferred()
    assert not runtime.registered and registry._pinned_bytes == 0


def test_a_lock_is_released_per_holder_of_the_storage(fake_streaming):
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])
    registry.lock(keys)
    registry.lock(keys)
    registry.release(holder)

    registry.unlock(keys)
    registry.drain_deferred()
    assert runtime.registered, "one lock is still held"

    registry.unlock(keys)
    registry.drain_deferred()
    assert not runtime.registered


def test_reclaiming_for_room_skips_a_storage_a_lock_holds(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, runtime = fake_streaming
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", MB)
    idle, taker = PinHolder(CUDA0), PinHolder(CUDA0)
    held, wanted = _storage(), _storage()
    registry.update(idle, [held])
    registry.idle(idle)
    registry.lock(storage_keys([held]))

    registry.update(taker, [wanted])

    assert registry.is_registered(held) and not registry.is_registered(wanted)
    registry.unlock(storage_keys([held]))
    registry.update(taker, [wanted])
    assert registry.is_registered(wanted) and not registry.is_registered(held)
    registry.release(taker)
    registry.release(idle)


def test_a_storage_registered_again_while_its_release_is_pending_is_not_registered_twice(fake_streaming):
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    first, second = PinHolder(CUDA0), PinHolder(CUDA0)
    registry.update(first, [storage])
    registry.lock(keys)
    registry.release(first)

    registry.update(second, [storage])

    assert registry.is_registered(storage)
    assert [call for call, _ in runtime.log] == ["register"], "the registration is taken back, not made again"
    assert registry._pinned_bytes == MB
    registry.unlock(keys)
    registry.release(second)
    assert not runtime.registered and registry._pinned_bytes == 0


def test_host_memory_available_drains_a_pending_release_after_a_lost_drain(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])
    registry.lock(keys)
    registry.release(holder)

    sample_entered, release_sample = threading.Event(), threading.Event()

    def sample() -> int:
        sample_entered.set()
        assert release_sample.wait(timeout=30)
        return MB

    monkeypatch.setattr(weight_streaming, "_host_memory_available", sample)
    result: list[int] = []
    accessor = threading.Thread(target=lambda: result.append(registry.host_memory_available()))
    accessor.start()
    assert sample_entered.wait(timeout=30)
    registry.unlock(keys)

    # The deferred drain loses the registration lock held by the accessor. The _registering() finally block must
    # retry it after the sampler returns, or this storage remains pinned until another registry call.
    drain = threading.Thread(target=registry.drain_deferred)
    drain.start()
    drain.join(timeout=30)
    release_sample.set()
    accessor.join(timeout=30)

    assert not accessor.is_alive() and not drain.is_alive()
    assert result == [MB]
    assert not runtime.registered
    assert registry._pinned_bytes == 0


def test_a_lock_waits_for_an_unregistration_in_progress(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    """The copies a lock issues must not start while the range is being unregistered: they would race it."""
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])
    in_unregister, proceed, waiting = threading.Event(), threading.Event(), threading.Event()
    registry._cond = _SignalingCondition(waiting)
    unregister = runtime.unregister

    def slow_unregister(ptr: int) -> int:
        in_unregister.set()
        assert proceed.wait(timeout=30)
        return unregister(ptr)

    monkeypatch.setattr(runtime, "unregister", slow_unregister)
    releaser = threading.Thread(target=registry.release, args=(holder,))
    releaser.start()
    assert in_unregister.wait(timeout=30)

    def lock_then_log() -> None:
        registry.lock(keys)
        runtime.log.append(("locked", 0))

    locker = threading.Thread(target=lock_then_log)
    locker.start()
    assert waiting.wait(timeout=30)
    assert ("locked", 0) not in runtime.log
    proceed.set()
    releaser.join(timeout=30)
    locker.join(timeout=30)

    assert [call for call, _ in runtime.log if call != "register"] == ["unregister", "locked"]
    registry.unlock(keys)


def test_a_release_does_not_wait_for_a_lock_that_is_held(fake_streaming):
    """The unregistering side waits for no lock: a cache lock waits for it, so waiting back would deadlock."""
    registry, runtime = fake_streaming
    storage = _storage()
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])
    registry.lock(storage_keys([storage]))
    done = threading.Event()

    def release() -> None:
        registry.release(holder)
        done.set()

    thread = threading.Thread(target=release)
    thread.start()

    assert done.wait(timeout=30)
    thread.join(timeout=30)
    registry.unlock(storage_keys([storage]))
    registry.drain_deferred()


def test_register_new_false_neither_registers_nor_joins_anything_new(fake_streaming):
    """A lock that planned no VRAM for copies in flight must not start streaming from what another device registered."""
    registry, runtime = fake_streaming
    shared, fresh = _storage(), _storage()
    first, second = PinHolder(CUDA0), PinHolder(CUDA0)
    registry.update(first, [shared])

    registry.update(second, [shared, fresh], register_new=False)

    assert not second.keys and not registry.is_registered(fresh)
    assert [call for call, _ in runtime.log] == ["register"]
    registry.release(first)
    registry.release(second)


def test_register_new_false_keeps_streaming_what_the_holder_holds(fake_streaming):
    registry, _ = fake_streaming
    held, other = _storage(), _storage()
    holder = PinHolder(CUDA0)
    registry.update(holder, [held])

    registry.update(holder, [held, other], register_new=False)

    assert registry.is_registered(held) and not registry.is_registered(other)
    assert holder.keys == storage_keys([held]) and holder.wanted >= storage_keys([held])
    assert registry._entries[next(iter(holder.keys))].wanted_by == 1
    registry.release(holder)


def test_room_is_found_for_a_storage_that_fits_or_is_registered_only(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", MB)
    holder = PinHolder(CUDA0)
    held, other = _storage(), _storage()
    registry.update(holder, [held])

    assert registry.has_room_for_any([held]), "registered already"
    assert not registry.has_room_for_any([other]), "the cap is held by a storage that is being streamed"
    registry.idle(holder)
    registry.drain_deferred()
    assert registry.has_room_for_any([other]), "its holder is idle: reclaimable"
    registry.release(holder)


class _InterruptingEntries(dict):
    """`_entries` whose first insert is interrupted, as a KeyboardInterrupt lands between two statements."""

    def __init__(self) -> None:
        super().__init__()
        self.interrupt = True

    def __setitem__(self, key, value) -> None:
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        super().__setitem__(key, value)


def test_an_interrupt_as_the_driver_call_returns_leaves_nothing_pinned_and_counted(fake_streaming, monkeypatch):
    registry, runtime = fake_streaming
    register = runtime.register

    def register_then_interrupt(ptr: int, nbytes: int) -> int:
        register(ptr, nbytes)  # The range is page-locked, and the interrupt arrives before the caller sees it.
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "register", register_then_interrupt)
    holder = PinHolder(CUDA0)

    with pytest.raises(KeyboardInterrupt):
        registry.update(holder, [_storage()])

    assert runtime.registered == set() and registry._pinned_bytes == 0 and not registry._entries
    registry.release(holder)


def test_an_interrupt_while_tracking_the_registration_leaves_nothing_pinned_and_counted(fake_streaming):
    registry, runtime = fake_streaming
    registry._entries = _InterruptingEntries()
    holder = PinHolder(CUDA0)

    with pytest.raises(KeyboardInterrupt):
        registry.update(holder, [_storage()])

    assert runtime.registered == set() and registry._pinned_bytes == 0 and not registry._entries
    registry.release(holder)


def test_an_interrupt_before_the_join_drops_the_entry_no_holder_holds(fake_streaming, monkeypatch):
    registry, runtime = fake_streaming
    holder = PinHolder(CUDA0)
    join = registry._join_locked
    interrupted: list[bool] = []

    def join_once_interrupted(holder_, key):
        if not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt
        join(holder_, key)

    monkeypatch.setattr(registry, "_join_locked", join_once_interrupted)

    with pytest.raises(KeyboardInterrupt):
        registry.update(holder, [_storage()])

    assert runtime.registered == set() and registry._pinned_bytes == 0 and not registry._entries
    registry.release(holder)


def test_an_interrupt_before_the_join_keeps_an_entry_another_holder_holds(fake_streaming, monkeypatch):
    registry, runtime = fake_streaming
    storage = _storage()
    other, holder = PinHolder(CUDA0), PinHolder(CUDA0)

    registered_meanwhile: list[bool] = []

    def another_registers_meanwhile(_storage_) -> None:
        # Between `update`'s read-in and its registration, as a second device's wrapper of shared weights would.
        if not registered_meanwhile:
            registered_meanwhile.append(True)
            registry.update(other, [storage])

    join = registry._join_locked

    def join_interrupted_for_one_holder(holder_, key):
        if holder_ is holder:
            raise KeyboardInterrupt
        join(holder_, key)

    monkeypatch.setattr(weight_streaming, "_fault_in", another_registers_meanwhile)
    monkeypatch.setattr(registry, "_join_locked", join_interrupted_for_one_holder)

    with pytest.raises(KeyboardInterrupt):
        registry.update(holder, [storage])

    assert registry.is_registered(storage) and runtime.registered, "the other holder still streams it"
    registry.release(other)
    assert not runtime.registered


def test_the_self_test_runs_with_the_models_device_current(monkeypatch: pytest.MonkeyPatch):
    current: list[int | None] = []
    seen: list[int | None] = []

    class _DeviceContext:
        def __init__(self, device) -> None:
            self.index = torch.device(device).index

        def __enter__(self) -> None:
            current.append(self.index)

        def __exit__(self, *exc) -> None:
            current.pop()

    monkeypatch.setattr(torch.cuda, "device", _DeviceContext)
    monkeypatch.setattr(
        weight_streaming, "_self_test_on_current_device", lambda device: seen.append(current[-1] if current else None)
    )

    weight_streaming._registration_self_test(torch.device("cuda", 1))

    assert seen == [1]


def test_registration_runs_with_the_models_device_current(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, runtime = fake_streaming
    current: list[int | None] = []
    at_call: list[int | None] = []

    class _DeviceContext:
        def __init__(self, device) -> None:
            self.index = torch.device(device).index

        def __enter__(self) -> None:
            current.append(self.index)

        def __exit__(self, *exc) -> None:
            current.pop()

    register = runtime.register
    monkeypatch.setattr(torch.cuda, "device", _DeviceContext)
    monkeypatch.setattr(
        runtime,
        "register",
        lambda ptr, nbytes: at_call.append(current[-1] if current else None) or register(ptr, nbytes),
    )
    holder = PinHolder(torch.device("cuda", 1))

    registry.update(holder, [_storage()])

    assert at_call == [1]
    registry.release(holder)


def test_a_device_streams_only_what_a_holder_computing_there_wants(fake_streaming):
    """Multi-GPU: the holder on cuda:1 registered the shared weights; the holder on cuda:0 planned no VRAM for copies
    in flight (`register_new=False`), so copies to cuda:0 of those weights stay synchronous."""
    registry, _ = fake_streaming
    cuda0, cuda1 = torch.device("cuda", 0), torch.device("cuda", 1)
    shared = [_storage(), _storage()]
    on1, on0 = PinHolder(cuda1), PinHolder(cuda0)
    registry.update(on1, shared)

    registry.update(on0, shared, register_new=False)

    for storage in shared:
        with registry.lookup(storage, cuda0) as streams:
            assert not streams
        with registry.lookup(storage, cuda1) as streams:
            assert streams
        assert registry.is_registered(storage) and not registry.is_wanted_on(storage, cuda0)
        assert registry.is_wanted_on(storage, cuda1)
    assert not registry.wants_anything_on(cuda0), "the staging gate is closed for the device that planned nothing"
    assert registry.wants_anything_on(cuda1)

    registry.update(on0, shared)  # A later lock that planned the reserve joins them.
    with registry.lookup(shared[0], cuda0) as streams:
        assert streams
    assert registry.wants_anything_on(cuda0)

    registry.release(on1)  # What cuda:0 holds stays wanted there, and stops being wanted on cuda:1.
    with registry.lookup(shared[0], cuda0) as streams:
        assert streams
    with registry.lookup(shared[0], cuda1) as streams:
        assert not streams
    assert not registry.wants_anything_on(cuda1)
    registry.release(on0)
    assert not registry.wants_anything_on(cuda0)


def test_an_idle_holder_does_not_stream(fake_streaming):
    registry, _ = fake_streaming
    cuda0 = torch.device("cuda", 0)
    storage = _storage()
    holder = PinHolder(cuda0)
    registry.update(holder, [storage])

    registry.idle(holder)
    registry.drain_deferred()

    assert registry.is_registered(storage) and not registry.is_wanted_on(storage, cuda0)
    with registry.lookup(storage, cuda0) as streams:
        assert not streams
    assert registry.activate(holder)
    assert registry.is_wanted_on(storage, cuda0)
    registry.release(holder)


def test_an_interrupt_while_unregistering_does_not_leave_keys_that_block_every_lock(fake_streaming):
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])

    class _Interrupt(BaseException):
        pass

    class _InterruptedSet(set):
        def add(self, key) -> None:
            super().add(key)
            raise _Interrupt

    registry._unregistering = _InterruptedSet()
    with pytest.raises(_Interrupt):
        registry.release(holder)

    assert not registry._unregistering
    assert runtime.registered and registry._pinned_bytes == MB, "what was not unregistered stays registered, counted"
    assert [entry.key for entry in registry._stranded] == list(keys), "and kept alive"
    waiter = threading.Thread(target=lambda: (registry.lock(keys), registry.unlock(keys)), daemon=True)
    waiter.start()
    waiter.join(timeout=10)
    assert not waiter.is_alive(), "a later lock waits for an unregistration that never ends"


def test_a_pending_release_whose_last_unlock_lost_the_drain_is_unregistered_by_the_registration_it_overlapped(
    fake_streaming, monkeypatch: pytest.MonkeyPatch
):
    """The last unlock lands while another thread registers: its non-blocking drain loses the lock, and the
    registration must apply it when it ends."""
    registry, runtime = fake_streaming
    released, other = _storage(), _storage()
    keys = storage_keys([released])
    first, second = PinHolder(CUDA0), PinHolder(CUDA0)
    registry.update(first, [released])
    registry.lock(keys)
    registry.release(first)
    assert runtime.registered, "pending behind the lock"
    register = registry._register_locked

    def register_while_the_lock_ends(*args, **kwargs):
        result = register(*args, **kwargs)
        registry.unlock(keys)
        registry.drain_deferred()  # Loses the registration lock, which this thread holds.
        return result

    monkeypatch.setattr(registry, "_register_locked", register_while_the_lock_ends)

    registry.update(second, [other])

    assert runtime.registered == {other.data_ptr()}
    assert registry._pinned_bytes == MB
    registry.release(second)


def test_pending_entries_cost_a_streamed_copy_no_lock_until_an_unlock_frees_one(fake_streaming):
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])
    registry.lock(keys)
    registry.release(holder)

    class _ForbiddenLock:
        def acquire(self, *args, **kwargs):
            raise AssertionError("the registration lock was taken with nothing to apply")

    real_lock = registry._registration_lock
    registry._registration_lock = _ForbiddenLock()
    try:
        for _ in range(3):
            registry.drain_deferred()
    finally:
        registry._registration_lock = real_lock
    assert runtime.registered

    registry.unlock(keys)
    registry.drain_deferred()

    assert not runtime.registered and registry._pinned_bytes == 0
    registry._registration_lock = _ForbiddenLock()
    try:
        registry.drain_deferred()  # The flag is cleared again: nothing is left to retry.
    finally:
        registry._registration_lock = real_lock


def test_a_last_unlock_during_another_drain_is_drained_by_it_after_it_releases_the_lock(fake_streaming, monkeypatch):
    """Thread A drains and finds the key still locked; B's last unlock then loses the non-blocking acquire, and A's
    drain must go round again after releasing."""
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])
    registry.lock(keys)
    registry.release_deferred(holder)
    unregister_pending = registry._unregister_pending_locked

    def unlock_during_the_drain() -> None:
        unregister_pending()  # The key is still locked: it stays pending, and the flag is cleared.
        monkeypatch.setattr(registry, "_unregister_pending_locked", unregister_pending)
        registry.unlock(keys)
        registry.drain_deferred()  # Loses the registration lock, which this thread holds.

    monkeypatch.setattr(registry, "_unregister_pending_locked", unlock_during_the_drain)

    registry.drain_deferred()

    assert not runtime.registered and registry._pinned_bytes == 0 and not registry._pending


def test_holders_on_one_device_count_a_shared_storage_once(fake_streaming):
    registry, _ = fake_streaming
    storage = _storage()
    first, second = PinHolder(CUDA0), PinHolder(CUDA0)
    registry.update(first, [storage])
    registry.update(second, [storage])
    assert registry._wanted_entries_on == {CUDA0: 1}

    registry.release(first)
    assert registry.wants_anything_on(CUDA0), "the second holder still wants it"
    registry.release(second)

    assert not registry.wants_anything_on(CUDA0)


def test_releasing_a_want_that_an_interrupt_cut_short_does_not_raise(fake_streaming):
    registry, _ = fake_streaming
    storage = _storage()
    holder = PinHolder(CUDA0)
    registry.update(holder, [storage])
    next(iter(registry._entries.values())).wanted_on.clear()
    registry._wanted_entries_on.clear()  # A want cut short before its counters were updated.

    registry.release(holder)

    assert not registry._entries
    assert registry._reclaimable_bytes == 0 and not registry._reclaimable
    assert registry._wanted_entries_on.get(CUDA0, 0) == 0


def test_an_unwant_of_an_already_reclaimable_storage_counts_its_bytes_once(fake_streaming):
    registry, _ = fake_streaming
    holder = PinHolder(CUDA0)
    registry.update(holder, [_storage()])
    registry.idle(holder)
    registry.drain_deferred()
    entry = next(iter(registry._entries.values()))
    assert registry._reclaimable_bytes == MB

    registry._unwant_locked(entry, holder)  # What a cut-short want leaves behind.

    assert registry._reclaimable_bytes == MB and list(registry._reclaimable) == [entry.key]
    assert registry._wanted_entries_on.get(CUDA0, 0) == 0
    registry.release(holder)


def test_low_live_host_headroom_leaves_storage_pageable(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    storage = _storage()
    faulted: list[torch.UntypedStorage] = []
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: MB - 1)
    monkeypatch.setattr(weight_streaming, "_fault_in", faulted.append)
    holder = PinHolder(CUDA0)

    registry.update(holder, [storage])

    assert faulted == []
    assert not registry.is_registered(storage)
    registry.release(holder)


def test_live_headroom_is_rechecked_before_each_storage(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    first, second = _storage(), _storage()
    faulted: list[torch.UntypedStorage] = []
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: MB)
    monkeypatch.setattr(weight_streaming, "_fault_in", faulted.append)
    holder = PinHolder(CUDA0)

    registry.update(holder, [first, second])

    assert faulted == [first]
    assert registry.is_registered(first)
    assert not registry.is_registered(second)
    registry.release(holder)


def test_headroom_is_checked_again_before_registration(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    storage = _storage()
    available = iter((MB, 0))
    faulted: list[torch.UntypedStorage] = []
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: next(available))
    monkeypatch.setattr(weight_streaming, "_fault_in", faulted.append)
    holder = PinHolder(CUDA0)

    registry.update(holder, [storage])

    assert faulted == [storage]
    assert not registry.is_registered(storage)
    registry.release(holder)


def test_registration_headroom_is_cumulative(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    first, second = _storage(), _storage()
    phase = ["fault"]
    faulted: list[torch.UntypedStorage] = []

    def available() -> int:
        return 2 * MB if phase[0] == "fault" else int(1.5 * MB)

    def fault_in(storage: torch.UntypedStorage) -> None:
        faulted.append(storage)
        if len(faulted) == 2:
            phase[0] = "register"

    monkeypatch.setattr(weight_streaming, "_host_memory_available", available)
    monkeypatch.setattr(weight_streaming, "_fault_in", fault_in)
    holder = PinHolder(CUDA0)

    registry.update(holder, [first, second])

    assert faulted == [first, second]
    assert registry.is_registered(first) != registry.is_registered(second)
    assert registry._pinned_bytes == MB
    registry.release(holder)


def test_live_headroom_is_shared_across_updates(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    first, second = _storage(), _storage()
    available = iter((2 * MB, int(1.5 * MB), int(1.5 * MB)))
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: next(available, int(1.5 * MB)))
    first_holder, second_holder = PinHolder(CUDA0), PinHolder(CUDA0)

    registry.update(first_holder, [first])
    registry.update(second_holder, [second])

    assert registry.is_registered(first)
    assert registry.is_registered(second)
    registry.release(second_holder)
    registry.release(first_holder)


def test_pending_pin_reservation_is_visible_to_host_admission(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    storage = _storage()
    entered, proceed = threading.Event(), threading.Event()
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: 2 * MB)

    def fault_in(_storage: torch.UntypedStorage) -> None:
        entered.set()
        assert proceed.wait(timeout=30)

    monkeypatch.setattr(weight_streaming, "_fault_in", fault_in)
    holder = PinHolder(CUDA0)
    thread = threading.Thread(target=registry.update, args=(holder, [storage]))
    thread.start()
    assert entered.wait(timeout=30)
    assert registry.host_memory_available() == MB
    proceed.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    registry.release(holder)


def test_host_memory_available_drains_queued_pin_release(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, runtime = fake_streaming
    holder = PinHolder(CUDA0)
    storage = _storage()
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: MB)
    registry.update(holder, [storage])
    registry.release_deferred(holder)

    assert registry.host_memory_available() == MB
    assert not registry.holds_registrations()
    assert not runtime.registered


def test_registration_headroom_failure_releases_reservation(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    storage = _storage()
    available_calls = 0

    def available() -> int:
        nonlocal available_calls
        available_calls += 1
        if available_calls == 1:
            return MB
        raise RuntimeError("headroom probe failed")

    monkeypatch.setattr(weight_streaming, "_host_memory_available", available)
    monkeypatch.setattr(weight_streaming, "_fault_in", lambda storage: None)
    holder = PinHolder(CUDA0)

    with pytest.raises(RuntimeError, match="headroom probe failed"):
        registry.update(holder, [storage])

    assert registry._host_reserved_bytes == 0
    assert not registry.is_registered(storage)
    registry.release(holder)


def test_second_fault_failure_releases_all_reservations_for_retry(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    first, second = _storage(), _storage()
    faulted: list[torch.UntypedStorage] = []

    def fault_in(storage: torch.UntypedStorage) -> None:
        faulted.append(storage)
        if storage is second:
            raise RuntimeError("fault failed")

    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: 2 * MB)
    monkeypatch.setattr(weight_streaming, "_fault_in", fault_in)
    holder = PinHolder(CUDA0)

    with pytest.raises(RuntimeError, match="fault failed"):
        registry.update(holder, [first, second])

    assert faulted == [first, second]
    assert registry._host_reserved_bytes == 0

    monkeypatch.setattr(weight_streaming, "_fault_in", lambda storage: None)
    registry.update(holder, [first])

    assert registry.is_registered(first)
    assert registry._host_reserved_bytes == 0
    registry.release(holder)


def test_registration_failure_releases_other_reserved_storages(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    first, second, third = _storage(), _storage(), _storage()
    original_register = registry._register_locked

    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: 3 * MB)
    monkeypatch.setattr(weight_streaming, "_fault_in", lambda storage: None)

    def fail_registration(key, storage, device):
        if storage is second:
            raise RuntimeError("registration failed")
        return original_register(key, storage, device)

    monkeypatch.setattr(registry, "_register_locked", fail_registration)
    holder = PinHolder(CUDA0)

    with pytest.raises(RuntimeError, match="registration failed"):
        registry.update(holder, [first, second, third])

    assert registry.is_registered(first)
    assert not registry.is_registered(second)
    assert not registry.is_registered(third)
    assert registry._host_reserved_bytes == 0

    monkeypatch.setattr(registry, "_register_locked", original_register)
    registry.update(holder, [second, third])

    assert registry.is_registered(second)
    assert registry.is_registered(third)
    assert registry._host_reserved_bytes == 0
    registry.release(holder)


def test_drain_failure_after_reservation_releases_headroom(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    storage = _storage()
    drain_calls = 0

    def drain_failure() -> None:
        nonlocal drain_calls
        drain_calls += 1
        if drain_calls == 2:
            raise RuntimeError("deferred drain failed")

    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: MB)
    monkeypatch.setattr(registry, "drain_deferred", drain_failure)
    monkeypatch.setattr(weight_streaming, "_fault_in", lambda storage: None)
    holder = PinHolder(CUDA0)

    with pytest.raises(RuntimeError, match="deferred drain failed"):
        registry.update(holder, [storage])

    assert registry._host_reserved_bytes == 0
    assert not registry.is_registered(storage)
    registry.release(holder)


def test_reclaimable_pins_do_not_count_as_returned_host_headroom(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    idle, wanted = _storage(), _storage()
    idle_holder, wanted_holder = PinHolder(CUDA0), PinHolder(CUDA0)
    registry.update(idle_holder, [idle])
    registry.idle(idle_holder)
    monkeypatch.setattr(weight_streaming, "_host_memory_available", lambda: MB - 1)

    registry.update(wanted_holder, [wanted])

    assert registry.is_registered(idle)
    assert not registry.is_registered(wanted)
    registry.release(wanted_holder)
    registry.release(idle_holder)

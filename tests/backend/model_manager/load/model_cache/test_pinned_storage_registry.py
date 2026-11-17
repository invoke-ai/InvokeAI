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
    holder = PinHolder()
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
    holder = PinHolder()
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
    idle, taker = PinHolder(), PinHolder()
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
    first, second = PinHolder(), PinHolder()
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


def test_a_lock_waits_for_an_unregistration_in_progress(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    """The copies a lock issues must not start while the range is being unregistered: they would race it."""
    registry, runtime = fake_streaming
    storage = _storage()
    keys = storage_keys([storage])
    holder = PinHolder()
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
    holder = PinHolder()
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
    first, second = PinHolder(), PinHolder()
    registry.update(first, [shared])

    registry.update(second, [shared, fresh], register_new=False)

    assert not second.keys and not registry.is_registered(fresh)
    assert [call for call, _ in runtime.log] == ["register"]
    registry.release(first)
    registry.release(second)


def test_register_new_false_keeps_streaming_what_the_holder_holds(fake_streaming):
    registry, _ = fake_streaming
    held, other = _storage(), _storage()
    holder = PinHolder()
    registry.update(holder, [held])

    registry.update(holder, [held, other], register_new=False)

    assert registry.is_registered(held) and not registry.is_registered(other)
    assert holder.keys == storage_keys([held]) and holder.wanted >= storage_keys([held])
    assert registry._entries[next(iter(holder.keys))].wanted_by == 1
    registry.release(holder)


def test_room_is_found_for_a_storage_that_fits_or_is_registered_only(fake_streaming, monkeypatch: pytest.MonkeyPatch):
    registry, _ = fake_streaming
    monkeypatch.setattr(weight_streaming, "PINNED_RAM_CAP_BYTES", MB)
    holder = PinHolder()
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
    holder = PinHolder()

    with pytest.raises(KeyboardInterrupt):
        registry.update(holder, [_storage()])

    assert runtime.registered == set() and registry._pinned_bytes == 0 and not registry._entries
    registry.release(holder)


def test_an_interrupt_while_tracking_the_registration_leaves_nothing_pinned_and_counted(fake_streaming):
    registry, runtime = fake_streaming
    registry._entries = _InterruptingEntries()
    holder = PinHolder()

    with pytest.raises(KeyboardInterrupt):
        registry.update(holder, [_storage()])

    assert runtime.registered == set() and registry._pinned_bytes == 0 and not registry._entries
    registry.release(holder)


def test_an_interrupt_before_the_join_drops_the_entry_no_holder_holds(fake_streaming, monkeypatch):
    registry, runtime = fake_streaming
    holder = PinHolder()
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
    other, holder = PinHolder(), PinHolder()

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
    holder = PinHolder()

    registry.update(holder, [_storage()], torch.device("cuda", 1))

    assert at_call == [1]
    registry.release(holder)

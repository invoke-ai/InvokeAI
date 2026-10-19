"""Worker notifications are broadcast, but only consumed by their owning worker."""

import asyncio
import threading
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.services.session_processor import session_processor_default as processor_module
from invokeai.app.services.session_processor.workflow_call_runtime import WorkflowCallQueueLifecycle
from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
from invokeai.backend.util.devices import TorchDevice


def _stop_and_join(processor, poll_events, gates=()):
    try:
        if hasattr(processor, "_stop_event"):
            processor.stop()
        for gate in gates:
            gate.set()
        for event in poll_events:
            event.set()
        for worker in processor._workers:
            if worker.thread is not None and worker.thread.ident is not None:
                worker.thread.join(20)
    finally:
        TorchDevice.clear_session_device()
        GENERATION_DEVICE_POOL.reset()


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
@pytest.mark.parametrize("producer", ["release", "batch", "stop"])
@pytest.mark.parametrize("order", ["no-competing-clear", "clear-before-notify", "notify-before-clear"])
def test_started_worker_keeps_notification_when_sibling_clears(monkeypatch, device_type, producer, order):
    devices = [torch.device(f"{device_type}:0"), torch.device(f"{device_type}:1")]
    pool = GENERATION_DEVICE_POOL
    pool.reset()
    gates = {name: Event() for name in ("a-clear", "b-clear", "b-continue", "a-wait")}
    a_before_clear, b_before_clear, b_cleared, a_at_wait, observed_ready, ran = (Event() for _ in range(6))
    polled_events = {}
    first_clears = set()
    errors = []
    error_lock = threading.Lock()
    observed = {}
    claims = []
    queue_lock = threading.Lock()
    item = SimpleNamespace(item_id=1, session_id="s", queue_id="default")
    available = producer == "release"

    def gate_wait(gate, message):
        if not gate.wait(20):
            with error_lock:
                errors.append(message)
            raise AssertionError(message)

    class ObservedEvent(Event):
        def clear(self):
            name = threading.current_thread().name
            if polled_events.get(name) is self and name not in first_clears:
                first_clears.add(name)
                if name == "session_processor_0":
                    a_before_clear.set()
                    gate_wait(gates["a-clear"], "worker A did not receive its initial-clear gate")
                else:
                    b_before_clear.set()
                    gate_wait(gates["b-clear"], "worker B did not receive its initial-clear gate")
                    super().clear()
                    b_cleared.set()
                    gate_wait(gates["b-continue"], "worker B did not receive its continuation gate")
                    return
            super().clear()

        def wait(self, timeout=None):
            name = threading.current_thread().name
            if name == "session_processor_0" and polled_events.get(name) is self and not observed:
                a_at_wait.set()
                gate_wait(gates["a-wait"], "worker A did not receive its wait gate")
                observed.update(lent=pool.is_lent(devices[0]), signalled=self.is_set())
                observed_ready.set()
            return super().wait(timeout)

    class ObservedThread(Thread):
        def __init__(self, *args, **kwargs):
            polled_events[kwargs["name"]] = kwargs["kwargs"]["poll_now_event"]
            super().__init__(*args, **kwargs)

    class Queue:
        def dequeue(self, device=None):
            with queue_lock:
                if not available or claims:
                    return None
                claims.append(device)
                return item

        def get_queue_item(self, item_id):
            return SimpleNamespace(item_id=item_id, status="in_progress")

    logger = MagicMock()
    invoker = SimpleNamespace(
        services=SimpleNamespace(
            session_queue=Queue(),
            logger=logger,
            image_moves=None,
            configuration=SimpleNamespace(
                generation_devices="auto",
                profile_graphs=False,
                multiuser=False,
                clear_vram_after_session=False,
            ),
        )
    )
    processor = processor_module.DefaultSessionProcessor(polling_interval=30)

    def run_item(_lifecycle, _item):
        ran.set()
        processor._stop_event.set()

    monkeypatch.setattr(processor_module, "ThreadEvent", ObservedEvent)
    monkeypatch.setattr(processor_module, "Thread", ObservedThread)
    # The ASGI event registry is transport, not the notification ownership under test.
    monkeypatch.setattr(processor_module, "register_events", lambda *_args: None)
    monkeypatch.setattr(TorchDevice, "get_generation_devices", lambda _setting: devices)
    monkeypatch.setattr(processor_module, "disable_conv_benchmark_empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.xpu, "set_device", lambda _device: None)
    monkeypatch.setattr(WorkflowCallQueueLifecycle, "run_queue_item", run_item)

    def notify():
        nonlocal available
        if producer == "release":
            pool.release_borrow(devices[0])
        elif producer == "batch":
            with queue_lock:
                available = True
            asyncio.run(processor._on_batch_enqueued(("batch_enqueued", None)))
        else:
            processor.stop()

    try:
        processor.start(invoker)
        assert a_before_clear.wait(20) and b_before_clear.wait(20)
        if producer == "release":
            assert pool.try_borrow_off_queue(device_type) == devices[0]
        gates["a-clear"].set()
        assert a_at_wait.wait(20)
        assert claims == []
        if order == "clear-before-notify":
            gates["b-clear"].set()
            assert b_cleared.wait(20)
            notify()
        else:
            notify()
            if order == "notify-before-clear":
                gates["b-clear"].set()
                assert b_cleared.wait(20)
        gates["a-wait"].set()
        assert observed_ready.wait(20)
        if observed["signalled"]:
            if producer == "stop":
                gates["b-continue"].set()
                gates["b-clear"].set()
                for worker in processor._workers:
                    worker.thread.join(20)
                    assert not worker.thread.is_alive()
                    assert worker.cancel_event.is_set()
            else:
                assert ran.wait(20), "notified worker did not execute pending work"
                assert claims == [str(devices[0])]
    finally:
        _stop_and_join(processor, polled_events.values(), gates.values())

    assert all(worker.thread is not None and not worker.thread.is_alive() for worker in processor._workers)
    assert errors == [], f"worker coordination failed: {errors}"
    logger.error.assert_not_called()
    assert observed["lent"] is False
    assert observed["signalled"] is True, f"another worker consumed the {producer} notification: {observed}"


@pytest.mark.parametrize("device_type", [None, "cuda", "xpu"])
def test_stop_wakes_paused_started_workers_and_preserves_cancellation(monkeypatch, device_type):
    devices = [] if device_type is None else [torch.device(f"{device_type}:{i}") for i in range(2)]
    poll_events, idle, paused = {}, {}, {}
    processor = processor_module.DefaultSessionProcessor(polling_interval=30)

    class ObservedEvent(Event):
        def wait(self, timeout=None):
            name = threading.current_thread().name
            if poll_events.get(name) is self:
                idle[name].set()
            if self is getattr(processor, "_resume_event", None) and not self.is_set():
                paused[name].set()
            return super().wait(timeout)

    class ObservedThread(Thread):
        def __init__(self, *args, **kwargs):
            name = kwargs["name"]
            poll_events[name] = kwargs["kwargs"]["poll_now_event"]
            idle[name], paused[name] = Event(), Event()
            super().__init__(*args, **kwargs)

    logger = MagicMock()
    invoker = SimpleNamespace(
        services=SimpleNamespace(
            session_queue=SimpleNamespace(dequeue=lambda **_kwargs: None),
            logger=logger,
            image_moves=None,
            configuration=SimpleNamespace(generation_devices="auto", profile_graphs=False),
        )
    )
    monkeypatch.setattr(processor_module, "ThreadEvent", ObservedEvent)
    monkeypatch.setattr(processor_module, "Thread", ObservedThread)
    monkeypatch.setattr(processor_module, "register_events", lambda *_args: None)
    monkeypatch.setattr(TorchDevice, "get_generation_devices", lambda _setting: devices)
    monkeypatch.setattr(processor_module, "disable_conv_benchmark_empty_cache", lambda: None)
    try:
        processor.start(invoker)
        assert all(event.wait(20) for event in idle.values())
        processor.pause()
        asyncio.run(processor._on_batch_enqueued(("batch_enqueued", None)))
        assert all(event.wait(20) for event in paused.values())
        processor.stop()
        for worker in processor._workers:
            worker.thread.join(20)
            assert not worker.thread.is_alive()
            assert worker.cancel_event.is_set()
        if device_type is None:
            assert len(processor._workers) == 1
            assert processor._workers[0].device is None
    finally:
        _stop_and_join(processor, poll_events.values())
    assert all(worker.thread is not None and not worker.thread.is_alive() for worker in processor._workers)
    logger.error.assert_not_called()


def test_release_wakes_every_started_worker_before_cleanup(monkeypatch):
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    pool = GENERATION_DEVICE_POOL
    pool.reset()
    poll_events, idle, returned, wakeups = {}, {}, {}, {}
    wait_gate = Event()
    processor = processor_module.DefaultSessionProcessor(polling_interval=30)

    class ObservedEvent(Event):
        def wait(self, timeout=None):
            name = threading.current_thread().name
            if poll_events.get(name) is self and name not in wakeups:
                idle[name].set()
                assert wait_gate.wait(20), f"{name} did not receive its wait gate"
                signalled = super().wait(timeout)
                wakeups[name] = signalled
                returned[name].set()
                return signalled
            return super().wait(timeout)

    class ObservedThread(Thread):
        def __init__(self, *args, **kwargs):
            name = kwargs["name"]
            poll_events[name] = kwargs["kwargs"]["poll_now_event"]
            idle[name], returned[name] = Event(), Event()
            super().__init__(*args, **kwargs)

    logger = MagicMock()
    invoker = SimpleNamespace(
        services=SimpleNamespace(
            session_queue=SimpleNamespace(dequeue=lambda **_kwargs: None),
            logger=logger,
            image_moves=None,
            configuration=SimpleNamespace(generation_devices="auto", profile_graphs=False),
        )
    )
    monkeypatch.setattr(processor_module, "ThreadEvent", ObservedEvent)
    monkeypatch.setattr(processor_module, "Thread", ObservedThread)
    monkeypatch.setattr(processor_module, "register_events", lambda *_args: None)
    monkeypatch.setattr(TorchDevice, "get_generation_devices", lambda _setting: devices)
    monkeypatch.setattr(processor_module, "disable_conv_benchmark_empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    try:
        processor.start(invoker)
        assert len(poll_events) == 2
        for name in poll_events:
            assert idle[name].wait(20), f"{name} did not reach its poll wait"
        assert all(not event.is_set() for event in poll_events.values())
        assert pool.try_borrow_off_queue("cuda") == devices[0]
        pool.release_borrow(devices[0])
        # Both workers remain gated so neither can clear a notification before it is checked.
        for name, event in poll_events.items():
            assert event.is_set(), f"release did not notify {name}"
        wait_gate.set()
        for name in poll_events:
            assert returned[name].wait(20), f"{name} did not return from its poll wait"
            assert wakeups[name] is True, f"{name}'s real poll wait did not consume the release notification"
        logger.error.assert_not_called()
    finally:
        _stop_and_join(processor, poll_events.values(), (wait_gate,))


def test_stop_after_failed_second_worker_start_wakes_first_worker_and_cancels_workers(monkeypatch):
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    processor = processor_module.DefaultSessionProcessor(polling_interval=30)
    poll_events = {}
    first_at_wait, wait_gate = Event(), Event()
    wakeups = []

    class ObservedEvent(Event):
        def wait(self, timeout=None):
            name = threading.current_thread().name
            if poll_events.get(name) is self:
                first_at_wait.set()
                assert wait_gate.wait(20), "first worker did not receive its wait gate"
                signalled = super().wait(timeout)
                wakeups.append(signalled)
                return signalled
            return super().wait(timeout)

    class FailingThread(Thread):
        def __init__(self, *args, **kwargs):
            poll_events[kwargs["name"]] = kwargs["kwargs"]["poll_now_event"]
            super().__init__(*args, **kwargs)

        def start(self):
            if self.name == "session_processor_1":
                assert first_at_wait.wait(20), "first worker did not reach its poll wait before startup failed"
                raise RuntimeError("injected second-worker startup failure")
            super().start()

    logger = MagicMock()
    invoker = SimpleNamespace(
        services=SimpleNamespace(
            session_queue=SimpleNamespace(dequeue=lambda **_kwargs: None),
            logger=logger,
            image_moves=None,
            configuration=SimpleNamespace(generation_devices="auto", profile_graphs=False),
        )
    )
    monkeypatch.setattr(processor_module, "ThreadEvent", ObservedEvent)
    monkeypatch.setattr(processor_module, "Thread", FailingThread)
    monkeypatch.setattr(processor_module, "register_events", lambda *_args: None)
    monkeypatch.setattr(TorchDevice, "get_generation_devices", lambda _setting: devices)
    monkeypatch.setattr(processor_module, "disable_conv_benchmark_empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    try:
        with pytest.raises(RuntimeError, match="injected second-worker startup failure"):
            processor.start(invoker)
        first, second = processor._workers
        assert first.thread.is_alive()
        assert second.thread.ident is None
        assert not poll_events["session_processor_0"].is_set()
        processor.stop()
        assert poll_events["session_processor_0"].is_set(), "stop did not notify the first worker's poll wait"
        wait_gate.set()
        first.thread.join(20)
        assert not first.thread.is_alive()
        assert wakeups == [True]
        assert all(worker.cancel_event.is_set() for worker in processor._workers)
        logger.error.assert_not_called()
    finally:
        _stop_and_join(processor, poll_events.values(), (wait_gate,))

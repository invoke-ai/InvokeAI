"""The invoker's service lifecycle: the session processor starts after every other service and stops before them, and
its workers leave their sessions before any other service stops."""

import asyncio
import inspect
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor import session_processor_default
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    WorkingMemoryCalibrationRequest,
)
from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
from tests.fixtures.working_memory_calibration import calibration_service


def test_the_session_processor_starts_after_and_stops_before_every_other_service() -> None:
    calls: list[tuple[str, str]] = []

    class Service:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self, invoker: Invoker) -> None:
            calls.append(("start", self.name))

        def stop(self, invoker: Invoker) -> None:
            calls.append(("stop", self.name))

    names = list(inspect.signature(InvocationServices).parameters)
    invoker = Invoker(InvocationServices(**{name: Service(name) for name in names}))
    started = [name for action, name in calls if action == "start"]
    calls.clear()
    invoker.stop()
    stopped = [name for action, name in calls if action == "stop"]

    assert sorted(started) == sorted(stopped) == sorted(names)
    assert started[-1] == stopped[0] == "session_processor"


class _ProcessorThatRunsTheResumedItemFirst(DefaultSessionProcessor):
    """Starts like the real processor, then returns only once the item pending at startup has run: a worker that is
    quicker than the rest of startup, which services started after the processor would not keep up with."""

    def __init__(self, queue: SessionQueue, item_id: int) -> None:
        super().__init__()
        self._resumed_queue = queue
        self._resumed_item_id = item_id

    def start(self, invoker: Invoker) -> None:
        super().start(invoker)
        deadline = time.monotonic() + 30
        while self._resumed_queue.get_queue_item(self._resumed_item_id).status in ("pending", "in_progress"):
            assert time.monotonic() < deadline, "the item pending at startup did not run"
            time.sleep(0.01)


def test_a_calibration_item_pending_at_startup_runs_once_every_service_started(
    tmp_path: Path, mock_invoker: Invoker, mock_sqlite_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.paged_bytes", lambda device: None
    )
    services = mock_invoker.services
    services.configuration.generation_devices = ["cpu"]
    services.configuration.offload_on_architecture_switch = False
    services.model_manager = MagicMock()
    services.model_manager.store.get_model.side_effect = lambda key: SimpleNamespace(key=key, hash=None)

    # The previous process enqueued a calibration run and stopped before its item ran.
    previous_queue = SessionQueue(mock_sqlite_database)
    services.session_queue = previous_queue
    previous_queue.start(mock_invoker)
    creator = calibration_service(tmp_path)
    creator.start(mock_invoker)
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    created = asyncio.run(creator.create_run("model-key", WorkingMemoryCalibrationRequest(graph=graph), "system"))
    [item_id] = created.item_ids

    # This process: new services over the same database and root, started by its invoker.
    queue = SessionQueue(mock_sqlite_database)
    calibration = calibration_service(tmp_path)
    processor = _ProcessorThatRunsTheResumedItemFirst(queue, item_id)
    services.session_queue, services.working_memory_calibration = queue, calibration
    services.session_processor = processor
    invoker = Invoker(services)
    try:
        item = queue.get_queue_item(item_id)
        assert item.status == "completed", item.error_message
        [run] = calibration.get_status().device_classes[0].runs
        assert (run.run_id, run.status) == (created.run_id, "completed")
    finally:
        invoker.stop()
        for worker in processor._workers:
            assert worker.thread is not None
            worker.thread.join(timeout=5)
        # The processor registered its devices and a release listener in the process-wide pool.
        GENERATION_DEVICE_POOL.reset()


# Bounds a hang only: every wait below is released by an event.
_HANG_S = 30


class _RunnerHeldInANode(DefaultSessionRunner):
    """Runs every queue item as one node with no cancel point, like a long decode: it returns once `release` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, queue_item: SessionQueueItem) -> None:
        self.entered.set()
        assert self.release.wait(timeout=_HANG_S), "the test never released the node"


class _OneItemQueue:
    def __init__(self) -> None:
        self._item: Optional[SimpleNamespace] = SimpleNamespace(
            item_id=1, session_id="session", queue_id="default", user_id="system"
        )

    def dequeue(self, device: Optional[str] = None) -> Optional[SimpleNamespace]:
        item, self._item = self._item, None
        return item

    def get_queue_item(self, item_id: int) -> SimpleNamespace:
        return SimpleNamespace(item_id=item_id, status="in_progress")


def _start_invoker(processor: DefaultSessionProcessor, teardown: list[tuple[str, bool]]) -> Invoker:
    """An invoker of `processor` and stand-in services. Each service's stop records whether a worker was alive then."""

    class Service:
        def __init__(self, name: str) -> None:
            self.name = name

        def stop(self, invoker: Invoker) -> None:
            alive = any(w.thread is not None and w.thread.is_alive() for w in processor._workers)
            teardown.append((self.name, alive))

    services = InvocationServices(**{name: Service(name) for name in inspect.signature(InvocationServices).parameters})
    services.configuration = InvokeAIAppConfig(generation_devices=["cpu"], offload_on_architecture_switch=False)
    services.logger = MagicMock()
    services.session_queue = _OneItemQueue()  # type: ignore[assignment]
    services.image_moves = None  # type: ignore[assignment]
    services.session_processor = processor
    return Invoker(services)


@pytest.fixture
def no_paging_reads(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.paged_bytes", lambda device: None
    )
    yield
    # The processors registered their devices and a release listener in the process-wide pool.
    GENERATION_DEVICE_POOL.reset()


def test_stopping_waits_for_a_worker_to_leave_its_node_before_any_other_service_stops(
    no_paging_reads: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _RunnerHeldInANode()
    processor = DefaultSessionProcessor(session_runner=runner)
    teardown: list[tuple[str, bool]] = []
    invoker = _start_invoker(processor, teardown)
    [worker] = processor._workers
    assert worker.thread is not None
    try:
        assert runner.entered.wait(timeout=_HANG_S)
        # The node returns only once stop() waits for the worker: released any earlier, it could return before a stop
        # that does not wait reached the other services.
        join = worker.thread.join

        def release_then_join(timeout: Optional[float] = None) -> None:
            runner.release.set()
            join(timeout)

        monkeypatch.setattr(worker.thread, "join", release_then_join)
        invoker.stop()

        assert not worker.thread.is_alive()
        assert teardown and [name for name, alive in teardown if alive] == []
    finally:
        runner.release.set()
        worker.thread.join(timeout=_HANG_S)


def test_a_stop_that_waits_says_which_worker_and_queue_item_it_waits_for(
    no_paging_reads: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_processor_default, "WORKER_STOP_NOTICE_SECONDS", 0.0)
    runner = _RunnerHeldInANode()
    processor = DefaultSessionProcessor(session_runner=runner)
    invoker = _start_invoker(processor, [])
    [worker] = processor._workers
    assert worker.thread is not None
    notices: list[str] = []

    def info(message: str) -> None:
        # The node returns once the stop has said it is waiting: a stop that waits in silence waits out its deadline.
        if message.startswith("Waiting"):
            notices.append(message)
            runner.release.set()

    invoker.services.logger.info.side_effect = info
    try:
        assert runner.entered.wait(timeout=_HANG_S)
        invoker.stop()

        assert not worker.thread.is_alive()
        assert len(notices) == 1
        assert "session_processor_0" in notices[0] and "queue item 1" in notices[0] and "up to 30s" in notices[0]
        invoker.services.logger.error.assert_not_called()
    finally:
        runner.release.set()
        worker.thread.join(timeout=_HANG_S)


def test_a_worker_still_in_its_node_after_the_stop_keeps_a_new_processor_from_starting(
    no_paging_reads: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop_timeout = session_processor_default.WORKER_STOP_TIMEOUT_SECONDS
    # A stop that gives up on the worker at once: the node outlasts any bound.
    monkeypatch.setattr(session_processor_default, "WORKER_STOP_TIMEOUT_SECONDS", 0.0)
    runner = _RunnerHeldInANode()
    processor = DefaultSessionProcessor(session_runner=runner)
    invoker = _start_invoker(processor, [])
    [worker] = processor._workers
    assert worker.thread is not None
    # Registered before the stop, so that its start is the one to refuse.
    successor = DefaultSessionProcessor(session_runner=DefaultSessionRunner())
    successor.register_generation_devices(InvokeAIAppConfig(generation_devices=["cpu"]))
    join = worker.thread.join
    try:
        assert runner.entered.wait(timeout=_HANG_S)
        invoker.stop()
        assert worker.thread.is_alive()
        [(message,), _] = invoker.services.logger.error.call_args
        assert "did not stop" in message and "queue item 1" in message

        # Ctrl+C stops the invoker a second time, with the full timeout: the abandoned worker gets no second wait.
        monkeypatch.setattr(session_processor_default, "WORKER_STOP_TIMEOUT_SECONDS", stop_timeout)
        joins: list[Optional[float]] = []
        monkeypatch.setattr(worker.thread, "join", lambda timeout=None: joins.append(timeout))
        invoker.stop()
        assert joins == []
        assert invoker.services.logger.error.call_count == 1

        # A restart in this process: neither a second worker nor a new device pool may replace the one still running.
        with pytest.raises(RuntimeError, match="still running"):
            _start_invoker(successor, [])
        assert successor._workers == []
        with pytest.raises(RuntimeError, match="still running"):
            DefaultSessionProcessor().register_generation_devices(InvokeAIAppConfig(generation_devices=["cpu"]))
    finally:
        runner.release.set()
        join(timeout=_HANG_S)

    # Once the old worker has exited, the process starts a processor again.
    restarted_runner = _RunnerHeldInANode()
    restarted_runner.release.set()
    restarted = DefaultSessionProcessor(session_runner=restarted_runner)
    _start_invoker(restarted, []).stop()
    assert [w.thread is not None and not w.thread.is_alive() for w in restarted._workers] == [True]

"""The invoker's service lifecycle: the session processor starts after every other service and stops before them."""

import asyncio
import inspect
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor
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

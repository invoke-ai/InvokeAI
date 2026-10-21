"""The session runner's calibration hook: calibration items run their nodes inside a probe scope, have the windows
persisted -- invalid unless the node completed -- and are admitted again at session start; every other item runs exactly
as before.

The probe itself needs a GPU, so a test node plays its part: it appends a valid window to the scope it runs in, as an
estimated `model_on_device` would when its operation finishes, and then completes, fails or is canceled.
"""

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event as ThreadEvent
from types import SimpleNamespace
from typing import Literal, Optional
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.primitives import IntegerOutput
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.session_queue.session_queue_common import Batch, BatchDatum, SessionQueueItem
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.services.users.users_common import UserCreateRequest, UserUpdateRequest
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    WorkingMemoryCalibrationRequest,
    calibration_origin,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_default import (
    WorkingMemoryCalibrationService,
)
from invokeai.backend.util.vae_working_memory import VAE_SD_SPEC
from invokeai.backend.working_memory import probe
from tests.fixtures.sqlite_database import sqlite_cursor_of
from tests.fixtures.working_memory_calibration import GPU, GPU_CLASS, OTHER_CLASS, calibration_service, measured_window

# What each test node saw: the scope it ran in, or None.
_SEEN: list[Optional[tuple[str, int, str, str]]] = []
_ON_CANCEL: list[Callable[[], None]] = []
# What a "hook" node does after its window closed, before it completes.
_WHILE_RUNNING: list[Callable[[], None]] = []


@invocation("test_working_memory_calibration_node", version="1.0.0")
class CalibrationProbeNodeInvocation(BaseInvocation):
    behaviour: Literal["complete", "fail", "cancel", "hook"] = InputField(default="complete")

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        scope = probe.current_scope()
        if scope is None:
            _SEEN.append(None)
        else:
            _SEEN.append((scope.run_id, scope.queue_item_id, scope.node_id, scope.node_type))
            scope.windows.append(measured_window(scope.run_id, scope.queue_item_id, scope.node_id))
        if self.behaviour == "fail":
            raise ValueError("decode failed")
        if self.behaviour == "cancel":
            _ON_CANCEL[0]()
            raise CanceledException()
        if self.behaviour == "hook":
            _WHILE_RUNNING[0]()
        return IntegerOutput(value=1)


class _Stats:
    @contextmanager
    def collect_stats(self, invocation, graph_execution_state_id) -> Iterator[None]:
        yield

    def log_stats(self, graph_execution_state_id) -> None:
        pass

    def reset_stats(self, graph_execution_state_id) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_seen() -> Iterator[None]:
    _SEEN.clear()
    _ON_CANCEL.clear()
    _WHILE_RUNNING.clear()
    yield
    _SEEN.clear()
    _ON_CANCEL.clear()
    _WHILE_RUNNING.clear()


@pytest.fixture
def queue(mock_invoker: Invoker, mock_sqlite_database: Database) -> SessionQueue:
    queue = SessionQueue(mock_sqlite_database)
    services = mock_invoker.services
    services.session_queue = queue
    services.performance_statistics = _Stats()
    model_manager = MagicMock()
    model_manager.store.get_model.side_effect = lambda key: SimpleNamespace(key=key, hash=None)
    services.model_manager = model_manager
    services.configuration.offload_on_architecture_switch = False
    queue.start(mock_invoker)
    return queue


@pytest.fixture
def service(tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue) -> Iterator[WorkingMemoryCalibrationService]:
    service = calibration_service(tmp_path)
    service.start(mock_invoker)
    mock_invoker.services.working_memory_calibration = service
    yield service
    service.stop(mock_invoker)


@pytest.fixture
def runner(mock_invoker: Invoker, queue: SessionQueue) -> tuple[DefaultSessionRunner, ThreadEvent]:
    runner = DefaultSessionRunner()
    cancel_event = ThreadEvent()
    runner.start(services=mock_invoker.services, cancel_event=cancel_event)
    return runner, cancel_event


def _graph(*behaviours: str) -> Graph:
    graph = Graph()
    for index, behaviour in enumerate(behaviours):
        graph.add_node(CalibrationProbeNodeInvocation(id=f"node{index}", behaviour=behaviour))
    return graph


def _create(service: WorkingMemoryCalibrationService, user_id: str, *behaviours: str) -> tuple[str, int]:
    created = asyncio.run(
        service.create_run("model-key", WorkingMemoryCalibrationRequest(graph=_graph(*behaviours)), user_id)
    )
    return created.run_id, created.item_ids[0]


def _run_next(queue: SessionQueue, runner: DefaultSessionRunner) -> SessionQueueItem:
    item = queue.dequeue()
    assert item is not None
    runner.workflow_call_queue_lifecycle.run_queue_item(item)
    return queue.get_queue_item(item.item_id)


def _stored_points(tmp_path: Path) -> list[dict]:
    path = tmp_path / "working_memory" / f"{GPU_CLASS.id}.json"
    return json.loads(path.read_bytes())["points"]


def _user(invoker: Invoker, email: str, is_admin: bool) -> str:
    request = UserCreateRequest(email=email, display_name=email, password="TestPass123", is_admin=is_admin)
    return invoker.services.users.create(request).user_id


def test_each_node_runs_in_its_own_scope_and_its_windows_are_stored(
    tmp_path: Path, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    run_id, item_id = _create(service, "system", "complete", "complete")

    item = _run_next(queue, runner[0])

    assert item.status == "completed"
    node_type = CalibrationProbeNodeInvocation.get_type()
    assert sorted(_SEEN) == [(run_id, item_id, "node0", node_type), (run_id, item_id, "node1", node_type)]
    assert sorted((point["node_id"], point["run_id"], point["valid"]) for point in _stored_points(tmp_path)) == [
        ("node0", run_id, True),
        ("node1", run_id, True),
    ]
    assert not probe.armed


@pytest.mark.parametrize(("behaviour", "reason"), [("fail", "exception"), ("cancel", "canceled")])
def test_the_windows_of_a_node_that_fails_or_is_canceled_are_stored_invalid(
    tmp_path: Path, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner, behaviour: str, reason: str
) -> None:
    """The window closed before the node raised, so the probe saw nothing wrong; `run_node` swallows the failure or the
    cancellation, and only the runner can tell the scope."""
    session_runner, cancel_event = runner
    run_id, item_id = _create(service, "system", behaviour)

    def cancel() -> None:
        queue.cancel_queue_item(item_id)
        cancel_event.set()

    _ON_CANCEL.append(cancel)
    item = _run_next(queue, session_runner)

    assert item.status == ("failed" if behaviour == "fail" else "canceled")
    [point] = _stored_points(tmp_path)
    assert point["node_id"] == "node0"
    assert (point["valid"], point["fit_eligible"], point["reasons"]) == (False, False, [reason])
    [run] = service.get_status().device_classes[0].runs
    # Only valid points make a run that ended without completing partial.
    assert run.status == ("failed" if behaviour == "fail" else "canceled")


def test_a_model_reset_while_the_node_runs_keeps_its_points_of_the_model_out(
    tmp_path: Path, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    """The node's window of the VAE closed, then the VAE was reset before the node ended: its windows are persisted
    only after that, and must not bring the VAE's points back. The next node measures after the reset."""
    _create(service, "system", "hook", "complete")
    _WHILE_RUNNING.append(lambda: service.reset_model("vae-key"))

    assert _run_next(queue, runner[0]).status == "completed"

    assert [point["node_id"] for point in _stored_points(tmp_path)] == ["node1"]


def test_an_item_whose_owner_is_not_an_admin_fails_before_any_node_runs(
    tmp_path: Path, mock_invoker: Invoker, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    mock_invoker.services.configuration.multiuser = True
    user_id = _user(mock_invoker, "user@test.com", is_admin=False)
    run_id, _ = _create(service, user_id, "complete")

    item = _run_next(queue, runner[0])

    assert item.status == "failed" and "administrators only" in (item.error_message or "")
    assert _SEEN == [] and _stored_points(tmp_path) == []
    [run] = service.get_status().device_classes[0].runs
    assert run.status == "failed" and "administrators only" in (run.error or "")


def test_an_admin_owned_item_runs_in_multiuser_mode(
    mock_invoker: Invoker, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    mock_invoker.services.configuration.multiuser = True
    admin_id = _user(mock_invoker, "admin@test.com", is_admin=True)
    _create(service, admin_id, "complete")

    assert _run_next(queue, runner[0]).status == "completed"
    assert len(_SEEN) == 1


def test_a_retry_of_a_reset_run_is_refused_at_session_start(
    service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    run_id, item_id = _create(service, "system", "fail")
    assert _run_next(queue, runner[0]).status == "failed"

    queue.retry_items_by_id("default", [item_id])
    service.reset_run(run_id)
    _SEEN.clear()
    retried = _run_next(queue, runner[0])

    assert retried.retried_from_item_id == item_id and retried.origin == calibration_origin(run_id)
    assert retried.status == "failed" and "does not exist" in (retried.error_message or "")
    assert _SEEN == []


def test_a_workflow_call_child_of_a_calibration_item_is_admitted_again(
    mock_invoker: Invoker, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    mock_invoker.services.configuration.multiuser = True
    user_id = _user(mock_invoker, "user@test.com", is_admin=False)
    run_id, _ = _create(service, user_id, "complete")
    queue.delete_queue_items_by_id(queue.get_queue_item_ids("default").item_ids)

    # A parent waiting on a saved-workflow call, as the queue would hold one: the child inherits origin and owner.
    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="workflow"))
    root = GraphExecutionState(graph=graph)
    call = root.next()
    assert isinstance(call, CallSavedWorkflowInvocation)
    frame = root.build_workflow_call_frame(call.id, "workflow")
    root.begin_waiting_on_workflow_call(frame)
    child_session = root.create_child_workflow_execution_state(_graph("complete"), frame)
    root.attach_waiting_workflow_call_child_sessions([child_session])
    with sqlite_cursor_of(queue) as cursor:
        cursor.execute(
            "INSERT INTO session_queue (queue_id, session, session_id, batch_id, priority, user_id, status, origin)"
            " VALUES ('default', ?, ?, ?, 0, ?, 'in_progress', ?);",
            (root.model_dump_json(), root.id, uuid.uuid4().hex, user_id, calibration_origin(run_id)),
        )
        root_id = int(cursor.lastrowid or 0)
    [child] = queue.enqueue_workflow_call_children(queue.get_queue_item(root_id), [(child_session, None)])
    assert child.origin == calibration_origin(run_id) and child.user_id == user_id

    item = _run_next(queue, runner[0])

    assert item.item_id == child.item_id
    assert item.status == "failed" and "administrators only" in (item.error_message or "")
    assert _SEEN == []


def test_an_ordinary_item_runs_without_a_probe_scope(
    tmp_path: Path, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    asyncio.run(queue.enqueue_batch("default", Batch(graph=_graph("complete"), origin="canvas"), prepend=False))

    item = _run_next(queue, runner[0])

    assert item.status == "completed"
    assert _SEEN == [None]
    assert not (tmp_path / "working_memory").exists()


_ESTIMATED: list[str] = []


@invocation("test_working_memory_estimate_node", version="1.0.0")
class EstimateNodeInvocation(BaseInvocation):
    """Prices a 1024px fp16 SD decode on the GPU, as `l2i` does, and records where the estimate's values came from."""

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        features = {
            "operation": "decode",
            "height": 1024,
            "width": 1024,
            "tile": None,
            "dtype": "fp16",
            "vae_class": "AutoencoderKL",
            "upcast_weight_bytes": 0,
            "score_matrix_bytes": 0,
        }
        _ESTIMATED.append(VAE_SD_SPEC.estimate(features, GPU).source)
        return IntegerOutput(value=1)


def _line_windows(run_id: str) -> list:
    """Windows that fit the fp16 decode line the estimate node prices with."""
    return [
        measured_window(run_id, estimator="vae.sd", version=VAE_SD_SPEC.version).model_copy(
            update={
                "features": {
                    "operation": "decode",
                    "height": side,
                    "width": side,
                    "tile": None,
                    "dtype": "fp16",
                    "upcast_weight_bytes": 0,
                },
                "demand_bytes": 900 * side**2 + 400 * 2**20,
            }
        )
        for side in (512, 1024)
    ]


def test_each_item_start_reads_the_calibrations_recorded_before_it(
    service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    _ESTIMATED.clear()
    graph = Graph()
    graph.add_node(EstimateNodeInvocation(id="estimate"))
    batch = Batch(graph=graph, origin="canvas")
    run_id, _ = _create(service, "system", "complete")

    asyncio.run(queue.enqueue_batch("default", batch, prepend=False))
    queue.cancel_queue_item(queue.dequeue().item_id)  # type: ignore[union-attr]  # the calibration item
    service.record_windows(run_id, _line_windows(run_id))
    _run_next(queue, runner[0])
    service.reset_run(run_id)
    asyncio.run(queue.enqueue_batch("default", batch, prepend=False))
    _run_next(queue, runner[0])

    first, second = _ESTIMATED
    assert first.startswith(f"user:{GPU_CLASS.id}:") and second.startswith("shipped:")


def test_windows_of_a_run_still_running_join_no_fit_until_it_finished(
    service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    """Every item start refreshes the fits, yet the windows of a run still running join none: an item that runs
    between a run's items (an ordinary generation here) is priced without them, and the next one after the run
    finished with them."""
    _ESTIMATED.clear()
    estimate = Graph()
    estimate.add_node(EstimateNodeInvocation(id="estimate"))
    two_items = WorkingMemoryCalibrationRequest(
        graph=_graph("complete"),
        data=[[BatchDatum(node_path="node0", field_name="behaviour", items=["complete", "complete"])]],
    )
    run_id = asyncio.run(service.create_run("model-key", two_items, "system")).run_id

    def generate() -> None:
        asyncio.run(queue.enqueue_batch("default", Batch(graph=estimate, origin="canvas"), prepend=True))
        assert _run_next(queue, runner[0]).status == "completed"

    assert _run_next(queue, runner[0]).status == "completed"  # the run's first item
    service.record_windows(run_id, _line_windows(run_id))
    generate()
    assert _run_next(queue, runner[0]).status == "completed"  # its second, which finishes it
    generate()

    assert [source.split(":")[0] for source in _ESTIMATED] == ["shipped", "user"]


def test_the_finished_steps_of_an_attempt_join_no_fit_while_a_later_step_waits(
    service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    """A client posts every step of an attempt up front, each as its own run: the first step finishing must not price
    the later, larger steps, so its windows wait for the whole attempt."""
    _ESTIMATED.clear()
    estimate = Graph()
    estimate.add_node(EstimateNodeInvocation(id="estimate"))

    def step() -> str:
        # As the client names an attempt: `<uuid>.<steps>`, the same for every step.
        request = WorkingMemoryCalibrationRequest(graph=_graph("complete"), attempt_id="attempt.2")
        return asyncio.run(service.create_run("model-key", request, "system")).run_id

    first, _ = step(), step()

    def generate() -> None:
        asyncio.run(queue.enqueue_batch("default", Batch(graph=estimate, origin="canvas"), prepend=True))
        assert _run_next(queue, runner[0]).status == "completed"

    assert _run_next(queue, runner[0]).status == "completed"  # step 1 finishes; step 2 still waits
    service.record_windows(first, _line_windows(first))
    generate()
    assert _run_next(queue, runner[0]).status == "completed"  # step 2, which finishes the attempt
    generate()

    assert [source.split(":")[0] for source in _ESTIMATED] == ["shipped", "user"]


def test_a_store_failure_fails_the_item_and_stops_the_session(
    tmp_path: Path, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner, monkeypatch
) -> None:
    run_id, _ = _create(service, "system", "complete", "complete")

    def refuse(run_id: str, windows, model_reset_epochs) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(service, "record_windows", refuse)
    item = _run_next(queue, runner[0])

    assert item.status == "failed" and "read-only file system" in (item.error_message or "")
    assert len(_SEEN) == 1


def test_a_store_that_cannot_be_read_at_node_start_fails_the_item_before_the_node_runs(
    tmp_path: Path, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner, monkeypatch
) -> None:
    """Without the reset epochs the node's windows could undo a reset, so the node does not run."""
    _create(service, "system", "complete", "complete")

    def unreadable(run_id: str) -> dict[str, int]:
        raise OSError("lock timed out")

    monkeypatch.setattr(service, "model_reset_epochs", unreadable)
    item = _run_next(queue, runner[0])

    assert item.status == "failed" and "lock timed out" in (item.error_message or "")
    assert _SEEN == [] and _stored_points(tmp_path) == []


def _refused_before_any_node(queue: SessionQueue, runner: DefaultSessionRunner, refusal: str) -> None:
    item = _run_next(queue, runner)
    assert item.status == "failed" and refusal in (item.error_message or "")
    assert _SEEN == []


def test_an_owner_deactivated_after_enqueue_is_refused(
    mock_invoker: Invoker, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    mock_invoker.services.configuration.multiuser = True
    _user(mock_invoker, "first-admin@test.com", is_admin=True)
    admin_id = _user(mock_invoker, "admin@test.com", is_admin=True)
    _create(service, admin_id, "complete")
    mock_invoker.services.users.update(admin_id, UserUpdateRequest(is_active=False))

    _refused_before_any_node(queue, runner[0], "administrators only")


def test_an_item_is_refused_once_the_machine_generates_on_two_gpus(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, runner
) -> None:
    devices = [GPU]
    service = calibration_service(tmp_path, devices=lambda: list(devices))
    service.start(mock_invoker)
    mock_invoker.services.working_memory_calibration = service
    _create(service, "system", "complete")
    devices.append(torch.device("cuda:1"))

    _refused_before_any_node(queue, runner[0], "single generation device")


def test_an_item_is_refused_on_a_gpu_of_another_device_class(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, runner
) -> None:
    creator = calibration_service(tmp_path)
    creator.start(mock_invoker)
    run_id, _ = _create(creator, "system", "complete")
    # The card was swapped: the same root, another class.
    swapped = calibration_service(tmp_path, device_class=OTHER_CLASS)
    swapped.start(mock_invoker)
    mock_invoker.services.working_memory_calibration = swapped

    _refused_before_any_node(queue, runner[0], f"created for device class {GPU_CLASS.id}")
    [run] = next(c for c in creator.get_status().device_classes if c.device_class_id == GPU_CLASS.id).runs
    assert run.run_id == run_id and run.status == "failed"


def test_an_item_removed_by_clear_queue_mid_run_is_recorded_as_canceled(
    tmp_path: Path, service: WorkingMemoryCalibrationService, queue: SessionQueue, runner
) -> None:
    session_runner, cancel_event = runner
    run_id, item_id = _create(service, "system", "cancel")

    def clear() -> None:
        queue.clear("default")
        cancel_event.set()

    _ON_CANCEL.append(clear)
    item = queue.dequeue()
    assert item is not None
    session_runner.workflow_call_queue_lifecycle.run_queue_item(item)

    assert queue.get_queue_item_ids("default").item_ids == []
    [run] = service.get_status().device_classes[0].runs
    # The node's window was kept, invalid since the node was canceled; the item, gone from the queue, will not finish.
    assert (run.status, run.point_count, run.item_ids) == ("canceled", 1, [item_id])


def test_a_runner_failure_outside_the_nodes_is_recorded_as_failed(
    service: WorkingMemoryCalibrationService, queue: SessionQueue, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_runner, _ = runner
    run_id, item_id = _create(service, "system", "complete")

    def explode(queue_item: SessionQueueItem) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(session_runner, "_on_after_run_session", explode)
    item = queue.dequeue()
    assert item is not None
    with pytest.raises(RuntimeError):
        session_runner.run(item)
    # The processor fails the item next; pruned afterwards, the run keeps the failure the runner recorded.
    queue.delete_queue_items_by_id([item_id])

    [run] = service.get_status().device_classes[0].runs
    assert (run.status, run.error) == ("failed", "RuntimeError: database is locked")

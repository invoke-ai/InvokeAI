"""The calibration store and service on a temporary root, with a real SQLite queue."""

import asyncio
import json
import logging
import os
import stat
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.session_queue.session_queue_common import (
    Batch,
    BatchDatum,
    EnqueueIdempotencyConflictError,
    SessionQueueItem,
)
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.working_memory_calibration import working_memory_calibration_store
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CALIBRATION_ORIGIN_PREFIX,
    CalibrationDeviceClass,
    CalibrationQueueFullError,
    CalibrationRunNotFoundError,
    CalibrationStoreError,
    CalibrationStoreUnusableError,
    WorkingMemoryCalibrationRequest,
    WorkingMemoryCalibrationRunCreated,
    WorkingMemoryCalibrationStatus,
    calibration_run_id,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_default import (
    WorkingMemoryCalibrationService,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_store import CalibrationStore
from tests.fixtures.sqlite_database import sqlite_cursor
from tests.fixtures.working_memory_calibration import (
    GPU_CLASS,
    OTHER_CLASS,
    HeldEnqueue,
    calibration_service,
    measured_window,
)


@pytest.fixture
def queue(mock_invoker: Invoker, mock_sqlite_database: Database) -> SessionQueue:
    queue = SessionQueue(mock_sqlite_database)
    mock_invoker.services.session_queue = queue
    model_manager = MagicMock()
    model_manager.store.get_model.side_effect = lambda key: SimpleNamespace(key=key, hash=f"blake3:{key}")
    mock_invoker.services.model_manager = model_manager
    queue.start(mock_invoker)
    return queue


def _service(root: Path, invoker: Invoker, **kwargs) -> WorkingMemoryCalibrationService:
    service = calibration_service(root, **kwargs)
    service.start(invoker)
    return service


def _request(
    values: tuple[int, ...] = (1,), *, key: Optional[str] = None, b: int = 2, graph_id: Optional[str] = None
) -> WorkingMemoryCalibrationRequest:
    # Without an id the graph gets a fresh one, as a client that sends none does.
    graph = Graph() if graph_id is None else Graph(id=graph_id)
    graph.add_node(AddInvocation(id="add", a=1, b=b))
    data = [[BatchDatum(node_path="add", field_name="a", items=list(values))]] if len(values) > 1 else None
    return WorkingMemoryCalibrationRequest(graph=graph, data=data, idempotency_key=key)


def _create(
    service: WorkingMemoryCalibrationService,
    model_key: str = "model-a",
    values: tuple[int, ...] = (1,),
    *,
    key: Optional[str] = None,
):
    return asyncio.run(service.create_run(model_key, _request(values, key=key), "system"))


def _calibration_item_ids(queue: SessionQueue) -> list[int]:
    return sorted(queue.get_queue_item_ids("default", origin_prefix=CALIBRATION_ORIGIN_PREFIX).item_ids)


def _refuse_writes(source: str, target: str) -> None:
    raise OSError("disk full")


async def _dequeued(queue: SessionQueue) -> SessionQueueItem:
    """The next item, once the queue has one: what the session processor claims as soon as an enqueue commits."""
    deadline = time.monotonic() + 10
    while (item := queue.dequeue()) is None:
        assert time.monotonic() < deadline, "nothing was enqueued"
        await asyncio.sleep(0.01)
    return item


def _store_path(root: Path, device_class_id: str = GPU_CLASS.id) -> Path:
    return root / "working_memory" / f"{device_class_id}.json"


def _run_status(status: WorkingMemoryCalibrationStatus, run_id: str):
    return next(run for device_class in status.device_classes for run in device_class.runs if run.run_id == run_id)


def test_runs_and_points_round_trip_through_a_fresh_service(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    service.record_windows(
        created.run_id,
        [
            measured_window(created.run_id, created.item_ids[0]),
            measured_window(created.run_id, created.item_ids[0], reasons=("partial_residency",)),
            measured_window(created.run_id, created.item_ids[0], reasons=("spill",)),
        ],
    )

    status = _service(tmp_path, mock_invoker).get_status()

    assert status.available and status.current_device_class_id == GPU_CLASS.id
    [device_class] = status.device_classes
    assert device_class.usable and device_class.is_current
    assert device_class.device_class == CalibrationDeviceClass.of(GPU_CLASS)
    assert device_class.device_class is not None and device_class.device_class.id == GPU_CLASS.id
    [run] = device_class.runs
    assert (run.run_id, run.model_key, run.model_hash, run.point_count) == (
        created.run_id,
        "model-a",
        "blake3:model-a",
        3,
    )
    assert run.item_ids == created.item_ids
    assert {(count.valid, count.fit_eligible): count.count for count in device_class.point_counts} == {
        (True, True): 1,
        (True, False): 1,
        (False, False): 1,
    }
    [env] = device_class.envs
    assert env.id == run.env_id == status.current_env_id
    assert env.hard["setting.attention_backend"] == str(mock_invoker.services.configuration.attention_backend)
    assert env.hard_matches_current and env.soft_matches_current and env.point_count == 3


def test_a_failed_write_keeps_the_previous_file_and_leaves_no_staging_file(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    before = _store_path(tmp_path).read_bytes()

    def refuse(source: str, target: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(CalibrationStoreError, match="disk full"):
        service.record_windows(created.run_id, [measured_window(created.run_id)])
    monkeypatch.undo()

    assert _store_path(tmp_path).read_bytes() == before
    assert sorted(path.name for path in (tmp_path / "working_memory").iterdir()) == [
        f"{GPU_CLASS.id}.json",
        f"{GPU_CLASS.id}.json.lock",
    ]


def test_a_replace_refused_by_another_processes_handle_is_retried(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows a scanner or indexer holding the file open makes the replace fail with PermissionError for a moment."""
    monkeypatch.setattr(working_memory_calibration_store, "REPLACE_RETRY_S", 0)
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    real_replace = os.replace
    refusals = [PermissionError("in use"), PermissionError("in use")]

    def busy_then_free(source: str, target: str) -> None:
        if refusals:
            raise refusals.pop()
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", busy_then_free)
    service.record_windows(created.run_id, [measured_window(created.run_id)])
    assert refusals == []
    assert len(json.loads(_store_path(tmp_path).read_bytes())["points"]) == 1

    def always_busy(source: str, target: str) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(os, "replace", always_busy)
    before = _store_path(tmp_path).read_bytes()
    with pytest.raises(CalibrationStoreError, match="in use"):
        service.record_windows(created.run_id, [measured_window(created.run_id)])
    monkeypatch.undo()
    assert _store_path(tmp_path).read_bytes() == before
    assert not any(path.name.endswith(".tmp") for path in (tmp_path / "working_memory").iterdir())


def test_two_services_sharing_a_root_keep_each_others_writes(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    # Two instances stand for two processes: separate process locks, so only the OS file lock serializes them.
    first, second = _service(tmp_path, mock_invoker), _service(tmp_path, mock_invoker)
    created = _create(first)
    writes_each = 15

    def write(service: WorkingMemoryCalibrationService, offset: int) -> None:
        for index in range(writes_each):
            service.record_windows(created.run_id, [measured_window(created.run_id, node_id=f"n{offset + index}")])

    threads = [
        threading.Thread(target=write, args=(service, offset)) for service, offset in ((first, 0), (second, 100))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    stored = json.loads(_store_path(tmp_path).read_bytes())
    assert sorted(point["node_id"] for point in stored["points"]) == sorted(
        [f"n{index}" for index in range(writes_each)] + [f"n{100 + index}" for index in range(writes_each)]
    )


def _with_extra(field: str):
    def mutate(valid: dict) -> bytes:
        target = valid["runs"][0] if field == "run" else valid["points"][0]
        target["added_by_a_newer_writer"] = True
        return json.dumps(valid).encode()

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        lambda valid: json.dumps({**valid, "schema_version": 2, "future": []}).encode(),
        lambda valid: b'{"schema_version": 1, "device_class": "truncated',
        lambda valid: json.dumps({**valid, "device_class": OTHER_CLASS.model_dump()}).encode(),
        lambda valid: json.dumps({**valid, "device_class": {**valid["device_class"], "total_mib": 1}}).encode(),
        _with_extra("run"),
        _with_extra("point"),
    ],
    ids=[
        "newer-schema",
        "invalid-json",
        "other-device-class",
        "edited-device-class",
        "extra-run-field",
        "extra-point-field",
    ],
)
def test_a_file_this_version_cannot_read_is_reported_and_never_written(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, mutate
) -> None:
    """One policy for every store model: an unknown field makes the file unusable rather than being dropped on the
    next rewrite."""
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    service.record_windows(created.run_id, [measured_window(created.run_id)])
    path = _store_path(tmp_path)
    content = mutate(json.loads(path.read_bytes()))
    path.write_bytes(content)
    item_ids = queue.get_queue_item_ids("default").item_ids

    status = service.get_status()
    [device_class] = status.device_classes
    assert not device_class.usable and device_class.problem and device_class.runs == []
    assert not status.available and "cannot be written" in (status.unavailable_reason or "")
    with pytest.raises(CalibrationStoreUnusableError):
        _create(service)
    with pytest.raises(CalibrationStoreUnusableError):
        service.reset_all()
    with pytest.raises(CalibrationStoreUnusableError):
        service.reset_model("model-a")
    service.record_windows(created.run_id, [measured_window(created.run_id)])

    assert path.read_bytes() == content
    assert service.export() == content
    # The failed create enqueued nothing.
    assert queue.get_queue_item_ids("default").item_ids == item_ids


def test_resets_remove_runs_points_and_unused_environments(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    service = _service(tmp_path, mock_invoker)
    run_a1, run_a2, run_b = _create(service, "model-a"), _create(service, "model-a"), _create(service, "model-b")
    for run in (run_a1, run_a2, run_b):
        service.record_windows(
            run.run_id, [measured_window(run.run_id, model_key="transformer"), measured_window(run.run_id)]
        )
    # A window of model-a measured inside model-b's run.
    service.record_windows(run_b.run_id, [measured_window(run_b.run_id, model_key="model-a")])

    removed = service.reset_run(run_a1.run_id)
    assert (removed.runs_removed, removed.points_removed) == (1, 2)

    removed = service.reset_model("model-a")
    assert (removed.runs_removed, removed.points_removed) == (1, 3)
    [device_class] = service.get_status().device_classes
    assert [run.run_id for run in device_class.runs] == [run_b.run_id]
    assert device_class.runs[0].point_count == 2

    removed = service.reset_all()
    assert (removed.runs_removed, removed.points_removed) == (1, 2)
    stored = json.loads(_store_path(tmp_path).read_bytes())
    assert (stored["runs"], stored["points"], stored["envs"]) == ([], [], [])
    assert stored["device_class"]["id"] == GPU_CLASS.id
    # Runs without a retry key can never be started again, so no reset keeps their ids.
    assert stored["removed_run_ids"] == []


def test_windows_of_a_reset_run_are_dropped(tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue) -> None:
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    service.reset_run(created.run_id)

    service.record_windows(created.run_id, [measured_window(created.run_id)])

    assert json.loads(_store_path(tmp_path).read_bytes())["points"] == []


def test_run_status_follows_its_items_retries_and_observations(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    service = _service(tmp_path, mock_invoker)
    created = _create(service, values=(1, 2))
    first, second = created.item_ids

    assert _run_status(service.get_status(), created.run_id).status == "running"

    for _ in range(2):
        item = queue.dequeue()
        assert item is not None
        assert service.admit(item, created.run_id) is None
        if item.item_id == first:
            queue.complete_queue_item(item.item_id)
        else:
            queue.fail_queue_item(item.item_id, "ValueError", "out of memory", "traceback")
        service.observe(item, created.run_id)
    run = _run_status(service.get_status(), created.run_id)
    assert (run.status, run.error) == ("failed", "out of memory")

    # The retry stands in for the failed item.
    queue.retry_items_by_id("default", [second])
    run = _run_status(service.get_status(), created.run_id)
    retry = max(run.item_ids)
    assert retry not in (first, second) and run.status == "running"
    retried = queue.dequeue()
    assert retried is not None and retried.item_id == retry
    assert service.admit(retried, created.run_id) is None
    queue.cancel_queue_item(retry)
    service.observe(retried, created.run_id)
    assert _run_status(service.get_status(), created.run_id).status == "canceled"

    # Only a valid point measured something: a canceled operation's own window does not make the run partial.
    service.record_windows(created.run_id, [measured_window(created.run_id, retry, reasons=("canceled",))])
    assert _run_status(service.get_status(), created.run_id).status == "canceled"
    service.record_windows(created.run_id, [measured_window(created.run_id, first)])
    assert _run_status(service.get_status(), created.run_id).status == "partial"

    # Pruned from the queue, the items keep the status the runner saw.
    queue.retry_items_by_id("default", [retry])
    rerun = queue.dequeue()
    assert rerun is not None
    assert service.admit(rerun, created.run_id) is None
    queue.complete_queue_item(rerun.item_id)
    service.observe(rerun, created.run_id)
    queue.delete_queue_items_by_id([first, second, retry, rerun.item_id])
    run = _run_status(service.get_status(), created.run_id)
    assert run.status == "completed"
    assert run.item_ids == sorted([first, second, retry, rerun.item_id])


def test_an_item_gone_from_the_queue_while_running_counts_as_canceled(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    """The process died mid-item, so the runner never recorded an outcome; then the queue history was pruned."""
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    item = queue.dequeue()
    assert item is not None and service.admit(item, created.run_id) is None
    assert _run_status(service.get_status(), created.run_id).status == "running"

    queue.delete_queue_items_by_id([item.item_id])

    assert _run_status(service.get_status(), created.run_id).status == "canceled"


def test_status_reports_machines_that_cannot_calibrate(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    import torch

    service = _service(tmp_path, mock_invoker, devices=lambda: [torch.device("cuda:0"), torch.device("cuda:1")])
    status = service.get_status()
    assert not status.available and "single generation device" in (status.unavailable_reason or "")
    assert status.current_device_class_id is None and status.device_classes == []

    service = _service(tmp_path, mock_invoker, devices=lambda: [torch.device("cpu")])
    assert "GPU memory" in (service.get_status().unavailable_reason or "")


def test_a_request_reusing_a_key_mid_enqueue_is_refused_without_taking_the_other_run(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two requests share a retry key with different graphs. The second commits while the first is still enqueuing:
    the first is refused and removes only the run it recorded."""
    service = _service(tmp_path, mock_invoker)
    held = HeldEnqueue(monkeypatch)

    async def race():
        first = asyncio.create_task(service.create_run("model-a", _request(key="shared", b=2), "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        second = await service.create_run("model-a", _request(key="shared", b=5), "system")
        held.release.set()
        with pytest.raises(EnqueueIdempotencyConflictError):
            await first
        return second

    second = asyncio.run(race())

    [item_id] = second.item_ids
    assert _calibration_item_ids(queue) == [item_id]
    [run] = service.get_status().device_classes[0].runs
    assert (run.run_id, run.item_ids) == (second.run_id, [item_id])
    item = queue.dequeue()
    assert item is not None and service.admit(item, second.run_id) is None


def test_a_request_canceled_while_enqueuing_keeps_the_run_of_the_items_it_enqueued(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client went away while the enqueue ran in its worker thread, which committed afterwards."""
    service = _service(tmp_path, mock_invoker)
    held = HeldEnqueue(monkeypatch)

    async def cancel_then_retry():
        creating = asyncio.create_task(service.create_run("model-a", _request(key="flaky-connection"), "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creating
        held.release.set()
        item = await _dequeued(queue)
        refusal = await asyncio.to_thread(service.admit, item, calibration_run_id(item.origin))
        # The client retries the lost request.
        retry = await service.create_run("model-a", _request(key="flaky-connection"), "system")
        return item, refusal, retry

    item, refusal, retry = asyncio.run(cancel_then_retry())

    assert refusal is None
    assert (retry.run_id, retry.item_ids) == (calibration_run_id(item.origin), [item.item_id])
    assert _calibration_item_ids(queue) == [item.item_id]
    [run] = service.get_status().device_classes[0].runs
    assert (run.run_id, run.item_ids, run.status) == (retry.run_id, [item.item_id], "running")


def _collect_receipt(queue: SessionQueue, database: Database, key: str) -> None:
    """The client acknowledged the key, the retention passed, and an admission collected the receipt."""
    queue.acknowledge_enqueue("default", key, "system")
    with sqlite_cursor(database) as cursor:
        cursor.execute(
            "UPDATE session_queue_enqueue_receipts SET acknowledged_at = '2000-01-01' WHERE idempotency_key = ?;",
            (key,),
        )
    collect = Batch(graph=_request().graph, origin="canvas", idempotency_key=f"collects-{key}")
    asyncio.run(queue.enqueue_batch("default", collect, False, "system"))
    assert queue.get_enqueue_receipt("default", key, "system") is None


@pytest.mark.parametrize(
    ("other_model", "other_values"), [("model-b", (1, 2)), ("model-a", (1, 3))], ids=["another-model", "other-data"]
)
def test_a_retry_key_names_one_request_across_models_batch_data_and_receipt_expiry(
    tmp_path: Path,
    mock_invoker: Invoker,
    queue: SessionQueue,
    mock_sqlite_database: Database,
    other_model: str,
    other_values: tuple[int, ...],
) -> None:
    service = _service(tmp_path, mock_invoker)
    first = _create(service, "model-a", (1, 2), key="calibrate")

    # The same key for another model or other batch data is another request, which the key's receipt refuses.
    with pytest.raises(EnqueueIdempotencyConflictError):
        _create(service, other_model, other_values, key="calibrate")
    assert [run.run_id for run in service.get_status().device_classes[0].runs] == [first.run_id]

    _collect_receipt(queue, mock_sqlite_database, "calibrate")

    # A late repeat of the first request still answers with its run and enqueues nothing.
    assert _create(service, "model-a", (1, 2), key="calibrate") == first
    assert _calibration_item_ids(queue) == first.item_ids
    # The other request now starts a run of its own.
    other = _create(service, other_model, other_values, key="calibrate")
    runs = {run.run_id: run for run in service.get_status().device_classes[0].runs}
    assert other.run_id != first.run_id and runs.keys() == {first.run_id, other.run_id}
    assert (runs[first.run_id].model_key, runs[first.run_id].item_ids) == ("model-a", first.item_ids)
    assert (runs[other.run_id].model_key, runs[other.run_id].item_ids) == (other_model, other.item_ids)
    assert _calibration_item_ids(queue) == sorted(first.item_ids + other.item_ids)


def test_a_batch_the_queue_cannot_take_whole_starts_no_run(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    """A run with only some of its cases would measure only some of them, so it gets all of its items or none."""
    service = _service(tmp_path, mock_invoker)
    mock_invoker.services.configuration.max_queue_size = 1

    with pytest.raises(CalibrationQueueFullError, match="all 2 items"):
        _create(service, values=(1, 2), key="two-sizes")

    assert queue.get_queue_item_ids("default").item_ids == []
    assert service.get_status().device_classes[0].runs == []
    # The refusal left the key unused: with room for both items, the same request starts the run.
    mock_invoker.services.configuration.max_queue_size = 2
    created = _create(service, values=(1, 2), key="two-sizes")
    assert len(created.item_ids) == 2 and _calibration_item_ids(queue) == created.item_ids


def test_a_request_canceled_while_enqueuing_into_a_queue_that_filled_up_leaves_no_run(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The creation goes on after the cancellation, finds no room left, and removes the run it recorded."""
    service = _service(tmp_path, mock_invoker)
    held = HeldEnqueue(monkeypatch)
    mock_invoker.services.configuration.max_queue_size = 1

    async def cancel_fill_then_repeat():
        creating = asyncio.create_task(service.create_run("model-a", _request(key="lost"), "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creating
        # Another run takes the last place while the canceled one waits between survey and admission.
        other = await service.create_run("model-b", _request(), "system")
        held.release.set()
        # A repeat waits for the creation still in flight, then finds no room either.
        with pytest.raises(CalibrationQueueFullError):
            await service.create_run("model-a", _request(key="lost"), "system")
        return other

    other = asyncio.run(cancel_fill_then_repeat())

    assert [run.run_id for run in service.get_status().device_classes[0].runs] == [other.run_id]
    assert _calibration_item_ids(queue) == other.item_ids


def test_an_enqueue_that_raised_after_its_commit_answers_with_the_queued_items(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue announces a batch after committing it: a failed announcement leaves the batch queued."""
    service = _service(tmp_path, mock_invoker)

    def announce(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the event bus is gone")

    monkeypatch.setattr(mock_invoker.services.events, "emit_batch_enqueued", announce)
    created = _create(service)

    [item_id] = created.item_ids
    assert _calibration_item_ids(queue) == [item_id]
    [run] = service.get_status().device_classes[0].runs
    assert (run.run_id, run.item_ids) == (created.run_id, [item_id])
    item = queue.dequeue()
    assert item is not None and service.admit(item, created.run_id) is None


def test_a_repeat_of_a_request_waits_for_the_attempt_in_flight(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, mock_invoker)
    held = HeldEnqueue(monkeypatch)

    async def repeat_while_in_flight():
        first = asyncio.create_task(service.create_run("model-a", _request(key="slow"), "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        repeat = asyncio.create_task(service.create_run("model-a", _request(key="slow"), "system"))
        # Half a second, as `tests/fixtures/races.waiting` gives a change that must wait.
        done, _ = await asyncio.wait({repeat}, timeout=0.5)
        held.release.set()
        return bool(done), await first, await repeat

    repeat_finished_early, first, repeat = asyncio.run(repeat_while_in_flight())

    assert not repeat_finished_early, "the repeat did not wait for the attempt in flight"
    assert repeat == first and _calibration_item_ids(queue) == first.item_ids


def _create_while_recording_fails(
    service: WorkingMemoryCalibrationService, monkeypatch: pytest.MonkeyPatch, request: WorkingMemoryCalibrationRequest
) -> WorkingMemoryCalibrationRunCreated:
    """Create a run whose store starts refusing writes once the run is recorded: recording its items fails."""
    held = HeldEnqueue(monkeypatch)
    replace = os.replace

    async def create():
        creating = asyncio.create_task(service.create_run("model-a", request, "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        monkeypatch.setattr(os, "replace", _refuse_writes)
        held.release.set()
        try:
            return await creating
        finally:
            monkeypatch.setattr(os, "replace", replace)

    return asyncio.run(create())


def test_a_request_without_a_key_answers_with_its_queued_items_when_the_store_fails_to_record_them(
    tmp_path: Path,
    mock_invoker: Invoker,
    queue: SessionQueue,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The items run all the same; sending the request again would queue the batch a second time."""
    service = _service(tmp_path, mock_invoker)

    with caplog.at_level(logging.WARNING):
        created = _create_while_recording_fails(service, monkeypatch, _request())

    [item_id] = created.item_ids
    assert _calibration_item_ids(queue) == [item_id]
    assert any(record.levelno == logging.WARNING and created.run_id in record.getMessage() for record in caplog.records)
    [run] = service.get_status().device_classes[0].runs
    assert (run.run_id, run.item_ids, run.status) == (created.run_id, [item_id], "running")


def test_a_repeat_answers_with_every_item_the_keys_receipt_names(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One of the unrecorded items left the queue meanwhile. The receipt still names it, so the run cannot read as
    completed once the others have run."""
    service = _service(tmp_path, mock_invoker)
    with pytest.raises(CalibrationStoreError):
        _create_while_recording_fails(service, monkeypatch, _request((1, 2, 3), key="three-sizes"))
    queued = _calibration_item_ids(queue)
    queue.delete_queue_items_by_id(queued[:1])

    repeat = _create(service, values=(1, 2, 3), key="three-sizes")

    assert len(queued) == 3 and repeat.item_ids == queued
    assert _calibration_item_ids(queue) == queued[1:]
    [stored] = json.loads(_store_path(tmp_path).read_bytes())["runs"]
    assert stored["queue_item_ids"] == queued


def test_a_repeat_takes_over_the_queued_items_once_the_keys_receipt_is_collected(
    tmp_path: Path,
    mock_invoker: Invoker,
    queue: SessionQueue,
    mock_sqlite_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enqueuing the batch again would queue it a second time."""
    service = _service(tmp_path, mock_invoker)
    with pytest.raises(CalibrationStoreError):
        _create_while_recording_fails(service, monkeypatch, _request(key="unrecorded"))
    [item_id] = _calibration_item_ids(queue)
    _collect_receipt(queue, mock_sqlite_database, "unrecorded")

    repeat = _create(service, key="unrecorded")

    assert repeat.item_ids == [item_id] and _calibration_item_ids(queue) == [item_id]
    [stored] = json.loads(_store_path(tmp_path).read_bytes())["runs"]
    assert (stored["run_id"], stored["queue_item_ids"]) == (repeat.run_id, [item_id])


def test_a_failed_enqueue_answers_with_its_own_error_when_the_queue_cannot_be_read(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether anything of the run was queued cannot be told then, so the run is kept."""
    service = _service(tmp_path, mock_invoker)

    def unreadable(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the database is gone")

    async def failing_enqueue(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the connection dropped")

    with monkeypatch.context() as patch:
        patch.setattr(queue, "enqueue_batch", failing_enqueue)
        patch.setattr(queue, "get_queue_item_ids", unreadable)
        with pytest.raises(RuntimeError, match="the connection dropped"):
            _create(service, key="unknown-outcome")

    [run] = service.get_status().device_classes[0].runs
    assert run.model_key == "model-a"


def test_requests_that_differ_only_in_their_graph_id_are_one_request(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    """A client that sends no graph id gets a fresh one with each request."""
    service = _service(tmp_path, mock_invoker)

    def create(graph_id: Optional[str]):
        return asyncio.run(service.create_run("model-a", _request(key="one-request", graph_id=graph_id), "system"))

    first = create(None)
    assert create("chosen-by-the-client") == first
    assert _calibration_item_ids(queue) == first.item_ids


def test_a_model_reset_removes_its_points_that_a_write_in_flight_brings(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window of model-a measured in model-b's run reaches the store only after model-a was reset."""
    service = _service(tmp_path, mock_invoker)
    run_b = _create(service, "model-b")
    entered, release = threading.Event(), threading.Event()
    change = CalibrationStore.change

    def held(self: CalibrationStore, *args, **kwargs):
        if threading.current_thread() is writer:
            entered.set()
            assert release.wait(timeout=10)
        return change(self, *args, **kwargs)

    monkeypatch.setattr(CalibrationStore, "change", held)
    windows = [measured_window(run_b.run_id, model_key="model-a"), measured_window(run_b.run_id)]
    writer = threading.Thread(target=service.record_windows, args=(run_b.run_id, windows))
    writer.start()
    assert entered.wait(timeout=10)
    service.reset_model("model-a")
    release.set()
    writer.join(timeout=30)

    assert not writer.is_alive()
    assert [point["model_key"] for point in json.loads(_store_path(tmp_path).read_bytes())["points"]] == ["vae-key"]


def test_a_model_reset_by_another_process_keeps_out_the_points_a_node_measured_before_it(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    """Two service instances stand for two processes sharing the root."""
    service, other = _service(tmp_path, mock_invoker), _service(tmp_path, mock_invoker)
    run_b = _create(service, "model-b")
    node_start = service.model_reset_epochs(run_b.run_id)

    other.reset_model("model-a")
    windows = [measured_window(run_b.run_id, model_key="model-a"), measured_window(run_b.run_id)]
    service.record_windows(run_b.run_id, windows, node_start)
    # A node that starts after the reset keeps its points of the model.
    service.record_windows(run_b.run_id, [measured_window(run_b.run_id, model_key="model-a")])

    points = json.loads(_store_path(tmp_path).read_bytes())["points"]
    assert [point["model_key"] for point in points] == ["vae-key", "model-a"]


def test_a_retry_stands_in_for_its_item_before_the_runner_starts_it(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    """The failed item was pruned from the queue, then its retry canceled before it ran: the run did not fail."""
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    [item_id] = created.item_ids
    item = queue.dequeue()
    assert item is not None and service.admit(item, created.run_id) is None
    queue.fail_queue_item(item_id, "ValueError", "out of memory", "traceback")
    service.observe(item, created.run_id)
    queue.retry_items_by_id("default", [item_id])
    [retry] = [queued for queued in _calibration_item_ids(queue) if queued != item_id]

    queue.prune("default")
    queue.cancel_queue_item(retry)

    run = _run_status(service.get_status(), created.run_id)
    assert (run.status, run.error, run.item_ids) == ("canceled", None, [item_id, retry])


def test_a_repeat_records_the_queued_items_of_a_model_removed_meanwhile(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key's receipt names the items; only a new request needs the model."""
    service = _service(tmp_path, mock_invoker)
    with pytest.raises(CalibrationStoreError):
        _create_while_recording_fails(service, monkeypatch, _request(key="model-removed"))
    [item_id] = _calibration_item_ids(queue)

    def removed(key: str) -> None:
        raise UnknownModelException(key)

    mock_invoker.services.model_manager.store.get_model.side_effect = removed
    repeat = _create(service, key="model-removed")

    assert repeat.item_ids == [item_id] and _calibration_item_ids(queue) == [item_id]
    [stored] = json.loads(_store_path(tmp_path).read_bytes())["runs"]
    assert (stored["run_id"], stored["queue_item_ids"]) == (repeat.run_id, [item_id])
    with pytest.raises(UnknownModelException):
        _create(service, key="another-request")


@pytest.mark.parametrize("receipt", ["retained", "collected"])
@pytest.mark.parametrize("reset", ["run", "model", "all"])
def test_a_reset_run_cannot_be_started_again_by_its_retry_key(
    tmp_path: Path,
    mock_invoker: Invoker,
    queue: SessionQueue,
    mock_sqlite_database: Database,
    reset: str,
    receipt: str,
) -> None:
    """Its items stay queued and fail at admission; repeating the request neither revives the run nor enqueues."""
    service = _service(tmp_path, mock_invoker)
    created = _create(service, key="reset-me")
    if reset == "run":
        service.reset_run(created.run_id)
    elif reset == "model":
        service.reset_model("model-a")
    else:
        service.reset_all()
    if receipt == "collected":
        _collect_receipt(queue, mock_sqlite_database, "reset-me")

    with pytest.raises(CalibrationRunNotFoundError, match="reset"):
        _create(service, key="reset-me")
    # Nor on a GPU of another class, whose file would otherwise take the run.
    with pytest.raises(CalibrationRunNotFoundError, match="reset"):
        _create(_service(tmp_path, mock_invoker, device_class=OTHER_CLASS), key="reset-me")

    assert _calibration_item_ids(queue) == created.item_ids
    assert service.get_status().device_classes[0].runs == []
    item = queue.dequeue()
    assert item is not None and item.item_id in created.item_ids
    assert "does not exist" in (service.admit(item, created.run_id) or "")


def _read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


@pytest.mark.parametrize("unwritable", ["root-is-a-file", "read-only-lock-file"])
def test_a_store_that_cannot_be_written_makes_calibration_unavailable(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, unwritable: str
) -> None:
    """The status says so, and a run cannot start: the two agree."""
    if unwritable == "root-is-a-file":
        root = tmp_path / "root"
        root.write_bytes(b"")
        service = _service(root, mock_invoker)
    else:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root writes read-only files")
        service = _service(tmp_path, mock_invoker)
        _create(service)
        lock = _store_path(tmp_path).with_name(f"{GPU_CLASS.id}.json.lock")
        _read_only(lock)
    try:
        status = service.get_status()

        assert not status.available and "cannot be written" in (status.unavailable_reason or "")
        with pytest.raises(CalibrationStoreError):
            _create(service, key="refused")
    finally:
        if unwritable != "root-is-a-file":
            os.chmod(lock, stat.S_IREAD | stat.S_IWRITE)


def test_a_repeat_enqueues_nothing_once_the_items_of_an_uncertain_admission_are_gone(
    tmp_path: Path,
    mock_invoker: Invoker,
    queue: SessionQueue,
    mock_sqlite_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The items were queued but not recorded (a 503); the client acknowledged the key, the queue collected the
    receipt and the items left the queue before the runner saw them. Enqueuing again would run the batch twice."""
    service = _service(tmp_path, mock_invoker)
    with pytest.raises(CalibrationStoreError):
        _create_while_recording_fails(service, monkeypatch, _request(key="uncertain"))
    [item_id] = _calibration_item_ids(queue)
    _collect_receipt(queue, mock_sqlite_database, "uncertain")
    queue.delete_queue_items_by_id([item_id])

    repeat = _create(service, key="uncertain")

    assert repeat.item_ids == [] and _calibration_item_ids(queue) == []
    [run] = service.get_status().device_classes[0].runs
    assert (run.run_id, run.status) == (repeat.run_id, "canceled")


def test_a_run_lists_the_item_its_retry_came_from(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run never recorded the item (its store failed, without a key), the runner never saw it, and the queue
    deleted it: only its retry still names it."""
    service = _service(tmp_path, mock_invoker)
    created = _create_while_recording_fails(service, monkeypatch, _request())
    [item_id] = created.item_ids
    item = queue.dequeue()
    assert item is not None
    queue.fail_queue_item(item_id, "ValueError", "out of memory", "traceback")
    queue.retry_items_by_id("default", [item_id])
    queue.delete_queue_items_by_id([item_id])
    [retry] = _calibration_item_ids(queue)

    run = _run_status(service.get_status(), created.run_id)

    assert run.item_ids == [item_id, retry]


def test_a_failed_replay_of_the_keys_receipt_keeps_the_runs_mark(
    tmp_path: Path,
    mock_invoker: Invoker,
    queue: SessionQueue,
    mock_sqlite_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first attempt queued the batch and could not record its items, which were pruned. A repeat replays the
    receipt and its enqueue raises: that attempt queued nothing, but the first may have."""
    service = _service(tmp_path, mock_invoker)
    with pytest.raises(CalibrationStoreError):
        _create_while_recording_fails(service, monkeypatch, _request(key="replayed"))
    queue.delete_queue_items_by_id(_calibration_item_ids(queue))

    async def failing_enqueue(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the connection dropped")

    with monkeypatch.context() as patch:
        patch.setattr(queue, "enqueue_batch", failing_enqueue)
        with pytest.raises(RuntimeError):
            _create(service, key="replayed")
    _collect_receipt(queue, mock_sqlite_database, "replayed")

    repeat = _create(service, key="replayed")

    assert repeat.item_ids == [] and _calibration_item_ids(queue) == []


def test_a_found_run_whose_enqueue_found_no_room_is_enqueued_by_a_later_repeat(
    tmp_path: Path,
    mock_invoker: Invoker,
    queue: SessionQueue,
    mock_sqlite_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run recorded without its items and without the mark (as a store written before the mark existed has it),
    with nothing queued: the repeat that marks it and finds the queue full clears the mark again."""
    service = _service(tmp_path, mock_invoker)
    with pytest.raises(CalibrationStoreError):
        _create_while_recording_fails(service, monkeypatch, _request((1, 2), key="unmarked"))
    queue.delete_queue_items_by_id(_calibration_item_ids(queue))
    _collect_receipt(queue, mock_sqlite_database, "unmarked")
    path = _store_path(tmp_path)
    stored = json.loads(path.read_bytes())
    del stored["runs"][0]["enqueue_attempted"]
    path.write_bytes(json.dumps(stored).encode())

    mock_invoker.services.configuration.max_queue_size = 1
    with pytest.raises(CalibrationQueueFullError):
        _create(service, values=(1, 2), key="unmarked")
    mock_invoker.services.configuration.max_queue_size = 10
    repeat = _create(service, values=(1, 2), key="unmarked")

    assert len(repeat.item_ids) == 2 and _calibration_item_ids(queue) == repeat.item_ids


def _in_thread(work) -> tuple[threading.Thread, list]:
    """Starts `work` on a thread of its own; the list receives what it returned or raised."""
    outcome: list = []

    def run() -> None:
        try:
            outcome.append(work())
        except BaseException as error:  # noqa: BLE001 - returned to the test
            outcome.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def test_a_repeat_from_another_process_waits_for_the_attempt_that_is_enqueuing(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first process recorded and marked the run and is enqueuing it. Two service instances on one root stand for
    two processes: they share nothing in memory, only the root's files and the database."""
    first, second = _service(tmp_path, mock_invoker), _service(tmp_path, mock_invoker)
    held = HeldEnqueue(monkeypatch)
    creating, created = _in_thread(lambda: _create(first, key="two-processes"))
    assert held.entered.wait(timeout=10)

    repeating, repeated = _in_thread(lambda: _create(second, key="two-processes"))
    repeating.join(timeout=0.5)
    waited = repeating.is_alive()
    held.release.set()
    creating.join(timeout=30)
    repeating.join(timeout=30)

    assert waited, f"the repeat did not wait for the attempt in flight; it answered {repeated!r}"
    assert repeated == created and len(created[0].item_ids) == 1
    assert _calibration_item_ids(queue) == created[0].item_ids


def test_a_request_from_another_process_cannot_prepare_while_the_first_attempt_records_the_run(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepared side by side, both attempts would enqueue: the one resuming after the queue collected the other's
    receipt queues the batch again, and its items replace the other's."""
    first, second = _service(tmp_path, mock_invoker), _service(tmp_path, mock_invoker)
    first_store = first._store
    entered, release = threading.Event(), threading.Event()
    change = CalibrationStore.change

    def held(self: CalibrationStore, *args, **kwargs):
        # The first change of the first process's store records the run.
        if self is first_store and not entered.is_set():
            entered.set()
            assert release.wait(timeout=10)
        return change(self, *args, **kwargs)

    monkeypatch.setattr(CalibrationStore, "change", held)
    creating, created = _in_thread(lambda: _create(first, key="side-by-side"))
    assert entered.wait(timeout=10)

    repeating, repeated = _in_thread(lambda: _create(second, key="side-by-side"))
    repeating.join(timeout=0.5)
    waited = repeating.is_alive()
    release.set()
    creating.join(timeout=30)
    repeating.join(timeout=30)

    assert waited, f"the other request prepared while the first recorded the run; it answered {repeated!r}"
    assert repeated == created and len(created[0].item_ids) == 1
    assert _calibration_item_ids(queue) == created[0].item_ids
    [stored] = json.loads(_store_path(tmp_path).read_bytes())["runs"]
    assert stored["queue_item_ids"] == created[0].item_ids


@pytest.mark.parametrize(("afterwards", "outcome"), [("pruned", "completed"), ("failed-in-queue", "failed")])
def test_the_final_observation_keeps_a_recorded_completion_of_an_item_the_queue_no_longer_has(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, afterwards: str, outcome: str
) -> None:
    """The runner recorded the completion, then the queue completed the item and pruned it before the runner's last
    record. A queue row that still exists and failed the item has the last word."""
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    item = queue.dequeue()
    assert item is not None and service.admit(item, created.run_id) is None
    service.observe_completed(item, created.run_id)
    if afterwards == "pruned":
        queue.complete_queue_item(item.item_id)
        queue.prune("default")
    else:
        queue.fail_queue_item(item.item_id, "RuntimeError", "the session ended badly", "traceback")

    service.observe(item, created.run_id)

    assert _run_status(service.get_status(), created.run_id).status == outcome


def test_repeats_in_one_process_leave_one_worker_waiting_for_the_file_lock(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other repeats wait on the event loop: a burst would otherwise hold a worker thread each, polling the lock."""
    service = _service(tmp_path, mock_invoker)
    held = HeldEnqueue(monkeypatch)
    count = threading.Lock()
    inside, most = [0], [0]
    creation_lock = service._store.creation_lock

    @contextmanager
    def counted(run_id: str):
        with count:
            inside[0] += 1
            most[0] = max(most[0], inside[0])
        try:
            with creation_lock(run_id):
                yield
        finally:
            with count:
                inside[0] -= 1

    monkeypatch.setattr(service._store, "creation_lock", counted)

    async def burst():
        first = asyncio.create_task(service.create_run("model-a", _request(key="burst"), "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        repeats = [
            asyncio.create_task(service.create_run("model-a", _request(key="burst"), "system")) for _ in range(8)
        ]
        # Half a second, as `tests/fixtures/races.waiting` gives a change that must wait.
        await asyncio.wait(repeats, timeout=0.5)
        most_while_held = most[0]
        held.release.set()
        return most_while_held, await first, await asyncio.gather(*repeats)

    most_while_held, first, repeats = asyncio.run(burst())

    assert most_while_held == 1
    assert all(repeat == first for repeat in repeats) and _calibration_item_ids(queue) == first.item_ids

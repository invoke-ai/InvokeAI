"""The calibration store and service on a temporary root, with a real SQLite queue."""

import asyncio
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import BatchDatum, EnqueueIdempotencyConflictError
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.working_memory_calibration import working_memory_calibration_store
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationDeviceClass,
    CalibrationStoreError,
    CalibrationStoreUnusableError,
    WorkingMemoryCalibrationRequest,
    WorkingMemoryCalibrationStatus,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_default import (
    WorkingMemoryCalibrationService,
)
from tests.fixtures.working_memory_calibration import (
    GPU_CLASS,
    OTHER_CLASS,
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


def _request(values: tuple[int, ...] = (1,)) -> WorkingMemoryCalibrationRequest:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    data = [[BatchDatum(node_path="add", field_name="a", items=list(values))]] if len(values) > 1 else None
    return WorkingMemoryCalibrationRequest(graph=graph, data=data)


def _create(service: WorkingMemoryCalibrationService, model_key: str = "model-a", values: tuple[int, ...] = (1,)):
    return asyncio.run(service.create_run(model_key, _request(values), "system"))


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


def _attempt_request(attempt_id: str | None, key: str | None = "retry") -> WorkingMemoryCalibrationRequest:
    return _request().model_copy(update={"attempt_id": attempt_id, "idempotency_key": key})


def test_a_runs_attempt_is_stored_and_returned_and_a_run_without_one_writes_none(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    service = _service(tmp_path, mock_invoker)
    with_attempt = asyncio.run(service.create_run("model-a", _attempt_request("attempt-1"), "system"))
    without = asyncio.run(service.create_run("model-a", _attempt_request(None, None), "system"))

    runs = {run["run_id"]: run for run in json.loads(_store_path(tmp_path).read_bytes())["runs"]}
    assert runs[with_attempt.run_id]["attempt_id"] == "attempt-1"
    # Absent rather than null: a version that forbids unknown fields still reads a run without one.
    assert "attempt_id" not in runs[without.run_id]
    status = {run.run_id: run.attempt_id for run in service.get_status().device_classes[0].runs}
    assert status == {with_attempt.run_id: "attempt-1", without.run_id: None}
    summary = {run.run_id: run.attempt_id for run in service.get_model_summary("model-a").runs}
    assert summary == status


def test_a_retry_key_names_one_run_per_attempt(tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue) -> None:
    """The run id derives from the attempt too, so a key reused under another attempt never returns the first
    attempt's run: the queue refuses the different batch, and the attempt's own run is discarded."""
    service = _service(tmp_path, mock_invoker)

    request = _attempt_request("a")
    first = asyncio.run(service.create_run("model-a", request, "system"))
    replay = asyncio.run(service.create_run("model-a", request.model_copy(deep=True), "system"))
    with pytest.raises(EnqueueIdempotencyConflictError):
        asyncio.run(service.create_run("model-a", _attempt_request("b"), "system"))

    assert replay.run_id == first.run_id and replay.item_ids == first.item_ids
    assert [run.run_id for run in service.get_status().device_classes[0].runs] == [first.run_id]
    assert asyncio.run(service.create_run("model-a", _attempt_request("b", "other-key"), "system")).run_id not in {
        first.run_id
    }


def test_runs_stored_before_attempts_and_soft_facts_existed_still_load(
    tmp_path: Path, mock_invoker: Invoker, queue: SessionQueue
) -> None:
    service = _service(tmp_path, mock_invoker)
    created = _create(service)
    path = _store_path(tmp_path)
    stored = json.loads(path.read_bytes())
    assert all("attempt_id" not in run for run in stored["runs"])
    assert [run["soft"] for run in stored["runs"]] == [stored["envs"][0]["soft"]]
    for run in stored["runs"]:
        del run["soft"]
    path.write_bytes(json.dumps(stored).encode())

    [run] = _service(tmp_path, mock_invoker).get_status().device_classes[0].runs

    assert (run.run_id, run.attempt_id) == (created.run_id, None)
    # Rewritten by a later change, the run stays without the field it never had.
    service.record_windows(created.run_id, [measured_window(created.run_id)])
    assert "soft" not in json.loads(path.read_bytes())["runs"][0]


@pytest.mark.parametrize("attempt_id", ["", "x" * 65, "has space", "slash/no"])
def test_an_attempt_id_outside_its_alphabet_or_length_is_rejected(attempt_id: str) -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    with pytest.raises(ValueError):
        WorkingMemoryCalibrationRequest(graph=graph, attempt_id=attempt_id)


@pytest.mark.parametrize(
    ("platform", "method", "verified"), [("win32", "windows-d3dkmt", True), ("linux", "nvml", False)]
)
def test_status_and_summary_say_whether_this_gpus_measurement_is_verified(
    tmp_path: Path,
    mock_invoker: Invoker,
    queue: SessionQueue,
    monkeypatch: pytest.MonkeyPatch,
    platform,
    method,
    verified,
) -> None:
    import sys

    import torch

    service = _service(tmp_path, mock_invoker)
    _create(service)
    # A file of a class no generation device here belongs to.
    _service(tmp_path, mock_invoker, device_class=OTHER_CLASS)._store.change(
        OTHER_CLASS.id, lambda file: (True, None), create=CalibrationDeviceClass.of(OTHER_CLASS)
    )
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(sys, "platform", platform)

    classes = {entry.device_class_id: entry for entry in service.get_status().device_classes}
    summary = service.get_model_summary("model-a")

    current = classes[GPU_CLASS.id]
    assert (current.measurement_method, current.measurement_verified) == (method, verified)
    assert (classes[OTHER_CLASS.id].measurement_method, classes[OTHER_CLASS.id].measurement_verified) == (None, None)
    assert (summary.measurement_method, summary.measurement_verified) == (method, verified)
    assert service.get_model_summary("model-a", OTHER_CLASS.id).measurement_verified is None

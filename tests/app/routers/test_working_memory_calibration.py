"""The calibration routes and the reserved origin, against a real SQLite queue and a store on a temporary root."""

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from invokeai.app.api_app import app
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CALIBRATION_DESTINATION,
    calibration_origin,
)
from tests.app.routers.conftest import MockApiDependencies, _auth
from tests.fixtures.working_memory_calibration import (
    GPU,
    GPU_CLASS,
    HeldEnqueue,
    calibration_service,
    measured_window,
)

_DEVICES: list[torch.device] = [GPU]


@pytest.fixture
def calibration(
    tmp_path: Path, monkeypatch: Any, mock_invoker: Invoker, mock_sqlite_database: Database, enable_multiuser: Any
):
    """A started queue, a model store that knows `model-key`, and the calibration service on `tmp_path`."""
    services = mock_invoker.services
    queue = SessionQueue(mock_sqlite_database)
    services.session_queue = queue
    queue.start(mock_invoker)

    def get_model(key: str) -> MagicMock:
        if key != "model-key":
            raise UnknownModelException(f"unknown model {key}")
        return MagicMock(key=key, hash="blake3:abc")

    services.model_manager.store.get_model.side_effect = get_model
    _DEVICES[:] = [GPU]
    service = calibration_service(tmp_path, devices=lambda: list(_DEVICES))
    service.start(mock_invoker)
    services.working_memory_calibration = service

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.session_queue.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    return queue, service


def _graph() -> dict:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2, use_cache=True))
    return graph.model_dump(mode="json")


def _calibrate(client: TestClient, token: str, body: dict | None = None, key: str = "model-key"):
    return client.post(
        f"/api/v2/models/i/{key}/working_memory/calibrate",
        json=body if body is not None else {"graph": _graph()},
        headers=_auth(token),
    )


def test_calibrate_enqueues_the_run_uncached_with_the_reserved_origin(
    client: TestClient, admin_token: str, calibration
) -> None:
    queue, service = calibration

    response = _calibrate(client, admin_token)

    assert response.status_code == 201, response.text
    created = response.json()
    assert created["device_class_id"] == GPU_CLASS.id
    [item_id] = created["item_ids"]
    item = queue.get_queue_item(item_id)
    assert item.origin == calibration_origin(created["run_id"]) and item.destination == CALIBRATION_DESTINATION
    assert [node.use_cache for node in item.session.graph.nodes.values()] == [False]
    [run] = service.get_status().device_classes[0].runs
    assert (run.run_id, run.model_key, run.item_ids, run.status) == (
        created["run_id"],
        "model-key",
        [item_id],
        "running",
    )


def test_non_admins_cannot_calibrate_reset_read_or_export(
    client: TestClient, admin_token: str, user1_token: str, calibration
) -> None:
    queue, _ = calibration
    run_id = _calibrate(client, admin_token).json()["run_id"]
    item_ids = queue.get_queue_item_ids("default").item_ids

    responses = [
        _calibrate(client, user1_token),
        client.get("/api/v2/models/working_memory/status", headers=_auth(user1_token)),
        client.get("/api/v2/models/working_memory/export", headers=_auth(user1_token)),
        client.delete(f"/api/v2/models/working_memory/runs/{run_id}", headers=_auth(user1_token)),
        client.delete("/api/v2/models/i/model-key/working_memory", headers=_auth(user1_token)),
        client.delete("/api/v2/models/working_memory", headers=_auth(user1_token)),
    ]

    assert [response.status_code for response in responses] == [403] * 6
    assert queue.get_queue_item_ids("default").item_ids == item_ids


@pytest.mark.parametrize("origin", [calibration_origin("forged"), "INVOKEAI:WM-Calibration:forged"])
def test_enqueue_batch_refuses_the_reserved_origin_even_to_admins(
    client: TestClient, admin_token: str, calibration, origin: str
) -> None:
    queue, _ = calibration

    response = client.post(
        "/api/v1/queue/default/enqueue_batch",
        json={"batch": {"graph": _graph(), "origin": origin, "runs": 1}},
        headers=_auth(admin_token),
    )

    assert response.status_code == 400 and "reserved" in response.json()["detail"]
    assert queue.get_queue_item_ids("default").item_ids == []


def test_enqueue_batch_documents_the_reserved_origin_refusal() -> None:
    responses = app.openapi()["paths"]["/api/v1/queue/{queue_id}/enqueue_batch"]["post"]["responses"]

    assert "reserved for working-memory calibration" in responses["400"]["description"]


def test_calibrate_refuses_more_than_one_generation_device_and_cpu(
    client: TestClient, admin_token: str, calibration
) -> None:
    queue, _ = calibration
    _DEVICES[:] = [GPU, torch.device("cuda:1")]
    assert _calibrate(client, admin_token).status_code == 409
    _DEVICES[:] = [torch.device("cpu")]
    assert _calibrate(client, admin_token).status_code == 409
    assert queue.get_queue_item_ids("default").item_ids == []


def test_calibrate_rejects_unknown_models_workflow_calls_and_server_owned_fields(
    client: TestClient, admin_token: str, calibration
) -> None:
    queue, service = calibration
    call_graph = Graph()
    call_graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="workflow"))
    use_cache_data = [[{"node_path": "add", "field_name": "use_cache", "items": [True]}]]

    assert _calibrate(client, admin_token, key="missing").status_code == 404
    assert _calibrate(client, admin_token, {"graph": call_graph.model_dump(mode="json")}).status_code == 422
    assert _calibrate(client, admin_token, {"graph": _graph(), "data": use_cache_data}).status_code == 422
    assert _calibrate(client, admin_token, {"graph": _graph(), "origin": "canvas"}).status_code == 422
    assert queue.get_queue_item_ids("default").item_ids == []
    assert service.get_status().device_classes[0].runs == []


def test_retrying_a_calibration_item_keeps_its_origin_and_owner(
    client: TestClient, admin_token: str, calibration
) -> None:
    queue, _ = calibration
    created = _calibrate(client, admin_token).json()
    [item_id] = created["item_ids"]
    item = queue.dequeue()
    assert item is not None
    queue.fail_queue_item(item_id, "ValueError", "failed", "traceback")

    response = client.put("/api/v1/queue/default/retry_items_by_id", json=[item_id], headers=_auth(admin_token))

    assert response.status_code == 200, response.text
    retried = queue.dequeue()
    assert retried is not None and retried.retried_from_item_id == item_id
    assert (retried.origin, retried.user_id) == (calibration_origin(created["run_id"]), item.user_id)


def test_status_reset_and_export_for_admins(client: TestClient, admin_token: str, calibration) -> None:
    _, service = calibration
    run_id = _calibrate(client, admin_token).json()["run_id"]
    service.record_windows(run_id, [measured_window(run_id), measured_window(run_id, model_key="model-key")])

    status = client.get("/api/v2/models/working_memory/status", headers=_auth(admin_token)).json()
    assert status["available"] and status["current_device_class_id"] == GPU_CLASS.id
    assert status["device_classes"][0]["runs"][0]["point_count"] == 2

    export = client.get("/api/v2/models/working_memory/export", headers=_auth(admin_token))
    assert export.status_code == 200 and export.headers["content-type"] == "application/json"
    assert [point["run_id"] for point in export.json()["points"]] == [run_id, run_id]

    reset = client.delete("/api/v2/models/i/model-key/working_memory", headers=_auth(admin_token))
    assert reset.json() == {"runs_removed": 1, "points_removed": 2}
    assert client.delete(f"/api/v2/models/working_memory/runs/{run_id}", headers=_auth(admin_token)).status_code == 404
    assert client.delete("/api/v2/models/working_memory", headers=_auth(admin_token)).json() == {
        "runs_removed": 0,
        "points_removed": 0,
    }
    missing_class = client.get(
        "/api/v2/models/working_memory/export", params={"device_class_id": "0" * 16}, headers=_auth(admin_token)
    )
    assert missing_class.status_code == 404


def test_a_repeated_calibrate_with_its_retry_key_returns_the_same_run(
    client: TestClient, admin_token: str, calibration
) -> None:
    queue, service = calibration
    body = {"graph": _graph(), "idempotency_key": "lost-response"}

    first = _calibrate(client, admin_token, body)
    replay = _calibrate(client, admin_token, body)

    assert first.status_code == replay.status_code == 201
    assert replay.json() == first.json()
    assert queue.get_queue_item_ids("default").item_ids == first.json()["item_ids"]
    assert [run.run_id for run in service.get_status().device_classes[0].runs] == [first.json()["run_id"]]

    other_graph = Graph()
    other_graph.add_node(AddInvocation(id="add", a=5, b=2))
    conflict = _calibrate(
        client, admin_token, {"graph": other_graph.model_dump(mode="json"), "idempotency_key": "lost-response"}
    )

    assert conflict.status_code == 409
    # The conflicting request does not take the original run with it.
    assert [run.run_id for run in service.get_status().device_classes[0].runs] == [first.json()["run_id"]]
    assert queue.get_queue_item_ids("default").item_ids == first.json()["item_ids"]


def test_calibrate_into_a_full_queue_is_refused_and_leaves_no_run(
    client: TestClient, admin_token: str, mock_invoker: Invoker, calibration
) -> None:
    _, service = calibration
    mock_invoker.services.configuration.max_queue_size = 1
    kept = _calibrate(client, admin_token).json()["run_id"]

    response = _calibrate(client, admin_token)

    assert response.status_code == 409 and "queue is full" in response.json()["detail"]
    assert [run.run_id for run in service.get_status().device_classes[0].runs] == [kept]


@pytest.mark.parametrize(
    ("idempotency_key", "lost", "status_code"),
    [(None, "run", 404), ("lost-midway", "run", 404), (None, "file", 409)],
    ids=["run-removed-without-key", "run-removed-with-key", "file-unusable-without-key"],
)
def test_a_run_lost_while_it_is_being_created_fails_the_request_with_its_items_queued(
    client: TestClient,
    admin_token: str,
    calibration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    idempotency_key: str | None,
    lost: str,
    status_code: int,
) -> None:
    """Admitted, the items fail when they start: without a key as well, the request is no success."""
    queue, service = calibration
    held = HeldEnqueue(monkeypatch)
    responses = []
    body = {"graph": _graph(), "idempotency_key": idempotency_key}
    request = threading.Thread(target=lambda: responses.append(_calibrate(client, admin_token, body)))
    request.start()
    assert held.entered.wait(10)
    [run] = service.get_status().device_classes[0].runs
    if lost == "run":
        service.reset_run(run.run_id)
    else:
        (tmp_path / "working_memory" / f"{GPU_CLASS.id}.json").write_bytes(b'{"schema_version": 2}')
    held.release.set()
    request.join(timeout=30)

    [response] = responses
    assert response.status_code == status_code, response.text
    [item_id] = queue.get_queue_item_ids("default").item_ids
    assert queue.get_queue_item(item_id).origin == calibration_origin(run.run_id)


def test_the_status_carries_the_device_class_id(client: TestClient, admin_token: str, calibration) -> None:
    status = client.get("/api/v2/models/working_memory/status", headers=_auth(admin_token)).json()

    assert status["device_classes"][0]["device_class"] == {
        "id": GPU_CLASS.id,
        **GPU_CLASS.model_dump(exclude={"id"}),
    }


def test_a_store_directory_that_cannot_be_listed_answers_the_status_with_503(
    client: TestClient, admin_token: str, calibration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "working_memory"
    store.mkdir()
    iterdir = Path.iterdir

    def denied(path: Path):
        if path == store:
            raise PermissionError(13, "Permission denied", str(path))
        return iterdir(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    response = client.get("/api/v2/models/working_memory/status", headers=_auth(admin_token))

    assert response.status_code == 503 and "Permission denied" in response.json()["detail"]


def test_a_retry_key_reused_for_an_unknown_model_is_a_conflict(
    client: TestClient, admin_token: str, calibration
) -> None:
    """Key reuse is the documented 409, whichever model the other request names."""
    body = {"graph": _graph(), "idempotency_key": "taken"}
    assert _calibrate(client, admin_token, body).status_code == 201

    response = _calibrate(client, admin_token, body, key="no-such-model")

    assert response.status_code == 409, response.text

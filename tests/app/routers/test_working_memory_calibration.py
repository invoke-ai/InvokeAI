"""The calibration routes and the reserved origin, against a real SQLite queue and a store on a temporary root."""

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
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

    model = SimpleNamespace(
        key="model-key", name="Model", hash="blake3:abc", base=BaseModelType.StableDiffusionXL, type=ModelType.Main
    )

    def get_model(key: str) -> SimpleNamespace:
        if key != "model-key":
            raise UnknownModelException(f"unknown model {key}")
        return model

    services.model_manager.store.get_model.side_effect = get_model
    services.model_manager.store.all_models.return_value = [model]
    _DEVICES[:] = [GPU]
    service = calibration_service(tmp_path, devices=lambda: list(_DEVICES))
    service.start(mock_invoker)
    services.working_memory_calibration = service

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.session_queue.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    yield queue, service
    service.stop(mock_invoker)


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
        client.get("/api/v2/models/i/model-key/working_memory/plan", headers=_auth(user1_token)),
        client.get("/api/v2/models/i/model-key/working_memory/summary", headers=_auth(user1_token)),
    ]

    assert [response.status_code for response in responses] == [403] * 8
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


def test_plan_and_summary_for_admins(client: TestClient, admin_token: str, calibration) -> None:
    _, service = calibration
    run_id = _calibrate(client, admin_token).json()["run_id"]
    service.record_windows(run_id, [measured_window(run_id, model_key="model-key")])

    plan = client.get("/api/v2/models/i/model-key/working_memory/plan", headers=_auth(admin_token))
    assert plan.status_code == 200, plan.text
    body = plan.json()
    assert body["supported"] and body["device_class"]["id"] == GPU_CLASS.id and body["estimators"] == ["vae.sd"]
    assert {case["line"] for case in body["cases"]} == {"decode_fp16", "decode_fp32"}
    assert client.get("/api/v2/models/i/missing/working_memory/plan", headers=_auth(admin_token)).status_code == 404

    summary = client.get("/api/v2/models/i/model-key/working_memory/summary", headers=_auth(admin_token))
    assert summary.status_code == 200, summary.text
    # The window's features name no line: it is counted, not fitted.
    assert summary.json()["excluded_windows"] == 1 and summary.json()["runs"][0]["run_id"] == run_id
    # A model removed meanwhile, as the plan says.
    assert client.get("/api/v2/models/i/missing/working_memory/summary", headers=_auth(admin_token)).status_code == 404
    _DEVICES[:] = [GPU, torch.device("cuda:1")]
    no_class = client.get("/api/v2/models/i/model-key/working_memory/summary", headers=_auth(admin_token))
    assert no_class.status_code == 409


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


def test_calibrate_records_the_attempt_and_rejects_a_malformed_one(
    client: TestClient, admin_token: str, calibration
) -> None:
    _, service = calibration

    created = _calibrate(client, admin_token, {"graph": _graph(), "attempt_id": "attempt-7", "attempt_steps": 3})
    malformed = _calibrate(client, admin_token, {"graph": _graph(), "attempt_id": "no spaces", "attempt_steps": 3})

    assert created.status_code == 201 and malformed.status_code == 422
    status = client.get("/api/v2/models/working_memory/status", headers=_auth(admin_token)).json()
    [run] = status["device_classes"][0]["runs"]
    assert (run["run_id"], run["attempt_id"], run["attempt_steps"]) == (created.json()["run_id"], "attempt-7", 3)
    assert "measurement_verified" in status["device_classes"][0]


@pytest.mark.parametrize(
    "attempt",
    [
        {"attempt_id": "alone"},
        {"attempt_steps": 2},
        {"attempt_id": "zero", "attempt_steps": 0},
        {"attempt_id": "too-many", "attempt_steps": 257},
    ],
    ids=["attempt-without-steps", "steps-without-attempt", "no-steps", "above-the-bound"],
)
def test_an_attempt_and_its_step_count_come_together_or_not_at_all(
    client: TestClient, admin_token: str, calibration, attempt: dict
) -> None:
    _, service = calibration

    response = _calibrate(client, admin_token, {"graph": _graph(), **attempt})

    assert response.status_code == 422, response.text
    assert [run for device_class in service.get_status().device_classes for run in device_class.runs] == []


def test_a_run_whose_step_count_differs_from_its_attempts_is_a_conflict(
    client: TestClient, admin_token: str, calibration
) -> None:
    """The attempt feeds a fit once it has its planned runs: a run that plans another count would make that count
    ambiguous, whether it is another run of the attempt or a repeat of one."""
    _, service = calibration
    first = {"graph": _graph(), "attempt_id": "attempt", "attempt_steps": 2, "idempotency_key": "step-1"}
    assert _calibrate(client, admin_token, first).status_code == 201

    other_run = _calibrate(client, admin_token, {**first, "attempt_steps": 3, "idempotency_key": "step-2"})
    repeat = _calibrate(client, admin_token, {**first, "attempt_steps": 3})

    assert (other_run.status_code, repeat.status_code) == (409, 409), (other_run.text, repeat.text)
    assert "attempt_steps" in other_run.json()["detail"]
    [run] = service.get_status().device_classes[0].runs
    assert run.attempt_steps == 2


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

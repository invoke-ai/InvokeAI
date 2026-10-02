"""The workflow endpoints expose content revisions, refuse stale updates, and make creation retry-safe."""

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowCategory, WorkflowRecordOrderBy
from tests.app.routers.test_workflows_multiuser import WORKFLOW_BODY, create_workflow

pytest_plugins = ("tests.app.routers.test_workflows_multiuser",)


@pytest.fixture(autouse=True)
def _no_thumbnails(enable_multiuser: Any, mock_invoker: Invoker) -> None:
    # The routers conftest also provides `enable_multiuser`; whichever wins, single-record reads need a real URL or None.
    mock_invoker.services.workflow_thumbnails.get_url.return_value = None


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _update(client: TestClient, token: str, workflow_id: str, name: str, expected_revision: int | None = None):
    body: dict[str, Any] = {"workflow": {**WORKFLOW_BODY, "id": workflow_id, "name": name}}
    if expected_revision is not None:
        body["expected_revision"] = expected_revision
    return client.patch(f"/api/v1/workflows/i/{workflow_id}", json=body, headers=_auth(token))


def test_records_report_their_revision(client: TestClient, user1_token: str) -> None:
    created = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY}, headers=_auth(user1_token))
    assert created.status_code == 200, created.text
    assert created.json()["revision"] == 1
    workflow_id = created.json()["workflow_id"]

    updated = _update(client, user1_token, workflow_id, "Second", expected_revision=1)
    assert updated.status_code == 200, updated.text
    assert updated.json()["revision"] == 2

    fetched = client.get(f"/api/v1/workflows/i/{workflow_id}", headers=_auth(user1_token))
    assert fetched.json()["revision"] == 2
    listed = client.get("/api/v1/workflows/?categories=user", headers=_auth(user1_token))
    assert next(item["revision"] for item in listed.json()["items"] if item["workflow_id"] == workflow_id) == 2


def test_a_stale_update_is_a_structured_conflict(client: TestClient, user1_token: str, mock_invoker: Invoker) -> None:
    workflow_id = create_workflow(client, user1_token)
    assert _update(client, user1_token, workflow_id, "Second", expected_revision=1).status_code == 200
    events_before = len(mock_invoker.services.events.events)

    stale = _update(client, user1_token, workflow_id, "Stale", expected_revision=1)

    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"] == {
        "message": stale.json()["detail"]["message"],
        "reason": "revision-conflict",
        "current_revision": 2,
        "expected_revision": 1,
    }
    assert len(mock_invoker.services.events.events) == events_before
    assert client.get(f"/api/v1/workflows/i/{workflow_id}", headers=_auth(user1_token)).json()["name"] == "Second"


def test_legacy_updates_without_an_expected_revision_still_write(client: TestClient, user1_token: str) -> None:
    workflow_id = create_workflow(client, user1_token)
    assert _update(client, user1_token, workflow_id, "Second", expected_revision=1).status_code == 200

    legacy = _update(client, user1_token, workflow_id, "Third")

    assert legacy.status_code == 200, legacy.text
    assert legacy.json()["revision"] == 3
    assert legacy.json()["name"] == "Third"


def test_the_url_and_body_must_name_the_same_workflow(client: TestClient, user1_token: str) -> None:
    first = create_workflow(client, user1_token)
    second = create_workflow(client, user1_token)

    response = client.patch(
        f"/api/v1/workflows/i/{first}",
        json={"workflow": {**WORKFLOW_BODY, "id": second, "name": "Crossed"}},
        headers=_auth(user1_token),
    )

    assert response.status_code == 400
    for workflow_id in (first, second):
        assert (
            client.get(f"/api/v1/workflows/i/{workflow_id}", headers=_auth(user1_token)).json()["name"]
            == (WORKFLOW_BODY["name"])
        )


def test_bundled_workflows_refuse_updates_even_from_admins(
    client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    bundled = mock_invoker.services.workflow_records.get_many(
        order_by=WorkflowRecordOrderBy.Name,
        direction=SQLiteDirection.Ascending,
        categories=[WorkflowCategory.Default],
        page=0,
        per_page=1,
        query=None,
        tags=None,
        has_been_opened=None,
    ).items[0]

    response = _update(client, admin_token, bundled.workflow_id, "Hijacked")

    assert response.status_code == 403, response.text
    assert client.delete(f"/api/v1/workflows/i/{bundled.workflow_id}", headers=_auth(admin_token)).status_code == 403
    assert mock_invoker.services.workflow_records.get(bundled.workflow_id).name == bundled.name


def test_bundled_workflows_answer_immutability_before_ownership(
    client: TestClient, user1_token: str, mock_invoker: Invoker
) -> None:
    bundled = mock_invoker.services.workflow_records.get_many(
        order_by=WorkflowRecordOrderBy.Name,
        direction=SQLiteDirection.Ascending,
        categories=[WorkflowCategory.Default],
        page=0,
        per_page=1,
        query=None,
        tags=None,
        has_been_opened=None,
    ).items[0]

    response = _update(client, user1_token, bundled.workflow_id, "Hijacked")

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Bundled workflows cannot be modified"
    deleted = client.delete(f"/api/v1/workflows/i/{bundled.workflow_id}", headers=_auth(user1_token))
    assert deleted.status_code == 403
    assert deleted.json()["detail"] == "Bundled workflows cannot be deleted"


def test_creation_with_a_reserved_id_is_idempotent(client: TestClient, user1_token: str, mock_invoker: Invoker) -> None:
    reserved = str(uuid.uuid4())
    body = {"workflow": WORKFLOW_BODY, "workflow_id": reserved}

    first = client.post("/api/v1/workflows/", json=body, headers=_auth(user1_token))
    retried = client.post("/api/v1/workflows/", json=body, headers=_auth(user1_token))

    assert first.status_code == 200, first.text
    assert retried.status_code == 200, retried.text
    assert first.json()["workflow_id"] == reserved
    assert retried.json()["workflow_id"] == reserved
    created_events = [e for e in mock_invoker.services.events.events if e.__event_name__ == "workflow_created"]
    assert [e.workflow_id for e in created_events].count(reserved) == 1


def test_a_reserved_id_never_exposes_another_accounts_record(
    client: TestClient, user1_token: str, user2_token: str
) -> None:
    reserved = str(uuid.uuid4())
    body = {"workflow": WORKFLOW_BODY, "workflow_id": reserved}
    assert client.post("/api/v1/workflows/", json=body, headers=_auth(user1_token)).status_code == 200

    stolen = client.post("/api/v1/workflows/", json=body, headers=_auth(user2_token))
    changed = client.post(
        "/api/v1/workflows/",
        json={"workflow": {**WORKFLOW_BODY, "name": "Different"}, "workflow_id": reserved},
        headers=_auth(user1_token),
    )
    malformed = client.post(
        "/api/v1/workflows/", json={"workflow": WORKFLOW_BODY, "workflow_id": "nope"}, headers=_auth(user1_token)
    )

    assert stolen.status_code == 409
    assert stolen.json()["detail"]["reason"] == "id-conflict"
    assert "user1" not in stolen.text
    assert changed.status_code == 409
    assert malformed.status_code == 400
    assert client.get(f"/api/v1/workflows/i/{reserved}", headers=_auth(user2_token)).status_code == 403

"""Tests for workflow CRUD live-update events with multiuser visibility rules."""

from typing import Any

from fastapi.testclient import TestClient

from tests.app.routers.test_workflows_multiuser import WORKFLOW_BODY

pytest_plugins = ("tests.app.routers.test_workflows_multiuser",)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _event_names(events: list[Any]) -> list[str]:
    return [event.__event_name__ for event in events]


def _get_last_event(events: list[Any], event_name: str) -> Any:
    matching = [event for event in events if event.__event_name__ == event_name]
    assert matching, f"Expected event '{event_name}' to be emitted"
    return matching[-1]


def test_create_private_workflow_emits_owner_scoped_created_event(
    client: TestClient, user1_token: str, mock_invoker: Any
) -> None:
    response = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY}, headers=_auth(user1_token))

    assert response.status_code == 200

    event = _get_last_event(mock_invoker.services.events.events, "workflow_created")
    assert event.workflow_id == response.json()["workflow_id"]
    assert event.user_id == response.json()["user_id"]
    assert event.is_public is False


def test_update_workflow_emits_updated_event_with_previous_visibility(
    client: TestClient, user1_token: str, mock_invoker: Any
) -> None:
    create_response = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY}, headers=_auth(user1_token))
    workflow_id = create_response.json()["workflow_id"]

    update_response = client.patch(
        f"/api/v1/workflows/i/{workflow_id}",
        json={"workflow": {**WORKFLOW_BODY, "id": workflow_id, "name": "Renamed Workflow"}},
        headers=_auth(user1_token),
    )

    assert update_response.status_code == 200

    event = _get_last_event(mock_invoker.services.events.events, "workflow_updated")
    assert event.workflow_id == workflow_id
    assert event.user_id == create_response.json()["user_id"]
    assert event.old_is_public is False
    assert event.new_is_public is False


def test_update_workflow_is_public_emits_visibility_transition_event(
    client: TestClient, user1_token: str, mock_invoker: Any
) -> None:
    create_response = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY}, headers=_auth(user1_token))
    workflow_id = create_response.json()["workflow_id"]

    update_response = client.patch(
        f"/api/v1/workflows/i/{workflow_id}/is_public",
        json={"is_public": True},
        headers=_auth(user1_token),
    )

    assert update_response.status_code == 200

    event = _get_last_event(mock_invoker.services.events.events, "workflow_updated")
    assert event.workflow_id == workflow_id
    assert event.user_id == create_response.json()["user_id"]
    assert event.old_is_public is False
    assert event.new_is_public is True


def test_delete_workflow_emits_deleted_event_with_last_known_visibility(
    client: TestClient, user1_token: str, mock_invoker: Any
) -> None:
    create_response = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY}, headers=_auth(user1_token))
    workflow_id = create_response.json()["workflow_id"]

    share_response = client.patch(
        f"/api/v1/workflows/i/{workflow_id}/is_public",
        json={"is_public": True},
        headers=_auth(user1_token),
    )
    assert share_response.status_code == 200

    delete_response = client.delete(f"/api/v1/workflows/i/{workflow_id}", headers=_auth(user1_token))

    assert delete_response.status_code == 200

    event = _get_last_event(mock_invoker.services.events.events, "workflow_deleted")
    assert event.workflow_id == workflow_id
    assert event.user_id == create_response.json()["user_id"]
    assert event.is_public is True


def test_failed_update_does_not_emit_workflow_live_update_event(
    client: TestClient, user1_token: str, user2_token: str, mock_invoker: Any
) -> None:
    create_response = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY}, headers=_auth(user1_token))
    workflow_id = create_response.json()["workflow_id"]
    before_event_names = _event_names(mock_invoker.services.events.events)

    update_response = client.patch(
        f"/api/v1/workflows/i/{workflow_id}",
        json={"workflow": {**WORKFLOW_BODY, "id": workflow_id, "name": "Hijacked"}},
        headers=_auth(user2_token),
    )

    assert update_response.status_code == 403
    assert _event_names(mock_invoker.services.events.events) == before_event_names

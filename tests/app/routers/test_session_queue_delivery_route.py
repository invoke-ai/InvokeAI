"""The delivery route lets only the owner of a queue item (or an admin) claim, confirm or release staging its results."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic_core import to_jsonable_python
from sqlalchemy import insert

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.execution_state_migration import dump_execution_state
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.app.routers.conftest import MockApiDependencies, _auth


@pytest.fixture
def queue_router(
    monkeypatch: pytest.MonkeyPatch, enable_multiuser: Any, mock_invoker: Invoker, mock_sqlite_database: Database
):
    mock_invoker.services.session_queue = SessionQueue(mock_sqlite_database)
    monkeypatch.setattr("invokeai.app.api.routers.session_queue.ApiDependencies", MockApiDependencies(mock_invoker))


def _item(database: Database, user_id: str, queue_id: str = "default") -> int:
    session = GraphExecutionState(graph=Graph())
    values = {
        "queue_id": queue_id,
        "batch_id": "batch",
        "session_id": session.id,
        "session": json.dumps(dump_execution_state(session), default=to_jsonable_python),
        "status": "completed",
        "user_id": user_id,
    }
    with database.begin(write=True) as conn:
        return int(conn.execute(insert(session_queue).values(**values)).inserted_primary_key[0])


def _deliver(
    client: TestClient, token: str, item_id: int, action: str, claimant: str = "tab-a", queue_id: str = "default"
):
    return client.post(
        f"/api/v1/queue/{queue_id}/i/{item_id}/delivery",
        json={"action": action, "claimant": claimant},
        headers=_auth(token),
    )


def test_the_owner_claims_confirms_and_releases(
    queue_router: None, mock_sqlite_database: Database, mock_invoker: Invoker, client: TestClient, user1_token: str
):
    user_id = mock_invoker.services.users.get_by_email("user1@test.com").user_id
    item_id = _item(mock_sqlite_database, user_id)

    assert _deliver(client, user1_token, item_id, "claim").json() == {"status": "claimed", "retry_after_ms": None}
    held = _deliver(client, user1_token, item_id, "claim", claimant="tab-b")
    assert held.status_code == 200
    assert held.json()["status"] == "held" and 0 < held.json()["retry_after_ms"] <= 120_000
    assert _deliver(client, user1_token, item_id, "release").json()["status"] == "released"
    assert _deliver(client, user1_token, item_id, "claim", claimant="tab-b").json()["status"] == "claimed"
    assert _deliver(client, user1_token, item_id, "confirm", claimant="tab-c").json()["status"] == "delivered"
    assert _deliver(client, user1_token, item_id, "claim").json()["status"] == "delivered"


def test_another_users_item_is_refused_and_an_admins_request_allowed(
    queue_router: None,
    mock_sqlite_database: Database,
    mock_invoker: Invoker,
    client: TestClient,
    admin_token: str,
    user1_token: str,
    user2_token: str,
):
    user_id = mock_invoker.services.users.get_by_email("user1@test.com").user_id
    item_id = _item(mock_sqlite_database, user_id)

    for action in ("claim", "confirm", "release"):
        assert _deliver(client, user2_token, item_id, action).status_code == 403
    assert _deliver(client, user1_token, item_id, "claim").json()["status"] == "claimed"
    assert _deliver(client, admin_token, item_id, "claim", claimant="admin-tab").json()["status"] == "held"


def test_a_request_is_authorized_without_reading_the_items_session(
    monkeypatch: pytest.MonkeyPatch,
    queue_router: None,
    mock_sqlite_database: Database,
    mock_invoker: Invoker,
    client: TestClient,
    user1_token: str,
    user2_token: str,
):
    """Waiting browsers poll the route: it must not load and validate the whole item each time."""
    user_id = mock_invoker.services.users.get_by_email("user1@test.com").user_id
    item_id = _item(mock_sqlite_database, user_id)

    def unread(self: SessionQueue, item_id: int) -> None:
        raise AssertionError("the delivery route read the whole queue item")

    monkeypatch.setattr(SessionQueue, "get_queue_item", unread)

    assert _deliver(client, user1_token, item_id, "claim").json()["status"] == "claimed"
    assert _deliver(client, user2_token, item_id, "claim").status_code == 403


def test_a_missing_item_or_one_of_another_queue_is_not_found(
    queue_router: None, mock_sqlite_database: Database, mock_invoker: Invoker, client: TestClient, user1_token: str
):
    user_id = mock_invoker.services.users.get_by_email("user1@test.com").user_id
    item_id = _item(mock_sqlite_database, user_id, queue_id="other")

    assert _deliver(client, user1_token, item_id + 1000, "claim").status_code == 404
    assert _deliver(client, user1_token, item_id, "claim").status_code == 404
    assert _deliver(client, user1_token, item_id, "claim", queue_id="other").status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"action": "steal", "claimant": "tab-a"},
        {"action": "claim", "claimant": ""},
        {"action": "claim", "claimant": "x" * 129},
        {"action": "claim"},
    ],
)
def test_a_malformed_request_is_rejected(
    queue_router: None,
    mock_sqlite_database: Database,
    mock_invoker: Invoker,
    client: TestClient,
    user1_token: str,
    body: dict[str, Any],
):
    user_id = mock_invoker.services.users.get_by_email("user1@test.com").user_id
    item_id = _item(mock_sqlite_database, user_id)

    response = client.post(f"/api/v1/queue/default/i/{item_id}/delivery", json=body, headers=_auth(user1_token))

    assert response.status_code == 422

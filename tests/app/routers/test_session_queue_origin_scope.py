"""A project's queue items are its owner's: reads scoped by an origin prefix return a non-admin only their own
items, so another account's items cannot join the project's view, nor can a prefix probe for other accounts'
project ids."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor.session_processor_common import SessionProcessorStatus
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from tests.app.routers.conftest import MockApiDependencies, _auth
from tests.app.services.session_queue.test_session_queue_status_user_scoping import _insert_queue_item
from tests.fixtures.sqlite_database import sqlite_cursor_of

PROJECT = "webv2:p:project-1:q:"


@pytest.fixture
def queue(
    monkeypatch: pytest.MonkeyPatch, mock_invoker: Invoker, mock_sqlite_database: Database, enable_multiuser: Any
) -> SessionQueue:
    queue = SessionQueue(mock_sqlite_database)
    queue.start(mock_invoker)
    mock_invoker.services.session_queue = queue
    mock_invoker.services.session_processor = MagicMock()
    mock_invoker.services.session_processor.get_status.return_value = SessionProcessorStatus(
        is_started=True, is_processing=True
    )
    monkeypatch.setattr("invokeai.app.api.routers.session_queue.ApiDependencies", MockApiDependencies(mock_invoker))
    return queue


def _user_id(client: TestClient, token: str) -> str:
    return client.get("/api/v1/auth/me", headers=_auth(token)).json()["user_id"]


def _put(queue: SessionQueue, user_id: str, status: str) -> int:
    item_id = _insert_queue_item(queue, user_id=user_id, origin=f"{PROJECT}{status}")
    with sqlite_cursor_of(queue) as cursor:
        cursor.execute("UPDATE session_queue SET status = ? WHERE item_id = ?", (status, item_id))
    return item_id


def _project_view(client: TestClient, token: str) -> dict[str, Any]:
    params = {"origin_prefix": PROJECT}
    status = client.get("/api/v1/queue/default/status", params=params, headers=_auth(token)).json()["queue"]
    current = client.get("/api/v1/queue/default/current", params=params, headers=_auth(token)).json()
    next_item = client.get("/api/v1/queue/default/next", params=params, headers=_auth(token)).json()
    ids = client.get("/api/v1/queue/default/item_ids", params=params, headers=_auth(token)).json()["item_ids"]
    return {
        "ids": sorted(ids),
        "pending": status["pending"],
        "in_progress": status["in_progress"],
        "current": current and current["item_id"],
        "status_current": status["item_id"],
        "next": next_item and next_item["item_id"],
    }


def test_an_origin_scoped_read_shows_a_non_admin_only_their_own_items(
    client: TestClient, queue: SessionQueue, user1_token: str, user2_token: str, admin_token: str
) -> None:
    owner, intruder = _user_id(client, user1_token), _user_id(client, user2_token)
    # Another account's items under the owner's project origin, as anyone could enqueue them, and enqueued first
    # (with the higher priority) so that an unscoped read would pick them.
    other_running, other_pending = _put(queue, intruder, "in_progress"), _put(queue, intruder, "pending")
    own_running, own_pending = _put(queue, owner, "in_progress"), _put(queue, owner, "pending")
    with sqlite_cursor_of(queue) as cursor:
        cursor.execute("UPDATE session_queue SET priority = 1 WHERE item_id = ?", (other_pending,))

    assert _project_view(client, user1_token) == {
        "ids": [own_running, own_pending],
        "pending": 1,
        "in_progress": 1,
        "current": own_running,
        "status_current": own_running,
        "next": own_pending,
    }
    assert _project_view(client, user2_token)["ids"] == [other_running, other_pending]
    # Admins see the whole queue as before.
    assert _project_view(client, admin_token) == {
        "ids": [other_running, other_pending, own_running, own_pending],
        "pending": 2,
        "in_progress": 2,
        "current": other_running,
        "status_current": other_running,
        "next": other_pending,
    }


def test_the_unscoped_queue_list_still_lists_every_accounts_items(
    client: TestClient, queue: SessionQueue, user1_token: str, user2_token: str
) -> None:
    mine, theirs = (
        _put(queue, _user_id(client, user1_token), "pending"),
        _put(queue, _user_id(client, user2_token), "pending"),
    )

    ids = client.get("/api/v1/queue/default/item_ids", headers=_auth(user1_token)).json()["item_ids"]

    assert sorted(ids) == [mine, theirs]

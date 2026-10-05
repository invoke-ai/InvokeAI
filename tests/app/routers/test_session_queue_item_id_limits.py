"""Guards the batch bound on the queue summary route.

`item_summaries_by_ids` takes a client-supplied list of ids and the SQLite layer binds one
parameter per id. Unbounded, a client could post tens of thousands of ids: past SQLite's
per-statement variable limit the query raises `OperationalError`, which the route reports as a
generic HTTP 500, and even below that limit it is an invitation to make the server do arbitrary
work per request. The route caps the list so oversized requests are rejected by validation
instead.

The id listing the client hydrates from takes an optional limit with the same bound. A limited listing
returns the head of the full order and counts only the ids it returns: counting every match would read
the queue's whole history, which is what the limit exists to avoid.
"""

import json
import uuid
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic_core import to_jsonable_python

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.session_queue import MAX_QUEUE_ITEM_IDS_PER_REQUEST
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState

SUMMARIES_ROUTE = "/api/v1/queue/default/item_summaries_by_ids"
FULL_ITEMS_ROUTE = "/api/v1/queue/default/items_by_ids"
ITEM_IDS_ROUTE = "/api/v1/queue/default/item_ids"

# SQLite's bind-parameter limit on builds >= 3.32. One id over it is what turns the unbounded
# version of this route into a 500.
SQLITE_MAX_VARIABLE_NUMBER = 32766


@pytest.fixture
def mock_queue_invoker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    invoker = MagicMock()
    # A bare MagicMock attribute is truthy, which would put the auth dependencies into multiuser
    # mode and answer with 401 before the route is reached.
    invoker.services.configuration.multiuser = False
    invoker.services.session_queue.get_queue_item_summaries_by_ids.return_value = []
    monkeypatch.setattr(ApiDependencies, "invoker", invoker, raising=False)
    return invoker


def test_summaries_by_ids_rejects_oversized_id_lists(mock_queue_invoker: MagicMock) -> None:
    client = TestClient(app)

    response = client.post(SUMMARIES_ROUTE, json={"item_ids": list(range(SQLITE_MAX_VARIABLE_NUMBER + 1))})

    assert response.status_code == 422, (
        f"Expected a controlled rejection, got {response.status_code}. An unbounded id list reaches "
        f"the SQLite layer, which binds one variable per id and raises OperationalError."
    )
    # The request must be turned away by validation, before any database work is attempted.
    mock_queue_invoker.services.session_queue.get_queue_item_summaries_by_ids.assert_not_called()


def test_summaries_by_ids_accepts_a_full_size_batch(mock_queue_invoker: MagicMock) -> None:
    client = TestClient(app)
    item_ids = list(range(MAX_QUEUE_ITEM_IDS_PER_REQUEST))

    response = client.post(SUMMARIES_ROUTE, json={"item_ids": item_ids})

    assert response.status_code == 200
    mock_queue_invoker.services.session_queue.get_queue_item_summaries_by_ids.assert_called_once_with(
        queue_id="default", item_ids=item_ids
    )


def test_full_items_by_ids_rejects_oversized_id_lists(mock_queue_invoker: MagicMock) -> None:
    client = TestClient(app)

    response = client.post(FULL_ITEMS_ROUTE, json={"item_ids": list(range(MAX_QUEUE_ITEM_IDS_PER_REQUEST + 1))})

    assert response.status_code == 422
    # Validation must reject the request before graph deserialization starts.
    mock_queue_invoker.services.session_queue.get_queue_item.assert_not_called()


@pytest.fixture
def listed_queue(monkeypatch: pytest.MonkeyPatch, mock_invoker: Invoker) -> list[int]:
    """A real queue of five items in two projects, the ids in insertion (and creation) order."""
    session_queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.session_queue = session_queue
    monkeypatch.setattr(ApiDependencies, "invoker", mock_invoker, raising=False)
    session = json.dumps(to_jsonable_python(GraphExecutionState(graph=Graph()).model_dump()))
    item_ids = []
    for index, origin in enumerate(["webv2:p:a:q:1", "webv2:p:b:q:2", "webv2:p:a:q:3", "webv2:p:a:q:4", "x"]):
        with session_queue._db.transaction() as cursor:
            cursor.execute(
                """--sql
                INSERT INTO session_queue (queue_id, session, session_id, batch_id, created_at, origin)
                VALUES ('default', ?, ?, 'batch', ?, ?);
                """,
                (session, str(uuid.uuid4()), f"2026-10-01 10:00:0{index}.000", origin),
            )
            item_ids.append(cast(int, cursor.lastrowid))
    return item_ids


def test_item_ids_without_a_limit_lists_and_counts_every_matching_item(listed_queue: list[int]) -> None:
    client = TestClient(app)

    response = client.get(ITEM_IDS_ROUTE)

    assert response.status_code == 200
    assert response.json() == {"item_ids": listed_queue[::-1], "total_count": 5}


@pytest.mark.parametrize(
    ("params", "expected_positions"),
    [
        ({"limit": 2}, [4, 3]),
        ({"limit": 2, "order_dir": "ASC"}, [0, 1]),
        # The project filter applies before the limit: its newest two, not the queue's.
        ({"limit": 2, "origin_prefix": "webv2:p:a:"}, [3, 2]),
        ({"limit": 1000, "origin_prefix": "webv2:p:a:"}, [3, 2, 0]),
    ],
)
def test_item_ids_with_a_limit_returns_the_head_and_counts_only_what_it_returns(
    listed_queue: list[int], params: dict[str, object], expected_positions: list[int]
) -> None:
    client = TestClient(app)

    response = client.get(ITEM_IDS_ROUTE, params=params)

    assert response.status_code == 200
    expected = [listed_queue[position] for position in expected_positions]
    assert response.json() == {"item_ids": expected, "total_count": len(expected)}


@pytest.mark.parametrize("limit", ["0", "-1", str(MAX_QUEUE_ITEM_IDS_PER_REQUEST + 1), "fifty", "2.5"])
def test_item_ids_rejects_a_limit_outside_the_hydration_bound(mock_queue_invoker: MagicMock, limit: str) -> None:
    client = TestClient(app)

    response = client.get(ITEM_IDS_ROUTE, params={"limit": limit})

    assert response.status_code == 422
    mock_queue_invoker.services.session_queue.get_queue_item_ids.assert_not_called()

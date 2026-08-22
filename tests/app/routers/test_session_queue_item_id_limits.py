"""Guards the batch bound on the queue summary route.

`item_summaries_by_ids` takes a client-supplied list of ids and the SQLite layer binds one
parameter per id. Unbounded, a client could post tens of thousands of ids: past SQLite's
per-statement variable limit the query raises `OperationalError`, which the route reports as a
generic HTTP 500, and even below that limit it is an invitation to make the server do arbitrary
work per request. The route caps the list so oversized requests are rejected by validation
instead.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.session_queue import MAX_QUEUE_ITEM_IDS_PER_REQUEST
from invokeai.app.api_app import app

SUMMARIES_ROUTE = "/api/v1/queue/default/item_summaries_by_ids"
FULL_ITEMS_ROUTE = "/api/v1/queue/default/items_by_ids"

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

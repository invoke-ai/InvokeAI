from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.session_queue.session_queue_common import (
    Batch,
    EnqueueBatchReceipt,
    EnqueueProjectNotFoundError,
    EnqueueReceiptLimitError,
)
from invokeai.app.services.shared.graph import Graph

RECEIPT_ROUTE = "/api/v1/queue/default/enqueue_batch/receipt"
ACKNOWLEDGE_ROUTE = "/api/v1/queue/default/enqueue_batch/acknowledge"
ENQUEUE_ROUTE = "/api/v1/queue/default/enqueue_batch"


@pytest.fixture
def mock_queue_invoker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    invoker = MagicMock()
    invoker.services.configuration.multiuser = False
    invoker.services.image_moves.is_maintenance_active.return_value = False
    monkeypatch.setattr(ApiDependencies, "invoker", invoker, raising=False)
    return invoker


def test_get_enqueue_receipt_returns_the_authenticated_owners_result(mock_queue_invoker: MagicMock) -> None:
    mock_queue_invoker.services.session_queue.get_enqueue_receipt.return_value = EnqueueBatchReceipt(
        batch_id="batch-1", enqueued=2, item_ids=[10, 11], requested=2
    )

    response = TestClient(app).get(RECEIPT_ROUTE, params={"idempotency_key": "webv2:project:item"})

    assert response.status_code == 200
    assert response.json() == {"batch_id": "batch-1", "enqueued": 2, "item_ids": [10, 11], "requested": 2}
    mock_queue_invoker.services.session_queue.get_enqueue_receipt.assert_called_once_with(
        queue_id="default", idempotency_key="webv2:project:item", user_id="system"
    )


def test_get_enqueue_receipt_returns_404_when_absent(mock_queue_invoker: MagicMock) -> None:
    mock_queue_invoker.services.session_queue.get_enqueue_receipt.return_value = None

    response = TestClient(app).get(RECEIPT_ROUTE, params={"idempotency_key": "missing"})

    assert response.status_code == 404


@pytest.mark.parametrize("idempotency_key", ["", "x" * 256])
def test_get_enqueue_receipt_validates_the_key_before_storage(
    mock_queue_invoker: MagicMock, idempotency_key: str
) -> None:
    response = TestClient(app).get(RECEIPT_ROUTE, params={"idempotency_key": idempotency_key})

    assert response.status_code == 422
    mock_queue_invoker.services.session_queue.get_enqueue_receipt.assert_not_called()


@pytest.mark.parametrize("idempotency_key", ["", "x" * 256])
def test_acknowledge_enqueue_validates_the_key_before_storage(
    mock_queue_invoker: MagicMock, idempotency_key: str
) -> None:
    response = TestClient(app).post(ACKNOWLEDGE_ROUTE, json={"idempotency_key": idempotency_key})

    assert response.status_code == 422
    mock_queue_invoker.services.session_queue.acknowledge_enqueue.assert_not_called()


def test_enqueue_receipt_capacity_is_a_retryable_http_error(mock_queue_invoker: MagicMock) -> None:
    mock_queue_invoker.services.session_queue.enqueue_batch = AsyncMock(
        side_effect=EnqueueReceiptLimitError("Too many unacknowledged enqueue requests")
    )

    response = TestClient(app).post(
        ENQUEUE_ROUTE,
        json={"batch": Batch(graph=Graph()).model_dump(mode="json"), "prepend": False},
    )

    assert response.status_code == 429


def test_project_scoped_enqueue_missing_project_is_404(mock_queue_invoker: MagicMock) -> None:
    mock_queue_invoker.services.session_queue.enqueue_batch = AsyncMock(
        side_effect=EnqueueProjectNotFoundError("project-1")
    )

    response = TestClient(app).post(
        ENQUEUE_ROUTE,
        json={"batch": Batch(graph=Graph(), project_id="project-1").model_dump(mode="json"), "prepend": False},
    )

    assert response.status_code == 404

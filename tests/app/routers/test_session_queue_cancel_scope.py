"""The scoped bulk-cancel routes pass the caller's origin prefix through to the service.

The webv2 queue widget cancels a project's items with one request instead of hydrating every id, so
a dropped or renamed query parameter would silently turn a project-scoped cancel into a queue-wide one.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app


@pytest.fixture
def mock_queue_invoker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    invoker = MagicMock()
    invoker.services.configuration.multiuser = False
    invoker.services.session_queue.cancel_by_queue_id.return_value = {"canceled": 0}
    invoker.services.session_queue.cancel_all_except_current.return_value = {"canceled": 0}
    monkeypatch.setattr(ApiDependencies, "invoker", invoker, raising=False)
    return invoker


def test_cancel_all_forwards_the_origin_prefix(mock_queue_invoker: MagicMock) -> None:
    client = TestClient(app)

    response = client.put("/api/v1/queue/default/cancel_all", params={"origin_prefix": "webv2:p:a:"})

    assert response.status_code == 200
    cancel = mock_queue_invoker.services.session_queue.cancel_by_queue_id
    cancel.assert_called_once()
    assert cancel.call_args.kwargs["origin_prefix"] == "webv2:p:a:"


def test_cancel_all_except_current_forwards_the_origin_prefix(mock_queue_invoker: MagicMock) -> None:
    client = TestClient(app)

    response = client.put("/api/v1/queue/default/cancel_all_except_current", params={"origin_prefix": "webv2:p:a:"})

    assert response.status_code == 200
    cancel = mock_queue_invoker.services.session_queue.cancel_all_except_current
    cancel.assert_called_once()
    assert cancel.call_args.kwargs["origin_prefix"] == "webv2:p:a:"

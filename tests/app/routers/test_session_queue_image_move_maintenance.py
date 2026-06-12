from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.session_queue import enqueue_batch
from invokeai.app.services.session_queue.session_queue_common import DEFAULT_QUEUE_ID, Batch
from invokeai.app.services.shared.graph import Graph


class MockApiDependencies(ApiDependencies):
    def __init__(self, invoker) -> None:
        self.invoker = invoker


@pytest.mark.anyio
async def test_enqueue_batch_is_blocked_during_image_move_maintenance(
    monkeypatch: pytest.MonkeyPatch, mock_invoker
) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = True
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)

    with pytest.raises(HTTPException) as exc:
        await enqueue_batch(
            current_user=MagicMock(user_id="user-id"),
            queue_id=DEFAULT_QUEUE_ID,
            batch=Batch(graph=Graph()),
            prepend=False,
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "Image storage maintenance is active"

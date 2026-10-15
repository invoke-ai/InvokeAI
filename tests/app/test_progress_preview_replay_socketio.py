"""On `subscribe_queue`, a socket is sent the latest preview frame of each running item its user owns."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.events.events_common import InvocationProgressEvent
from invokeai.app.services.progress_previews.progress_previews_common import ProgressPreviewDTO
from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _event(item_id: int, user_id: str, queue_id: str = "default") -> InvocationProgressEvent:
    return InvocationProgressEvent(
        queue_id=queue_id,
        item_id=item_id,
        batch_id="batch-1",
        user_id=user_id,
        session_id=f"session-{item_id}",
        invocation={"type": "add", "id": "node-1", "a": 1, "b": 2},
        invocation_source_id="node-1",
        message="Denoising",
        percentage=0.5,
        image={"width": 64, "height": 64, "dataURL": "data:image/jpeg;base64,frame"},
        revision=3,
    )


def test_preview_dto_preserves_workflow_call_routing_metadata() -> None:
    event = _event(2, "owner-1").model_copy(
        update={
            "parent_item_id": 1,
            "root_item_id": 1,
            "workflow_call_parent_source_id": "call-node",
        }
    )

    preview = ProgressPreviewDTO.from_event(event)

    assert preview.parent_item_id == 1
    assert preview.root_item_id == 1
    assert preview.workflow_call_parent_source_id == "call-node"


@pytest.mark.anyio
async def test_subscribe_replays_only_the_users_previews_to_that_socket(monkeypatch) -> None:
    previews = MemoryProgressPreviews()
    previews.record(_event(1, "owner-1"))
    previews.record(_event(2, "owner-2"))
    previews.record(_event(3, "owner-1", queue_id="other"))
    monkeypatch.setattr(
        ApiDependencies,
        "invoker",
        SimpleNamespace(services=SimpleNamespace(progress_previews=previews)),
        raising=False,
    )

    socketio = SocketIO(FastAPI())
    socketio._sio.enter_room = AsyncMock()
    socketio._sio.emit = AsyncMock()
    socketio._socket_users["sid-1"] = {"user_id": "owner-1", "is_admin": True, "authenticated": True}

    await socketio._handle_sub_queue("sid-1", {"queue_id": "default"})

    socketio._sio.emit.assert_awaited_once_with(
        event="invocation_progress",
        data=previews.get(1).model_dump(mode="json"),
        to="sid-1",
    )


@pytest.mark.anyio
async def test_subscribe_without_an_invoker_replays_nothing(monkeypatch) -> None:
    monkeypatch.delattr(ApiDependencies, "invoker", raising=False)
    socketio = SocketIO(FastAPI())
    socketio._sio.enter_room = AsyncMock()
    socketio._sio.emit = AsyncMock()
    socketio._socket_users["sid-1"] = {"user_id": "owner-1", "is_admin": False, "authenticated": True}

    await socketio._handle_sub_queue("sid-1", {"queue_id": "default"})

    socketio._sio.emit.assert_not_awaited()

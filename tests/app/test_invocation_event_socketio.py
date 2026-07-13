"""Tests for socket routing of invocation events in multiuser mode.

Invocation progress events drive personal UI (the global progress bar and progress image
previews) and must be delivered only to the owner - admins receiving other users' progress
would see their own progress display hijacked. The other invocation events (started,
complete, error) also feed admins' gallery cache updates, so they go to the owner and the
admin room - but in a single emit so that an admin who owns the queue item (which includes
the "system" user in single-user mode) receives exactly one copy.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.events.events_common import (
    InvocationCompleteEvent,
    InvocationErrorEvent,
    InvocationProgressEvent,
    InvocationStartedEvent,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_COMMON_FIELDS = {
    "queue_id": "default",
    "item_id": 1,
    "batch_id": "batch-1",
    "user_id": "owner-1",
    "session_id": "session-1",
    "invocation": {"type": "add", "id": "node-1", "a": 1, "b": 2},
    "invocation_source_id": "node-1",
}


@pytest.mark.anyio
async def test_progress_event_is_emitted_only_to_owner() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event = InvocationProgressEvent(**_COMMON_FIELDS, message="denoising", percentage=0.5)

    await socketio._handle_queue_event(("invocation_progress", event))

    socketio._sio.emit.assert_awaited_once_with(
        event="invocation_progress",
        data=event.model_dump(mode="json"),
        room="user:owner-1",
    )


@pytest.mark.anyio
async def test_complete_event_is_emitted_once_to_owner_and_admin_rooms() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event = InvocationCompleteEvent(**_COMMON_FIELDS, result={"type": "integer_output", "value": 3})

    await socketio._handle_queue_event(("invocation_complete", event))

    # A single emit to the union of rooms - python-socketio dedupes recipients across a room
    # list, so an admin owner (or the single-user "system" user) receives exactly one copy.
    socketio._sio.emit.assert_awaited_once_with(
        event="invocation_complete",
        data=event.model_dump(mode="json"),
        room=["user:owner-1", "admin"],
    )


@pytest.mark.anyio
async def test_started_event_is_emitted_once_to_owner_and_admin_rooms() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event = InvocationStartedEvent(**_COMMON_FIELDS)

    await socketio._handle_queue_event(("invocation_started", event))

    socketio._sio.emit.assert_awaited_once_with(
        event="invocation_started",
        data=event.model_dump(mode="json"),
        room=["user:owner-1", "admin"],
    )


@pytest.mark.anyio
async def test_error_event_is_emitted_once_to_owner_and_admin_rooms() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event = InvocationErrorEvent(
        **_COMMON_FIELDS,
        error_type="ValueError",
        error_message="oops",
        error_traceback="Traceback (most recent call last): ...",
    )

    await socketio._handle_queue_event(("invocation_error", event))

    socketio._sio.emit.assert_awaited_once_with(
        event="invocation_error",
        data=event.model_dump(mode="json"),
        room=["user:owner-1", "admin"],
    )

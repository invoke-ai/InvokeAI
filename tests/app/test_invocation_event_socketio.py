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
from pydantic.networks import AnyHttpUrl

from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.events.events_common import (
    DownloadStartedEvent,
    InvocationCompleteEvent,
    InvocationErrorEvent,
    InvocationProgressEvent,
    InvocationStartedEvent,
    ModelInstallStartedEvent,
)
from invokeai.app.services.model_install.model_install_common import URLModelSource


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


def _make_model_config():
    from invokeai.backend.model_manager.configs.factory import AnyModelConfigValidator

    return AnyModelConfigValidator.validate_python(
        {
            "key": "model-key-1",
            "hash": "hash-1",
            "path": "/models/some-model",
            "name": "some-model",
            "base": "sd-1",
            "type": "vae",
            "format": "diffusers",
            "source": "/models/some-model",
            "source_type": "path",
            "file_size": 1024,
        }
    )


@pytest.mark.anyio
async def test_model_load_events_are_emitted_only_to_triggering_user() -> None:
    from invokeai.app.services.events.events_common import ModelLoadCompleteEvent, ModelLoadStartedEvent

    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    config = _make_model_config()
    started = ModelLoadStartedEvent.build(config, user_id="owner-1")
    complete = ModelLoadCompleteEvent.build(config, user_id="owner-1")

    await socketio._handle_model_event(("model_load_started", started))
    await socketio._handle_model_event(("model_load_complete", complete))

    socketio._sio.emit.assert_any_await(
        event="model_load_started", data=started.model_dump(mode="json"), room="user:owner-1"
    )
    socketio._sio.emit.assert_any_await(
        event="model_load_complete", data=complete.model_dump(mode="json"), room="user:owner-1"
    )
    assert socketio._sio.emit.await_count == 2


@pytest.mark.anyio
async def test_model_install_events_are_emitted_only_to_admins() -> None:
    from invokeai.app.services.events.events_common import ModelInstallStartedEvent

    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event = ModelInstallStartedEvent(id=1, source={"type": "url", "url": "https://example.com/model.safetensors"})

    await socketio._handle_model_event(("model_install_started", event))

    socketio._sio.emit.assert_awaited_once_with(
        event="model_install_started", data=event.model_dump(mode="json"), room="admin"
    )


@pytest.mark.anyio
async def test_generic_download_events_remain_broadcast() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    from types import SimpleNamespace

    event = SimpleNamespace(model_dump=lambda mode="json": {"id": 1})

    await socketio._handle_model_event(("download_started", event))

    socketio._sio.emit.assert_awaited_once_with(event="download_started", data={"id": 1})


@pytest.mark.anyio
async def test_llm_task_progress_is_emitted_once_to_owner_and_admin_rooms() -> None:
    from invokeai.app.services.events.events_common import LLMTaskProgressEvent

    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event = LLMTaskProgressEvent(
        task_id="task-1",
        user_id="owner-1",
        phase="generating",
        message="Generating",
        percentage=0.5,
        current_tokens=10,
        total_tokens=20,
    )

    await socketio._handle_llm_task_event(("llm_task_progress", event))

    socketio._sio.emit.assert_awaited_once_with(
        event="llm_task_progress",
        data=event.model_dump(mode="json"),
        room=["user:owner-1", "admin"],
    )


@pytest.mark.anyio
async def test_download_and_model_install_events_are_admin_only() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    download_event = DownloadStartedEvent(source="https://example.com/model", download_path="/cache/model")
    install_event = ModelInstallStartedEvent(
        id=1,
        source=URLModelSource(url=AnyHttpUrl("https://example.com/model")),
    )

    await socketio._handle_model_event(("download_started", download_event))
    await socketio._handle_model_event(("model_install_started", install_event))

    socketio._sio.emit.assert_any_await(
        event="download_started", data=download_event.model_dump(mode="json"), room="admin"
    )
    socketio._sio.emit.assert_any_await(
        event="model_install_started", data=install_event.model_dump(mode="json"), room="admin"
    )
    assert socketio._sio.emit.await_count == 2

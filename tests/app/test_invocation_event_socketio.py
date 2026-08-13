"""Tests for socket routing of model and download events in multiuser mode.

Model install and download events carry the source URL and the server-side path a file is
being written to, so they are routed to the admin room rather than broadcast. Model load
events carry neither and drive the loading-models spinner in every client, so they stay
broadcast.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from pydantic.networks import AnyHttpUrl

from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.events.events_common import (
    DownloadStartedEvent,
    ModelInstallStartedEvent,
)
from invokeai.app.services.model_install.model_install_common import URLModelSource


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
async def test_model_load_events_remain_broadcast() -> None:
    from invokeai.app.services.events.events_common import ModelLoadCompleteEvent, ModelLoadStartedEvent

    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    config = _make_model_config()
    started = ModelLoadStartedEvent.build(config)
    complete = ModelLoadCompleteEvent.build(config)

    await socketio._handle_model_event(("model_load_started", started))
    await socketio._handle_model_event(("model_load_complete", complete))

    socketio._sio.emit.assert_any_await(event="model_load_started", data=started.model_dump(mode="json"))
    socketio._sio.emit.assert_any_await(event="model_load_complete", data=complete.model_dump(mode="json"))
    assert socketio._sio.emit.await_count == 2


def _download_event() -> DownloadStartedEvent:
    return DownloadStartedEvent(source="https://example.com/model", download_path="/cache/model")


def _install_event() -> ModelInstallStartedEvent:
    return ModelInstallStartedEvent(id=1, source=URLModelSource(url=AnyHttpUrl("https://example.com/model")))


@pytest.mark.anyio
async def test_download_and_model_install_events_are_admin_only() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()
    socketio._socket_users = {
        "admin-sid": {"user_id": "admin-1", "is_admin": True},
        "user-sid": {"user_id": "user-1", "is_admin": False},
    }

    download_event = _download_event()
    install_event = _install_event()

    await socketio._handle_model_event(("download_started", download_event))
    await socketio._handle_model_event(("model_install_started", install_event))

    socketio._sio.emit.assert_any_await(
        event="download_started", data=download_event.model_dump(mode="json"), to=["admin-sid"]
    )
    socketio._sio.emit.assert_any_await(
        event="model_install_started", data=install_event.model_dump(mode="json"), to=["admin-sid"]
    )
    assert socketio._sio.emit.await_count == 2


@pytest.mark.anyio
async def test_admin_receives_install_events_before_subscribing_to_a_queue() -> None:
    """The "admin" room is only entered on queue subscription.

    A socket that has connected but not yet sent `subscribe_queue` is in no rooms, so
    addressing these events to the room would drop them for the whole connect/subscribe
    window - and permanently for any client that never subscribes.
    """
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()
    socketio._sio.enter_room = AsyncMock()
    # State after _handle_connect only: user info recorded, no rooms joined.
    socketio._socket_users = {"admin-sid": {"user_id": "admin-1", "is_admin": True}}

    event = _install_event()
    await socketio._handle_model_event(("model_install_started", event))

    socketio._sio.emit.assert_awaited_once_with(
        event="model_install_started", data=event.model_dump(mode="json"), to=["admin-sid"]
    )
    socketio._sio.enter_room.assert_not_awaited()


@pytest.mark.anyio
async def test_install_events_are_dropped_when_no_admin_is_connected() -> None:
    """An empty recipient list must not fall through to a broadcast.

    python-socketio indexes `room[0]` for a list, so passing `[]` would also raise.
    """
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()
    socketio._socket_users = {"user-sid": {"user_id": "user-1", "is_admin": False}}

    await socketio._handle_model_event(("model_install_started", _install_event()))

    socketio._sio.emit.assert_not_awaited()

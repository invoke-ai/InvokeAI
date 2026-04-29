from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from invokeai.app.api.sockets import SocketIO


def _patch_multiuser_context(monkeypatch: pytest.MonkeyPatch, *, user_id: str, is_admin: bool) -> None:
    user = SimpleNamespace(user_id=user_id, is_active=True)
    invoker = SimpleNamespace(
        services=SimpleNamespace(
            configuration=SimpleNamespace(multiuser=True),
            users=SimpleNamespace(get=lambda candidate_user_id: user if candidate_user_id == user_id else None),
        )
    )
    monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))
    monkeypatch.setattr(
        "invokeai.app.api.sockets.verify_token",
        lambda token: SimpleNamespace(user_id=user_id, is_admin=is_admin) if token == "valid-token" else None,
    )


def _patch_single_user_context(monkeypatch: pytest.MonkeyPatch) -> None:
    invoker = SimpleNamespace(services=SimpleNamespace(configuration=SimpleNamespace(multiuser=False)))
    monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))


@pytest.mark.anyio
async def test_authenticated_user_joins_workflow_rooms_on_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.enter_room = AsyncMock()
    _patch_multiuser_context(monkeypatch, user_id="user-1", is_admin=False)

    accepted = await socketio._handle_connect("sid-1", {}, {"token": "valid-token"})

    assert accepted is True
    socketio._sio.enter_room.assert_any_call("sid-1", "user:user-1")
    socketio._sio.enter_room.assert_any_call("sid-1", "workflows:shared")


@pytest.mark.anyio
async def test_admin_joins_admin_room_on_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.enter_room = AsyncMock()
    _patch_multiuser_context(monkeypatch, user_id="admin-1", is_admin=True)

    accepted = await socketio._handle_connect("sid-1", {}, {"token": "valid-token"})

    assert accepted is True
    socketio._sio.enter_room.assert_any_call("sid-1", "user:admin-1")
    socketio._sio.enter_room.assert_any_call("sid-1", "workflows:shared")
    socketio._sio.enter_room.assert_any_call("sid-1", "admin")


@pytest.mark.anyio
async def test_single_user_socket_joins_workflow_rooms_on_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.enter_room = AsyncMock()
    _patch_single_user_context(monkeypatch)

    accepted = await socketio._handle_connect("sid-1", {}, None)

    assert accepted is True
    socketio._sio.enter_room.assert_any_call("sid-1", "user:system")
    socketio._sio.enter_room.assert_any_call("sid-1", "workflows:shared")
    socketio._sio.enter_room.assert_any_call("sid-1", "admin")


@pytest.mark.anyio
async def test_private_workflow_event_is_emitted_only_to_owner_and_admin() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event_payload = SimpleNamespace(
        __event_name__="workflow_created",
        workflow_id="wf-1",
        user_id="owner-1",
        is_public=False,
        model_dump=lambda mode="json": {"workflow_id": "wf-1", "user_id": "owner-1", "is_public": False},
    )

    await socketio._handle_workflow_event(("workflow_created", event_payload))

    socketio._sio.emit.assert_any_call(
        event="workflow_created",
        data={"workflow_id": "wf-1", "user_id": "owner-1", "is_public": False},
        room="user:owner-1",
    )
    socketio._sio.emit.assert_any_call(
        event="workflow_created",
        data={"workflow_id": "wf-1", "user_id": "owner-1", "is_public": False},
        room="admin",
    )
    assert socketio._sio.emit.await_count == 2


@pytest.mark.anyio
async def test_single_user_workflow_event_is_emitted_once_to_admin_room(monkeypatch: pytest.MonkeyPatch) -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()
    _patch_single_user_context(monkeypatch)

    event_payload = SimpleNamespace(
        __event_name__="workflow_created",
        workflow_id="wf-1",
        user_id="system",
        is_public=False,
        model_dump=lambda mode="json": {"workflow_id": "wf-1", "user_id": "system", "is_public": False},
    )

    await socketio._handle_workflow_event(("workflow_created", event_payload))

    socketio._sio.emit.assert_awaited_once_with(
        event="workflow_created",
        data={"workflow_id": "wf-1", "user_id": "system", "is_public": False},
        room="admin",
    )


@pytest.mark.anyio
async def test_shared_workflow_event_is_emitted_to_shared_room() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event_payload = SimpleNamespace(
        __event_name__="workflow_updated",
        workflow_id="wf-1",
        user_id="owner-1",
        old_is_public=False,
        new_is_public=True,
        model_dump=lambda mode="json": {
            "workflow_id": "wf-1",
            "user_id": "owner-1",
            "old_is_public": False,
            "new_is_public": True,
        },
    )

    await socketio._handle_workflow_event(("workflow_updated", event_payload))

    socketio._sio.emit.assert_any_call(
        event="workflow_updated",
        data={"workflow_id": "wf-1", "user_id": "owner-1", "old_is_public": False, "new_is_public": True},
        room="workflows:shared",
    )


@pytest.mark.anyio
async def test_shared_to_private_transition_emits_removal_to_shared_room() -> None:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()

    event_payload = SimpleNamespace(
        __event_name__="workflow_updated",
        workflow_id="wf-1",
        user_id="owner-1",
        old_is_public=True,
        new_is_public=False,
        model_dump=lambda mode="json": {
            "workflow_id": "wf-1",
            "user_id": "owner-1",
            "old_is_public": True,
            "new_is_public": False,
        },
    )

    await socketio._handle_workflow_event(("workflow_updated", event_payload))

    socketio._sio.emit.assert_any_call(
        event="workflow_deleted",
        data={"workflow_id": "wf-1"},
        room="workflows:shared",
    )

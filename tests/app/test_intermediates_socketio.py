"""Intermediates cleanup events reach the confirming account and admins, and nobody else."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.events.events_common import IntermediatesOperationChangedEvent
from invokeai.app.services.intermediates.intermediates_common import (
    IntermediatesOperation,
    IntermediatesOperationProgress,
    IntermediatesScope,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _patch_multiuser(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    invoker = SimpleNamespace(services=SimpleNamespace(configuration=SimpleNamespace(multiuser=enabled)))
    monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))


def _event() -> IntermediatesOperationChangedEvent:
    return IntermediatesOperationChangedEvent.build(
        IntermediatesOperation(
            operation_id="op-1",
            user_id="user-1",
            mode="safe",
            scope=IntermediatesScope(kind="owner", user_id="user-1"),
            status="running",
            created_at=datetime.now(timezone.utc),
            target_images=3,
            target_videos=0,
            progress=IntermediatesOperationProgress(),
        )
    )


async def _route(monkeypatch: pytest.MonkeyPatch, multiuser: bool) -> list:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()
    _patch_multiuser(monkeypatch, multiuser)
    event = _event()

    await socketio._handle_intermediates_event((event.__event_name__, event))

    return [call.kwargs for call in socketio._sio.emit.await_args_list]


def test_the_event_is_registered_with_the_socket_handler() -> None:
    SocketIO(FastAPI())

    from fastapi_events.handlers.local import local_handler

    assert any(
        getattr(handler, "__func__", handler).__name__ == "_handle_intermediates_event"
        for handler in local_handler._registry.get("intermediates_operation_changed", [])
    )


@pytest.mark.anyio
async def test_single_user_events_go_to_the_admin_room(monkeypatch: pytest.MonkeyPatch) -> None:
    emits = await _route(monkeypatch, multiuser=False)

    assert [emit["room"] for emit in emits] == ["admin"]
    assert emits[0]["event"] == "intermediates_operation_changed"
    assert emits[0]["data"]["operation"]["operation_id"] == "op-1"


@pytest.mark.anyio
async def test_multiuser_events_reach_the_confirming_account_and_admins(monkeypatch: pytest.MonkeyPatch) -> None:
    emits = await _route(monkeypatch, multiuser=True)

    assert len(emits) == 1
    assert emits[0]["room"] == ["user:user-1", "admin"]

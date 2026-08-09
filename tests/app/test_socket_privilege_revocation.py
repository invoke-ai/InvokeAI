"""Tests that socket connections lose (or gain) privileges when the backing user
record changes.

Socket room membership is established at connect time. Without live re-authorization,
a demoted administrator's sockets would keep receiving other users' private events via
the admin room, and a deactivated user's sockets would keep receiving events
indefinitely; a demoted admin could also reconnect with an old token and rejoin the
admin room.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.events.events_common import UserAccessChangedEvent


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _patch_multiuser_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    user_id: str,
    token_is_admin: bool,
    db_is_admin: bool,
    db_is_active: bool = True,
    db_epoch: int = 0,
    token_epoch: int = 0,
) -> None:
    """Multiuser context where the token's claims and the database record can differ."""
    user = SimpleNamespace(user_id=user_id, is_active=db_is_active, is_admin=db_is_admin, token_epoch=db_epoch)
    invoker = SimpleNamespace(
        services=SimpleNamespace(
            configuration=SimpleNamespace(multiuser=True),
            users=SimpleNamespace(get=lambda candidate_user_id: user if candidate_user_id == user_id else None),
        )
    )
    monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))
    # The connect handler resolves the record through `resolve_authorized_user`, which binds
    # ApiDependencies at import time in auth_dependencies — patching the defining module alone
    # would not reach it.
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))
    monkeypatch.setattr(
        "invokeai.app.api.sockets.verify_token",
        lambda token: SimpleNamespace(user_id=user_id, is_admin=token_is_admin, token_epoch=token_epoch)
        if token == "valid-token"
        else None,
    )


class TestConnectDerivesRoleFromDatabase:
    """_handle_connect must trust the database record, not the token's is_admin claim."""

    @pytest.mark.anyio
    async def test_demoted_admin_reconnecting_with_old_token_does_not_join_admin_room(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        _patch_multiuser_context(monkeypatch, user_id="user-1", token_is_admin=True, db_is_admin=False)

        accepted = await socketio._handle_connect("sid-1", {}, {"token": "valid-token"})

        assert accepted is True
        rooms_entered = [call.args[1] for call in socketio._sio.enter_room.await_args_list]
        assert "admin" not in rooms_entered
        assert socketio._socket_users["sid-1"]["is_admin"] is False

    @pytest.mark.anyio
    async def test_promoted_user_connecting_with_old_token_joins_admin_room(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        _patch_multiuser_context(monkeypatch, user_id="user-1", token_is_admin=False, db_is_admin=True)

        accepted = await socketio._handle_connect("sid-1", {}, {"token": "valid-token"})

        assert accepted is True
        rooms_entered = [call.args[1] for call in socketio._sio.enter_room.await_args_list]
        assert "admin" in rooms_entered
        assert socketio._socket_users["sid-1"]["is_admin"] is True

    @pytest.mark.anyio
    async def test_deactivated_user_cannot_reconnect_with_old_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        _patch_multiuser_context(
            monkeypatch, user_id="user-1", token_is_admin=False, db_is_admin=False, db_is_active=False
        )

        accepted = await socketio._handle_connect("sid-1", {}, {"token": "valid-token"})

        assert accepted is False
        assert "sid-1" not in socketio._socket_users

    @pytest.mark.anyio
    async def test_active_admin_still_joins_admin_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive case: an unchanged administrator keeps full admin connectivity."""
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        _patch_multiuser_context(monkeypatch, user_id="admin-1", token_is_admin=True, db_is_admin=True)

        accepted = await socketio._handle_connect("sid-1", {}, {"token": "valid-token"})

        assert accepted is True
        rooms_entered = [call.args[1] for call in socketio._sio.enter_room.await_args_list]
        assert "admin" in rooms_entered


class TestUserAccessChangedHandler:
    """_handle_user_access_changed re-authorizes already-connected sockets."""

    def _connected_socketio(self) -> SocketIO:
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        socketio._sio.leave_room = AsyncMock()
        socketio._sio.disconnect = AsyncMock()
        socketio._socket_users = {
            "sid-admin": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
            "sid-user-a": {"user_id": "user-1", "is_admin": False, "token_epoch": 0},
            "sid-user-b": {"user_id": "user-1", "is_admin": False, "token_epoch": 0},
            "sid-other": {"user_id": "user-2", "is_admin": False, "token_epoch": 0},
        }
        return socketio

    @pytest.mark.anyio
    async def test_demoted_admin_sockets_leave_admin_room(self) -> None:
        socketio = self._connected_socketio()
        event = UserAccessChangedEvent.build(user_id="admin-1", is_admin=False, is_active=True)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        socketio._sio.leave_room.assert_awaited_once_with("sid-admin", "admin")
        assert socketio._socket_users["sid-admin"]["is_admin"] is False
        socketio._sio.disconnect.assert_not_awaited()

    @pytest.mark.anyio
    async def test_demoted_admin_cannot_rejoin_admin_room_via_queue_subscription(self) -> None:
        """After demotion, the cached is_admin is False, so _handle_sub_queue does not
        re-add the socket to the admin room."""
        socketio = self._connected_socketio()
        event = UserAccessChangedEvent.build(user_id="admin-1", is_admin=False, is_active=True)
        await socketio._handle_user_access_changed(("user_access_changed", event))

        await socketio._handle_sub_queue("sid-admin", {"queue_id": "default"})

        rooms_entered = [call.args[1] for call in socketio._sio.enter_room.await_args_list]
        assert "admin" not in rooms_entered

    @pytest.mark.anyio
    async def test_deactivated_user_sockets_are_disconnected(self) -> None:
        socketio = self._connected_socketio()
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert disconnected == {"sid-user-a", "sid-user-b"}

    @pytest.mark.anyio
    async def test_deleted_user_sockets_are_disconnected(self) -> None:
        """Deletion is emitted as is_active=False and disconnects the user's sockets."""
        socketio = self._connected_socketio()
        event = UserAccessChangedEvent.build(user_id="user-2", is_admin=False, is_active=False)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert disconnected == {"sid-other"}

    @pytest.mark.anyio
    async def test_promoted_user_sockets_join_admin_room(self) -> None:
        socketio = self._connected_socketio()
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=True, is_active=True)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        rooms_entered = [(call.args[0], call.args[1]) for call in socketio._sio.enter_room.await_args_list]
        assert ("sid-user-a", "admin") in rooms_entered
        assert ("sid-user-b", "admin") in rooms_entered
        assert socketio._socket_users["sid-user-a"]["is_admin"] is True

    @pytest.mark.anyio
    async def test_other_users_sockets_are_untouched(self) -> None:
        """Positive case: an access change for one user does not affect other users'
        sockets — an unchanged administrator keeps receiving admin-room events."""
        socketio = self._connected_socketio()
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert "sid-admin" not in disconnected
        assert "sid-other" not in disconnected
        socketio._sio.leave_room.assert_not_awaited()
        assert socketio._socket_users["sid-admin"]["is_admin"] is True

    @pytest.mark.anyio
    async def test_a_socket_dropping_mid_loop_is_skipped_not_redisconnected(self) -> None:
        """A socket that goes away while the loop is suspended is skipped on its turn.

        The deactivation branch never indexed `_socket_users`, so it did not raise on a
        mid-loop removal — see `test_demotion_survives_a_socket_dropping_mid_loop` for the
        branch that did. What this pins is the weaker half: the loop's snapshot is not
        treated as still-live, so an already-disconnected socket is not disconnected twice.
        """
        socketio = self._connected_socketio()
        socketio._socket_users["sid-user-c"] = {"user_id": "user-1", "is_admin": True, "token_epoch": 0}

        async def disconnect(sid: str) -> None:
            # Stand in for the packet flush: yield, and drop an as-yet-unvisited socket
            # of the same user while suspended, as a client disconnect would.
            await asyncio.sleep(0)
            socketio._socket_users.pop("sid-user-b", None)
            await socketio._handle_disconnect(sid)

        socketio._sio.disconnect = AsyncMock(side_effect=disconnect)
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        # sid-user-a is disconnected first and takes sid-user-b with it; sid-user-c must
        # still be reached rather than left connected on revoked credentials.
        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert disconnected == {"sid-user-a", "sid-user-c"}
        assert "sid-user-c" not in socketio._socket_users

    @pytest.mark.anyio
    async def test_demotion_survives_a_socket_dropping_mid_loop(self) -> None:
        """Same interleaving on the demotion path, where an abandoned socket is worse: it
        stays in the admin room *and* keeps a cached is_admin of True, which
        `_handle_sub_queue` would use to re-add it on the next subscription."""
        socketio = self._connected_socketio()
        socketio._socket_users = {
            "sid-a": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
            "sid-b": {"user_id": "admin-1", "is_admin": True, "token_epoch": 1},
            "sid-c": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
        }

        async def disconnect(sid: str) -> None:
            await asyncio.sleep(0)
            socketio._socket_users.pop("sid-b", None)
            await socketio._handle_disconnect(sid)

        socketio._sio.disconnect = AsyncMock(side_effect=disconnect)
        # Epoch 1: sid-a and sid-c hold superseded tokens, sid-b is current.
        event = UserAccessChangedEvent.build(user_id="admin-1", is_admin=False, is_active=True, token_epoch=1)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert disconnected == {"sid-a", "sid-c"}


class TestTokenEpochOnSockets:
    """A revoked token must not open a socket, and must not keep an open one alive.

    Sockets authenticate once, at connect. Without both halves of this, HTTP would be
    locked out while an already-connected socket kept streaming the same user's events.
    """

    @pytest.mark.anyio
    async def test_revoked_token_cannot_connect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        _patch_multiuser_context(
            monkeypatch, user_id="user-1", token_is_admin=False, db_is_admin=False, db_epoch=1, token_epoch=0
        )

        accepted = await socketio._handle_connect("sid-1", {}, {"token": "valid-token"})

        assert accepted is False
        assert "sid-1" not in socketio._socket_users
        socketio._sio.enter_room.assert_not_awaited()

    @pytest.mark.anyio
    async def test_current_token_still_connects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The epoch check must not reject an ordinary, current token."""
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        _patch_multiuser_context(
            monkeypatch, user_id="user-1", token_is_admin=False, db_is_admin=False, db_epoch=3, token_epoch=3
        )

        accepted = await socketio._handle_connect("sid-1", {}, {"token": "valid-token"})

        assert accepted is True
        assert socketio._socket_users["sid-1"]["token_epoch"] == 3

    @pytest.mark.anyio
    async def test_password_change_disconnects_superseded_sockets_only(self) -> None:
        """The account stays active, so the deactivation branch does not fire — the epoch
        is what identifies which of the user's sockets are now holding dead tokens."""
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        socketio._sio.leave_room = AsyncMock()
        socketio._sio.disconnect = AsyncMock()
        socketio._socket_users = {
            "sid-old": {"user_id": "user-1", "is_admin": False, "token_epoch": 0},
            "sid-new": {"user_id": "user-1", "is_admin": False, "token_epoch": 1},
            "sid-other": {"user_id": "user-2", "is_admin": False, "token_epoch": 0},
        }
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=True, token_epoch=1)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert disconnected == {"sid-old"}

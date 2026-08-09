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


class TestRevalidationSweep:
    """`_revalidate_socket_users` catches user changes made by another process.

    `invoke-usermod --no-admin` and `invoke-userdel` open the database directly, from
    their own process. Nothing in the server process can raise `user_access_changed` for
    them, so a socket would otherwise keep the rooms it joined at connect time until it
    happened to reconnect. REST is unaffected either way — every request re-reads the
    record — which is exactly why the socket cache is the thing that needs a sweep.
    """

    def _socketio(self, socket_users: dict[str, dict]) -> SocketIO:
        socketio = SocketIO(FastAPI())
        socketio._socket_users = socket_users
        return socketio

    def _patch_services(
        self,
        monkeypatch: pytest.MonkeyPatch,
        users_by_id: dict[str, SimpleNamespace | None],
        *,
        multiuser: bool = True,
        lookups: list[str] | None = None,
    ) -> list:
        """Bind ApiDependencies to a stub and return the list emitted events land in."""
        emitted: list = []

        def get(user_id: str):
            if lookups is not None:
                lookups.append(user_id)
            user = users_by_id[user_id]
            if isinstance(user, Exception):
                raise user
            return user

        invoker = SimpleNamespace(
            services=SimpleNamespace(
                configuration=SimpleNamespace(multiuser=multiuser),
                users=SimpleNamespace(get=get),
                events=SimpleNamespace(emit_user_access_changed=lambda **kwargs: emitted.append(kwargs)),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))
        return emitted

    def _user(self, user_id: str, *, is_admin: bool, is_active: bool = True, token_epoch: int = 0) -> SimpleNamespace:
        return SimpleNamespace(user_id=user_id, is_admin=is_admin, is_active=is_active, token_epoch=token_epoch)

    @pytest.mark.anyio
    async def test_cli_demotion_is_published(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio({"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}})
        emitted = self._patch_services(monkeypatch, {"admin-1": self._user("admin-1", is_admin=False)})

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "admin-1", "is_admin": False, "is_active": True, "token_epoch": 0}]

    @pytest.mark.anyio
    async def test_cli_deletion_is_published(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        emitted = self._patch_services(monkeypatch, {"user-1": None})

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "user-1", "is_admin": False, "is_active": False, "token_epoch": 0}]

    @pytest.mark.anyio
    async def test_cli_deactivation_is_published(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        emitted = self._patch_services(monkeypatch, {"user-1": self._user("user-1", is_admin=False, is_active=False)})

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "user-1", "is_admin": False, "is_active": False, "token_epoch": 0}]

    @pytest.mark.anyio
    async def test_cli_password_reset_is_published(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The account stays active and keeps its role; only the epoch moves."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        emitted = self._patch_services(monkeypatch, {"user-1": self._user("user-1", is_admin=False, token_epoch=1)})

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "user-1", "is_admin": False, "is_active": True, "token_epoch": 1}]

    @pytest.mark.anyio
    async def test_unchanged_users_emit_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sweep runs forever; a no-op sweep must stay a no-op."""
        socketio = self._socketio(
            {
                "sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 2},
                "sid-2": {"user_id": "user-1", "is_admin": False, "token_epoch": 0},
            }
        )
        emitted = self._patch_services(
            monkeypatch,
            {
                "admin-1": self._user("admin-1", is_admin=True, token_epoch=2),
                "user-1": self._user("user-1", is_admin=False),
            },
        )

        await socketio._revalidate_socket_users()

        assert emitted == []

    @pytest.mark.anyio
    async def test_a_user_with_several_sockets_is_looked_up_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Several open tabs are the common case, and every socket of a user shares one answer."""
        socketio = self._socketio(
            {
                "sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
                "sid-2": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
                "sid-3": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
            }
        )
        lookups: list[str] = []
        emitted = self._patch_services(monkeypatch, {"admin-1": self._user("admin-1", is_admin=False)}, lookups=lookups)

        await socketio._revalidate_socket_users()

        assert lookups == ["admin-1"]
        assert len(emitted) == 1

    @pytest.mark.anyio
    async def test_an_unreadable_record_is_left_for_the_next_sweep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Opposite of the queue gate's fail-closed policy, deliberately: nothing runs on the
        user's behalf because a socket stays open one more interval, whereas tearing down
        every live session on a transient database error would be a self-inflicted outage —
        and `_handle_connect` fails closed, so the clients would not get back in."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0}})
        emitted = self._patch_services(monkeypatch, {"user-1": RuntimeError("database is locked")})

        await socketio._revalidate_socket_users()

        assert emitted == []

    @pytest.mark.anyio
    async def test_one_unreadable_record_does_not_abandon_the_others(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio(
            {
                "sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0},
                "sid-2": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
            }
        )
        emitted = self._patch_services(
            monkeypatch,
            {"user-1": RuntimeError("database is locked"), "admin-1": self._user("admin-1", is_admin=False)},
        )

        await socketio._revalidate_socket_users()

        assert [event["user_id"] for event in emitted] == ["admin-1"]

    @pytest.mark.anyio
    async def test_single_user_mode_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single-user sockets are cached as the system admin and have no record to check."""
        socketio = self._socketio({"sid-1": {"user_id": "system", "is_admin": True}})
        emitted = self._patch_services(monkeypatch, {"system": None}, multiuser=False)

        await socketio._revalidate_socket_users()

        assert emitted == []

    @pytest.mark.anyio
    async def test_a_stale_socket_is_found_even_beside_an_up_to_date_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The user's sockets need not agree. A session that reconnected after a CLI
        password reset holds the current epoch while the superseded session is still
        connected under the old one; sampling only one of them would find nothing to do."""
        socketio = self._socketio(
            {
                "sid-new": {"user_id": "user-1", "is_admin": False, "token_epoch": 1},
                "sid-old": {"user_id": "user-1", "is_admin": False, "token_epoch": 0},
            }
        )
        emitted = self._patch_services(monkeypatch, {"user-1": self._user("user-1", is_admin=False, token_epoch=1)})

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "user-1", "is_admin": False, "is_active": True, "token_epoch": 1}]


class TestHandlerRereadsAtThePointOfDecision:
    """`_handle_user_access_changed` applies the *record*, using the event only as a trigger.

    Handlers are dispatched as independent tasks, and the revalidation sweep's payload is a
    snapshot taken before an await, so an event can arrive already superseded.
    """

    def _socketio(self, socket_users: dict[str, dict]) -> SocketIO:
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        socketio._sio.leave_room = AsyncMock()
        socketio._sio.disconnect = AsyncMock()
        socketio._socket_users = socket_users
        return socketio

    def _patch_users(self, monkeypatch: pytest.MonkeyPatch, user) -> None:
        invoker = SimpleNamespace(
            services=SimpleNamespace(
                configuration=SimpleNamespace(multiuser=True),
                users=SimpleNamespace(get=lambda user_id: user),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))

    @pytest.mark.anyio
    async def test_a_superseded_promotion_does_not_regrant_the_admin_room(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sweep that read `is_admin=True`, then a demotion committed while it was
        suspended. Applying the stale payload would put a demoted user back in the admin
        room — receiving every other user's events — until the next sweep."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        self._patch_users(monkeypatch, SimpleNamespace(user_id="user-1", is_admin=False, is_active=True, token_epoch=0))
        stale = UserAccessChangedEvent.build(user_id="user-1", is_admin=True, is_active=True)

        await socketio._handle_user_access_changed(("user_access_changed", stale))

        socketio._sio.enter_room.assert_not_awaited()
        socketio._sio.leave_room.assert_awaited_once_with("sid-1", "admin")
        assert socketio._socket_users["sid-1"]["is_admin"] is False

    @pytest.mark.anyio
    async def test_a_superseded_epoch_does_not_disconnect_the_replacement_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The user changed their password and reconnected with the replacement token. A
        sweep still carrying the pre-change epoch must not kick that new session."""
        socketio = self._socketio({"sid-new": {"user_id": "user-1", "is_admin": False, "token_epoch": 1}})
        self._patch_users(monkeypatch, SimpleNamespace(user_id="user-1", is_admin=False, is_active=True, token_epoch=1))
        stale = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=True, token_epoch=0)

        await socketio._handle_user_access_changed(("user_access_changed", stale))

        socketio._sio.disconnect.assert_not_awaited()

    @pytest.mark.anyio
    async def test_a_deleted_record_disconnects_even_if_the_event_says_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        self._patch_users(monkeypatch, None)
        stale = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=True)

        await socketio._handle_user_access_changed(("user_access_changed", stale))

        socketio._sio.disconnect.assert_awaited_once_with("sid-1")

    @pytest.mark.anyio
    async def test_an_unreadable_record_leaves_the_event_standing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The event is evidence of a committed change; a read that cannot contradict it
        does not get to override it."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0}})

        def explode(user_id: str):
            raise RuntimeError("database is locked")

        invoker = SimpleNamespace(
            services=SimpleNamespace(
                configuration=SimpleNamespace(multiuser=True),
                users=SimpleNamespace(get=explode),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        socketio._sio.disconnect.assert_awaited_once_with("sid-1")

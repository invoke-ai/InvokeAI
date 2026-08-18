"""Tests that socket connections lose (or gain) privileges when the backing user
record changes.

Socket room membership is established at connect time. Without live re-authorization,
a demoted administrator's sockets would keep receiving other users' private events via
the admin room, and a deactivated user's sockets would keep receiving events
indefinitely; a demoted admin could also reconnect with an old token and rejoin the
admin room.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from invokeai.app.api.sockets import SOCKET_REVALIDATION_FAILURE_LIMIT, SocketIO
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

    def _connected_socketio(self, monkeypatch: pytest.MonkeyPatch, event: UserAccessChangedEvent) -> SocketIO:
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
        self._patch_record_agreeing_with(monkeypatch, event)
        return socketio

    def _patch_record_agreeing_with(self, monkeypatch: pytest.MonkeyPatch, event: UserAccessChangedEvent) -> None:
        """Back the event with a database record that says the same thing.

        The handler re-reads and applies the *record*, so a test that leaves
        `ApiDependencies` unbound is not testing what it looks like: the read raises, and
        the handler falls to its failure path. That path deliberately refuses to grant
        privileges, so an unbound promotion test would fail — and, before that refusal
        existed, passed for the wrong reason. The disagreeing cases live in
        `TestHandlerRereadsAtThePointOfDecision`.
        """
        record = (
            None
            if not event.is_active
            else SimpleNamespace(
                user_id=event.user_id,
                is_admin=event.is_admin,
                is_active=True,
                token_epoch=event.token_epoch,
            )
        )
        invoker = SimpleNamespace(
            services=SimpleNamespace(
                configuration=SimpleNamespace(multiuser=True),
                users=SimpleNamespace(get=lambda user_id: record),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))

    @pytest.mark.anyio
    async def test_demoted_admin_sockets_leave_admin_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = UserAccessChangedEvent.build(user_id="admin-1", is_admin=False, is_active=True)
        socketio = self._connected_socketio(monkeypatch, event)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        socketio._sio.leave_room.assert_awaited_once_with("sid-admin", "admin")
        assert socketio._socket_users["sid-admin"]["is_admin"] is False
        socketio._sio.disconnect.assert_not_awaited()

    @pytest.mark.anyio
    async def test_demoted_admin_cannot_rejoin_admin_room_via_queue_subscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After demotion, the cached is_admin is False, so _handle_sub_queue does not
        re-add the socket to the admin room."""
        event = UserAccessChangedEvent.build(user_id="admin-1", is_admin=False, is_active=True)
        socketio = self._connected_socketio(monkeypatch, event)
        await socketio._handle_user_access_changed(("user_access_changed", event))

        await socketio._handle_sub_queue("sid-admin", {"queue_id": "default"})

        rooms_entered = [call.args[1] for call in socketio._sio.enter_room.await_args_list]
        assert "admin" not in rooms_entered

    @pytest.mark.anyio
    async def test_deactivated_user_sockets_are_disconnected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)
        socketio = self._connected_socketio(monkeypatch, event)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert disconnected == {"sid-user-a", "sid-user-b"}

    @pytest.mark.anyio
    async def test_deleted_user_sockets_are_disconnected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deletion is emitted as is_active=False and disconnects the user's sockets."""
        event = UserAccessChangedEvent.build(user_id="user-2", is_admin=False, is_active=False)
        socketio = self._connected_socketio(monkeypatch, event)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert disconnected == {"sid-other"}

    @pytest.mark.anyio
    async def test_promoted_user_sockets_join_admin_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=True, is_active=True)
        socketio = self._connected_socketio(monkeypatch, event)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        rooms_entered = [(call.args[0], call.args[1]) for call in socketio._sio.enter_room.await_args_list]
        assert ("sid-user-a", "admin") in rooms_entered
        assert ("sid-user-b", "admin") in rooms_entered
        assert socketio._socket_users["sid-user-a"]["is_admin"] is True

    @pytest.mark.anyio
    async def test_other_users_sockets_are_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive case: an access change for one user does not affect other users'
        sockets — an unchanged administrator keeps receiving admin-room events."""
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)
        socketio = self._connected_socketio(monkeypatch, event)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert "sid-admin" not in disconnected
        assert "sid-other" not in disconnected
        socketio._sio.leave_room.assert_not_awaited()
        assert socketio._socket_users["sid-admin"]["is_admin"] is True

    @pytest.mark.anyio
    async def test_a_socket_dropping_mid_loop_is_skipped_not_redisconnected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A socket that goes away while the loop is suspended is skipped on its turn.

        The deactivation branch never indexed `_socket_users`, so it did not raise on a
        mid-loop removal — see `test_demotion_survives_a_socket_dropping_mid_loop` for the
        branch that did. What this pins is the weaker half: the loop's snapshot is not
        treated as still-live, so an already-disconnected socket is not disconnected twice.
        """
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)
        socketio = self._connected_socketio(monkeypatch, event)
        socketio._socket_users["sid-user-c"] = {"user_id": "user-1", "is_admin": True, "token_epoch": 0}

        async def disconnect(sid: str) -> None:
            # Stand in for the packet flush: yield, and drop an as-yet-unvisited socket
            # of the same user while suspended, as a client disconnect would.
            await asyncio.sleep(0)
            socketio._socket_users.pop("sid-user-b", None)
            await socketio._handle_disconnect(sid)

        socketio._sio.disconnect = AsyncMock(side_effect=disconnect)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        # sid-user-a is disconnected first and takes sid-user-b with it; sid-user-c must
        # still be reached rather than left connected on revoked credentials.
        disconnected = {call.args[0] for call in socketio._sio.disconnect.await_args_list}
        assert disconnected == {"sid-user-a", "sid-user-c"}
        assert "sid-user-c" not in socketio._socket_users

    @pytest.mark.anyio
    async def test_demotion_survives_a_socket_dropping_mid_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same interleaving on the demotion path, where an abandoned socket is worse: it
        stays in the admin room *and* keeps a cached is_admin of True, which
        `_handle_sub_queue` would use to re-add it on the next subscription."""
        # Epoch 1: sid-a and sid-c hold superseded tokens, sid-b is current.
        event = UserAccessChangedEvent.build(user_id="admin-1", is_admin=False, is_active=True, token_epoch=1)
        socketio = self._connected_socketio(monkeypatch, event)
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
    async def test_password_change_disconnects_superseded_sockets_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The account stays active, so the deactivation branch does not fire — the epoch
        is what identifies which of the user's sockets are now holding dead tokens."""
        # Bind the record the handler re-reads. Without this the read raises
        # (`ApiDependencies.invoker` is an unset annotation) and the whole re-read block
        # could be deleted with this test still green — it would be pinning the failure
        # path, not the epoch logic it is named for.
        _patch_multiuser_context(
            monkeypatch, user_id="user-1", token_is_admin=False, db_is_admin=False, db_epoch=1, token_epoch=1
        )
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


class _SweepHarness:
    """Stubs shared by the sweep test classes. Not collected — the name does not start with Test."""

    def _socketio(self, socket_users: dict[str, dict]) -> SocketIO:
        socketio = SocketIO(FastAPI())
        socketio._sio.leave_room = AsyncMock()
        socketio._socket_users = socket_users
        return socketio

    def _patch_services(
        self,
        monkeypatch: pytest.MonkeyPatch,
        users_by_id: dict[str, SimpleNamespace | None],
        *,
        multiuser: bool = True,
        lookups: list[str] | None = None,
        running_owners: set[str] | None = None,
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
                session_processor=SimpleNamespace(
                    get_running_queue_item_owners=lambda: set(running_owners or set()),
                ),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))
        return emitted

    def _user(self, user_id: str, *, is_admin: bool, is_active: bool = True, token_epoch: int = 0) -> SimpleNamespace:
        return SimpleNamespace(user_id=user_id, is_admin=is_admin, is_active=is_active, token_epoch=token_epoch)


class TestRevalidationSweep(_SweepHarness):
    """`_revalidate_socket_users` catches user changes made by another process.

    `invoke-usermod --no-admin` and `invoke-userdel` open the database directly, from
    their own process. Nothing in the server process can raise `user_access_changed` for
    them, so a socket would otherwise keep the rooms it joined at connect time until it
    happened to reconnect. REST is unaffected either way — every request re-reads the
    record — which is exactly why the socket cache is the thing that needs a sweep.
    """

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
        """Below the failure limit the socket is left alone, opposite to the queue gate's
        fail-closed policy and deliberately so: nothing runs on the user's behalf because a
        socket stays open one more interval, whereas tearing down every live session on a
        transient database error would be a self-inflicted outage — and `_handle_connect`
        fails closed, so the clients would not get back in."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0}})
        emitted = self._patch_services(monkeypatch, {"user-1": RuntimeError("database is locked")})

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT - 1):
            await socketio._revalidate_socket_users()

        assert emitted == []
        assert socketio._socket_users["sid-1"]["is_admin"] is True
        socketio._sio.leave_room.assert_not_awaited()

    @pytest.mark.anyio
    async def test_repeated_failures_drop_the_admin_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The retry cannot be unbounded. A CLI demotion during a database outage would
        otherwise leave the socket in the admin room — reading every other user's private
        events — for as long as the reads kept failing."""
        socketio = self._socketio(
            {
                "sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
                "sid-2": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
            }
        )
        self._patch_services(monkeypatch, {"admin-1": RuntimeError("database is locked")})

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT):
            await socketio._revalidate_socket_users()

        assert socketio._socket_users["sid-1"]["is_admin"] is False
        assert socketio._socket_users["sid-2"]["is_admin"] is False
        assert sorted(call.args for call in socketio._sio.leave_room.await_args_list) == [
            ("sid-1", "admin"),
            ("sid-2", "admin"),
        ]

    @pytest.mark.anyio
    async def test_the_dropped_privilege_is_restored_once_the_record_reads_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Revocation on failure is only safe because it is self-healing: a still-genuine
        admin now has a cached `is_admin` the record disagrees with, which is precisely
        what the staleness check publishes."""
        socketio = self._socketio({"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}})
        users: dict[str, object] = {"admin-1": RuntimeError("database is locked")}
        emitted = self._patch_services(monkeypatch, users)

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT):
            await socketio._revalidate_socket_users()
        assert socketio._socket_users["sid-1"]["is_admin"] is False

        users["admin-1"] = self._user("admin-1", is_admin=True)
        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "admin-1", "is_admin": True, "is_active": True, "token_epoch": 0}]

    @pytest.mark.anyio
    async def test_a_successful_read_resets_the_failure_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`SOCKET_REVALIDATION_FAILURE_LIMIT` counts *consecutive* failures. A database
        that fails one sweep in two is being read successfully in between, so nothing is
        stale and there is nothing to revoke."""
        socketio = self._socketio({"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}})
        users: dict[str, object] = {"admin-1": RuntimeError("database is locked")}
        self._patch_services(monkeypatch, users)

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT * 3):
            await socketio._revalidate_socket_users()
            users["admin-1"] = (
                self._user("admin-1", is_admin=True)
                if isinstance(users["admin-1"], Exception)
                else RuntimeError("database is locked")
            )

        assert socketio._socket_users["sid-1"]["is_admin"] is True
        socketio._sio.leave_room.assert_not_awaited()

    @pytest.mark.anyio
    async def test_a_reconnecting_user_does_not_inherit_the_old_failure_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The counter is keyed by user, but it stands in for the state of *sockets*. A user
        whose last socket closed mid-outage and who reconnected onto a healthy database must
        not be one failed read away from losing the admin room."""
        socket_users: dict[str, dict] = {"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}}
        socketio = self._socketio(socket_users)
        self._patch_services(monkeypatch, {"admin-1": RuntimeError("database is locked")})

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT - 1):
            await socketio._revalidate_socket_users()

        # Disconnect for real rather than reaching into the dict: `_revalidation_loop`
        # skips the sweep entirely while no sockets are open, so the sweep's own orphan
        # cleanup never runs in this window. The counter has to be dropped at the
        # disconnect or it survives to meet the returning socket.
        await socketio._handle_disconnect("sid-1")
        assert socketio._revalidation_failures == {}

        socket_users["sid-2"] = {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}
        await socketio._revalidate_socket_users()

        assert socketio._socket_users["sid-2"]["is_admin"] is True
        socketio._sio.leave_room.assert_not_awaited()

    @pytest.mark.anyio
    async def test_a_disconnect_during_the_lookup_does_not_resurrect_the_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lookup awaits, and the failure is recorded after it returns. A last socket
        closing inside that window runs the disconnect's cleanup *before* the write, so a
        naive write puts back a counter for a user with no sockets — and nothing would ever
        collect it, because the loop skips sweeps entirely while none are open."""
        socket_users: dict[str, dict] = {"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}}
        socketio = self._socketio(socket_users)

        # The lookup runs on a worker thread via `run_in_threadpool`, so the event loop is
        # genuinely free while it is in flight. Park it there, run the real disconnect
        # handler in that window, and only then let it fail — the actual interleaving,
        # rather than a simulation of it.
        lookup_entered = threading.Event()
        release_lookup = threading.Event()

        def get(user_id: str):
            lookup_entered.set()
            assert release_lookup.wait(5), "test deadlock: the disconnect never ran"
            raise RuntimeError("database is locked")

        invoker = SimpleNamespace(
            services=SimpleNamespace(
                configuration=SimpleNamespace(multiuser=True),
                users=SimpleNamespace(get=get),
                events=SimpleNamespace(emit_user_access_changed=lambda **kwargs: None),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))

        sweep = asyncio.create_task(socketio._revalidate_socket_users())
        await asyncio.to_thread(lookup_entered.wait, 5)
        await socketio._handle_disconnect("sid-1")
        assert socketio._revalidation_failures == {}, "precondition: the disconnect cleared the count"
        release_lookup.set()
        await sweep

        assert socketio._socket_users == {}
        assert socketio._revalidation_failures == {}

    @pytest.mark.anyio
    async def test_a_successful_connect_time_read_clears_the_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`_handle_connect` resolves the same record the sweep does. A socket admitted as
        an administrator by a read that succeeded must get the full retry budget, not
        inherit a count from an outage that had demonstrably ended before it connected."""
        socketio = self._socketio({"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}})
        self._patch_services(monkeypatch, {"admin-1": RuntimeError("database is locked")})

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT - 1):
            await socketio._revalidate_socket_users()
        assert socketio._revalidation_failures["admin-1"] == SOCKET_REVALIDATION_FAILURE_LIMIT - 1

        # The database recovers and the admin opens a second tab.
        socketio._sio.enter_room = AsyncMock()
        _patch_multiuser_context(monkeypatch, user_id="admin-1", token_is_admin=True, db_is_admin=True)
        assert await socketio._handle_connect("sid-2", {}, {"token": "valid-token"}) is True

        assert socketio._revalidation_failures == {}

    @pytest.mark.anyio
    async def test_a_successful_handler_reread_clears_the_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same rule for the other reader. A promotion applied from a re-read that
        succeeded must not be one failed sweep away from being taken back."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        socketio._sio.enter_room = AsyncMock()
        self._patch_services(monkeypatch, {"user-1": RuntimeError("database is locked")})

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT):
            await socketio._revalidate_socket_users()
        assert socketio._revalidation_failures["user-1"] == SOCKET_REVALIDATION_FAILURE_LIMIT

        _patch_multiuser_context(monkeypatch, user_id="user-1", token_is_admin=False, db_is_admin=True)
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=True, is_active=True)
        await socketio._handle_user_access_changed(("user_access_changed", event))

        assert socketio._socket_users["sid-1"]["is_admin"] is True
        assert socketio._revalidation_failures == {}

    @pytest.mark.anyio
    async def test_the_sweep_drops_a_counter_whose_user_has_no_sockets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backstop for anything that removes sockets without going through
        `_handle_disconnect`. Exercised directly because no path through the public API
        should be able to leave such an entry behind in the first place."""
        socketio = self._socketio({"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}})
        self._patch_services(monkeypatch, {"admin-1": self._user("admin-1", is_admin=True)})
        socketio._revalidation_failures["ghost-user"] = 2

        await socketio._revalidate_socket_users()

        assert socketio._revalidation_failures == {}

    @pytest.mark.anyio
    async def test_one_of_several_sockets_closing_keeps_the_failure_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The counter is dropped when the user's *last* socket goes, not the first. A user
        with three tabs who closes one has not had their remaining sockets checked any more
        recently than before."""
        socket_users: dict[str, dict] = {
            "sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
            "sid-2": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0},
        }
        socketio = self._socketio(socket_users)
        self._patch_services(monkeypatch, {"admin-1": RuntimeError("database is locked")})

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT - 1):
            await socketio._revalidate_socket_users()
        await socketio._handle_disconnect("sid-1")

        assert socketio._revalidation_failures["admin-1"] == SOCKET_REVALIDATION_FAILURE_LIMIT - 1

        await socketio._revalidate_socket_users()

        assert socketio._socket_users["sid-2"]["is_admin"] is False

    @pytest.mark.anyio
    async def test_repeated_failures_do_not_disconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The privilege is dropped, not the connection. Disconnecting every socket because
        the database is unreadable is the outage this sweep exists to avoid, and
        `_handle_connect` fails closed, so the clients could not get back in."""
        socketio = self._socketio({"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}})
        socketio._sio.disconnect = AsyncMock()
        self._patch_services(monkeypatch, {"admin-1": RuntimeError("database is locked")})

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT * 2):
            await socketio._revalidate_socket_users()

        socketio._sio.disconnect.assert_not_awaited()
        assert "sid-1" in socketio._socket_users

    @pytest.mark.anyio
    async def test_a_non_admin_users_failures_revoke_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """There is no privilege to drop, and dropping the connection is not the policy."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        emitted = self._patch_services(monkeypatch, {"user-1": RuntimeError("database is locked")})

        for _ in range(SOCKET_REVALIDATION_FAILURE_LIMIT * 2):
            await socketio._revalidate_socket_users()

        assert emitted == []
        socketio._sio.leave_room.assert_not_awaited()
        assert "sid-1" in socketio._socket_users

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


class TestRunningQueueOwnersAreSwept(_SweepHarness):
    """The sweep also covers the owner of every queue item that is executing.

    A queue item is not attached to a socket. An account deleted with `invoke-userdel`
    while its owner has no browser open raises no event in this process, and the owner
    check in `_run_session_loop` only runs between nodes — so a graph spending minutes
    inside a single node had nothing that would stop it. Publishing the event for running
    owners routes that case through `DefaultSessionProcessor._on_user_access_changed`,
    which cancels the item and so sets the worker's cancel event.
    """

    @pytest.mark.anyio
    async def test_a_deleted_socketless_owner_is_published(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio({})
        emitted = self._patch_services(monkeypatch, {"user-1": None}, running_owners={"user-1"})

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "user-1", "is_admin": False, "is_active": False, "token_epoch": 0}]

    @pytest.mark.anyio
    async def test_a_deactivated_socketless_owner_is_published(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio({})
        emitted = self._patch_services(
            monkeypatch,
            {"user-1": self._user("user-1", is_admin=False, is_active=False)},
            running_owners={"user-1"},
        )

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "user-1", "is_admin": False, "is_active": False, "token_epoch": 0}]

    @pytest.mark.anyio
    async def test_an_active_socketless_owner_publishes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An owner with no socket has no cached authorization to be stale, so a healthy
        record must not produce an event every interval for the length of every job."""
        socketio = self._socketio({})
        emitted = self._patch_services(
            monkeypatch, {"user-1": self._user("user-1", is_admin=True)}, running_owners={"user-1"}
        )

        await socketio._revalidate_socket_users()

        assert emitted == []

    @pytest.mark.anyio
    async def test_an_owner_who_also_holds_a_socket_is_looked_up_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        lookups: list[str] = []
        self._patch_services(
            monkeypatch,
            {"user-1": self._user("user-1", is_admin=False)},
            lookups=lookups,
            running_owners={"user-1"},
        )

        await socketio._revalidate_socket_users()

        assert lookups == ["user-1"]

    @pytest.mark.anyio
    async def test_an_owners_socket_state_is_still_what_decides_staleness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Being a running owner must not suppress the socket comparison: the empty list a
        queue owner contributes would satisfy `all(...)` vacuously if it replaced the
        socket's cached state instead of being merged with it."""
        socketio = self._socketio({"sid-1": {"user_id": "admin-1", "is_admin": True, "token_epoch": 0}})
        emitted = self._patch_services(
            monkeypatch, {"admin-1": self._user("admin-1", is_admin=False)}, running_owners={"admin-1"}
        )

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "admin-1", "is_admin": False, "is_active": True, "token_epoch": 0}]

    async def _run_one_loop_tick(self, socketio: SocketIO, monkeypatch: pytest.MonkeyPatch) -> bool:
        """Drive `_revalidation_loop` once and report whether it ran the sweep."""
        monkeypatch.setattr("invokeai.app.api.sockets.SOCKET_REVALIDATION_INTERVAL_SECONDS", 0)
        swept = asyncio.Event()

        async def fake_sweep() -> None:
            swept.set()

        monkeypatch.setattr(socketio, "_revalidate_socket_users", fake_sweep)
        task = asyncio.create_task(socketio._revalidation_loop())
        try:
            await asyncio.wait_for(swept.wait(), 0.25)
        except TimeoutError:
            return False
        finally:
            task.cancel()
        return True

    @pytest.mark.anyio
    async def test_the_loop_does_not_skip_a_sweep_when_only_a_queue_item_is_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_revalidation_loop` skips the sweep when nothing is live. A running item with no
        socket open is exactly the case this fix exists for, so it must count as live."""
        socketio = self._socketio({})
        self._patch_services(monkeypatch, {}, running_owners={"user-1"})

        assert await self._run_one_loop_tick(socketio, monkeypatch) is True

    @pytest.mark.anyio
    async def test_the_loop_still_skips_when_nothing_at_all_is_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio({})
        self._patch_services(monkeypatch, {}, running_owners=set())

        assert await self._run_one_loop_tick(socketio, monkeypatch) is False

    @pytest.mark.anyio
    async def test_the_loop_does_not_look_for_running_items_in_single_user_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep has nothing to do without records to check, so it should not be
        reaching into the session processor on every tick to find that out."""
        socketio = self._socketio({})
        self._patch_services(monkeypatch, {}, multiuser=False, running_owners={"user-1"})
        looked = []
        monkeypatch.setattr(socketio, "_running_queue_item_owners", lambda: looked.append(1) or set())

        assert await self._run_one_loop_tick(socketio, monkeypatch) is False
        assert looked == []

    def test_running_owners_are_empty_when_dependencies_are_not_initialized(self) -> None:
        """Called on a timer, so a missing `ApiDependencies.invoker` must not raise (or log
        a line every interval) — an idle server and a test that never builds dependencies
        both land here."""
        socketio = self._socketio({})

        assert socketio._running_queue_item_owners() == set()


class TestRoomChangesAndCacheStayInStep:
    """The cached `is_admin` must never claim more than the socket's actual room membership.

    The sweep decides there is nothing to do by comparing the record against that cache, so
    a cache written for a room change that then failed would report agreement and leave the
    socket in the admin room with nothing left to notice it.
    """

    def _socketio(self, socket_users: dict[str, dict]) -> SocketIO:
        socketio = SocketIO(FastAPI())
        socketio._sio.enter_room = AsyncMock()
        socketio._sio.leave_room = AsyncMock(side_effect=RuntimeError("room manager is unhappy"))
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
    async def test_a_failed_demotion_leaves_the_cache_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0}})
        self._patch_users(monkeypatch, SimpleNamespace(user_id="user-1", is_admin=False, is_active=True, token_epoch=0))
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=True)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        assert socketio._socket_users["sid-1"]["is_admin"] is True, (
            "the room change failed, so the cache must not claim it succeeded"
        )

    @pytest.mark.anyio
    async def test_the_sweep_republishes_after_a_failed_demotion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The point of leaving it stale: the next sweep still sees a difference and tries
        again, instead of finding cache and record in agreement."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0}})
        self._patch_users(monkeypatch, SimpleNamespace(user_id="user-1", is_admin=False, is_active=True, token_epoch=0))
        await socketio._handle_user_access_changed(
            ("user_access_changed", UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=True))
        )

        emitted: list[dict] = []
        invoker = SimpleNamespace(
            services=SimpleNamespace(
                configuration=SimpleNamespace(multiuser=True),
                users=SimpleNamespace(
                    get=lambda user_id: SimpleNamespace(user_id="user-1", is_admin=False, is_active=True, token_epoch=0)
                ),
                events=SimpleNamespace(emit_user_access_changed=lambda **kwargs: emitted.append(kwargs)),
                session_processor=SimpleNamespace(get_running_queue_item_owners=lambda: set()),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))

        await socketio._revalidate_socket_users()

        assert emitted == [{"user_id": "user-1", "is_admin": False, "is_active": True, "token_epoch": 0}]

    @pytest.mark.anyio
    async def test_a_failed_drop_on_the_failure_path_leaves_the_cache_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same rule where there is no record at all to fall back on — and where a stale
        cache is the only thing that gets the drop retried, since the next failed sweep
        re-runs it."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0}})
        socketio._revalidation_failures["user-1"] = SOCKET_REVALIDATION_FAILURE_LIMIT - 1

        await socketio._note_revalidation_failure("user-1")

        socketio._sio.leave_room.assert_awaited_once_with("sid-1", "admin")
        assert socketio._socket_users["sid-1"]["is_admin"] is True

    @pytest.mark.anyio
    async def test_a_failed_room_change_does_not_abandon_the_other_sockets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One socket's room manager error must not leave the user's remaining sockets
        holding the admin room untouched."""
        socketio = self._socketio(
            {
                "sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0},
                "sid-2": {"user_id": "user-1", "is_admin": True, "token_epoch": 0},
            }
        )
        socketio._sio.leave_room = AsyncMock(side_effect=[RuntimeError("room manager is unhappy"), None])
        self._patch_users(monkeypatch, SimpleNamespace(user_id="user-1", is_admin=False, is_active=True, token_epoch=0))

        await socketio._handle_user_access_changed(
            ("user_access_changed", UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=True))
        )

        assert socketio._sio.leave_room.await_count == 2
        assert socketio._socket_users["sid-1"]["is_admin"] is True
        assert socketio._socket_users["sid-2"]["is_admin"] is False

    @pytest.mark.anyio
    async def test_a_failed_promotion_leaves_the_cache_unpromoted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reachable half of this: python-socketio's `enter_room` raises for a sid that
        left between the snapshot and the call, where `leave_room` swallows the same
        condition. The cache must not record a room the socket never joined."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        socketio._sio.enter_room = AsyncMock(side_effect=ValueError("sid is not connected to requested namespace"))
        self._patch_users(monkeypatch, SimpleNamespace(user_id="user-1", is_admin=True, is_active=True, token_epoch=0))

        await socketio._handle_user_access_changed(
            ("user_access_changed", UserAccessChangedEvent.build(user_id="user-1", is_admin=True, is_active=True))
        )

        assert socketio._socket_users["sid-1"]["is_admin"] is False

    @pytest.mark.anyio
    async def test_a_failed_disconnect_does_not_abandon_the_other_sockets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`disconnect` flushes a packet and can raise on a half-closed transport. Letting
        that escape would strand the user's other sockets — and skip the session
        processor's handler for the same event, which is what cancels their queue item."""
        socketio = self._socketio(
            {
                "sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0},
                "sid-2": {"user_id": "user-1", "is_admin": False, "token_epoch": 0},
            }
        )
        socketio._sio.disconnect = AsyncMock(side_effect=[RuntimeError("transport is gone"), None])
        self._patch_users(monkeypatch, None)

        await socketio._handle_user_access_changed(
            ("user_access_changed", UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False))
        )

        assert socketio._sio.disconnect.await_count == 2


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

    def _patch_unreadable_users(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(user_id: str):
            raise RuntimeError("database is locked")

        invoker = SimpleNamespace(
            services=SimpleNamespace(
                configuration=SimpleNamespace(multiuser=True),
                users=SimpleNamespace(get=explode),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))

    @pytest.mark.anyio
    async def test_an_unreadable_record_does_not_disconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A payload that cannot be checked is not trusted to close a connection.

        The same payload that might be superseded in the promotion direction might be
        superseded here, and this direction is unrecoverable while the database is
        unreadable: `_handle_connect` fails closed, so a socket dropped on a stale
        `is_active` cannot get back in. The admin room still goes."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0}})
        self._patch_unreadable_users(monkeypatch)
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)

        await socketio._handle_user_access_changed(("user_access_changed", event))

        socketio._sio.disconnect.assert_not_awaited()
        socketio._sio.leave_room.assert_awaited_once_with("sid-1", "admin")
        assert socketio._socket_users["sid-1"]["is_admin"] is False

    @pytest.mark.anyio
    async def test_an_unreadable_record_does_not_disconnect_on_a_stale_epoch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The replacement-session case, with the re-read that normally catches it failing.
        The socket holds the epoch issued by the password change; the event still carries
        the one from before it."""
        socketio = self._socketio({"sid-new": {"user_id": "user-1", "is_admin": False, "token_epoch": 1}})
        self._patch_unreadable_users(monkeypatch)
        stale = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=True, token_epoch=0)

        await socketio._handle_user_access_changed(("user_access_changed", stale))

        socketio._sio.disconnect.assert_not_awaited()

    @pytest.mark.anyio
    async def test_a_deferred_disconnect_is_applied_once_the_record_reads_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing is lost by deferring: the sweep republishes the difference it finds
        between the record and the socket, and the handler then applies it against a record
        it could read."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        self._patch_unreadable_users(monkeypatch)
        event = UserAccessChangedEvent.build(user_id="user-1", is_admin=False, is_active=False)
        await socketio._handle_user_access_changed(("user_access_changed", event))
        socketio._sio.disconnect.assert_not_awaited()

        # The database comes back, and the account really was deactivated.
        emitted: list[dict] = []
        deactivated = SimpleNamespace(user_id="user-1", is_admin=False, is_active=False, token_epoch=0)
        invoker = SimpleNamespace(
            services=SimpleNamespace(
                configuration=SimpleNamespace(multiuser=True),
                users=SimpleNamespace(get=lambda user_id: deactivated),
                events=SimpleNamespace(emit_user_access_changed=lambda **kwargs: emitted.append(kwargs)),
                session_processor=SimpleNamespace(get_running_queue_item_owners=lambda: set()),
            )
        )
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))

        await socketio._revalidate_socket_users()
        assert emitted == [{"user_id": "user-1", "is_admin": False, "is_active": False, "token_epoch": 0}]

        await socketio._handle_user_access_changed(("user_access_changed", UserAccessChangedEvent.build(**emitted[0])))
        socketio._sio.disconnect.assert_awaited_once_with("sid-1")

    @pytest.mark.anyio
    async def test_an_unreadable_record_never_grants_the_admin_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The superseded-promotion case again, with no record to check it against. A sweep
        read `is_admin=True`, a demotion committed while it was suspended, and the re-read
        that would have caught it fails. Honoring the payload here would put a demoted user
        back in the admin room; the handler may demote on an unreadable database, never
        promote."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": False, "token_epoch": 0}})
        self._patch_unreadable_users(monkeypatch)
        stale = UserAccessChangedEvent.build(user_id="user-1", is_admin=True, is_active=True)

        await socketio._handle_user_access_changed(("user_access_changed", stale))

        socketio._sio.enter_room.assert_not_awaited()
        socketio._sio.leave_room.assert_awaited_once_with("sid-1", "admin")
        assert socketio._socket_users["sid-1"]["is_admin"] is False

    @pytest.mark.anyio
    async def test_an_unreadable_record_demotes_a_currently_admin_socket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same rule seen from the other side: a socket already holding the admin room does
        not get to keep it just because the payload said so and the read failed."""
        socketio = self._socketio({"sid-1": {"user_id": "user-1", "is_admin": True, "token_epoch": 0}})
        self._patch_unreadable_users(monkeypatch)
        stale = UserAccessChangedEvent.build(user_id="user-1", is_admin=True, is_active=True)

        await socketio._handle_user_access_changed(("user_access_changed", stale))

        socketio._sio.enter_room.assert_not_awaited()
        socketio._sio.leave_room.assert_awaited_once_with("sid-1", "admin")
        assert socketio._socket_users["sid-1"]["is_admin"] is False

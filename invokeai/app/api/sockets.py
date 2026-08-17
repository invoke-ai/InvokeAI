# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import asyncio
from collections.abc import Collection
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from socketio import ASGIApp, AsyncServer
from starlette.concurrency import run_in_threadpool

from invokeai.app.services.auth.token_service import verify_token
from invokeai.app.services.config.config_default import get_config
from invokeai.app.services.events.events_common import (
    BatchEnqueuedEvent,
    BulkDownloadCompleteEvent,
    BulkDownloadErrorEvent,
    BulkDownloadEventBase,
    BulkDownloadStartedEvent,
    DownloadCancelledEvent,
    DownloadCompleteEvent,
    DownloadErrorEvent,
    DownloadEventBase,
    DownloadProgressEvent,
    DownloadStartedEvent,
    FastAPIEvent,
    InvocationCompleteEvent,
    InvocationErrorEvent,
    InvocationProgressEvent,
    InvocationStartedEvent,
    LLMTaskCompleteEvent,
    LLMTaskErrorEvent,
    LLMTaskEventBase,
    LLMTaskProgressEvent,
    ModelEventBase,
    ModelInstallCancelledEvent,
    ModelInstallCompleteEvent,
    ModelInstallDownloadProgressEvent,
    ModelInstallDownloadsCompleteEvent,
    ModelInstallErrorEvent,
    ModelInstallStartedEvent,
    ModelLoadCompleteEvent,
    ModelLoadStartedEvent,
    QueueClearedEvent,
    QueueEventBase,
    QueueItemsCanceledEvent,
    QueueItemsRetriedEvent,
    QueueItemStatusChangedEvent,
    RecallParametersUpdatedEvent,
    UserAccessChangedEvent,
    WorkflowAccessRevokedEvent,
    WorkflowCreatedEvent,
    WorkflowDeletedEvent,
    WorkflowEventBase,
    WorkflowUpdatedEvent,
    register_events,
)
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()

# How often open sockets are re-derived from the database. Route handlers emit
# `user_access_changed` the moment they mutate a user, so in-process changes are already
# immediate; this sweep exists for mutations this process never sees — the `invoke-usermod`
# / `invoke-userdel` CLIs run against the database file from a *separate* process, so no
# in-process event can be raised for them. It bounds how long a socket can outlive the
# authorization it was granted under. See `SocketIO._revalidate_socket_users`.
SOCKET_REVALIDATION_INTERVAL_SECONDS = 30.0

# How many consecutive sweeps may fail to read a user's record before that user's sockets
# lose their administrator privileges. A single failed read is retried rather than acted on
# — a busy-timeout on the shared SQLite connection is transient, and a demotion is far more
# likely to arrive as a clean read on the next sweep than to coincide with an outage — but
# the retry cannot be unbounded, or an unreadable database would preserve a stale admin
# socket indefinitely. At the interval above this bounds stale admin access to ~90 seconds.
# See `SocketIO._note_revalidation_failure`.
SOCKET_REVALIDATION_FAILURE_LIMIT = 3


class QueueSubscriptionEvent(BaseModel):
    """Event data for subscribing to the socket.io queue room.
    This is a pydantic model to ensure the data is in the correct format."""

    queue_id: str


class BulkDownloadSubscriptionEvent(BaseModel):
    """Event data for subscribing to the socket.io bulk downloads room.
    This is a pydantic model to ensure the data is in the correct format."""

    bulk_download_id: str


QUEUE_EVENTS = {
    InvocationStartedEvent,
    InvocationProgressEvent,
    InvocationCompleteEvent,
    InvocationErrorEvent,
    QueueItemStatusChangedEvent,
    BatchEnqueuedEvent,
    QueueItemsRetriedEvent,
    QueueItemsCanceledEvent,
    QueueClearedEvent,
    RecallParametersUpdatedEvent,
}

MODEL_EVENTS = {
    DownloadCancelledEvent,
    DownloadCompleteEvent,
    DownloadErrorEvent,
    DownloadProgressEvent,
    DownloadStartedEvent,
    ModelLoadStartedEvent,
    ModelLoadCompleteEvent,
    ModelInstallDownloadProgressEvent,
    ModelInstallDownloadsCompleteEvent,
    ModelInstallStartedEvent,
    ModelInstallCompleteEvent,
    ModelInstallCancelledEvent,
    ModelInstallErrorEvent,
}

BULK_DOWNLOAD_EVENTS = {BulkDownloadStartedEvent, BulkDownloadCompleteEvent, BulkDownloadErrorEvent}
WORKFLOW_EVENTS = {WorkflowCreatedEvent, WorkflowUpdatedEvent, WorkflowDeletedEvent}
USER_EVENTS = {UserAccessChangedEvent}

LLM_TASK_EVENTS = {LLMTaskProgressEvent, LLMTaskCompleteEvent, LLMTaskErrorEvent}


class SocketIO:
    _sub_queue = "subscribe_queue"
    _unsub_queue = "unsubscribe_queue"

    _sub_bulk_download = "subscribe_bulk_download"
    _unsub_bulk_download = "unsubscribe_bulk_download"

    def __init__(self, app: FastAPI):
        self._sio = AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        # When deployed behind a reverse proxy under a sub-path, `base_url` is set and the
        # SubPathASGIMiddleware advertises it via `root_path`. Starlette then hands mounted
        # sub-apps the full public path (e.g. `/invoke/ws/socket.io`). Unlike routers and
        # StaticFiles, engine.io is not root_path-aware: it matches the raw `scope["path"]`
        # against `socketio_path`, so the path must include the sub-path prefix or every
        # handshake 404s. The frontend already targets `{basePath}/ws/socket.io`.
        base_url = get_config().base_url or ""
        self._app = ASGIApp(socketio_server=self._sio, socketio_path=f"{base_url}/ws/socket.io")
        app.mount("/ws", self._app)

        # Track user information for each socket connection
        self._socket_users: dict[str, dict[str, Any]] = {}

        # Periodic re-derivation of that cached state from the database; see `start`.
        self._revalidation_task: asyncio.Task[None] | None = None

        # Consecutive failed lookups per user id, for the sweep's bounded-retry policy.
        # Only written by `_revalidate_socket_users`, which the loop runs one call at a
        # time; entries are cleared on a successful read and dropped when the user's last
        # socket goes away, so this cannot grow with reconnect churn.
        self._revalidation_failures: dict[str, int] = {}

        # Set up authentication middleware
        self._sio.on("connect", handler=self._handle_connect)
        self._sio.on("disconnect", handler=self._handle_disconnect)

        self._sio.on(self._sub_queue, handler=self._handle_sub_queue)
        self._sio.on(self._unsub_queue, handler=self._handle_unsub_queue)
        self._sio.on(self._sub_bulk_download, handler=self._handle_sub_bulk_download)
        self._sio.on(self._unsub_bulk_download, handler=self._handle_unsub_bulk_download)

        register_events(QUEUE_EVENTS, self._handle_queue_event)
        register_events(MODEL_EVENTS, self._handle_model_event)
        register_events(BULK_DOWNLOAD_EVENTS, self._handle_bulk_image_download_event)
        register_events(LLM_TASK_EVENTS, self._handle_llm_task_event)
        register_events(WORKFLOW_EVENTS, self._handle_workflow_event)
        register_events(USER_EVENTS, self._handle_user_access_changed)

    async def _handle_connect(self, sid: str, environ: dict, auth: dict | None) -> bool:
        """Handle socket connection and authenticate the user.

        Returns True to accept the connection, False to reject it.
        Stores user_id in the internal socket users dict for later use.

        In multiuser mode, connections without a valid token are rejected outright
        so that anonymous clients cannot subscribe to queue rooms and observe
        queue activity belonging to other users. In single-user mode, unauthenticated
        connections are accepted as the system admin user.
        """
        # Extract token from auth data or headers
        token = None
        if auth and isinstance(auth, dict):
            token = auth.get("token")

        if not token and environ:
            # Try to get token from headers
            headers = environ.get("HTTP_AUTHORIZATION", "")
            if headers.startswith("Bearer "):
                token = headers[7:]

        # Verify the token
        if token:
            token_data = verify_token(token)
            if token_data:
                is_admin = token_data.is_admin
                # In multiuser mode, also verify the backing user record still
                # exists and is active — mirrors the REST auth check in
                # auth_dependencies.py.  A deleted or deactivated user whose
                # JWT has not yet expired must not be allowed to open a socket.
                token_epoch = token_data.token_epoch
                if self._is_multiuser_enabled():
                    try:
                        from invokeai.app.api.auth_dependencies import resolve_authorized_user

                        user = resolve_authorized_user(token_data)
                        if user is None:
                            logger.warning(
                                f"Rejecting socket {sid}: user {token_data.user_id} not found, inactive, or revoked"
                            )
                            return False
                        # The token proves identity only — authorization comes from the
                        # database. A demoted admin reconnecting with an old token must
                        # not rejoin the admin room; a promoted user gets admin rooms
                        # without re-login.
                        is_admin = user.is_admin
                        token_epoch = user.token_epoch
                        self._note_revalidation_success(token_data.user_id)
                    except Exception:
                        # If user service is unavailable, fail closed
                        logger.warning(f"Rejecting socket {sid}: unable to verify user record")
                        return False

                # Store user_id, is_admin and the epoch this socket authenticated under.
                # The epoch lets `_handle_user_access_changed` drop exactly the sockets
                # whose token a later revocation invalidated.
                self._socket_users[sid] = {
                    "user_id": token_data.user_id,
                    "is_admin": is_admin,
                    "token_epoch": token_epoch,
                }
                logger.info(f"Socket {sid} connected with user_id: {token_data.user_id}, is_admin: {is_admin}")
                await self._sio.enter_room(sid, f"user:{token_data.user_id}")
                await self._sio.enter_room(sid, "workflows:shared")
                if is_admin:
                    await self._sio.enter_room(sid, "admin")
                return True

        # No valid token provided. In multiuser mode this is not allowed — reject
        # the connection so anonymous clients cannot subscribe to queue rooms.
        # In single-user mode, fall through and accept the socket as system admin.
        if self._is_multiuser_enabled():
            logger.warning(
                f"Rejecting socket {sid} connection: multiuser mode is enabled and no valid auth token was provided"
            )
            return False

        self._socket_users[sid] = {
            "user_id": "system",
            "is_admin": True,
        }
        logger.debug(f"Socket {sid} connected as system admin (single-user mode)")
        await self._sio.enter_room(sid, "user:system")
        await self._sio.enter_room(sid, "workflows:shared")
        await self._sio.enter_room(sid, "admin")
        return True

    @staticmethod
    def _is_multiuser_enabled() -> bool:
        """Check whether multiuser mode is enabled. Fails closed if configuration
        is not yet initialized, which should not happen in practice but prevents
        accidentally opening the socket during startup races."""
        try:
            # Imported here to avoid a circular import at module load time.
            from invokeai.app.api.dependencies import ApiDependencies

            return bool(ApiDependencies.invoker.services.configuration.multiuser)
        except Exception:
            # If dependencies are not initialized, fail closed (treat as multiuser)
            # so we never accidentally admit an anonymous socket.
            return True

    async def _handle_disconnect(self, sid: str) -> None:
        """Handle socket disconnection and cleanup user info."""
        if sid in self._socket_users:
            user_id = self._socket_users[sid].get("user_id")
            del self._socket_users[sid]
            # Forget the user's revalidation failures once their last socket is gone. The
            # count stands in for how long *these* sockets have gone unchecked, so a user
            # who reconnects must start over. The sweep drops orphaned counters too, but it
            # cannot be relied on here: `_revalidation_loop` still skips the sweep entirely
            # when nothing is live, which a departing last socket can leave it. Without
            # this, a user whose only socket closed part-way through a database outage
            # would come back one failed read away from losing the admin room.
            if user_id is not None and not any(info.get("user_id") == user_id for info in self._socket_users.values()):
                self._revalidation_failures.pop(user_id, None)
            logger.debug(f"Socket {sid} disconnected and cleaned up")

    def start(self) -> None:
        """Start background work. Called from the app's startup hook.

        Not started in `__init__`: this object is constructed at import time, where there
        is no running event loop to attach a task to.
        """
        if self._revalidation_task is None or self._revalidation_task.done():
            self._revalidation_task = asyncio.create_task(self._revalidation_loop())

    def stop(self) -> None:
        """Cancel background work. Called from the app's shutdown hook."""
        if self._revalidation_task is not None:
            self._revalidation_task.cancel()
            self._revalidation_task = None

    @staticmethod
    def _running_queue_item_owners() -> set[str]:
        """The user ids that own queue items currently executing, or an empty set.

        Errors are swallowed rather than logged: this runs on a timer, and the one way it
        fails in practice is dependencies not being initialized (at startup, or in a test
        that never builds them), which would otherwise produce a line every interval
        forever. A miss costs nothing that the sweep does not already tolerate — the
        owner is simply not revalidated this round, exactly as if their item had started
        a moment later.
        """
        try:
            from invokeai.app.api.dependencies import ApiDependencies

            return ApiDependencies.invoker.services.session_processor.get_running_queue_item_owners()
        except Exception:
            return set()

    async def _revalidation_loop(self) -> None:
        """Re-derive the authorization of every live user from the database, periodically."""
        while True:
            await asyncio.sleep(SOCKET_REVALIDATION_INTERVAL_SECONDS)
            # Skip the work, not just its result, when there is nothing to revalidate:
            # single-user mode has no records to check, and an idle server has nobody to
            # check them for. Running queue items count as live even with no socket open —
            # that is the whole point of including them.
            if not self._is_multiuser_enabled():
                continue
            if not self._socket_users and not self._running_queue_item_owners():
                continue
            try:
                await self._revalidate_socket_users()
            except Exception:
                logger.exception("Error revalidating socket authorization")

    async def _revalidate_socket_users(self) -> None:
        """Emit `user_access_changed` for any live user the database no longer agrees with.

        The room membership and `is_admin` flag a socket carries are established at connect
        time and refreshed by `_handle_user_access_changed`, which fires on an in-process
        event. Nothing in this process emits that event for a change made by another
        process — `invoke-usermod --no-admin` and `invoke-userdel` open the database
        directly — so without this sweep a demoted administrator's socket would sit in the
        admin room until it happened to reconnect. REST is unaffected either way: every
        request re-reads the record.

        Rather than adjusting rooms here, differences are published as the same event the
        route handlers emit. That keeps one implementation of "what a change to this user
        means" — sockets re-authorize *and* the session processor cancels the user's running
        items — instead of a second copy that drifts from the first.

        A lookup that fails is retried on the next sweep, but only
        `SOCKET_REVALIDATION_FAILURE_LIMIT` times; after that the user's sockets lose the
        admin room and their cached `is_admin`. Retrying at all is the opposite of the
        queue gate's fail-closed policy, deliberately: nothing runs on the user's behalf
        because a socket stays open for another interval, whereas tearing down every live
        session on a transient database error would be an outage this sweep caused by
        itself — and `_handle_connect` fails closed, so the clients would not get back in.
        But retrying *forever* is what makes an unreadable database indistinguishable from
        a quiet one: a socket demoted by the CLI would sit in the admin room, reading every
        other user's private events, for as long as the reads kept failing. Dropping the
        privilege rather than the connection bounds that without causing the outage —
        the socket keeps working, it just stops being an administrator's.

        That revocation is self-healing, and deliberately relies on the staleness check
        below to be: once reads succeed again, a still-genuine admin's cached `is_admin` is
        `False` while the record says `True`, so the sweep sees a difference and publishes
        it, and `_handle_user_access_changed` re-reads and rejoins the admin room. Nothing
        re-grants privileges on the failure path itself.

        What the bound covers is `is_admin`, and only that. A deactivation, deletion or
        password change made out-of-process is still preserved for the whole outage, since
        acting on those means disconnecting and that is the outage this avoids. So during
        an outage a deleted user's socket stops seeing *other* users' events but keeps
        receiving its own, while every HTTP request it makes is already refused. Bounding
        that too would mean disconnecting on an unreadable database — for the epoch case
        specifically the replacement session is by definition already connected, so the
        argument is weaker there — but it is a policy change beyond this one, not an
        oversight.

        Scope: every user holding an open socket, plus the owner of every queue item that
        is currently executing. The second set is not about caches — a queue item holds no
        cached authorization — but about reach. An out-of-process deletion of a user with
        no socket open raises no event at all, and while
        `DefaultSessionRunner._run_session_loop` re-reads the owner around every node, a
        graph with a single long node spends that whole node inside one interval. Sweeping
        running owners gives that case what a connected user's gets: the published event
        reaches `DefaultSessionProcessor._on_user_access_changed`, which cancels the item,
        and the cancellation sets the worker's cancel event so a node with step callbacks
        stops part-way rather than finishing.
        """
        if not self._is_multiuser_enabled():
            return

        from invokeai.app.api.dependencies import ApiDependencies

        services = ApiDependencies.invoker.services

        # One lookup per distinct user, not per socket: a user with several tabs open is
        # the common case, and every socket of a user gets the same answer.
        cached_by_user: dict[str, list[dict[str, Any]]] = {}
        for info in list(self._socket_users.values()):
            cached_by_user.setdefault(info["user_id"], []).append(info)

        # Owners of running queue items are swept too, whether or not they hold a socket.
        # They have no cached socket state, so they contribute no staleness of their own —
        # an active owner matches the vacuous `all(...)` below and is skipped. What they
        # add is the deleted or deactivated case: publishing the event is what reaches
        # `DefaultSessionProcessor._on_user_access_changed` and cancels the item, which in
        # turn sets the worker's cancel event and stops the node at its next step callback
        # instead of at the next node boundary that a one-node graph never reaches.
        for owner_id in self._running_queue_item_owners():
            cached_by_user.setdefault(owner_id, [])

        # Forget the failure history of anyone who no longer has a socket. Without this the
        # dict would accumulate an entry per user across reconnect churn, and — worse — a
        # user who disconnected mid-outage and came back on a healthy database would be
        # judged against failures their current sockets never experienced.
        for tracked_user_id in list(self._revalidation_failures):
            if tracked_user_id not in cached_by_user:
                del self._revalidation_failures[tracked_user_id]

        for user_id, cached_infos in cached_by_user.items():
            try:
                user = await run_in_threadpool(services.users.get, user_id)
            except Exception:
                await self._note_revalidation_failure(user_id)
                continue

            self._note_revalidation_success(user_id)

            if user is None:
                # Deleted. `_handle_user_access_changed` only reads `is_active` on this
                # path, so the other fields just need to be inert.
                is_admin, is_active, token_epoch = False, False, 0
            else:
                is_admin, is_active, token_epoch = user.is_admin, user.is_active, user.token_epoch

            # Staleness is judged against *every* socket of the user, not a representative
            # one. They need not agree: a session that reconnected after a password change
            # holds the current epoch while the superseded session is still connected under
            # the old one, and sampling only the first would find nothing to do and leave
            # the revoked socket in place.
            if is_active and all(
                is_admin == cached.get("is_admin") and token_epoch == cached.get("token_epoch", 0)
                for cached in cached_infos
            ):
                continue

            logger.info(f"Revalidation found stale socket authorization for user {user_id}; re-authorizing")
            services.events.emit_user_access_changed(
                user_id=user_id,
                is_admin=is_admin,
                is_active=is_active,
                token_epoch=token_epoch,
            )

    def _note_revalidation_success(self, user_id: str) -> None:
        """Forget a user's failure history, because their record was just read successfully.

        The counter answers one question: how long have this user's sockets gone unchecked?
        So *any* successful read of the record answers it, not only the sweep's own. The
        sweep is simply not the only reader — `_handle_connect` resolves the same record
        through `resolve_authorized_user`, and `_handle_user_access_changed` re-reads it —
        and a counter that ignored those would punish a socket for an outage that had
        demonstrably ended before it connected: admitted as an administrator by a read that
        succeeded, then demoted by the single next failure rather than by three.
        """
        self._revalidation_failures.pop(user_id, None)

    async def _note_revalidation_failure(self, user_id: str) -> None:
        """Record a failed sweep lookup, and revoke admin rooms once the retries run out.

        Called only from `_revalidate_socket_users`'s exception path. The privilege is
        dropped, not the connection: see that method's docstring for why an unreadable
        database must not become a self-inflicted outage, and why this is safe to leave to
        the ordinary staleness check to undo.
        """
        # The caller's lookup awaited, so the user may have closed their last socket while
        # it was suspended — in which case `_handle_disconnect` has already dropped their
        # counter and recording this failure would resurrect it. Nothing would ever collect
        # it again: the sweep's orphan cleanup only runs from a sweep, and the loop skips
        # sweeps entirely while no sockets are open. There is also nothing left to revoke.
        if not any(info.get("user_id") == user_id for info in self._socket_users.values()):
            self._revalidation_failures.pop(user_id, None)
            return

        failures = self._revalidation_failures.get(user_id, 0) + 1
        self._revalidation_failures[user_id] = failures

        if failures < SOCKET_REVALIDATION_FAILURE_LIMIT:
            logger.warning(
                f"Could not revalidate socket user {user_id} (failure {failures} of "
                f"{SOCKET_REVALIDATION_FAILURE_LIMIT}); will retry",
                exc_info=True,
            )
            return

        # Re-read `_socket_users` rather than using the caller's snapshot: this coroutine
        # awaits, so a socket in that snapshot may already have disconnected and been
        # removed. Leaving a room for a dead sid is harmless, but mutating the dict of a
        # socket that is gone would silently resurrect nothing and hide a real one.
        stale_admin_sids = [
            sid for sid, info in self._socket_users.items() if info.get("user_id") == user_id and info.get("is_admin")
        ]
        if not stale_admin_sids:
            # Nothing privileged to drop: the user has sockets, but none of them holds the
            # admin room. The counter still stands rather than being cleared, because the
            # reads are still failing and that is what it counts. It cannot strand a later
            # promotion at one-failure-from-revocation — every route that could promote
            # this user emits an event whose successful re-read clears the counter, and a
            # new socket only becomes an admin via a successful read at connect.
            logger.warning(
                f"Could not revalidate socket user {user_id} after {failures} attempts; "
                "it holds no admin sockets, so there is nothing to revoke",
                exc_info=True,
            )
            return

        logger.warning(
            f"Could not revalidate socket user {user_id} after {failures} attempts; "
            "dropping admin privileges on its sockets until the record can be read again"
        )
        for sid in stale_admin_sids:
            # Re-look-up rather than re-index, the same way `_handle_user_access_changed`
            # does: `leave_room` yields, so a socket later in the list may have gone away
            # while this loop was suspended. The user id is re-checked too — a sid is only
            # reused after its connection is gone, but this way nothing here can demote a
            # socket that belongs to somebody else.
            info = self._socket_users.get(sid)
            if info is None or info.get("user_id") != user_id:
                continue
            await self._set_socket_admin(sid, info, False)

    async def _disconnect_socket(self, sid: str) -> None:
        """Close one socket, without letting its failure abandon the rest of the batch.

        `disconnect` flushes a packet over a transport that may already be half-closed. An
        exception here used to escape `_handle_user_access_changed`, which the dispatcher
        runs as a bare task — so nothing observed it, the user's remaining sockets kept
        their rooms, and (because every handler for an event shares one task, this one
        first) `DefaultSessionProcessor._on_user_access_changed` never ran and the user's
        queue item was never canceled. The revalidation sweep republishes, so a socket
        missed here is retried rather than lost.
        """
        try:
            await self._sio.disconnect(sid)
        except Exception:
            logger.warning(f"Could not disconnect socket {sid}; revalidation will retry", exc_info=True)

    async def _set_socket_admin(self, sid: str, info: dict[str, Any], is_admin: bool) -> bool:
        """Move a socket into or out of the admin room, keeping its cached role in step.

        The cache is written before the room call and rolled back if that call raises, so
        the cache never records a room change that did not happen. That matters beyond the
        moment itself: the revalidation sweep decides there is nothing left to do by
        comparing the record against this cache, and `_handle_sub_queue` re-derives room
        membership from it on every subscription. A cache updated for a failed demotion
        would report agreement with a record that says the user is not an administrator,
        so nothing would ever retry — the socket would hold the admin room, reading other
        users' events, with the one signal that could have caught it already overwritten.
        Rolling back leaves the socket visibly more privileged than its record, which is
        the state the sweep is built to find: it republishes, and this runs again. The
        failed-lookup path retries too, since every sweep past the limit re-attempts the
        drop.

        On the connect-time and demotion paths this is atomic in practice — python-socketio's
        in-memory manager implements both room calls without awaiting anything — but the
        ordering is written to be correct without relying on that, since an external client
        manager (Redis) does suspend and can fail.

        Returns whether the change was applied.
        """
        previous = info.get("is_admin", False)
        info["is_admin"] = is_admin
        try:
            if is_admin:
                await self._sio.enter_room(sid, "admin")
            else:
                await self._sio.leave_room(sid, "admin")
        except Exception:
            info["is_admin"] = previous
            action = "join" if is_admin else "leave"
            logger.warning(
                f"Socket {sid} could not {action} the admin room; leaving its cached role stale "
                "so revalidation retries",
                exc_info=True,
            )
            return False
        return True

    async def _handle_user_access_changed(self, event: FastAPIEvent[UserAccessChangedEvent]) -> None:
        """Re-authorize a user's open sockets when their role or active status changes.

        Socket room membership is established at connect time; without this handler a
        demoted administrator's sockets would remain in the admin room (receiving other
        users' private events) and a deactivated or deleted user's sockets would keep
        receiving their own private events indefinitely.

        - Deactivated/deleted: disconnect every socket belonging to the user. A
          reconnect attempt with the old token is rejected by ``_handle_connect``'s
          database check.
        - Sessions revoked (password change): disconnect the sockets that authenticated
          under a superseded token epoch. Without this the account stays active, so the
          branch above does not fire and revoked sessions keep streaming events from an
          already-open socket — HTTP would be locked out while the socket was not. The
          session that performed the change reconnects with its replacement token; the
          others fail ``_handle_connect`` and stay out.
        - Demoted: leave the admin room and update the cached ``is_admin`` so
          ``_handle_sub_queue`` cannot re-add it.
        - Promoted: join the admin room, matching the DB-derived REST behavior.

        The event is a trigger; the record is re-read here and *that* is what is applied,
        the same way ``DefaultSessionProcessor._on_user_access_changed`` does. Handlers are
        dispatched as independent tasks, and the revalidation sweep's payload is a snapshot
        taken before an await — so an event can arrive already superseded. Acting on it
        would then re-grant the admin room to someone just demoted, or disconnect the
        replacement session a password change had just issued, and nothing would correct
        either until the next sweep.

        When the read fails there is no record to check the payload against, so the only
        thing applied is the one revocation that cannot cost anything: ``is_admin`` is
        forced to ``False`` and the socket leaves the admin room. Nothing is disconnected.
        The payload cannot be trusted to disconnect on, for the same reason it cannot be
        trusted to promote on — it may already be superseded, and a stale ``is_active`` or
        ``token_epoch`` would then close the replacement session a password change had
        just issued, or a reactivated account's socket, with no way back in while
        ``_handle_connect`` is also failing closed. Neither decision is lost, only
        deferred: once reads succeed the sweep compares the record against every socket of
        the user and publishes any difference — a promotion, a deactivation, or an epoch
        the socket no longer matches — and this handler applies it against a record it
        could actually read.
        """
        _, event_data = event
        is_admin, is_active, token_epoch = event_data.is_admin, event_data.is_active, event_data.token_epoch
        record_unreadable = False
        try:
            from invokeai.app.api.dependencies import ApiDependencies

            user = await run_in_threadpool(ApiDependencies.invoker.services.users.get, event_data.user_id)
        except Exception:
            logger.warning(
                f"Could not re-read user {event_data.user_id} while re-authorizing its sockets; "
                "dropping administrator privileges and deferring every other decision to revalidation",
                exc_info=True,
            )
            record_unreadable = True
            is_admin = False
        else:
            self._note_revalidation_success(event_data.user_id)
            if user is None:
                is_admin, is_active, token_epoch = False, False, 0
            else:
                is_admin, is_active, token_epoch = user.is_admin, user.is_active, user.token_epoch

        affected_sids = [sid for sid, info in self._socket_users.items() if info.get("user_id") == event_data.user_id]
        for sid in affected_sids:
            # `affected_sids` is a snapshot, and `disconnect()` below yields to the event
            # loop while it flushes the packet — the socket's own disconnect handler then
            # removes it from `_socket_users`. Re-look-up rather than re-index: a socket
            # that dropped (client, ping timeout, or an interleaved event for the same
            # user) would otherwise raise KeyError here and abandon re-authorization for
            # every remaining sid in the loop, leaving them on stale privileges with no
            # retry. Nothing observes the failure either — the dispatcher runs this
            # handler as a bare task.
            info = self._socket_users.get(sid)
            if info is None:
                continue
            # Both of these close the connection, so neither is taken on the payload
            # alone — see the docstring. On an unreadable record only the admin-room
            # change below is applied.
            if not record_unreadable:
                if not is_active:
                    logger.info(f"Disconnecting socket {sid}: user {event_data.user_id} deactivated or deleted")
                    await self._disconnect_socket(sid)
                    continue
                if info.get("token_epoch", 0) != token_epoch:
                    logger.info(f"Disconnecting socket {sid}: user {event_data.user_id} revoked its earlier sessions")
                    await self._disconnect_socket(sid)
                    continue
            if not await self._set_socket_admin(sid, info, is_admin):
                continue
            if is_admin:
                logger.info(f"Socket {sid} joined admin room: user {event_data.user_id} promoted")
            else:
                logger.info(f"Socket {sid} left admin room: user {event_data.user_id} demoted")

    async def _handle_sub_queue(self, sid: str, data: Any) -> None:
        """Handle queue subscription and add socket to both queue and user-specific rooms."""
        queue_id = QueueSubscriptionEvent(**data).queue_id

        # Check if we have user info for this socket. In multiuser mode _handle_connect
        # will have already rejected any socket without a valid token, so missing user
        # info here is a bug — refuse the subscription rather than silently falling back
        # to an anonymous system user who could then receive queue item events.
        if sid not in self._socket_users:
            if self._is_multiuser_enabled():
                logger.warning(
                    f"Refusing queue subscription for socket {sid}: no user info (socket not authenticated via connect event)"
                )
                return
            # Single-user mode: safe to fall back to the system admin user.
            self._socket_users[sid] = {
                "user_id": "system",
                "is_admin": True,
            }

        user_id = self._socket_users[sid]["user_id"]
        is_admin = self._socket_users[sid]["is_admin"]

        # Add socket to the queue room
        await self._sio.enter_room(sid, queue_id)

        # Also add socket to a user-specific room for event filtering
        user_room = f"user:{user_id}"
        await self._sio.enter_room(sid, user_room)

        # If admin, also add to admin room to receive all events
        if is_admin:
            await self._sio.enter_room(sid, "admin")

        logger.debug(
            f"Socket {sid} (user_id: {user_id}, is_admin: {is_admin}) subscribed to queue {queue_id} and user room {user_room}"
        )

    async def _handle_unsub_queue(self, sid: str, data: Any) -> None:
        await self._sio.leave_room(sid, QueueSubscriptionEvent(**data).queue_id)

    async def _handle_sub_bulk_download(self, sid: str, data: Any) -> None:
        # In multiuser mode, only allow authenticated sockets to subscribe.
        # Bulk download events are routed to user-specific rooms, so the
        # bulk_download_id room subscription is only kept for single-user
        # backward compatibility.
        if self._is_multiuser_enabled() and sid not in self._socket_users:
            logger.warning(f"Refusing bulk download subscription for unknown socket {sid} in multiuser mode")
            return
        await self._sio.enter_room(sid, BulkDownloadSubscriptionEvent(**data).bulk_download_id)

    async def _handle_unsub_bulk_download(self, sid: str, data: Any) -> None:
        await self._sio.leave_room(sid, BulkDownloadSubscriptionEvent(**data).bulk_download_id)

    def _admin_sids(self) -> list[str]:
        """Sids belonging to admin sockets."""
        return [sid for sid, info in self._socket_users.items() if info.get("is_admin")]

    def _owner_and_admin_sids(self, owner_user_ids: str | Collection[str]) -> list[str]:
        """Sids belonging to the event's owner(s) or to any admin.

        Used as `skip_sid` when broadcasting a sanitized companion event to the queue room,
        so the owners and admins (who already received the full event) don't get a second
        copy that would clobber their cache with redacted values.

        Accepts a collection because a single event can have several owners — a retry batch may
        span multiple users' queue items.
        """
        owners = {owner_user_ids} if isinstance(owner_user_ids, str) else set(owner_user_ids)
        return [
            sid for sid, info in self._socket_users.items() if info.get("user_id") in owners or info.get("is_admin")
        ]

    async def _emit_bulk_queue_item_event(
        self,
        event_name: str,
        event_data: QueueItemsRetriedEvent | QueueItemsCanceledEvent,
        item_ids_field: str,
        item_ids_by_user_field: str,
    ) -> None:
        """Routes a multi-owner bulk queue item event (retried/canceled).

        These events carry queue item ids grouped by owner and are the only signal other clients
        get for a bulk operation, which changes many rows in one SQL statement and emits no
        per-item queue_item_status_changed events. Routing:

        - Each owner's room gets the event scoped to that owner's own item ids. Admin sids are
          skipped here because admins receive the full event below — without the skip, an admin
          who also owns affected items (including the "system" user in single-user mode, whose
          sockets are all in both rooms) would receive two copies and double-refetch the queue
          caches.
        - The admin room gets the full event: all item ids, all owners.
        - The rest of the queue room gets a sanitized companion carrying no item ids and no
          owners — just the queue_id its badge-count refetch needs.
        """
        item_ids_by_user: dict[str, list[int]] = getattr(event_data, item_ids_by_user_field)
        admin_sids = self._admin_sids()
        for user_id in event_data.user_ids:
            user_room = f"user:{user_id}"
            owner_item_ids = item_ids_by_user.get(user_id, [])
            owner_event_data = event_data.model_copy(
                update={
                    item_ids_field: owner_item_ids,
                    "user_ids": [user_id],
                    item_ids_by_user_field: {user_id: owner_item_ids},
                }
            )
            await self._sio.emit(
                event=event_name,
                data=owner_event_data.model_dump(mode="json"),
                room=user_room,
                skip_sid=admin_sids,
            )
        await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")

        sanitized = event_data.model_copy(update={item_ids_field: [], "user_ids": [], item_ids_by_user_field: {}})
        await self._sio.emit(
            event=event_name,
            data=sanitized.model_dump(mode="json"),
            room=event_data.queue_id,
            skip_sid=self._owner_and_admin_sids(event_data.user_ids),
        )
        logger.debug(
            f"Emitted {event_name}: full to user rooms {event_data.user_ids} and admin, "
            f"sanitized to queue {event_data.queue_id}"
        )

    async def _handle_queue_event(self, event: FastAPIEvent[QueueEventBase]):
        """Handle queue events with user isolation.

        Queue events split into two routing paths:

        1. The owner and admins receive the full unsanitized event in their `user:{id}` /
           `admin` rooms. The full payload may include batch_id, session_id, origin,
           destination, error metadata, etc.

        2. For events that other authenticated users need to know about so their queue list
           and badge counts stay in sync (QueueItemStatusChangedEvent and BatchEnqueuedEvent),
           a sanitized companion event is also emitted to the full queue room with the
           owner's and admins' sids in `skip_sid`. The companion uses `user_id="redacted"`
           as a sentinel so the frontend handler knows to do tag invalidation only and skip
           per-session side effects.

        InvocationEventBase events stay private (owner + admins only). RecallParametersUpdatedEvent
        is also private. QueueClearedEvent is broadcast to the queue room when unscoped (an admin or
        single-user clear that deleted every user's items); a user-scoped clear goes full to
        owner + admins with a sanitized companion to the rest of the queue room, so other users
        refresh their queue lists without treating the clear as their own.

        IMPORTANT: Check InvocationEventBase BEFORE QueueItemEventBase since InvocationEventBase
        inherits from QueueItemEventBase. The order of isinstance checks matters!
        """
        try:
            event_name, event_data = event

            # Import here to avoid circular dependency
            from invokeai.app.services.events.events_common import InvocationEventBase, QueueItemEventBase

            # Check InvocationEventBase FIRST (before QueueItemEventBase) since it's a subclass
            # Invocation events (progress, started, complete, error) are private to owner + admins
            if isinstance(event_data, InvocationEventBase) and hasattr(event_data, "user_id"):
                user_room = f"user:{event_data.user_id}"

                if isinstance(event_data, InvocationProgressEvent):
                    # Progress events only drive personal UI (the global progress bar and progress
                    # image previews) and are high-frequency. No admin UI consumes other users'
                    # progress, so emit to the owner only. This also keeps other users' progress
                    # from hijacking an admin's progress bar and image previews.
                    await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=user_room)
                    logger.debug(f"Emitted invocation progress event to user room {user_room}")
                else:
                    # started/complete/error also feed admins' gallery cache updates, so admins
                    # receive them for all users. Emit to the union of owner + admin rooms in a
                    # SINGLE call so an admin owner receives exactly one copy (see the
                    # RecallParametersUpdatedEvent note below on python-socketio's recipient
                    # dedup across a room list).
                    await self._sio.emit(
                        event=event_name, data=event_data.model_dump(mode="json"), room=[user_room, "admin"]
                    )
                    logger.debug(
                        f"Emitted private invocation event {event_name} to user room {user_room} and admin room"
                    )

            # QueueItemStatusChangedEvent: full to owner+admin, sanitized to everyone else in
            # the queue room so their queue list, badge, and item caches refresh.
            elif isinstance(event_data, QueueItemStatusChangedEvent):
                user_room = f"user:{event_data.user_id}"
                # Single emit to the union of rooms — python-socketio dedups recipients across a
                # room list, so an admin owner (or the single-user "system" user, which is in both
                # rooms) receives exactly one copy instead of running the frontend handler twice.
                await self._sio.emit(
                    event=event_name, data=event_data.model_dump(mode="json"), room=[user_room, "admin"]
                )

                sanitized = event_data.model_copy(
                    update={
                        "user_id": "redacted",
                        "batch_id": "redacted",
                        "session_id": "redacted",
                        "origin": None,
                        "destination": None,
                        "error_type": None,
                        "error_message": None,
                        "error_traceback": None,
                    }
                )
                # Strip identifying fields out of the embedded batch_status / queue_status too.
                sanitized.batch_status = sanitized.batch_status.model_copy(
                    update={"batch_id": "redacted", "origin": None, "destination": None}
                )
                sanitized.queue_status = sanitized.queue_status.model_copy(
                    update={
                        "item_id": None,
                        "session_id": None,
                        "batch_id": None,
                        "user_pending": None,
                        "user_in_progress": None,
                    }
                )
                await self._sio.emit(
                    event=event_name,
                    data=sanitized.model_dump(mode="json"),
                    room=event_data.queue_id,
                    skip_sid=self._owner_and_admin_sids(event_data.user_id),
                )

                logger.debug(
                    f"Emitted queue_item_status_changed: full to {user_room}+admin, sanitized to queue {event_data.queue_id}"
                )

            # Other queue item events (currently none beyond QueueItemStatusChangedEvent that
            # carry user_id) stay private to owner + admins.
            elif isinstance(event_data, QueueItemEventBase) and hasattr(event_data, "user_id"):
                user_room = f"user:{event_data.user_id}"
                await self._sio.emit(
                    event=event_name, data=event_data.model_dump(mode="json"), room=[user_room, "admin"]
                )

                logger.debug(f"Emitted private queue item event {event_name} to user room {user_room} and admin room")

            # RecallParametersUpdatedEvent is private - only emit to owner + admins.
            #
            # Emit to the union of the owner room and the admin room in a SINGLE
            # call. python-socketio deduplicates recipients across a room list,
            # so a socket that belongs to BOTH rooms — e.g. the "system" user in
            # single-user mode, which is also an admin — receives the event
            # exactly once. Two separate emits would deliver it twice: harmless
            # for the idempotent scalar recall fields (the frontend just re-sets
            # them), but the append-mode reference-image recall *pushes* rather
            # than replaces, so a double delivery adds the same reference image
            # twice.
            elif isinstance(event_data, RecallParametersUpdatedEvent):
                user_room = f"user:{event_data.user_id}"
                await self._sio.emit(
                    event=event_name, data=event_data.model_dump(mode="json"), room=[user_room, "admin"]
                )
                logger.debug(f"Emitted private recall_parameters_updated event to user room {user_room} and admin room")

            # BatchEnqueuedEvent: full to owner+admin, sanitized to everyone else in the queue
            # room so their badge total and queue list pick up the new items.
            elif isinstance(event_data, BatchEnqueuedEvent):
                user_room = f"user:{event_data.user_id}"
                await self._sio.emit(
                    event=event_name, data=event_data.model_dump(mode="json"), room=[user_room, "admin"]
                )

                sanitized = event_data.model_copy(
                    update={"user_id": "redacted", "batch_id": "redacted", "origin": None}
                )
                await self._sio.emit(
                    event=event_name,
                    data=sanitized.model_dump(mode="json"),
                    room=event_data.queue_id,
                    skip_sid=self._owner_and_admin_sids(event_data.user_id),
                )
                logger.debug(
                    f"Emitted batch_enqueued: full to {user_room}+admin, sanitized to queue {event_data.queue_id}"
                )

            # QueueItemsRetriedEvent: retried items are re-enqueued, raising the queue's global
            # total, but retrying emits no per-item queue_item_status_changed — this event is the
            # only signal. Item ids are only visible to their owners + admins; everyone else gets
            # the sanitized companion so their badge total refetches (the frontend handler skips
            # its per-item invalidation when retried_item_ids is empty).
            elif isinstance(event_data, QueueItemsRetriedEvent):
                await self._emit_bulk_queue_item_event(
                    event_name, event_data, "retried_item_ids", "retried_item_ids_by_user"
                )

            # QueueItemsCanceledEvent: a bulk cancel/delete (e.g. cancel-all-except-current) emits
            # no per-item queue_item_status_changed, so this event is the only signal that pending
            # items were removed from the queue.
            elif isinstance(event_data, QueueItemsCanceledEvent):
                await self._emit_bulk_queue_item_event(
                    event_name, event_data, "canceled_item_ids", "canceled_item_ids_by_user"
                )

            # QueueClearedEvent: an unscoped clear (user_id=None — admin or single-user mode)
            # deleted every user's items, so everyone gets the full event. A user-scoped clear
            # only deleted that user's rows: full event to owner+admin (single emit to a room
            # list so a socket in both rooms receives it once), sanitized companion to the rest
            # of the queue room so their queue lists and badge counts refetch without treating
            # the clear as their own.
            elif isinstance(event_data, QueueClearedEvent):
                if event_data.user_id is None:
                    await self._sio.emit(
                        event=event_name, data=event_data.model_dump(mode="json"), room=event_data.queue_id
                    )
                    logger.debug(f"Emitted unscoped queue_cleared to all subscribers in queue {event_data.queue_id}")
                else:
                    user_room = f"user:{event_data.user_id}"
                    await self._sio.emit(
                        event=event_name, data=event_data.model_dump(mode="json"), room=[user_room, "admin"]
                    )
                    sanitized = event_data.model_copy(update={"user_id": "redacted"})
                    await self._sio.emit(
                        event=event_name,
                        data=sanitized.model_dump(mode="json"),
                        room=event_data.queue_id,
                        skip_sid=self._owner_and_admin_sids(event_data.user_id),
                    )
                    logger.debug(
                        f"Emitted queue_cleared: full to {user_room}+admin, sanitized to queue {event_data.queue_id}"
                    )

            else:
                # Fail closed: an event type without explicit routing above must not leak user
                # identity or item ids to the whole queue room. If it carries user identity in
                # any form, deliver it to the identified owners + admins only and log loudly —
                # the event needs an explicit branch (and probably a sanitized companion) added
                # above. Only identity-free events are broadcast.
                owner_user_ids = getattr(event_data, "user_ids", None) or []
                single_user_id = getattr(event_data, "user_id", None)
                if single_user_id is not None:
                    owner_user_ids = [*owner_user_ids, single_user_id]
                if owner_user_ids:
                    rooms = [f"user:{user_id}" for user_id in owner_user_ids] + ["admin"]
                    await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=rooms)
                    logger.warning(
                        f"Queue event {event_name} carries user identity but has no explicit routing; "
                        f"emitted to owner + admin rooms only. Add a routing branch for it in _handle_queue_event."
                    )
                else:
                    await self._sio.emit(
                        event=event_name, data=event_data.model_dump(mode="json"), room=event_data.queue_id
                    )
                    logger.debug(
                        f"Emitted general queue event {event_name} to all subscribers in queue {event_data.queue_id}"
                    )
        except Exception as e:
            # Log any unhandled exceptions in event handling to prevent silent failures
            logger.error(f"Error handling queue event {event[0]}: {e}", exc_info=True)

    async def _handle_model_event(self, event: FastAPIEvent[ModelEventBase | DownloadEventBase]) -> None:
        event_name, event_data = event

        # Model load events only drive personal UI (the loading-models spinner that puts the
        # progress bar into indeterminate mode), so they are routed to the user whose action
        # triggered the load. Broadcasting them made every user's progress bar animate whenever
        # any user's generation loaded a model. In single-user mode the owner is "system" and
        # every socket is in user:system, preserving the old behavior.
        if isinstance(event_data, (ModelLoadStartedEvent, ModelLoadCompleteEvent)):
            await self._sio.emit(
                event=event_name, data=event_data.model_dump(mode="json"), room=f"user:{event_data.user_id}"
            )
            return

        # Model install and download events contain signed source URLs and server filesystem
        # paths. They feed the admin-only model manager UI and must not be broadcast to users.
        if isinstance(event_data, (DownloadEventBase, ModelEventBase)):
            await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")
            return

        await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"))

    async def _handle_llm_task_event(self, event: FastAPIEvent[LLMTaskEventBase]) -> None:
        """Route LLM utility task events privately to the originating user + admins.

        These events carry partial prompt content (via the task_id correlation) and
        must not be broadcast to other users.
        """
        event_name, event_data = event
        user_room = f"user:{event_data.user_id}"
        payload = event_data.model_dump(mode="json")
        await self._sio.emit(event=event_name, data=payload, room=user_room)
        await self._sio.emit(event=event_name, data=payload, room="admin")

    async def _handle_bulk_image_download_event(self, event: FastAPIEvent[BulkDownloadEventBase]) -> None:
        event_name, event_data = event
        # Route to user-specific + admin rooms so that other authenticated
        # users cannot learn the bulk_download_item_name (the capability token
        # needed to fetch the zip from the unauthenticated GET endpoint).
        # In single-user mode (user_id="system"), fall back to the shared
        # bulk_download_id room for backward compatibility.
        if hasattr(event_data, "user_id") and event_data.user_id != "system":
            user_room = f"user:{event_data.user_id}"
            await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=user_room)
            await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")
        else:
            await self._sio.emit(
                event=event_name, data=event_data.model_dump(mode="json"), room=event_data.bulk_download_id
            )

    async def _handle_workflow_event(self, event: FastAPIEvent[WorkflowEventBase]) -> None:
        event_name, event_data = event
        payload = event_data.model_dump(mode="json")

        if not self._is_multiuser_enabled():
            await self._sio.emit(event=event_name, data=payload, room="admin")
            return

        await self._sio.emit(event=event_name, data=payload, room=f"user:{event_data.user_id}")
        await self._sio.emit(event=event_name, data=payload, room="admin")

        if event_name == "workflow_created":
            if getattr(event_data, "is_public", False):
                await self._sio.emit(event=event_name, data=payload, room="workflows:shared")
            return

        if event_name == "workflow_deleted":
            if getattr(event_data, "is_public", False):
                await self._sio.emit(event=event_name, data=payload, room="workflows:shared")
            return

        if event_name == "workflow_updated":
            if getattr(event_data, "new_is_public", False):
                await self._sio.emit(event=event_name, data=payload, room="workflows:shared")
            elif getattr(event_data, "old_is_public", False):
                access_revoked = WorkflowAccessRevokedEvent.build(
                    workflow_id=event_data.workflow_id, user_id=event_data.user_id
                )
                await self._sio.emit(
                    event=access_revoked.__event_name__,
                    data=access_revoked.model_dump(mode="json"),
                    room="workflows:shared",
                )

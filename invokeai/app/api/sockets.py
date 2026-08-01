# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from collections.abc import Collection
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from socketio import ASGIApp, AsyncServer

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
    WorkflowAccessRevokedEvent,
    WorkflowCreatedEvent,
    WorkflowDeletedEvent,
    WorkflowEventBase,
    WorkflowUpdatedEvent,
    register_events,
)
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()


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
                # In multiuser mode, also verify the backing user record still
                # exists and is active — mirrors the REST auth check in
                # auth_dependencies.py.  A deleted or deactivated user whose
                # JWT has not yet expired must not be allowed to open a socket.
                if self._is_multiuser_enabled():
                    try:
                        from invokeai.app.api.dependencies import ApiDependencies

                        user = ApiDependencies.invoker.services.users.get(token_data.user_id)
                        if user is None or not user.is_active:
                            logger.warning(f"Rejecting socket {sid}: user {token_data.user_id} not found or inactive")
                            return False
                    except Exception:
                        # If user service is unavailable, fail closed
                        logger.warning(f"Rejecting socket {sid}: unable to verify user record")
                        return False

                # Store user_id and is_admin in socket users dict
                self._socket_users[sid] = {
                    "user_id": token_data.user_id,
                    "is_admin": token_data.is_admin,
                }
                logger.info(
                    f"Socket {sid} connected with user_id: {token_data.user_id}, is_admin: {token_data.is_admin}"
                )
                await self._sio.enter_room(sid, f"user:{token_data.user_id}")
                await self._sio.enter_room(sid, "workflows:shared")
                if token_data.is_admin:
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
            del self._socket_users[sid]
            logger.debug(f"Socket {sid} disconnected and cleaned up")

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

        # Model install / download events remain broadcast to all connected sockets - they feed
        # the model manager UI, which is not per-user.
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

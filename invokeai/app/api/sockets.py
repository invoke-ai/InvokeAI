# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from socketio import ASGIApp, AsyncServer

from invokeai.app.services.auth.token_service import verify_token
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
    QueueItemStatusChangedEvent,
    RecallParametersUpdatedEvent,
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


class SocketIO:
    _sub_queue = "subscribe_queue"
    _unsub_queue = "unsubscribe_queue"

    _sub_bulk_download = "subscribe_bulk_download"
    _unsub_bulk_download = "unsubscribe_bulk_download"

    def __init__(self, app: FastAPI):
        self._sio = AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self._app = ASGIApp(socketio_server=self._sio, socketio_path="/ws/socket.io")
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

    def _owner_and_admin_sids(self, owner_user_id: str) -> list[str]:
        """Sids belonging to the event's owner or to any admin.

        Used as `skip_sid` when broadcasting a sanitized companion event to the queue room,
        so the owner and admins (who already received the full event) don't get a second
        copy that would clobber their cache with redacted values.
        """
        return [
            sid
            for sid, info in self._socket_users.items()
            if info.get("user_id") == owner_user_id or info.get("is_admin")
        ]

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
        is also private. QueueClearedEvent has no user identity and is broadcast to the queue room.

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

                # Emit to the user's room
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=user_room)

                # Also emit to admin room so admins can see all events, but strip image preview data
                # from InvocationProgressEvent to prevent admins from seeing other users' image content
                if isinstance(event_data, InvocationProgressEvent):
                    admin_event_data = event_data.model_copy(update={"image": None})
                    await self._sio.emit(event=event_name, data=admin_event_data.model_dump(mode="json"), room="admin")
                else:
                    await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")

                logger.debug(f"Emitted private invocation event {event_name} to user room {user_room} and admin room")

            # QueueItemStatusChangedEvent: full to owner+admin, sanitized to everyone else in
            # the queue room so their queue list, badge, and item caches refresh.
            elif isinstance(event_data, QueueItemStatusChangedEvent):
                user_room = f"user:{event_data.user_id}"
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=user_room)
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")

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
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=user_room)
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")

                logger.debug(f"Emitted private queue item event {event_name} to user room {user_room} and admin room")

            # RecallParametersUpdatedEvent is private - only emit to owner + admins
            elif isinstance(event_data, RecallParametersUpdatedEvent):
                user_room = f"user:{event_data.user_id}"
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=user_room)
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")
                logger.debug(f"Emitted private recall_parameters_updated event to user room {user_room} and admin room")

            # BatchEnqueuedEvent: full to owner+admin, sanitized to everyone else in the queue
            # room so their badge total and queue list pick up the new items.
            elif isinstance(event_data, BatchEnqueuedEvent):
                user_room = f"user:{event_data.user_id}"
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room=user_room)
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")

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

            else:
                # For remaining queue events (e.g. QueueClearedEvent) that do not
                # carry user identity, emit to all subscribers in the queue room.
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
        await self._sio.emit(event=event[0], data=event[1].model_dump(mode="json"))

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

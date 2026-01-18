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
                # Store user_id and is_admin in socket users dict
                self._socket_users[sid] = {
                    "user_id": token_data.user_id,
                    "is_admin": token_data.is_admin,
                }
                logger.info(
                    f"Socket {sid} connected with user_id: {token_data.user_id}, is_admin: {token_data.is_admin}"
                )
                return True

        # If no valid token, store system user for backward compatibility
        self._socket_users[sid] = {
            "user_id": "system",
            "is_admin": False,
        }
        logger.info(f"Socket {sid} connected as system user (no valid token)")
        return True

    async def _handle_disconnect(self, sid: str) -> None:
        """Handle socket disconnection and cleanup user info."""
        if sid in self._socket_users:
            del self._socket_users[sid]
            logger.debug(f"Socket {sid} disconnected and cleaned up")

    async def _handle_sub_queue(self, sid: str, data: Any) -> None:
        """Handle queue subscription and add socket to both queue and user-specific rooms."""
        queue_id = QueueSubscriptionEvent(**data).queue_id

        # Check if we have user info for this socket
        if sid not in self._socket_users:
            logger.warning(
                f"Socket {sid} subscribing to queue {queue_id} but has no user info - need to authenticate via connect event"
            )
            # Store as system user temporarily - real auth should happen in connect
            self._socket_users[sid] = {
                "user_id": "system",
                "is_admin": False,
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

        logger.info(
            f"Socket {sid} (user_id: {user_id}, is_admin: {is_admin}) subscribed to queue {queue_id} and user room {user_room}"
        )

    async def _handle_unsub_queue(self, sid: str, data: Any) -> None:
        await self._sio.leave_room(sid, QueueSubscriptionEvent(**data).queue_id)

    async def _handle_sub_bulk_download(self, sid: str, data: Any) -> None:
        await self._sio.enter_room(sid, BulkDownloadSubscriptionEvent(**data).bulk_download_id)

    async def _handle_unsub_bulk_download(self, sid: str, data: Any) -> None:
        await self._sio.leave_room(sid, BulkDownloadSubscriptionEvent(**data).bulk_download_id)

    async def _handle_queue_event(self, event: FastAPIEvent[QueueEventBase]):
        """Handle queue events with user isolation.

        Invocation events (progress, started, complete) are private - only emit to owner and admins.
        Queue item status events are public - emit to all users (field values hidden via API).
        Other queue events emit to all subscribers.

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

                # Also emit to admin room so admins can see all events
                await self._sio.emit(event=event_name, data=event_data.model_dump(mode="json"), room="admin")

                logger.info(f"Emitted private invocation event {event_name} to user room {user_room} and admin room")

            # Queue item status events are visible to all users (field values masked via API)
            # This catches QueueItemStatusChangedEvent but NOT InvocationEvents (already handled above)
            elif isinstance(event_data, QueueItemEventBase) and hasattr(event_data, "user_id"):
                # Emit to all subscribers in the queue
                await self._sio.emit(
                    event=event_name, data=event_data.model_dump(mode="json"), room=event_data.queue_id
                )

                logger.info(
                    f"Emitted public queue item event {event_name} to all subscribers in queue {event_data.queue_id}"
                )

            else:
                # For other queue events (like QueueClearedEvent, BatchEnqueuedEvent), emit to all subscribers
                await self._sio.emit(
                    event=event_name, data=event_data.model_dump(mode="json"), room=event_data.queue_id
                )
                logger.info(
                    f"Emitted general queue event {event_name} to all subscribers in queue {event_data.queue_id}"
                )
        except Exception as e:
            # Log any unhandled exceptions in event handling to prevent silent failures
            logger.error(f"Error handling queue event {event[0]}: {e}", exc_info=True)

    async def _handle_model_event(self, event: FastAPIEvent[ModelEventBase | DownloadEventBase]) -> None:
        await self._sio.emit(event=event[0], data=event[1].model_dump(mode="json"))

    async def _handle_bulk_image_download_event(self, event: FastAPIEvent[BulkDownloadEventBase]) -> None:
        await self._sio.emit(event=event[0], data=event[1].model_dump(mode="json"), room=event[1].bulk_download_id)

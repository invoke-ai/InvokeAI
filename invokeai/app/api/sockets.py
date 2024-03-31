# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from socketio import ASGIApp, AsyncServer

from invokeai.app.services.events.events_common import (
    BatchEnqueuedEvent,
    BulkDownloadCompleteEvent,
    BulkDownloadErrorEvent,
    BulkDownloadEventBase,
    BulkDownloadStartedEvent,
    DownloadCancelledEvent,
    DownloadCompleteEvent,
    DownloadErrorEvent,
    DownloadProgressEvent,
    DownloadStartedEvent,
    FastAPIEvent,
    InvocationCompleteEvent,
    InvocationDenoiseProgressEvent,
    InvocationErrorEvent,
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
    SessionCanceledEvent,
    SessionCompleteEvent,
    SessionStartedEvent,
    register_events,
)


class QueueSubscriptionEvent(BaseModel):
    queue_id: str


class BulkDownloadSubscriptionEvent(BaseModel):
    bulk_download_id: str


class SocketIO:
    _sub_queue = "subscribe_queue"
    _unsub_queue = "unsubscribe_queue"

    _sub_bulk_download = "subscribe_bulk_download"
    _unsub_bulk_download = "unsubscribe_bulk_download"

    def __init__(self, app: FastAPI):
        self._sio = AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self._app = ASGIApp(socketio_server=self._sio, socketio_path="/ws/socket.io")
        app.mount("/ws", self._app)

        self._sio.on(self._sub_queue, handler=self._handle_sub_queue)
        self._sio.on(self._unsub_queue, handler=self._handle_unsub_queue)
        self._sio.on(self._sub_bulk_download, handler=self._handle_sub_bulk_download)
        self._sio.on(self._unsub_bulk_download, handler=self._handle_unsub_bulk_download)

        register_events(
            {
                InvocationStartedEvent,
                InvocationDenoiseProgressEvent,
                InvocationCompleteEvent,
                InvocationErrorEvent,
                SessionStartedEvent,
                SessionCompleteEvent,
                SessionCanceledEvent,
                QueueItemStatusChangedEvent,
                BatchEnqueuedEvent,
                QueueClearedEvent,
            },
            self._handle_queue_event,
        )

        register_events(
            {
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
            },
            self._handle_model_event,
        )

        register_events(
            {BulkDownloadStartedEvent, BulkDownloadCompleteEvent, BulkDownloadErrorEvent},
            self._handle_bulk_image_download_event,
        )

    async def _handle_sub_queue(self, sid: str, data: Any) -> None:
        await self._sio.enter_room(sid, QueueSubscriptionEvent(**data).queue_id)

    async def _handle_unsub_queue(self, sid: str, data: Any) -> None:
        await self._sio.leave_room(sid, QueueSubscriptionEvent(**data).queue_id)

    async def _handle_sub_bulk_download(self, sid: str, data: Any) -> None:
        await self._sio.enter_room(sid, BulkDownloadSubscriptionEvent(**data).bulk_download_id)

    async def _handle_unsub_bulk_download(self, sid: str, data: Any) -> None:
        await self._sio.leave_room(sid, BulkDownloadSubscriptionEvent(**data).bulk_download_id)

    async def _handle_queue_event(self, event: FastAPIEvent[QueueEventBase]):
        event_name, payload = event
        await self._sio.emit(event=event_name, data=payload.model_dump(mode="json"), room=payload.queue_id)

    async def _handle_model_event(self, event: FastAPIEvent[ModelEventBase]) -> None:
        event_name, payload = event
        await self._sio.emit(event=event_name, data=payload.model_dump(mode="json"))

    async def _handle_bulk_image_download_event(self, event: FastAPIEvent[BulkDownloadEventBase]) -> None:
        event_name, payload = event
        await self._sio.emit(event=event_name, data=payload.model_dump(mode="json"), room=payload.bulk_download_id)

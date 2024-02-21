# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from fastapi import FastAPI
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event
from socketio import ASGIApp, AsyncServer

from ..services.events.events_base import EventServiceBase


class SocketIO:
    __sio: AsyncServer
    __app: ASGIApp

    __sub_queue: str = "subscribe_queue"
    __unsub_queue: str = "unsubscribe_queue"

    __sub_bulk_download: str = "subscribe_bulk_download"
    __unsub_bulk_download: str = "unsubscribe_bulk_download"

    def __init__(self, app: FastAPI):
        self.__sio = AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.__app = ASGIApp(socketio_server=self.__sio, socketio_path="/ws/socket.io")
        app.mount("/ws", self.__app)

        self.__sio.on(self.__sub_queue, handler=self._handle_sub_queue)
        self.__sio.on(self.__unsub_queue, handler=self._handle_unsub_queue)
        local_handler.register(event_name=EventServiceBase.queue_event, _func=self._handle_queue_event)
        local_handler.register(event_name=EventServiceBase.model_event, _func=self._handle_model_event)

        self.__sio.on(self.__sub_bulk_download, handler=self._handle_sub_bulk_download)
        self.__sio.on(self.__unsub_bulk_download, handler=self._handle_unsub_bulk_download)
        local_handler.register(event_name=EventServiceBase.bulk_download_event, _func=self._handle_bulk_download_event)

    async def _handle_queue_event(self, event: Event):
        await self.__sio.emit(
            event=event[1]["event"],
            data=event[1]["data"],
            room=event[1]["data"]["queue_id"],
        )

    async def _handle_sub_queue(self, sid, data, *args, **kwargs) -> None:
        if "queue_id" in data:
            await self.__sio.enter_room(sid, data["queue_id"])

    async def _handle_unsub_queue(self, sid, data, *args, **kwargs) -> None:
        if "queue_id" in data:
            await self.__sio.leave_room(sid, data["queue_id"])

    async def _handle_model_event(self, event: Event) -> None:
        await self.__sio.emit(event=event[1]["event"], data=event[1]["data"])

    async def _handle_bulk_download_event(self, event: Event):
        await self.__sio.emit(
            event=event[1]["event"],
            data=event[1]["data"],
            room=event[1]["data"]["bulk_download_id"],
        )

    async def _handle_sub_bulk_download(self, sid, data, *args, **kwargs):
        if "bulk_download_id" in data:
            await self.__sio.enter_room(sid, data["bulk_download_id"])

    async def _handle_unsub_bulk_download(self, sid, data, *args, **kwargs):
        if "bulk_download_id" in data:
            await self.__sio.leave_room(sid, data["bulk_download_id"])

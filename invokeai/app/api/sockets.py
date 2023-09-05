# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from fastapi import FastAPI
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event
from fastapi_socketio import SocketManager

from ..services.events import EventServiceBase


class SocketIO:
    __sio: SocketManager

    def __init__(self, app: FastAPI):
        self.__sio = SocketManager(app=app)

        self.__sio.on("subscribe_session", handler=self._handle_sub_session)
        self.__sio.on("unsubscribe_session", handler=self._handle_unsub_session)
        local_handler.register(event_name=EventServiceBase.session_event, _func=self._handle_session_event)

        self.__sio.on("subscribe_batch", handler=self._handle_sub_batch)
        self.__sio.on("unsubscribe_batch", handler=self._handle_unsub_batch)
        local_handler.register(event_name=EventServiceBase.batch_event, _func=self._handle_batch_event)

    async def _handle_session_event(self, event: Event):
        await self.__sio.emit(
            event=event[1]["event"],
            data=event[1]["data"],
            room=event[1]["data"]["graph_execution_state_id"],
        )

    async def _handle_sub_session(self, sid, data, *args, **kwargs):
        if "session" in data:
            self.__sio.enter_room(sid, data["session"])

    async def _handle_unsub_session(self, sid, data, *args, **kwargs):
        if "session" in data:
            self.__sio.leave_room(sid, data["session"])

    async def _handle_batch_event(self, event: Event):
        await self.__sio.emit(
            event=event[1]["event"],
            data=event[1]["data"],
            room=event[1]["data"]["batch_id"],
        )

    async def _handle_sub_batch(self, sid, data, *args, **kwargs):
        if "batch_id" in data:
            self.__sio.enter_room(sid, data["batch_id"])

    async def _handle_unsub_batch(self, sid, data, *args, **kwargs):
        if "batch_id" in data:
            self.__sio.enter_room(sid, data["batch_id"])

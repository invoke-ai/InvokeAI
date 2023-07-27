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
        self.__sio.on("subscribe", handler=self._handle_sub)
        self.__sio.on("unsubscribe", handler=self._handle_unsub)

        local_handler.register(event_name=EventServiceBase.session_event, _func=self._handle_session_event)

    async def _handle_session_event(self, event: Event):
        await self.__sio.emit(
            event=event[1]["event"],
            data=event[1]["data"],
            room=event[1]["data"]["graph_execution_state_id"],
        )

    async def _handle_sub(self, sid, data, *args, **kwargs):
        if "session" in data:
            self.__sio.enter_room(sid, data["session"])

        # @app.sio.on('unsubscribe')

    async def _handle_unsub(self, sid, data, *args, **kwargs):
        if "session" in data:
            self.__sio.leave_room(sid, data["session"])

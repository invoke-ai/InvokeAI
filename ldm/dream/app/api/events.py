# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import asyncio
from queue import Empty, Queue
from typing import Any
from fastapi_events.dispatcher import dispatch
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event
from ..service_bases import EventServiceBase


class FastAPIEventService(EventServiceBase):
    event_handler_id: int
    __queue: Queue

    def __init__(self, event_handler_id: int) -> None:
        self.event_handler_id = event_handler_id
        self.__queue = Queue()
        asyncio.create_task(self.__dispatch_from_queue())
        super().__init__()



    def dispatch(self, event_name: str, payload: Any) -> None:
        self.__queue.put(dict(
            event_name = event_name,
            payload = payload
        ))


    async def __dispatch_from_queue(self):
        """Get events on from the queue and dispatch them, from the correct thread"""
        while True: # TODO: figure out how to gracefully exit
            try:
                event = self.__queue.get(block = False)
                dispatch(
                    event.get('event_name'),
                    payload       = event.get('payload'),
                    middleware_id = self.event_handler_id)

            except Empty:
                await asyncio.sleep(0.001)
                pass


@local_handler.register(event_name="progress")
def handle_progress_event(event: Event):
    print(event)

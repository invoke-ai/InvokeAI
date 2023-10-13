# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import asyncio
import threading
from queue import Empty, Queue
from typing import Any

from fastapi_events.dispatcher import dispatch

from ..services.events import EventServiceBase


class FastAPIEventService(EventServiceBase):
    event_handler_id: int
    __queue: Queue
    __stop_event: threading.Event

    def __init__(self, event_handler_id: int) -> None:
        self.event_handler_id = event_handler_id
        self.__queue = Queue()
        self.__stop_event = threading.Event()
        asyncio.create_task(self.__dispatch_from_queue(stop_event=self.__stop_event))

        super().__init__()

    def stop(self, *args, **kwargs):
        self.__stop_event.set()
        self.__queue.put(None)

    def dispatch(self, event_name: str, payload: Any) -> None:
        self.__queue.put(dict(event_name=event_name, payload=payload))

    async def __dispatch_from_queue(self, stop_event: threading.Event):
        """Get events on from the queue and dispatch them, from the correct thread"""
        while not stop_event.is_set():
            try:
                event = self.__queue.get(block=False)
                if not event:  # Probably stopping
                    continue

                dispatch(
                    event.get("event_name"),
                    payload=event.get("payload"),
                    middleware_id=self.event_handler_id,
                )

            except Empty:
                await asyncio.sleep(0.1)
                pass

            except asyncio.CancelledError as e:
                raise e  # Raise a proper error

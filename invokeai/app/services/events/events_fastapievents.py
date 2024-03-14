# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import asyncio
import threading
from queue import Empty, Queue

from fastapi_events.dispatcher import dispatch

from invokeai.app.services.events.events_common import (
    BaseEvent,
)

from .events_base import EventServiceBase


class FastAPIEventService(EventServiceBase):
    def __init__(self, event_handler_id: int) -> None:
        self.event_handler_id = event_handler_id
        self._queue = Queue[BaseEvent | None]()
        self._stop_event = threading.Event()
        asyncio.create_task(self._dispatch_from_queue(stop_event=self._stop_event))

        super().__init__()

    def stop(self, *args, **kwargs):
        self._stop_event.set()
        self._queue.put(None)

    def dispatch(self, event: BaseEvent) -> None:
        self._queue.put(event)

    async def _dispatch_from_queue(self, stop_event: threading.Event):
        """Get events on from the queue and dispatch them, from the correct thread"""
        while not stop_event.is_set():
            try:
                event = self._queue.get(block=False)
                if not event:  # Probably stopping
                    continue
                dispatch(event, middleware_id=self.event_handler_id, payload_schema_dump=False)

            except Empty:
                await asyncio.sleep(0.1)
                pass

            except asyncio.CancelledError as e:
                raise e  # Raise a proper error

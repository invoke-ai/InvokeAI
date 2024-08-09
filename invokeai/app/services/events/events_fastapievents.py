import asyncio
import threading

from fastapi_events.dispatcher import dispatch

from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.events.events_common import EventBase


class FastAPIEventService(EventServiceBase):
    def __init__(self, event_handler_id: int, loop: asyncio.AbstractEventLoop) -> None:
        self.event_handler_id = event_handler_id
        self._queue = asyncio.Queue[EventBase | None]()
        self._stop_event = threading.Event()
        self._loop = loop

        # We need to store a reference to the task so it doesn't get GC'd
        # See: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
        self._background_tasks: set[asyncio.Task[None]] = set()
        task = self._loop.create_task(self._dispatch_from_queue(stop_event=self._stop_event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.remove)

        super().__init__()

    def stop(self, *args, **kwargs):
        self._stop_event.set()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def dispatch(self, event: EventBase) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def _dispatch_from_queue(self, stop_event: threading.Event):
        """Get events on from the queue and dispatch them, from the correct thread"""
        while not stop_event.is_set():
            try:
                event = await self._queue.get()
                if not event:  # Probably stopping
                    continue
                # Leave the payloads as live pydantic models
                dispatch(event, middleware_id=self.event_handler_id, payload_schema_dump=False)

            except asyncio.CancelledError as e:
                raise e  # Raise a proper error

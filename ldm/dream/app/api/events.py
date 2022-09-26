# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Any
from fastapi_events.dispatcher import dispatch
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event

from ..service_bases import EventServiceBase


class FastAPIEventService(EventServiceBase):
    event_handler_id: int

    def __init__(self, event_handler_id: int) -> None:
        self.event_handler_id = event_handler_id
        super().__init__()


    def dispatch(self, event_name: str, payload: Any) -> None:
        dispatch(event_name, payload=payload, middleware_id = self.event_handler_id)


@local_handler.register(event_name="progress")
def handle_progress_event(event: Event):
    print(event)

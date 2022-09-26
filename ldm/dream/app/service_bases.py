# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Any


class EventServiceBase:
    """Basic event bus, to have an empty stand-in when not needed"""
    def dispatch(self, event_name: str, payload: Any) -> None:
        pass


# TODO: figure out the right architecture for context storage/management
# class ContextStorageBase:
#     from .services.invocation_context import InvocationContext
#     def get(context_id: str) -> InvocationContext:
#         ...

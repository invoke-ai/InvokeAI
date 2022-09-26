# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)
from ..service_bases import EventServiceBase
from ....generate import Generate


class InvocationServices():
    """Services that can be used by invocations"""
    generate: Generate
    events: EventServiceBase

    def __init__(self,
        generate: Generate,
        events: EventServiceBase,
    ):
        self.generate = generate
        self.events = events

# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)
from ldm.dream.app.services.image_storage import ImageStorageBase
from ..service_bases import EventServiceBase
from ....generate import Generate


class InvocationServices():
    """Services that can be used by invocations"""
    generate: Generate # TODO: wrap Generate, or split it up from model?
    events: EventServiceBase
    images: ImageStorageBase

    def __init__(self,
        generate: Generate,
        events: EventServiceBase,
        images: ImageStorageBase
    ):
        self.generate = generate
        self.events = events
        self.images = images

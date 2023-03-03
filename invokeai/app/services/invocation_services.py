# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)
from invokeai.backend import Generate

from .events import EventServiceBase
from .image_storage import ImageStorageBase
from .invocation_queue import InvocationQueueABC
from .item_storage import ItemStorageABC


class InvocationServices:
    """Services that can be used by invocations"""

    generate: Generate  # TODO: wrap Generate, or split it up from model?
    events: EventServiceBase
    images: ImageStorageBase
    queue: InvocationQueueABC

    # NOTE: we must forward-declare any types that include invocations, since invocations can use services
    graph_execution_manager: ItemStorageABC["GraphExecutionState"]
    processor: "InvocationProcessorABC"

    def __init__(
        self,
        generate: Generate,
        events: EventServiceBase,
        images: ImageStorageBase,
        queue: InvocationQueueABC,
        graph_execution_manager: ItemStorageABC["GraphExecutionState"],
        processor: "InvocationProcessorABC",
    ):
        self.generate = generate
        self.events = events
        self.images = images
        self.queue = queue
        self.graph_execution_manager = graph_execution_manager
        self.processor = processor

# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)
from invokeai.backend import ModelManager

from .events import EventServiceBase
from .image_storage import ImageStorageBase
from .restoration_services import RestorationServices
from .invocation_queue import InvocationQueueABC
from .item_storage import ItemStorageABC

class InvocationServices:
    """Services that can be used by invocations"""

    events: EventServiceBase
    images: ImageStorageBase
    queue: InvocationQueueABC
    model_manager: ModelManager
    restoration: RestorationServices

    # NOTE: we must forward-declare any types that include invocations, since invocations can use services
    graph_execution_manager: ItemStorageABC["GraphExecutionState"]
    processor: "InvocationProcessorABC"

    def __init__(
            self,
            model_manager: ModelManager,
            events: EventServiceBase,
            images: ImageStorageBase,
            queue: InvocationQueueABC,
            graph_execution_manager: ItemStorageABC["GraphExecutionState"],
            processor: "InvocationProcessorABC",
            restoration: RestorationServices,
    ):
        self.model_manager = model_manager
        self.events = events
        self.images = images
        self.queue = queue
        self.graph_execution_manager = graph_execution_manager
        self.processor = processor
        self.restoration = restoration

# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team

from typing import types
from invokeai.app.services.metadata import MetadataServiceBase
from invokeai.backend import ModelManager

from .events import EventServiceBase
from .latent_storage import LatentsStorageBase
from .image_storage import ImageStorageBase
from .restoration_services import RestorationServices
from .invocation_queue import InvocationQueueABC
from .item_storage import ItemStorageABC

class InvocationServices:
    """Services that can be used by invocations"""

    events: EventServiceBase
    latents: LatentsStorageBase
    images: ImageStorageBase
    metadata: MetadataServiceBase
    queue: InvocationQueueABC
    model_manager: ModelManager
    restoration: RestorationServices

    # NOTE: we must forward-declare any types that include invocations, since invocations can use services
    graph_library: ItemStorageABC["LibraryGraph"]
    graph_execution_manager: ItemStorageABC["GraphExecutionState"]
    processor: "InvocationProcessorABC"

    def __init__(
            self,
            model_manager: ModelManager,
            events: EventServiceBase,
            logger: types.ModuleType,
            latents: LatentsStorageBase,
            images: ImageStorageBase,
            metadata: MetadataServiceBase,
            queue: InvocationQueueABC,
            graph_library: ItemStorageABC["LibraryGraph"],
            graph_execution_manager: ItemStorageABC["GraphExecutionState"],
            processor: "InvocationProcessorABC",
            restoration: RestorationServices,
    ):
        self.model_manager = model_manager
        self.events = events
        self.logger = logger
        self.latents = latents
        self.images = images
        self.metadata = metadata
        self.queue = queue
        self.graph_library = graph_library
        self.graph_execution_manager = graph_execution_manager
        self.processor = processor
        self.restoration = restoration

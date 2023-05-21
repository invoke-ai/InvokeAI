# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team

from logging import Logger
from invokeai.app.services.images import ImageService
from invokeai.backend import ModelManager

from .events import EventServiceBase
from .latent_storage import LatentsStorageBase
from .image_file_storage import ImageFileStorageBase
from .restoration_services import RestorationServices
from .invocation_queue import InvocationQueueABC
from .item_storage import ItemStorageABC


class InvocationServices:
    """Services that can be used by invocations"""

    events: EventServiceBase
    latents: LatentsStorageBase
    images: ImageFileStorageBase
    queue: InvocationQueueABC
    model_manager: ModelManager
    restoration: RestorationServices
    images_new: ImageService

    # NOTE: we must forward-declare any types that include invocations, since invocations can use services
    graph_library: ItemStorageABC["LibraryGraph"]
    graph_execution_manager: ItemStorageABC["GraphExecutionState"]
    processor: "InvocationProcessorABC"

    def __init__(
        self,
        model_manager: ModelManager,
        events: EventServiceBase,
        logger: Logger,
        latents: LatentsStorageBase,
        images: ImageFileStorageBase,
        queue: InvocationQueueABC,
        images_new: ImageService,
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
        self.queue = queue
        self.images_new = images_new
        self.graph_library = graph_library
        self.graph_execution_manager = graph_execution_manager
        self.processor = processor
        self.restoration = restoration

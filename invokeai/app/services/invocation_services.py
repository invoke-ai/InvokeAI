# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team

from types import ModuleType
from invokeai.app.services.image_db import (
    ImageRecordServiceBase,
)
from invokeai.app.services.images import ImageService
from invokeai.app.services.metadata import MetadataServiceBase
from invokeai.app.services.urls import UrlServiceBase
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
    images_db: ImageRecordServiceBase
    urls: UrlServiceBase
    images_new: ImageService

    # NOTE: we must forward-declare any types that include invocations, since invocations can use services
    graph_library: ItemStorageABC["LibraryGraph"]
    graph_execution_manager: ItemStorageABC["GraphExecutionState"]
    processor: "InvocationProcessorABC"

    def __init__(
        self,
        model_manager: ModelManager,
        events: EventServiceBase,
        logger: ModuleType,
        latents: LatentsStorageBase,
        images: ImageStorageBase,
        metadata: MetadataServiceBase,
        queue: InvocationQueueABC,
        images_db: ImageRecordServiceBase,
        images_new: ImageService,
        urls: UrlServiceBase,
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
        self.images_db = images_db
        self.images_new = images_new
        self.urls = urls
        self.graph_library = graph_library
        self.graph_execution_manager = graph_execution_manager
        self.processor = processor
        self.restoration = restoration

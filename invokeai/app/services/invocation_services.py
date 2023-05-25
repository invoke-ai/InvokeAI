# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger
    from invokeai.app.services.images import ImageService
    from invokeai.backend import ModelManager
    from invokeai.app.services.events import EventServiceBase
    from invokeai.app.services.latent_storage import LatentsStorageBase
    from invokeai.app.services.restoration_services import RestorationServices
    from invokeai.app.services.invocation_queue import InvocationQueueABC
    from invokeai.app.services.item_storage import ItemStorageABC
    from invokeai.app.services.config import InvokeAISettings
    from invokeai.app.services.graph import GraphExecutionState, LibraryGraph
    from invokeai.app.services.invoker import InvocationProcessorABC


class InvocationServices:
    """Services that can be used by invocations"""

    # TODO: Just forward-declared everything due to circular dependencies. Fix structure.
    events: "EventServiceBase"
    latents: "LatentsStorageBase"
    queue: "InvocationQueueABC"
    model_manager: "ModelManager"
    restoration: "RestorationServices"
    configuration: "InvokeAISettings"
    images: "ImageService"

    # NOTE: we must forward-declare any types that include invocations, since invocations can use services
    graph_library: "ItemStorageABC"["LibraryGraph"]
    graph_execution_manager: "ItemStorageABC"["GraphExecutionState"]
    processor: "InvocationProcessorABC"

    def __init__(
        self,
        model_manager: "ModelManager",
        events: "EventServiceBase",
        logger: "Logger",
        latents: "LatentsStorageBase",
        images: "ImageService",
        queue: "InvocationQueueABC",
        graph_library: "ItemStorageABC"["LibraryGraph"],
        graph_execution_manager: "ItemStorageABC"["GraphExecutionState"],
        processor: "InvocationProcessorABC",
        restoration: "RestorationServices",
        configuration: "InvokeAISettings",
    ):
        self.model_manager = model_manager
        self.events = events
        self.logger = logger
        self.latents = latents
        self.images = images
        self.queue = queue
        self.graph_library = graph_library
        self.graph_execution_manager = graph_execution_manager
        self.processor = processor
        self.restoration = restoration
        self.configuration = configuration

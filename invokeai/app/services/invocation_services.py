# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger
    from invokeai.app.services.board_images import BoardImagesServiceABC
    from invokeai.app.services.boards import BoardServiceABC
    from invokeai.app.services.images import ImageServiceABC
    from invokeai.app.services.invocation_stats import InvocationStatsServiceBase
    from invokeai.app.services.model_manager_service import ModelManagerServiceBase
    from invokeai.app.services.events import EventServiceBase
    from invokeai.app.services.latent_storage import LatentsStorageBase
    from invokeai.app.services.invocation_queue import InvocationQueueABC
    from invokeai.app.services.item_storage import ItemStorageABC
    from invokeai.app.services.config import InvokeAIAppConfig
    from invokeai.app.services.graph import GraphExecutionState, LibraryGraph
    from invokeai.app.services.invoker import InvocationProcessorABC


class InvocationServices:
    """Services that can be used by invocations"""

    # TODO: Just forward-declared everything due to circular dependencies. Fix structure.
    board_images: "BoardImagesServiceABC"
    boards: "BoardServiceABC"
    configuration: "InvokeAIAppConfig"
    events: "EventServiceBase"
    graph_execution_manager: "ItemStorageABC"["GraphExecutionState"]
    graph_library: "ItemStorageABC"["LibraryGraph"]
    images: "ImageServiceABC"
    latents: "LatentsStorageBase"
    logger: "Logger"
    model_manager: "ModelManagerServiceBase"
    processor: "InvocationProcessorABC"
    performance_statistics: "InvocationStatsServiceBase"
    queue: "InvocationQueueABC"

    def __init__(
        self,
        board_images: "BoardImagesServiceABC",
        boards: "BoardServiceABC",
        configuration: "InvokeAIAppConfig",
        events: "EventServiceBase",
        graph_execution_manager: "ItemStorageABC"["GraphExecutionState"],
        graph_library: "ItemStorageABC"["LibraryGraph"],
        images: "ImageServiceABC",
        latents: "LatentsStorageBase",
        logger: "Logger",
        model_manager: "ModelManagerServiceBase",
        processor: "InvocationProcessorABC",
        performance_statistics: "InvocationStatsServiceBase",
        queue: "InvocationQueueABC",
    ):
        self.board_images = board_images
        self.boards = boards
        self.boards = boards
        self.configuration = configuration
        self.events = events
        self.graph_execution_manager = graph_execution_manager
        self.graph_library = graph_library
        self.images = images
        self.latents = latents
        self.logger = logger
        self.model_manager = model_manager
        self.processor = processor
        self.performance_statistics = performance_statistics
        self.queue = queue

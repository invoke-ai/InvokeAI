# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger

    from invokeai.app.services.board_images import BoardImagesServiceABC
    from invokeai.app.services.boards import BoardServiceABC
    from invokeai.app.services.config import InvokeAIAppConfig
    from invokeai.app.services.download_manager import DownloadQueueServiceBase
    from invokeai.app.services.events import EventServiceBase
    from invokeai.app.services.graph import GraphExecutionState, LibraryGraph
    from invokeai.app.services.images import ImageServiceABC
    from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
    from invokeai.app.services.invocation_queue import InvocationQueueABC
    from invokeai.app.services.invocation_stats import InvocationStatsServiceBase
    from invokeai.app.services.invoker import InvocationProcessorABC
    from invokeai.app.services.item_storage import ItemStorageABC
    from invokeai.app.services.latent_storage import LatentsStorageBase
    from invokeai.app.services.model_install_service import ModelInstallServiceBase
    from invokeai.app.services.model_loader_service import ModelLoadServiceBase
    from invokeai.app.services.model_record_service import ModelRecordServiceBase
    from invokeai.app.services.session_processor.session_processor_base import SessionProcessorBase
    from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase


class InvocationServices:
    """Services that can be used by invocations"""

    # TODO: Just forward-declared everything due to circular dependencies. Fix structure.
    board_images: "BoardImagesServiceABC"
    boards: "BoardServiceABC"
    configuration: "InvokeAIAppConfig"
    events: "EventServiceBase"
    graph_execution_manager: "ItemStorageABC[GraphExecutionState]"
    graph_library: "ItemStorageABC[LibraryGraph]"
    images: "ImageServiceABC"
    latents: "LatentsStorageBase"
    download_queue: "DownloadQueueServiceBase"
    model_record_store: "ModelRecordServiceBase"
    model_loader: "ModelLoadServiceBase"
    model_installer: "ModelInstallServiceBase"
    logger: "Logger"
    processor: "InvocationProcessorABC"
    performance_statistics: "InvocationStatsServiceBase"
    queue: "InvocationQueueABC"
    session_queue: "SessionQueueBase"
    session_processor: "SessionProcessorBase"
    invocation_cache: "InvocationCacheBase"

    def __init__(
        self,
        board_images: "BoardImagesServiceABC",
        boards: "BoardServiceABC",
        configuration: "InvokeAIAppConfig",
        events: "EventServiceBase",
        graph_execution_manager: "ItemStorageABC[GraphExecutionState]",
        graph_library: "ItemStorageABC[LibraryGraph]",
        images: "ImageServiceABC",
        latents: "LatentsStorageBase",
        logger: "Logger",
        download_queue: "DownloadQueueServiceBase",
        model_record_store: "ModelRecordServiceBase",
        model_loader: "ModelLoadServiceBase",
        model_installer: "ModelInstallServiceBase",
        processor: "InvocationProcessorABC",
        performance_statistics: "InvocationStatsServiceBase",
        queue: "InvocationQueueABC",
        session_queue: "SessionQueueBase",
        session_processor: "SessionProcessorBase",
        invocation_cache: "InvocationCacheBase",
    ):
        self.board_images = board_images
        self.boards = boards
        self.configuration = configuration
        self.events = events
        self.graph_execution_manager = graph_execution_manager
        self.graph_library = graph_library
        self.images = images
        self.latents = latents
        self.logger = logger
        self.download_queue = download_queue
        self.model_record_store = model_record_store
        self.model_loader = model_loader
        self.model_installer = model_installer
        self.processor = processor
        self.performance_statistics = performance_statistics
        self.queue = queue
        self.session_queue = session_queue
        self.session_processor = session_processor
        self.invocation_cache = invocation_cache

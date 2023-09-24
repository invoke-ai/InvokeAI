# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger

    from invokeai.app.services.board_image_record_storage import BoardImageRecordStorageBase
    from invokeai.app.services.board_images import BoardImagesServiceABC
    from invokeai.app.services.board_record_storage import BoardRecordStorageBase
    from invokeai.app.services.boards import BoardServiceABC
    from invokeai.app.services.config import InvokeAIAppConfig
    from invokeai.app.services.events import EventServiceBase
    from invokeai.app.services.graph import GraphExecutionState, LibraryGraph
    from invokeai.app.services.image_file_storage import ImageFileStorageBase
    from invokeai.app.services.image_record_storage import ImageRecordStorageBase
    from invokeai.app.services.images import ImageServiceABC
    from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
    from invokeai.app.services.invocation_queue import InvocationQueueABC
    from invokeai.app.services.invocation_stats import InvocationStatsServiceBase
    from invokeai.app.services.invoker import InvocationProcessorABC
    from invokeai.app.services.item_storage import ItemStorageABC
    from invokeai.app.services.latent_storage import LatentsStorageBase
    from invokeai.app.services.model_manager_service import ModelManagerServiceBase
    from invokeai.app.services.resource_name import NameServiceBase
    from invokeai.app.services.session_processor.session_processor_base import SessionProcessorBase
    from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase
    from invokeai.app.services.urls import UrlServiceBase


class InvocationServices:
    """Services that can be used by invocations"""

    # TODO: Just forward-declared everything due to circular dependencies. Fix structure.
    board_images: "BoardImagesServiceABC"
    board_image_record_storage: "BoardImageRecordStorageBase"
    boards: "BoardServiceABC"
    board_records: "BoardRecordStorageBase"
    configuration: "InvokeAIAppConfig"
    events: "EventServiceBase"
    graph_execution_manager: "ItemStorageABC[GraphExecutionState]"
    graph_library: "ItemStorageABC[LibraryGraph]"
    images: "ImageServiceABC"
    image_records: "ImageRecordStorageBase"
    image_files: "ImageFileStorageBase"
    latents: "LatentsStorageBase"
    logger: "Logger"
    model_manager: "ModelManagerServiceBase"
    processor: "InvocationProcessorABC"
    performance_statistics: "InvocationStatsServiceBase"
    queue: "InvocationQueueABC"
    session_queue: "SessionQueueBase"
    session_processor: "SessionProcessorBase"
    invocation_cache: "InvocationCacheBase"
    names: "NameServiceBase"
    urls: "UrlServiceBase"

    def __init__(
        self,
        board_images: "BoardImagesServiceABC",
        board_image_records: "BoardImageRecordStorageBase",
        boards: "BoardServiceABC",
        board_records: "BoardRecordStorageBase",
        configuration: "InvokeAIAppConfig",
        events: "EventServiceBase",
        graph_execution_manager: "ItemStorageABC[GraphExecutionState]",
        graph_library: "ItemStorageABC[LibraryGraph]",
        images: "ImageServiceABC",
        image_files: "ImageFileStorageBase",
        image_records: "ImageRecordStorageBase",
        latents: "LatentsStorageBase",
        logger: "Logger",
        model_manager: "ModelManagerServiceBase",
        processor: "InvocationProcessorABC",
        performance_statistics: "InvocationStatsServiceBase",
        queue: "InvocationQueueABC",
        session_queue: "SessionQueueBase",
        session_processor: "SessionProcessorBase",
        invocation_cache: "InvocationCacheBase",
        names: "NameServiceBase",
        urls: "UrlServiceBase",
    ):
        self.board_images = board_images
        self.board_image_records = board_image_records
        self.boards = boards
        self.board_records = board_records
        self.configuration = configuration
        self.events = events
        self.graph_execution_manager = graph_execution_manager
        self.graph_library = graph_library
        self.images = images
        self.image_files = image_files
        self.image_records = image_records
        self.latents = latents
        self.logger = logger
        self.model_manager = model_manager
        self.processor = processor
        self.performance_statistics = performance_statistics
        self.queue = queue
        self.session_queue = session_queue
        self.session_processor = session_processor
        self.invocation_cache = invocation_cache
        self.names = names
        self.urls = urls

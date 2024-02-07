# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger

    import torch

    from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData

    from .board_image_records.board_image_records_base import BoardImageRecordStorageBase
    from .board_images.board_images_base import BoardImagesServiceABC
    from .board_records.board_records_base import BoardRecordStorageBase
    from .boards.boards_base import BoardServiceABC
    from .config import InvokeAIAppConfig
    from .download import DownloadQueueServiceBase
    from .events.events_base import EventServiceBase
    from .image_files.image_files_base import ImageFileStorageBase
    from .image_records.image_records_base import ImageRecordStorageBase
    from .images.images_base import ImageServiceABC
    from .invocation_cache.invocation_cache_base import InvocationCacheBase
    from .invocation_processor.invocation_processor_base import InvocationProcessorABC
    from .invocation_queue.invocation_queue_base import InvocationQueueABC
    from .invocation_stats.invocation_stats_base import InvocationStatsServiceBase
    from .item_storage.item_storage_base import ItemStorageABC
    from .model_install import ModelInstallServiceBase
    from .model_manager.model_manager_base import ModelManagerServiceBase
    from .model_records import ModelRecordServiceBase
    from .names.names_base import NameServiceBase
    from .pickle_storage.pickle_storage_base import PickleStorageBase
    from .session_processor.session_processor_base import SessionProcessorBase
    from .session_queue.session_queue_base import SessionQueueBase
    from .shared.graph import GraphExecutionState
    from .urls.urls_base import UrlServiceBase
    from .workflow_records.workflow_records_base import WorkflowRecordsStorageBase


class InvocationServices:
    """Services that can be used by invocations"""

    def __init__(
        self,
        board_images: "BoardImagesServiceABC",
        board_image_records: "BoardImageRecordStorageBase",
        boards: "BoardServiceABC",
        board_records: "BoardRecordStorageBase",
        configuration: "InvokeAIAppConfig",
        events: "EventServiceBase",
        graph_execution_manager: "ItemStorageABC[GraphExecutionState]",
        images: "ImageServiceABC",
        image_files: "ImageFileStorageBase",
        image_records: "ImageRecordStorageBase",
        logger: "Logger",
        model_manager: "ModelManagerServiceBase",
        model_records: "ModelRecordServiceBase",
        download_queue: "DownloadQueueServiceBase",
        model_install: "ModelInstallServiceBase",
        processor: "InvocationProcessorABC",
        performance_statistics: "InvocationStatsServiceBase",
        queue: "InvocationQueueABC",
        session_queue: "SessionQueueBase",
        session_processor: "SessionProcessorBase",
        invocation_cache: "InvocationCacheBase",
        names: "NameServiceBase",
        urls: "UrlServiceBase",
        workflow_records: "WorkflowRecordsStorageBase",
        tensors: "PickleStorageBase[torch.Tensor]",
        conditioning: "PickleStorageBase[ConditioningFieldData]",
    ):
        self.board_images = board_images
        self.board_image_records = board_image_records
        self.boards = boards
        self.board_records = board_records
        self.configuration = configuration
        self.events = events
        self.graph_execution_manager = graph_execution_manager
        self.images = images
        self.image_files = image_files
        self.image_records = image_records
        self.logger = logger
        self.model_manager = model_manager
        self.model_records = model_records
        self.download_queue = download_queue
        self.model_install = model_install
        self.processor = processor
        self.performance_statistics = performance_statistics
        self.queue = queue
        self.session_queue = session_queue
        self.session_processor = session_processor
        self.invocation_cache = invocation_cache
        self.names = names
        self.urls = urls
        self.workflow_records = workflow_records
        self.tensors = tensors
        self.conditioning = conditioning

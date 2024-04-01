# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from __future__ import annotations

from typing import TYPE_CHECKING

from invokeai.app.services.object_serializer.object_serializer_base import ObjectSerializerBase

if TYPE_CHECKING:
    from logging import Logger

    import torch

    from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData

    from .board_image_records.board_image_records_base import BoardImageRecordStorageBase
    from .board_images.board_images_base import BoardImagesServiceABC
    from .board_records.board_records_base import BoardRecordStorageBase
    from .boards.boards_base import BoardServiceABC
    from .bulk_download.bulk_download_base import BulkDownloadBase
    from .config import InvokeAIAppConfig
    from .download import DownloadQueueServiceBase
    from .events.events_base import EventServiceBase
    from .image_files.image_files_base import ImageFileStorageBase
    from .image_records.image_records_base import ImageRecordStorageBase
    from .images.images_base import ImageServiceABC
    from .invocation_cache.invocation_cache_base import InvocationCacheBase
    from .model_images.model_images_base import ModelImageFileStorageBase
    from .model_manager.model_manager_base import ModelManagerServiceBase
    from .names.names_base import NameServiceBase
    from .session_processor.session_processor_base import SessionProcessorBase
    from .session_queue.session_queue_base import SessionQueueBase
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
        bulk_download: "BulkDownloadBase",
        configuration: "InvokeAIAppConfig",
        events: "EventServiceBase",
        images: "ImageServiceABC",
        image_files: "ImageFileStorageBase",
        image_records: "ImageRecordStorageBase",
        logger: "Logger",
        model_images: "ModelImageFileStorageBase",
        model_manager: "ModelManagerServiceBase",
        download_queue: "DownloadQueueServiceBase",
        session_queue: "SessionQueueBase",
        session_processor: "SessionProcessorBase",
        invocation_cache: "InvocationCacheBase",
        names: "NameServiceBase",
        urls: "UrlServiceBase",
        workflow_records: "WorkflowRecordsStorageBase",
        tensors: "ObjectSerializerBase[torch.Tensor]",
        conditioning: "ObjectSerializerBase[ConditioningFieldData]",
    ):
        self.board_images = board_images
        self.board_image_records = board_image_records
        self.boards = boards
        self.board_records = board_records
        self.bulk_download = bulk_download
        self.configuration = configuration
        self.events = events
        self.images = images
        self.image_files = image_files
        self.image_records = image_records
        self.logger = logger
        self.model_images = model_images
        self.model_manager = model_manager
        self.download_queue = download_queue
        self.session_queue = session_queue
        self.session_processor = session_processor
        self.invocation_cache = invocation_cache
        self.names = names
        self.urls = urls
        self.workflow_records = workflow_records
        self.tensors = tensors
        self.conditioning = conditioning

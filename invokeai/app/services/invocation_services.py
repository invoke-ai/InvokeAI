# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from __future__ import annotations

from typing import TYPE_CHECKING

from invokeai.app.services.object_serializer.object_serializer_base import ObjectSerializerBase
from invokeai.app.services.style_preset_images.style_preset_images_base import StylePresetImageFileStorageBase
from invokeai.app.services.style_preset_records.style_preset_records_base import StylePresetRecordsStorageBase

if TYPE_CHECKING:
    from logging import Logger

    import torch

    from invokeai.app.services.board_image_records.board_image_records_base import BoardImageRecordStorageBase
    from invokeai.app.services.board_images.board_images_base import BoardImagesServiceABC
    from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
    from invokeai.app.services.boards.boards_base import BoardServiceABC
    from invokeai.app.services.bulk_download.bulk_download_base import BulkDownloadBase
    from invokeai.app.services.config import InvokeAIAppConfig
    from invokeai.app.services.download import DownloadQueueServiceBase
    from invokeai.app.services.events.events_base import EventServiceBase
    from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
    from invokeai.app.services.image_records.image_records_base import ImageRecordStorageBase
    from invokeai.app.services.images.images_base import ImageServiceABC
    from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
    from invokeai.app.services.invocation_stats.invocation_stats_base import InvocationStatsServiceBase
    from invokeai.app.services.model_images.model_images_base import ModelImageFileStorageBase
    from invokeai.app.services.model_manager.model_manager_base import ModelManagerServiceBase
    from invokeai.app.services.names.names_base import NameServiceBase
    from invokeai.app.services.session_processor.session_processor_base import SessionProcessorBase
    from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase
    from invokeai.app.services.urls.urls_base import UrlServiceBase
    from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
    from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_base import WorkflowThumbnailServiceBase
    from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData


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
        performance_statistics: "InvocationStatsServiceBase",
        session_queue: "SessionQueueBase",
        session_processor: "SessionProcessorBase",
        invocation_cache: "InvocationCacheBase",
        names: "NameServiceBase",
        urls: "UrlServiceBase",
        workflow_records: "WorkflowRecordsStorageBase",
        tensors: "ObjectSerializerBase[torch.Tensor]",
        conditioning: "ObjectSerializerBase[ConditioningFieldData]",
        style_preset_records: "StylePresetRecordsStorageBase",
        style_preset_image_files: "StylePresetImageFileStorageBase",
        workflow_thumbnails: "WorkflowThumbnailServiceBase",
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
        self.performance_statistics = performance_statistics
        self.session_queue = session_queue
        self.session_processor = session_processor
        self.invocation_cache = invocation_cache
        self.names = names
        self.urls = urls
        self.workflow_records = workflow_records
        self.tensors = tensors
        self.conditioning = conditioning
        self.style_preset_records = style_preset_records
        self.style_preset_image_files = style_preset_image_files
        self.workflow_thumbnails = workflow_thumbnails

# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team
"""Implementation of ModelManagerServiceBase."""

from typing_extensions import Self

from invokeai.backend.model_manager.load import ModelCache, ModelConvertCache
from invokeai.backend.model_manager.metadata import ModelMetadataStore
from invokeai.backend.util.logging import InvokeAILogger

from ..config import InvokeAIAppConfig
from ..download import DownloadQueueServiceBase
from ..events.events_base import EventServiceBase
from ..model_install import ModelInstallService
from ..model_load import ModelLoadService
from ..model_records import ModelRecordServiceSQL
from ..shared.sqlite.sqlite_database import SqliteDatabase
from .model_manager_base import ModelManagerServiceBase


class ModelManagerService(ModelManagerServiceBase):
    """
    The ModelManagerService handles various aspects of model installation, maintenance and loading.

    It bundles three distinct services:
    model_manager.store   -- Routines to manage the database of model configuration records.
    model_manager.install -- Routines to install, move and delete models.
    model_manager.load    -- Routines to load models into memory.
    """

    @classmethod
    def build_model_manager(
        cls,
        app_config: InvokeAIAppConfig,
        db: SqliteDatabase,
        download_queue: DownloadQueueServiceBase,
        events: EventServiceBase,
    ) -> Self:
        """
        Construct the model manager service instance.

        For simplicity, use this class method rather than the __init__ constructor.
        """
        logger = InvokeAILogger.get_logger(cls.__name__)
        logger.setLevel(app_config.log_level.upper())

        ram_cache = ModelCache(
            max_cache_size=app_config.ram_cache_size, max_vram_cache_size=app_config.vram_cache_size, logger=logger
        )
        convert_cache = ModelConvertCache(
            cache_path=app_config.models_convert_cache_path, max_size=app_config.convert_cache_size
        )
        record_store = ModelRecordServiceSQL(db=db)
        loader = ModelLoadService(
            app_config=app_config,
            record_store=record_store,
            ram_cache=ram_cache,
            convert_cache=convert_cache,
        )
        record_store._loader = loader  # yeah, there is a circular reference here
        installer = ModelInstallService(
            app_config=app_config,
            record_store=record_store,
            download_queue=download_queue,
            metadata_store=ModelMetadataStore(db=db),
            event_bus=events,
        )
        return cls(store=record_store, install=installer, load=loader)

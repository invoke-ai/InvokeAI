# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team
"""Implementation of ModelManagerServiceBase."""

from typing_extensions import Self

from invokeai.app.services.invoker import Invoker
from invokeai.backend.model_manager.load import ModelCache, ModelConvertCache, ModelLoaderRegistry
from invokeai.backend.util.logging import InvokeAILogger

from ..config import InvokeAIAppConfig
from ..download import DownloadQueueServiceBase
from ..events.events_base import EventServiceBase
from ..model_install import ModelInstallService, ModelInstallServiceBase
from ..model_load import ModelLoadService, ModelLoadServiceBase
from ..model_records import ModelRecordServiceBase
from .model_manager_base import ModelManagerServiceBase


class ModelManagerService(ModelManagerServiceBase):
    """
    The ModelManagerService handles various aspects of model installation, maintenance and loading.

    It bundles three distinct services:
    model_manager.store   -- Routines to manage the database of model configuration records.
    model_manager.install -- Routines to install, move and delete models.
    model_manager.load    -- Routines to load models into memory.
    """

    def __init__(
        self,
        store: ModelRecordServiceBase,
        install: ModelInstallServiceBase,
        load: ModelLoadServiceBase,
    ):
        self._store = store
        self._install = install
        self._load = load

    @property
    def store(self) -> ModelRecordServiceBase:
        return self._store

    @property
    def install(self) -> ModelInstallServiceBase:
        return self._install

    @property
    def load(self) -> ModelLoadServiceBase:
        return self._load

    def start(self, invoker: Invoker) -> None:
        for service in [self._store, self._install, self._load]:
            if hasattr(service, "start"):
                service.start(invoker)

    def stop(self, invoker: Invoker) -> None:
        for service in [self._store, self._install, self._load]:
            if hasattr(service, "stop"):
                service.stop(invoker)

    @classmethod
    def build_model_manager(
        cls,
        app_config: InvokeAIAppConfig,
        model_record_service: ModelRecordServiceBase,
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
            max_cache_size=app_config.ram,
            max_vram_cache_size=app_config.vram,
            lazy_offloading=app_config.lazy_offload,
            logger=logger,
        )
        convert_cache = ModelConvertCache(cache_path=app_config.convert_cache_path, max_size=app_config.convert_cache)
        loader = ModelLoadService(
            app_config=app_config,
            ram_cache=ram_cache,
            convert_cache=convert_cache,
            registry=ModelLoaderRegistry,
        )
        installer = ModelInstallService(
            app_config=app_config,
            record_store=model_record_service,
            download_queue=download_queue,
            event_bus=events,
        )
        return cls(store=model_record_service, install=installer, load=loader)

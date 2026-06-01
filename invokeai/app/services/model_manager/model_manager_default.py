# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team
"""Implementation of ModelManagerServiceBase."""

from typing import Optional

import torch
from typing_extensions import Self

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.download.download_base import DownloadQueueServiceBase
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_install.model_install_base import ModelInstallServiceBase
from invokeai.app.services.model_install.model_install_default import ModelInstallService
from invokeai.app.services.model_load.model_load_base import ModelLoadServiceBase
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.app.services.model_manager.model_manager_base import ModelManagerServiceBase
from invokeai.app.services.model_records.model_records_base import ModelRecordServiceBase
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


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
        # Shutdown every per-device model cache to cancel any pending keep-alive timers.
        if hasattr(self._load, "ram_caches"):
            for cache in self._load.ram_caches.values():
                cache.shutdown()

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
        execution_device: Optional[torch.device] = None,
    ) -> Self:
        """
        Construct the model manager service instance.

        For simplicity, use this class method rather than the __init__ constructor.
        """
        logger = InvokeAILogger.get_logger(cls.__name__)
        logger.setLevel(app_config.log_level.upper())

        def build_cache(device: torch.device) -> ModelCache:
            return ModelCache(
                execution_device_working_mem_gb=app_config.device_working_mem_gb,
                enable_partial_loading=app_config.enable_partial_loading,
                keep_ram_copy_of_weights=app_config.keep_ram_copy_of_weights,
                max_ram_cache_size_gb=app_config.max_cache_ram_gb,
                max_vram_cache_size_gb=app_config.max_cache_vram_gb,
                execution_device=device,
                storage_device="cpu",
                log_memory_usage=app_config.log_memory_usage,
                logger=logger,
                keep_alive_minutes=app_config.model_cache_keep_alive_min,
            )

        # The default cache for callers without a pinned device (API threads, single-device installs).
        default_device = execution_device or TorchDevice.choose_torch_device()
        ram_cache = build_cache(default_device)

        # In multi-GPU mode, build one independent cache per generation device. Each session-processor
        # worker is pinned to a device (see TorchDevice.set_session_device) and resolves to its own
        # cache. The default cache is always included by ModelLoadService.
        ram_caches: dict[str, ModelCache] = {str(TorchDevice.normalize(default_device)): ram_cache}
        if app_config.generation_devices:
            for device_str in app_config.generation_devices:
                key = str(TorchDevice.normalize(device_str))
                if key not in ram_caches:
                    ram_caches[key] = build_cache(torch.device(key))

        loader = ModelLoadService(
            app_config=app_config,
            ram_cache=ram_cache,
            registry=ModelLoaderRegistry,
            ram_caches=ram_caches,
        )
        installer = ModelInstallService(
            app_config=app_config,
            record_store=model_record_service,
            download_queue=download_queue,
            event_bus=events,
        )
        return cls(store=model_record_service, install=installer, load=loader)

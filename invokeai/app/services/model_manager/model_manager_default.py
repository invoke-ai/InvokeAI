# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team
"""Implementation of ModelManagerServiceBase."""

from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from pydantic.networks import AnyHttpUrl
from typing_extensions import Self

from invokeai.app.services.invoker import Invoker
from invokeai.backend.model_manager.load import LoadedModel, ModelCache, ModelConvertCache, ModelLoaderRegistry
from invokeai.backend.util.devices import TorchDevice
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
        execution_device: Optional[torch.device] = None,
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
            execution_device=execution_device or TorchDevice.choose_torch_device(),
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

    def load_ckpt_from_url(
        self,
        source: str | AnyHttpUrl,
        access_token: Optional[str] = None,
        timeout: Optional[int] = 0,
        loader: Optional[Callable[[Path], Dict[str, torch.Tensor]]] = None,
    ) -> LoadedModel:
        """
        Download, cache, and Load the model file located at the indicated URL.

        This will check the model download cache for the model designated
        by the provided URL and download it if needed using download_and_cache_ckpt().
        It will then load the model into the RAM cache. If the optional loader
        argument is provided, the loader will be invoked to load the model into
        memory. Otherwise the method will call safetensors.torch.load_file() or
        torch.load() as appropriate to the file suffix.

        Be aware that the LoadedModel object will have a `config` attribute of None.

        Args:
          source: A URL or a string that can be converted in one. Repo_ids
                  do not work here.
          access_token: Optional access token for restricted resources.
          timeout: Wait up to the indicated number of seconds before timing
                   out long downloads.
          loader: A Callable that expects a Path and returns a Dict[str|int, Any]

        Returns:
          A LoadedModel object.
        """
        model_path = self.install.download_and_cache_ckpt(source=source, access_token=access_token, timeout=timeout)
        return self.load.load_ckpt_from_path(model_path=model_path, loader=loader)

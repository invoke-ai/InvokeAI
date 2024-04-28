# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from pydantic.networks import AnyHttpUrl
from typing_extensions import Self

from invokeai.app.services.invoker import Invoker
from invokeai.backend.model_manager.load import LoadedModel

from ..config import InvokeAIAppConfig
from ..download import DownloadQueueServiceBase
from ..events.events_base import EventServiceBase
from ..model_install import ModelInstallServiceBase
from ..model_load import ModelLoadServiceBase
from ..model_records import ModelRecordServiceBase


class ModelManagerServiceBase(ABC):
    """Abstract base class for the model manager service."""

    # attributes:
    # store: ModelRecordServiceBase = Field(description="An instance of the model record configuration service.")
    # install: ModelInstallServiceBase = Field(description="An instance of the model install service.")
    # load: ModelLoadServiceBase = Field(description="An instance of the model load service.")

    @classmethod
    @abstractmethod
    def build_model_manager(
        cls,
        app_config: InvokeAIAppConfig,
        model_record_service: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
        events: EventServiceBase,
        execution_device: torch.device,
    ) -> Self:
        """
        Construct the model manager service instance.

        Use it rather than the __init__ constructor. This class
        method simplifies the construction considerably.
        """
        pass

    @property
    @abstractmethod
    def store(self) -> ModelRecordServiceBase:
        """Return the ModelRecordServiceBase used to store and retrieve configuration records."""
        pass

    @property
    @abstractmethod
    def load(self) -> ModelLoadServiceBase:
        """Return the ModelLoadServiceBase used to load models from their configuration records."""
        pass

    @property
    @abstractmethod
    def install(self) -> ModelInstallServiceBase:
        """Return the ModelInstallServiceBase used to download and manipulate model files."""
        pass

    @abstractmethod
    def start(self, invoker: Invoker) -> None:
        pass

    @abstractmethod
    def stop(self, invoker: Invoker) -> None:
        pass

    @abstractmethod
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

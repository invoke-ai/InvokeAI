# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from abc import ABC, abstractmethod

from typing_extensions import Self

from invokeai.app.services.invoker import Invoker

from ..config import InvokeAIAppConfig
from ..download import DownloadQueueServiceBase
from ..events.events_base import EventServiceBase
from ..model_install import ModelInstallServiceBase
from ..model_load import ModelLoadServiceBase
from ..model_records import ModelRecordServiceBase
from ..shared.sqlite.sqlite_database import SqliteDatabase


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
        db: SqliteDatabase,
        download_queue: DownloadQueueServiceBase,
        events: EventServiceBase,
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

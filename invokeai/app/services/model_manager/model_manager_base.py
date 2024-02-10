# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from typing_extensions import Self

from ..config import InvokeAIAppConfig
from ..events.events_base import EventServiceBase
from ..download import DownloadQueueServiceBase
from ..model_install import ModelInstallServiceBase
from ..model_load import ModelLoadServiceBase
from ..model_records import ModelRecordServiceBase
from ..shared.sqlite.sqlite_database import SqliteDatabase


class ModelManagerServiceBase(BaseModel, ABC):
    """Abstract base class for the model manager service."""

    store: ModelRecordServiceBase = Field(description="An instance of the model record configuration service.")
    install: ModelInstallServiceBase = Field(description="An instance of the model install service.")
    load: ModelLoadServiceBase = Field(description="An instance of the model load service.")

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

from abc import ABC, abstractmethod

import torch
from typing_extensions import Self

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.download.download_base import DownloadQueueServiceBase
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_install.model_install_base import ModelInstallServiceBase
from invokeai.app.services.model_load.model_load_base import ModelLoadServiceBase
from invokeai.app.services.model_manager.model_manager_common import ModelIdentificationPreview
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges, ModelRecordServiceBase
from invokeai.backend.model_manager.configs.factory import AnyModelConfig


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
    def update_model_record(self, key: str, changes: ModelRecordChanges) -> AnyModelConfig:
        """Apply `changes` to `key`'s record, which may change its config class, and return the stored result.
        Cached instances the change made stale are dropped (see `ModelLoadServiceBase.apply_record_change`).

        Raises `UnknownModelException` for an unknown key, and `ValueError` when the changed record is invalid or
        conflicts with another record."""

    @abstractmethod
    def identify_model(self, key: str) -> ModelIdentificationPreview:
        """Probe `key`'s files as `reidentify_model` would and report the record it would write, without writing
        anything or touching caches. The files are not hashed again, so the proposal carries the stored hash; the
        probe reads a safetensors or GGUF file's header and a pickled checkpoint (.ckpt, .pt, .bin) by a pickle
        scan and a meta-device load that skips the tensor data.

        Raises `UnknownModelException` for an unknown key, and `InvalidModelException` when the files are missing
        or unreadable. Files no model class accepts are not an error: `proposed` is None and `refusal` says why a
        class rejected them, if one did. A record that is not Unknown never gets an Unknown proposal: `proposed` is
        None then, whereas `reidentify_model` still files such a record as Unknown."""

    @abstractmethod
    def reidentify_model(self, key: str) -> AnyModelConfig:
        """Identify `key`'s files again and replace its record with the result, keeping what the user set: the
        name, description, cover image, trigger phrases, install source and CPU/FP8 preferences. Cached instances
        the new record makes stale are dropped (see `ModelLoadServiceBase.apply_record_change`).

        Raises `UnknownModelException` for an unknown key, and `InvalidModelException` when the files are not
        identified as any model."""

    @abstractmethod
    def start(self, invoker: Invoker) -> None:
        pass

    @abstractmethod
    def stop(self, invoker: Invoker) -> None:
        pass

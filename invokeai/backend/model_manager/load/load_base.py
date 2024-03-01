# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""
Base class for model loading in InvokeAI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Optional

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager.config import (
    AnyModel,
    AnyModelConfig,
    SubModelType,
)
from invokeai.backend.model_manager.load.convert_cache.convert_cache_base import ModelConvertCacheBase
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase, ModelLockerBase


@dataclass
class LoadedModel:
    """Context manager object that mediates transfer from RAM<->VRAM."""

    config: AnyModelConfig
    _locker: ModelLockerBase

    def __enter__(self) -> AnyModel:
        """Context entry."""
        self._locker.lock()
        return self.model

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        """Context exit."""
        self._locker.unlock()

    @property
    def model(self) -> AnyModel:
        """Return the model without locking it."""
        return self._locker.model


# TODO(MM2):
# Some "intermediary" subclasses in the ModelLoaderBase class hierarchy define methods that their subclasses don't
# know about. I think the problem may be related to this class being an ABC.
#
# For example, GenericDiffusersLoader defines `get_hf_load_class()`, and StableDiffusionDiffusersModel attempts to
# call it. However, the method is not defined in the ABC, so it is not guaranteed to be implemented.


class ModelLoaderBase(ABC):
    """Abstract base class for loading models into RAM/VRAM."""

    @abstractmethod
    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCacheBase[AnyModel],
        convert_cache: ModelConvertCacheBase,
    ):
        """Initialize the loader."""
        pass

    @abstractmethod
    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its confguration.

        Given a model identified in the model configuration backend,
        return a ModelInfo object that can be used to retrieve the model.

        :param model_config: Model configuration, as returned by ModelConfigRecordStore
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        pass

    @abstractmethod
    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Return size in bytes of the model, calculated before loading."""
        pass

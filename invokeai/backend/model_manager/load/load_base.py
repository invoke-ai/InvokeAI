# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""
Base class for model loading in InvokeAI.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import torch

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
)
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.taxonomy import AnyModel, SubModelType


class LoadedModelWithoutConfig:
    """Context manager object that mediates transfer from RAM<->VRAM.

    This is a context manager object that has two distinct APIs:

    1. Older API (deprecated):
    Use the LoadedModel object directly as a context manager.  It will move the model into VRAM (on CUDA devices), and
    return the model in a form suitable for passing to torch.
    Example:
    ```
    loaded_model_= loader.get_model_by_key('f13dd932', SubModelType('vae'))
    with loaded_model as vae:
      image = vae.decode(latents)[0]
    ```

    2. Newer API (recommended):
    Call the LoadedModel's `model_on_device()` method in a context. It returns a tuple consisting of a copy of the
    model's state dict in CPU RAM followed by a copy of the model in VRAM. The state dict is provided to allow LoRAs and
    other model patchers to return the model to its unpatched state without expensive copy and restore operations.

    Example:
    ```
    loaded_model_= loader.get_model_by_key('f13dd932', SubModelType('vae'))
    with loaded_model.model_on_device() as (state_dict, vae):
        image = vae.decode(latents)[0]
    ```

    The state_dict should be treated as a read-only object and never modified. Also be aware that some loadable models
    do not have a state_dict, in which case this value will be None.
    """

    def __init__(self, cache_record: CacheRecord, cache: ModelCache):
        self._cache_record = cache_record
        self._cache = cache

    def __enter__(self) -> AnyModel:
        self._cache.lock(self._cache_record, None)
        return self.model

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._cache.unlock(self._cache_record)

    @contextmanager
    def model_on_device(
        self, working_mem_bytes: Optional[int] = None
    ) -> Generator[Tuple[Optional[Dict[str, torch.Tensor]], AnyModel], None, None]:
        """Return a tuple consisting of the model's state dict (if it exists) and the locked model on execution device.

        :param working_mem_bytes: The amount of working memory to keep available on the compute device when loading the
            model.
        """
        self._cache.lock(self._cache_record, working_mem_bytes)
        try:
            yield (self._cache_record.cached_model.get_cpu_state_dict(), self._cache_record.cached_model.model)
        finally:
            self._cache.unlock(self._cache_record)

    @property
    def model(self) -> AnyModel:
        """Return the model without locking it."""
        return self._cache_record.cached_model.model


class LoadedModel(LoadedModelWithoutConfig):
    """Context manager object that mediates transfer from RAM<->VRAM."""

    def __init__(self, config: Optional[AnyModelConfig], cache_record: CacheRecord, cache: ModelCache):
        super().__init__(cache_record=cache_record, cache=cache)
        self.config = config


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
        ram_cache: ModelCache,
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

    @property
    @abstractmethod
    def ram_cache(self) -> ModelCache:
        """Return the ram cache associated with this loader."""
        pass

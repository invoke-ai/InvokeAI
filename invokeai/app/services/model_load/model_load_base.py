# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Base class for model loader."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load import LoadedModel, LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.taxonomy import AnyModel, SubModelType


class ModelLoadServiceBase(ABC):
    """Wrapper around AnyModelLoader."""

    @abstractmethod
    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Given a model's configuration, load it and return the LoadedModel object.

        :param model_config: Model configuration record (as returned by ModelRecordBase.get_model())
        :param submodel: For main (pipeline models), the submodel to fetch.
        """

    @property
    @abstractmethod
    def ram_cache(self) -> ModelCache:
        """Return the RAM cache for the current thread's execution device.

        In multi-GPU mode, each session-processor worker is pinned to a device and gets its own
        cache; this resolves to the calling thread's cache. Outside a worker (e.g. API threads),
        it resolves to the default device's cache.
        """

    @property
    @abstractmethod
    def ram_caches(self) -> dict[str, ModelCache]:
        """Return all per-device RAM caches, keyed by normalized device string.

        Use this for maintenance operations that must apply to every device (clear cache, drop a
        model from all devices, shutdown).
        """

    @abstractmethod
    def load_model_from_path(
        self, model_path: Path, loader: Optional[Callable[[Path], AnyModel]] = None
    ) -> LoadedModelWithoutConfig:
        """
        Load the model file or directory located at the indicated Path.

        This will load an arbitrary model file into the RAM cache. If the optional loader
        argument is provided, the loader will be invoked to load the model into
        memory. Otherwise the method will call safetensors.torch.load_file() or
        torch.load() as appropriate to the file suffix.

        Be aware that this returns a LoadedModelWithoutConfig object, which is the same as
        LoadedModel, but without the config attribute.

        Args:
          model_path: A pathlib.Path to a checkpoint-style models file
          loader: A Callable that expects a Path and returns a Dict[str, Tensor]

        Returns:
          A LoadedModel object.
        """

# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Base class for model loader."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional

from torch import Tensor

from invokeai.app.services.shared.invocation_context import InvocationContextData
from invokeai.backend.model_manager import AnyModel, AnyModelConfig, SubModelType
from invokeai.backend.model_manager.load import LoadedModel
from invokeai.backend.model_manager.load.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase


class ModelLoadServiceBase(ABC):
    """Wrapper around AnyModelLoader."""

    @abstractmethod
    def load_model(
        self,
        model_config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
        context_data: Optional[InvocationContextData] = None,
    ) -> LoadedModel:
        """
        Given a model's configuration, load it and return the LoadedModel object.

        :param model_config: Model configuration record (as returned by ModelRecordBase.get_model())
        :param submodel: For main (pipeline models), the submodel to fetch.
        :param context_data: Invocation context data used for event reporting
        """

    @property
    @abstractmethod
    def ram_cache(self) -> ModelCacheBase[AnyModel]:
        """Return the RAM cache used by this loader."""

    @property
    @abstractmethod
    def convert_cache(self) -> ModelConvertCacheBase:
        """Return the checkpoint convert cache used by this loader."""

    @abstractmethod
    def load_ckpt_from_path(
        self, model_path: Path, loader: Optional[Callable[[Path], Dict[str, Tensor]]] = None
    ) -> LoadedModel:
        """
        Load the checkpoint-format model file located at the indicated Path.

        This will load an arbitrary model file into the RAM cache. If the optional loader
        argument is provided, the loader will be invoked to load the model into
        memory. Otherwise the method will call safetensors.torch.load_file() or
        torch.load() as appropriate to the file suffix.

        Be aware that the LoadedModel object will have a `config` attribute of None.

        Args:
          model_path: A pathlib.Path to a checkpoint-style models file
          loader: A Callable that expects a Path and returns a Dict[str, Tensor]

        Returns:
          A LoadedModel object.
        """

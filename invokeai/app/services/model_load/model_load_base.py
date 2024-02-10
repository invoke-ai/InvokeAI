# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Base class for model loader."""

from abc import ABC, abstractmethod
from typing import Optional

from invokeai.backend.model_manager import AnyModelConfig, SubModelType
from invokeai.backend.model_manager.load import LoadedModel


class ModelLoadServiceBase(ABC):
    """Wrapper around AnyModelLoader."""

    @abstractmethod
    def load_model_by_key(self, key: str, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """Given a model's key, load it and return the LoadedModel object."""
        pass

    @abstractmethod
    def load_model_by_config(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """Given a model's configuration, load it and return the LoadedModel object."""
        pass

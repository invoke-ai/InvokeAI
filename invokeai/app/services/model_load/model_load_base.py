# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Base class for model loader."""

from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.shared.invocation_context import InvocationContextData
from invokeai.backend.model_manager import AnyModelConfig, SubModelType
from invokeai.backend.model_manager.any_model_type import AnyModel
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

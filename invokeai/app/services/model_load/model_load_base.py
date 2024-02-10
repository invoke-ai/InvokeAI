# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Base class for model loader."""

from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.backend.model_manager import AnyModelConfig, BaseModelType, ModelType, SubModelType
from invokeai.backend.model_manager.load import LoadedModel


class ModelLoadServiceBase(ABC):
    """Wrapper around AnyModelLoader."""

    @abstractmethod
    def load_model_by_key(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> LoadedModel:
        """
        Given a model's key, load it and return the LoadedModel object.

        :param key: Key of model config to be fetched.
        :param submodel: For main (pipeline models), the submodel to fetch.
        :param context: Invocation context used for event reporting
        """
        pass

    @abstractmethod
    def load_model_by_config(
        self,
        model_config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> LoadedModel:
        """
        Given a model's configuration, load it and return the LoadedModel object.

        :param model_config: Model configuration record (as returned by ModelRecordBase.get_model())
        :param submodel: For main (pipeline models), the submodel to fetch.
        :param context: Invocation context used for event reporting
        """
        pass

    @abstractmethod
    def load_model_by_attr(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> LoadedModel:
        """
        Given a model's attributes, search the database for it, and if found, load and return the LoadedModel object.

        This is provided for API compatability with the get_model() method
        in the original model manager. However, note that LoadedModel is
        not the same as the original ModelInfo that ws returned.

        :param model_name: Name of to be fetched.
        :param base_model: Base model
        :param model_type: Type of the model
        :param submodel: For main (pipeline models), the submodel to fetch
        :param context: The invocation context.

        Exceptions: UnknownModelException -- model with these attributes not known
                    NotImplementedException -- a model loader was not provided at initialization time
                    ValueError -- more than one model matches this combination
        """

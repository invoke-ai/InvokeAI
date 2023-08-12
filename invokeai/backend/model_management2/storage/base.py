# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Abstract base class for storing and retrieving model configuration records.
"""


from abc import ABC, abstractmethod
from typing import Union

from ..model_config import ModelConfigBase


class DuplicateModelException(Exception):
    """Raised on an attempt to add a model with the same key twice."""

    pass


class UnknownModelException(Exception):
    """Raised on an attempt to delete a model with a nonexistent key."""

    pass


class ModelConfigStore(ABC):
    """Abstract base class for storage and retrieval of model configs."""

    @abstractmethod
    def add_model(self, key: str, config: Union[dict, ModelConfigBase]) -> None:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise a DuplicateModelException error.
        """
        pass

    @abstractmethod
    def del_model(self, key: str) -> None:
        """
        Delete a model.

        :param key: Unique key for the model to be deleted

        Can raise an UnknownModelException
        """
        pass

    @abstractmethod
    def update_model(self, key: str, config: Union[dict, ModelConfigBase]) -> ModelConfigBase:
        """
        Update the model, returning the updated version.

        :param key: Unique key for the model to be updated
        :param config: Model configuration record. Either a dict with the
         required fields, or a ModelConfigBase instance.
        """
        pass

    @abstractmethod
    def get_model(self, key: str) -> ModelConfigBase:
        """
        Retrieve the ModelConfigBase instance for the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        pass

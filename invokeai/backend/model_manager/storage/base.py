# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Abstract base class for storing and retrieving model configuration records.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Union

from ..config import AnyModelConfig, BaseModelType, ModelConfigBase, ModelType

# should match the InvokeAI version when this is first released.
CONFIG_FILE_VERSION = "3.2"


class DuplicateModelException(Exception):
    """Raised on an attempt to add a model with the same key twice."""


class InvalidModelException(Exception):
    """Raised when an invalid model is detected."""


class UnknownModelException(Exception):
    """Raised on an attempt to fetch or delete a model with a nonexistent key."""


class ConfigFileVersionMismatchException(Exception):
    """Raised on an attempt to open a config with an incompatible version."""


class ModelConfigStore(ABC):
    """Abstract base class for storage and retrieval of model configs."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the config file/database schema version."""
        pass

    @abstractmethod
    def add_model(self, key: str, config: Union[dict, AnyModelConfig]) -> ModelConfigBase:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfig exceptions.
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
    def update_model(self, key: str, config: Union[dict, AnyModelConfig]) -> AnyModelConfig:
        """
        Update the model, returning the updated version.

        :param key: Unique key for the model to be updated
        :param config: Model configuration record. Either a dict with the
         required fields, or a ModelConfigBase instance.
        """
        pass

    @abstractmethod
    def get_model(self, key: str) -> AnyModelConfig:
        """
        Retrieve the configuration for the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Return True if a model with the indicated key exists in the databse.

        :param key: Unique key for the model to be deleted
        """
        pass

    @abstractmethod
    def search_by_tag(self, tags: Set[str]) -> List[AnyModelConfig]:
        """
        Return models containing all of the listed tags.

        :param tags: Set of tags to search on.
        """
        pass

    @abstractmethod
    def search_by_path(
        self,
        path: Union[str, Path],
    ) -> Optional[AnyModelConfig]:
        """Return the model having the indicated path."""
        pass

    @abstractmethod
    def search_by_name(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
    ) -> List[AnyModelConfig]:
        """
        Return models matching name, base and/or type.

        :param model_name: Filter by name of model (optional)
        :param base_model: Filter by base model (optional)
        :param model_type: Filter by type of model (optional)

        If none of the optional filters are passed, will return all
        models in the database.
        """
        pass

    def all_models(self) -> List[AnyModelConfig]:
        """Return all the model configs in the database."""
        return self.search_by_name()

    def model_info_by_name(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> ModelConfigBase:
        """
        Return information about a single model using its name, base type and model type.

        If there are more than one model that match, raises a DuplicateModelException.
        If no model matches, raises an UnknownModelException
        """
        model_configs = self.search_by_name(model_name=model_name, base_model=base_model, model_type=model_type)
        if len(model_configs) > 1:
            raise DuplicateModelException(
                "More than one model share the same name and type: {base_model}/{model_type}/{model_name}"
            )
        if len(model_configs) == 0:
            raise UnknownModelException("No known model with name and type: {base_model}/{model_type}/{model_name}")
        return model_configs[0]

    def rename_model(
        self,
        key: str,
        new_name: str,
    ) -> ModelConfigBase:
        """
        Rename the indicated model. Just a special case of update_model().

        In some implementations, renaming the model may involve changing where
        it is stored on the filesystem. So this is broken out.

        :param key: Model key
        :param new_name: New name for model
        """
        return self.update_model(key, {"name": new_name})

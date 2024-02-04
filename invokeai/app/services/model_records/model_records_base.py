# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Abstract base class for storing and retrieving model configuration records.
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.backend.model_manager import LoadedModel, AnyModelConfig, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.metadata import AnyModelRepoMetadata, ModelMetadataStore


class DuplicateModelException(Exception):
    """Raised on an attempt to add a model with the same key twice."""


class InvalidModelException(Exception):
    """Raised when an invalid model is detected."""


class UnknownModelException(Exception):
    """Raised on an attempt to fetch or delete a model with a nonexistent key."""


class ConfigFileVersionMismatchException(Exception):
    """Raised on an attempt to open a config with an incompatible version."""


class ModelRecordOrderBy(str, Enum):
    """The order in which to return model summaries."""

    Default = "default"  # order by type, base, format and name
    Type = "type"
    Base = "base"
    Name = "name"
    Format = "format"


class ModelSummary(BaseModel):
    """A short summary of models for UI listing purposes."""

    key: str = Field(description="model key")
    type: ModelType = Field(description="model type")
    base: BaseModelType = Field(description="base model")
    format: ModelFormat = Field(description="model format")
    name: str = Field(description="model name")
    description: str = Field(description="short description of model")
    tags: Set[str] = Field(description="tags associated with model")


class ModelRecordServiceBase(ABC):
    """Abstract base class for storage and retrieval of model configs."""

    @abstractmethod
    def add_model(self, key: str, config: Union[Dict[str, Any], AnyModelConfig]) -> AnyModelConfig:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfigException exceptions.
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
    def update_model(self, key: str, config: Union[Dict[str, Any], AnyModelConfig]) -> AnyModelConfig:
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
    def load_model(self, key: str, submodel_type: Optional[SubModelType]) -> LoadedModel:
        """
        Load the indicated model into memory and return a LoadedModel object.

        :param key: Key of model config to be fetched.
        :param submodel_type: For main (pipeline models), the submodel to fetch 

        Exceptions: UnknownModelException -- model with this key not known
                    NotImplementedException -- a model loader was not provided at initialization time
        """
        pass

    @property
    @abstractmethod
    def metadata_store(self) -> ModelMetadataStore:
        """Return a ModelMetadataStore initialized on the same database."""
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[AnyModelRepoMetadata]:
        """
        Retrieve metadata (if any) from when model was downloaded from a repo.

        :param key: Model key
        """
        pass

    @abstractmethod
    def list_all_metadata(self) -> List[Tuple[str, AnyModelRepoMetadata]]:
        """List metadata for all models that have it."""
        pass

    @abstractmethod
    def search_by_metadata_tag(self, tags: Set[str]) -> List[AnyModelConfig]:
        """
        Search model metadata for ones with all listed tags and return their corresponding configs.

        :param tags: Set of tags to search for. All tags must be present.
        """
        pass

    @abstractmethod
    def list_tags(self) -> Set[str]:
        """Return a unique set of all the model tags in the metadata database."""
        pass

    @abstractmethod
    def list_models(
        self, page: int = 0, per_page: int = 10, order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default
    ) -> PaginatedResults[ModelSummary]:
        """Return a paginated summary listing of each model in the database."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Return True if a model with the indicated key exists in the databse.

        :param key: Unique key for the model to be deleted
        """
        pass

    @abstractmethod
    def search_by_path(
        self,
        path: Union[str, Path],
    ) -> List[AnyModelConfig]:
        """Return the model(s) having the indicated path."""
        pass

    @abstractmethod
    def search_by_hash(
        self,
        hash: str,
    ) -> List[AnyModelConfig]:
        """Return the model(s) having the indicated original hash."""
        pass

    @abstractmethod
    def search_by_attr(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
        model_format: Optional[ModelFormat] = None,
    ) -> List[AnyModelConfig]:
        """
        Return models matching name, base and/or type.

        :param model_name: Filter by name of model (optional)
        :param base_model: Filter by base model (optional)
        :param model_type: Filter by type of model (optional)
        :param model_format: Filter by model format (e.g. "diffusers") (optional)

        If none of the optional filters are passed, will return all
        models in the database.
        """
        pass

    def all_models(self) -> List[AnyModelConfig]:
        """Return all the model configs in the database."""
        return self.search_by_attr()

    def model_info_by_name(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> AnyModelConfig:
        """
        Return information about a single model using its name, base type and model type.

        If there are more than one model that match, raises a DuplicateModelException.
        If no model matches, raises an UnknownModelException
        """
        model_configs = self.search_by_attr(model_name=model_name, base_model=base_model, model_type=model_type)
        if len(model_configs) > 1:
            raise DuplicateModelException(
                f"More than one model matched the search criteria: base_model='{base_model}', model_type='{model_type}', model_name='{model_name}'."
            )
        if len(model_configs) == 0:
            raise UnknownModelException(
                f"More than one model matched the search criteria: base_model='{base_model}', model_type='{model_type}', model_name='{model_name}'."
            )
        return model_configs[0]

    def rename_model(
        self,
        key: str,
        new_name: str,
    ) -> AnyModelConfig:
        """
        Rename the indicated model. Just a special case of update_model().

        In some implementations, renaming the model may involve changing where
        it is stored on the filesystem. So this is broken out.

        :param key: Model key
        :param new_name: New name for model
        """
        config = self.get_model(key)
        config.name = new_name
        return self.update_model(key, config)

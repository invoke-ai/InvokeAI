# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Abstract base class for storing and retrieving model configuration records.
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Union

from pydantic import BaseModel, Field

from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ClipVariantType,
    ControlAdapterDefaultSettings,
    MainModelDefaultSettings,
    ModelFormat,
    ModelSourceType,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
)


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


class ModelRecordChanges(BaseModelExcludeNull):
    """A set of changes to apply to a model."""

    # Changes applicable to all models
    source: Optional[str] = Field(description="original source of the model", default=None)
    source_type: Optional[ModelSourceType] = Field(description="type of model source", default=None)
    source_api_response: Optional[str] = Field(description="metadata from remote source", default=None)
    name: Optional[str] = Field(description="Name of the model.", default=None)
    path: Optional[str] = Field(description="Path to the model.", default=None)
    description: Optional[str] = Field(description="Model description", default=None)
    base: Optional[BaseModelType] = Field(description="The base model.", default=None)
    type: Optional[ModelType] = Field(description="Type of model", default=None)
    key: Optional[str] = Field(description="Database ID for this model", default=None)
    hash: Optional[str] = Field(description="hash of model file", default=None)
    format: Optional[str] = Field(description="format of model file", default=None)
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: Optional[MainModelDefaultSettings | ControlAdapterDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )

    # Checkpoint-specific changes
    # TODO(MM2): Should we expose these? Feels footgun-y...
    variant: Optional[ModelVariantType | ClipVariantType] = Field(description="The variant of the model.", default=None)
    prediction_type: Optional[SchedulerPredictionType] = Field(
        description="The prediction type of the model.", default=None
    )
    upcast_attention: Optional[bool] = Field(description="Whether to upcast attention.", default=None)
    config_path: Optional[str] = Field(description="Path to config file for model", default=None)


class ModelRecordServiceBase(ABC):
    """Abstract base class for storage and retrieval of model configs."""

    @abstractmethod
    def add_model(self, config: AnyModelConfig) -> AnyModelConfig:
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
    def update_model(self, key: str, changes: ModelRecordChanges) -> AnyModelConfig:
        """
        Update the model, returning the updated version.

        :param key: Unique key for the model to be updated.
        :param changes: A set of changes to apply to this model. Changes are validated before being written.
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
    def get_model_by_hash(self, hash: str) -> AnyModelConfig:
        """
        Retrieve the configuration for the indicated model.

        :param hash: Hash of model config to be fetched.

        Exceptions: UnknownModelException
        """
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
        Return True if a model with the indicated key exists in the database.

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

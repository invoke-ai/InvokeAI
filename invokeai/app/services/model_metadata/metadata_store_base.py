# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Storage for Model Metadata
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

from pydantic import Field
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull

from invokeai.backend.model_manager.metadata import AnyModelRepoMetadata
from invokeai.backend.model_manager.metadata.metadata_base import ModelDefaultSettings

class ModelMetadataChanges(BaseModelExcludeNull, extra="allow"):
    """A set of changes to apply to model metadata.

    Only limited changes are valid:
      - `trigger_phrases`: the list of trigger phrases for this model
      - `default_settings`: the user-configured default settings for this model
    """

    trigger_phrases: Optional[List[str]] = Field(default=None, description="The model's list of trigger phrases")
    """The model's list of trigger phrases"""

    default_settings: Optional[ModelDefaultSettings] = Field(default=None, description="The user-configured default settings for this model")
    """The user-configured default settings for this model"""


class ModelMetadataStoreBase(ABC):
    """Store, search and fetch model metadata retrieved from remote repositories."""

    @abstractmethod
    def add_metadata(self, model_key: str, metadata: AnyModelRepoMetadata) -> None:
        """
        Add a block of repo metadata to a model record.

        The model record config must already exist in the database with the
        same key. Otherwise a FOREIGN KEY constraint exception will be raised.

        :param model_key: Existing model key in the `model_config` table
        :param metadata: ModelRepoMetadata object to store
        """

    @abstractmethod
    def get_metadata(self, model_key: str) -> AnyModelRepoMetadata:
        """Retrieve the ModelRepoMetadata corresponding to model key."""

    @abstractmethod
    def list_all_metadata(self) -> List[Tuple[str, AnyModelRepoMetadata]]:  # key, metadata
        """Dump out all the metadata."""

    @abstractmethod
    def update_metadata(self, model_key: str, metadata: AnyModelRepoMetadata) -> AnyModelRepoMetadata:
        """
        Update metadata corresponding to the model with the indicated key.

        :param model_key: Existing model key in the `model_config` table
        :param metadata: ModelRepoMetadata object to update
        """

    @abstractmethod
    def list_tags(self) -> Set[str]:
        """Return all tags in the tags table."""

    @abstractmethod
    def search_by_tag(self, tags: Set[str]) -> Set[str]:
        """Return the keys of models containing all of the listed tags."""

    @abstractmethod
    def search_by_author(self, author: str) -> Set[str]:
        """Return the keys of models authored by the indicated author."""

    @abstractmethod
    def search_by_name(self, name: str) -> Set[str]:
        """
        Return the keys of models with the indicated name.

        Note that this is the name of the model given to it by
        the remote source. The user may have changed the local
        name. The local name will be located in the model config
        record object.
        """

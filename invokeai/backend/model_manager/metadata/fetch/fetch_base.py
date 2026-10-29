"""
This module is the base class for subclasses that fetch metadata from model repositories

Usage:

from invokeai.backend.model_manager.metadata.fetch import HuggingFaceMetadataFetch

data = HuggingFaceMetadataFetch().from_id("<REPO_ID>")
assert isinstance(data, HuggingFaceMetadata)
"""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from invokeai.backend.model_manager.metadata.metadata_base import (
    AnyModelRepoMetadata,
    AnyModelRepoMetadataValidator,
    BaseMetadata,
)
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant


class ModelMetadataFetchBase(ABC):
    """Fetch metadata from remote generative model repositories."""

    @abstractmethod
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the fetcher with an optional requests.sessions.Session object.

        By providing a configurable Session object, we can support unit tests on
        this module without an internet connection.
        """
        pass

    @abstractmethod
    def from_url(
        self, url: AnyHttpUrl, revision: Optional[str] = None, pin_to_commit: bool = False
    ) -> AnyModelRepoMetadata:
        """
        Given a URL to a model repository, return a ModelMetadata object.

        :param revision: Fetch this repository revision rather than the current default.
        :param pin_to_commit: Without `revision`, name the current commit in file URLs rather than the branch.

        This method will raise a `UnknownMetadataException`
        in the event that the requested model metadata is not found at the provided location.
        """
        pass

    @abstractmethod
    def from_id(
        self,
        id: str,
        variant: Optional[ModelRepoVariant] = None,
        revision: Optional[str] = None,
        pin_to_commit: bool = False,
    ) -> AnyModelRepoMetadata:
        """
        Given an ID for a model, return a ModelMetadata object.

        :param id: An ID.
        :param variant: A model variant from the ModelRepoVariant enum.
        :param revision: Fetch this repository revision rather than the current default.
        :param pin_to_commit: Without `revision`, name the current commit in file URLs rather than the branch.

        This method will raise a `UnknownMetadataException`
        in the event that the requested model's metadata is not found at the provided id.
        """
        pass

    @classmethod
    def from_json(cls, json: str) -> AnyModelRepoMetadata:
        """Given the JSON representation of the metadata, return the corresponding Pydantic object."""
        metadata: BaseMetadata = AnyModelRepoMetadataValidator.validate_json(json)  # type: ignore
        return metadata

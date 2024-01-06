# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

"""
This module fetches model metadata objects from the HuggingFace model repository,
using either a `repo_id` or the model page URL.

Usage:

from invokeai.backend.model_manager.metadata.fetch import HuggingFaceMetadataFetch

fetcher = HuggingFaceMetadataFetch()
metadata = fetcher.from_url("https://huggingface.co/stabilityai/sdxl-turbo")
print(metadata.tags)
"""

import re
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import HfApi, configure_http_backend, hf_hub_url
from huggingface_hub.utils._errors import RepositoryNotFoundError
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from ..metadata_base import (
    AnyModelRepoMetadata,
    HuggingFaceMetadata,
    RemoteModelFile,
    UnknownMetadataException,
)
from .fetch_base import ModelMetadataFetchBase

HF_MODEL_RE = r"https?://huggingface.co/([\w\-.]+/[\w\-.]+)"


class HuggingFaceMetadataFetch(ModelMetadataFetchBase):
    """Fetch model metadata from HuggingFace."""

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the fetcher with an optional requests.sessions.Session object.

        By providing a configurable Session object, we can support unit tests on
        this module without an internet connection.
        """
        self._requests = session or requests.Session()
        configure_http_backend(backend_factory=lambda: self._requests)

    @classmethod
    def from_json(cls, json: str) -> HuggingFaceMetadata:
        """Given the JSON representation of the metadata, return the corresponding Pydantic object."""
        metadata = HuggingFaceMetadata.model_validate_json(json)
        return metadata

    def from_id(self, id: str) -> AnyModelRepoMetadata:
        """Return a HuggingFaceMetadata object given the model's repo_id."""
        try:
            model_info = HfApi().model_info(repo_id=id, files_metadata=True)
        except RepositoryNotFoundError as excp:
            raise UnknownMetadataException(f"'{id}' not found. See trace for details.") from excp

        _, name = id.split("/")
        return HuggingFaceMetadata(
            id=model_info.modelId,
            author=model_info.author,
            name=name,
            last_modified=model_info.lastModified,
            tag_dict=model_info.card_data.to_dict(),
            tags=model_info.tags,
            files=[
                RemoteModelFile(
                    url=hf_hub_url(id, x.rfilename),
                    path=Path(name, x.rfilename),
                    size=x.size,
                )
                for x in model_info.siblings
            ],
        )

    def from_url(self, url: AnyHttpUrl) -> AnyModelRepoMetadata:
        """
        Return a HuggingFaceMetadata object given the model's web page URL.

        In the case of an invalid or missing URL, raises a ModelNotFound exception.
        """
        if match := re.match(HF_MODEL_RE, str(url)):
            repo_id = match.group(1)
            return self.from_id(repo_id)
        else:
            raise UnknownMetadataException(f"'{url}' does not look like a HuggingFace model page")

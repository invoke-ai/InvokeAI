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

import json
import re
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import HfApi, configure_http_backend, hf_hub_url
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from invokeai.backend.model_manager.metadata.fetch.fetch_base import ModelMetadataFetchBase
from invokeai.backend.model_manager.metadata.metadata_base import (
    AnyModelRepoMetadata,
    HuggingFaceMetadata,
    RemoteModelFile,
    UnknownMetadataException,
)
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant

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

    def from_id(self, id: str, variant: Optional[ModelRepoVariant] = None) -> AnyModelRepoMetadata:
        """Return a HuggingFaceMetadata object given the model's repo_id."""
        # Little loop which tries fetching a revision corresponding to the selected variant.
        # If not available, then set variant to None and get the default.
        # If this too fails, raise exception.

        model_info = None

        # Handling for our special syntax - we only want the base HF `org/repo` here.
        repo_id = id.split("::")[0] or id
        while not model_info:
            try:
                model_info = HfApi().model_info(repo_id=repo_id, files_metadata=True, revision=variant)
            except RepositoryNotFoundError as excp:
                raise UnknownMetadataException(f"'{repo_id}' not found. See trace for details.") from excp
            except RevisionNotFoundError:
                if variant is None:
                    raise
                else:
                    variant = None

        files: list[RemoteModelFile] = []

        _, name = repo_id.split("/")

        for s in model_info.siblings or []:
            assert s.rfilename is not None
            assert s.size is not None
            files.append(
                RemoteModelFile(
                    url=hf_hub_url(repo_id, s.rfilename, revision=variant or "main"),
                    path=Path(name, s.rfilename),
                    size=s.size,
                    sha256=s.lfs.get("sha256") if s.lfs else None,
                )
            )

        # diffusers models have a `model_index.json` or `config.json` file
        is_diffusers = any(str(f.url).endswith(("model_index.json", "config.json")) for f in files)

        # These URLs will be exposed to the user - I think these are the only file types we fully support
        ckpt_urls = (
            None
            if is_diffusers
            else [
                f.url
                for f in files
                if str(f.url).endswith(
                    (
                        ".safetensors",
                        ".bin",
                        ".pth",
                        ".pt",
                        ".ckpt",
                    )
                )
            ]
        )

        return HuggingFaceMetadata(
            id=model_info.id,
            name=name,
            files=files,
            api_response=json.dumps(model_info.__dict__, default=str),
            is_diffusers=is_diffusers,
            ckpt_urls=ckpt_urls,
        )

    def from_url(self, url: AnyHttpUrl) -> AnyModelRepoMetadata:
        """
        Return a HuggingFaceMetadata object given the model's web page URL.

        In the case of an invalid or missing URL, raises a ModelNotFound exception.
        """
        if match := re.match(HF_MODEL_RE, str(url), re.IGNORECASE):
            repo_id = match.group(1)
            return self.from_id(repo_id)
        else:
            raise UnknownMetadataException(f"'{url}' does not look like a HuggingFace model page")

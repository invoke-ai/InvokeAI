# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

"""
This module fetches model metadata objects from the Civitai model repository.
In addition to the `from_url()` and `from_id()` methods inherited from the
`ModelMetadataFetchBase` base class.

Civitai has two separate ID spaces: a model ID and a version ID. The
version ID corresponds to a specific model, and is the ID accepted by
`from_id()`. The model ID corresponds to a family of related models,
such as different training checkpoints or 16 vs 32-bit versions. The
`from_civitai_modelid()` method will accept a model ID and return the
metadata from the default version within this model set. The default
version is the same as what the user sees when they click on a model's
thumbnail.

Usage:

from invokeai.backend.model_manager.metadata.fetch import CivitaiMetadataFetch

fetcher = CivitaiMetadataFetch()
metadata = fetcher.from_url("https://civitai.com/models/206883/split")
print(metadata.trained_words)
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

import requests
from pydantic import TypeAdapter, ValidationError
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from invokeai.backend.model_manager.config import ModelRepoVariant

from ..metadata_base import (
    AnyModelRepoMetadata,
    CivitaiMetadata,
    RemoteModelFile,
    UnknownMetadataException,
)
from .fetch_base import ModelMetadataFetchBase

CIVITAI_MODEL_PAGE_RE = r"https?://civitai.com/models/(\d+)"
CIVITAI_VERSION_PAGE_RE = r"https?://civitai.com/models/(\d+)\?modelVersionId=(\d+)"
CIVITAI_DOWNLOAD_RE = r"https?://civitai.com/api/download/models/(\d+)"

CIVITAI_VERSION_ENDPOINT = "https://civitai.com/api/v1/model-versions/"
CIVITAI_MODEL_ENDPOINT = "https://civitai.com/api/v1/models/"


StringSetAdapter = TypeAdapter(set[str])


class CivitaiMetadataFetch(ModelMetadataFetchBase):
    """Fetch model metadata from Civitai."""

    def __init__(self, session: Optional[Session] = None, api_key: Optional[str] = None):
        """
        Initialize the fetcher with an optional requests.sessions.Session object.

        By providing a configurable Session object, we can support unit tests on
        this module without an internet connection.
        """
        self._requests = session or requests.Session()
        self._api_key = api_key

    def from_url(self, url: AnyHttpUrl) -> AnyModelRepoMetadata:
        """
        Given a URL to a CivitAI model or version page, return a ModelMetadata object.

        In the event that the URL points to a model page without the particular version
        indicated, the default model version is returned. Otherwise, the requested version
        is returned.
        """
        if match := re.match(CIVITAI_VERSION_PAGE_RE, str(url), re.IGNORECASE):
            model_id = match.group(1)
            version_id = match.group(2)
            return self.from_civitai_versionid(int(version_id), int(model_id))
        elif match := re.match(CIVITAI_MODEL_PAGE_RE, str(url), re.IGNORECASE):
            model_id = match.group(1)
            return self.from_civitai_modelid(int(model_id))
        elif match := re.match(CIVITAI_DOWNLOAD_RE, str(url), re.IGNORECASE):
            version_id = match.group(1)
            return self.from_civitai_versionid(int(version_id))
        raise UnknownMetadataException("The url '{url}' does not match any known Civitai URL patterns")

    def from_id(self, id: str, variant: Optional[ModelRepoVariant] = None) -> AnyModelRepoMetadata:
        """
        Given a Civitai model version ID, return a ModelRepoMetadata object.

        :param id: An ID.
        :param variant: A model variant from the ModelRepoVariant enum (currently ignored)

        May raise an `UnknownMetadataException`.
        """
        return self.from_civitai_versionid(int(id))

    def from_civitai_modelid(self, model_id: int) -> CivitaiMetadata:
        """
        Return metadata from the default version of the indicated model.

        May raise an `UnknownMetadataException`.
        """
        model_url = CIVITAI_MODEL_ENDPOINT + str(model_id)
        model_json = self._requests.get(self._get_url_with_api_key(model_url)).json()
        return self._from_api_response(model_json)

    def _from_api_response(self, api_response: dict[str, Any], version_id: Optional[int] = None) -> CivitaiMetadata:
        try:
            version_id = version_id or api_response["modelVersions"][0]["id"]
        except TypeError as excp:
            raise UnknownMetadataException from excp

        # loop till we find the section containing the version requested
        version_sections = [x for x in api_response["modelVersions"] if x["id"] == version_id]
        if not version_sections:
            raise UnknownMetadataException(f"Version {version_id} not found in model metadata")

        version_json = version_sections[0]

        # Civitai has one "primary" file plus others such as VAEs. We only fetch the primary.
        primary = [x for x in version_json["files"] if x.get("primary")]
        assert len(primary) == 1
        primary_file = primary[0]

        url = primary_file["downloadUrl"]
        if "?" not in url:  # work around apparent bug in civitai api
            metadata_string = ""
            for key, value in primary_file["metadata"].items():
                if not value:
                    continue
                metadata_string += f"&{key}={value}"
            url = url + f"?type={primary_file['type']}{metadata_string}"
        model_files = [
            RemoteModelFile(
                url=self._get_url_with_api_key(url),
                path=Path(primary_file["name"]),
                size=int(primary_file["sizeKB"] * 1024),
                sha256=primary_file["hashes"]["SHA256"],
            )
        ]

        try:
            trigger_phrases = StringSetAdapter.validate_python(version_json.get("trainedWords"))
        except ValidationError:
            trigger_phrases: set[str] = set()

        return CivitaiMetadata(
            name=version_json["name"],
            files=model_files,
            trigger_phrases=trigger_phrases,
            api_response=json.dumps(version_json),
        )

    def from_civitai_versionid(self, version_id: int, model_id: Optional[int] = None) -> CivitaiMetadata:
        """
        Return a CivitaiMetadata object given a model version id.

        May raise an `UnknownMetadataException`.
        """
        if model_id is None:
            version_url = CIVITAI_VERSION_ENDPOINT + str(version_id)
            version = self._requests.get(self._get_url_with_api_key(version_url)).json()
            if error := version.get("error"):
                raise UnknownMetadataException(error)
            model_id = version["modelId"]

        model_url = CIVITAI_MODEL_ENDPOINT + str(model_id)
        model_json = self._requests.get(self._get_url_with_api_key(model_url)).json()
        return self._from_api_response(model_json, version_id)

    @classmethod
    def from_json(cls, json: str) -> CivitaiMetadata:
        """Given the JSON representation of the metadata, return the corresponding Pydantic object."""
        metadata = CivitaiMetadata.model_validate_json(json)
        return metadata

    def _get_url_with_api_key(self, url: str) -> str:
        if not self._api_key:
            return url

        if "?" in url:
            return f"{url}&token={self._api_key}"

        return f"{url}?token={self._api_key}"

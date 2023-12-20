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

import re
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from invokeai.app.services.model_records import UnknownModelException

from ..metadata_base import (
    AnyModelRepoMetadata,
    AnyModelRepoMetadataValidator,
    CivitaiMetadata,
    CommercialUsage,
    LicenseRestrictions,
)
from .fetch_base import ModelMetadataFetchBase

CIVITAI_MODEL_PAGE_RE = r"https?://civitai.com/models/(\d+)"
CIVITAI_VERSION_PAGE_RE = r"https?://civitai.com/models/(\d+)\?modelVersionId=(\d+)"
CIVITAI_DOWNLOAD_RE = r"https?://civitai.com/api/download/models/(\d+)"

CIVITAI_VERSION_ENDPOINT = "https://civitai.com/api/v1/model-versions/"
CIVITAI_MODEL_ENDPOINT = "https://civitai.com/api/v1/models/"


class CivitaiMetadataFetch(ModelMetadataFetchBase):
    """Fetch model metadata from Civitai."""

    _requests: Session

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the fetcher with an optional requests.sessions.Session object.

        By providing a configurable Session object, we can support unit tests on
        this module without an internet connection.
        """
        self._requests = session or requests.Session()

    def from_url(self, url: AnyHttpUrl) -> AnyModelRepoMetadata:
        """
        Given a URL to a CivitAI model or version page, return a ModelMetadata object.

        In the event that the URL points to a model page without the particular version
        indicated, the default model version is returned. Otherwise, the requested version
        is returned.
        """
        if match := re.match(CIVITAI_MODEL_PAGE_RE, str(url)):
            model_id = match.group(1)
            return self.from_civitai_modelid(int(model_id))
        elif match := re.match(CIVITAI_VERSION_PAGE_RE, str(url)):
            version_id = match.group(1)
            return self.from_civitai_versionid(int(version_id))
        elif match := re.match(CIVITAI_DOWNLOAD_RE, str(url)):
            version_id = match.group(1)
            return self.from_civitai_versionid(int(version_id))
        raise UnknownModelException("The url '{url}' does not match any known Civitai URL patterns")

    def from_id(self, id: str) -> AnyModelRepoMetadata:
        """
        Given a Civitai model version ID, return a ModelRepoMetadata object.

        May raise an `UnknownModelException`.
        """
        return self.from_civitai_versionid(int(id))

    def from_civitai_modelid(self, model_id: int) -> CivitaiMetadata:
        """
        Return metadata from the default version of the indicated model.

        May raise an `UnknownModelException`.
        """
        model_url = CIVITAI_MODEL_ENDPOINT + str(model_id)
        model = self._requests.get(model_url).json()
        default_version = model["modelVersions"][0]["id"]
        return self.from_civitai_versionid(default_version, model)

    def from_civitai_versionid(
        self, version_id: int, model_metadata: Optional[Dict[str, Any]] = None
    ) -> CivitaiMetadata:
        version_url = CIVITAI_VERSION_ENDPOINT + str(version_id)
        version = self._requests.get(version_url).json()

        model_url = CIVITAI_MODEL_ENDPOINT + str(version["modelId"])
        model = model_metadata or self._requests.get(model_url).json()
        safe_thumbnails = [x["url"] for x in version["images"] if x["nsfw"] == "None"]

        # It would be more elegant to define a Pydantic BaseModel that matches the Civitai metadata JSON.
        # However the contents of the JSON does not exactly match the documentation at
        # https://github.com/civitai/civitai/wiki/REST-API-Reference, and it feels safer to cherry pick
        # a subset of the fields.
        #
        # In addition, there are some fields that I want to pick up from the model JSON, such as `tags`,
        # that are not present in the version JSON.
        return CivitaiMetadata(
            id=version["modelId"],
            name=version["model"]["name"],
            version_id=version["id"],
            version_name=version["name"],
            created=datetime.fromisoformat(re.sub(r"Z$", "+00:00", version["createdAt"])),
            base_model_trained_on=version["baseModel"],  # note - need a dictionary to turn into a BaseModelType
            download_url=version["downloadUrl"],
            thumbnail_url=safe_thumbnails[0] if safe_thumbnails else None,
            author=model["creator"]["username"],
            description=model["description"],
            version_description=version["description"] or "",
            tags=model["tags"],
            trained_words=version["trainedWords"],
            nsfw=version["model"]["nsfw"],
            restrictions=LicenseRestrictions(
                AllowNoCredit=model["allowNoCredit"],
                AllowCommercialUse=CommercialUsage(model["allowCommercialUse"]),
                AllowDerivatives=model["allowDerivatives"],
                AllowDifferentLicense=model["allowDifferentLicense"],
            ),
        )

    @classmethod
    def from_json(cls, json: str) -> CivitaiMetadata:
        """Given the JSON representation of the metadata, return the corresponding Pydantic object."""
        metadata = AnyModelRepoMetadataValidator.validate_json(json)
        assert isinstance(metadata, CivitaiMetadata)
        return metadata

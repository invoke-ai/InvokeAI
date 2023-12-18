# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

"""
This module fetches metadata objects from HuggingFace and Civitai.

Usage:

from invokeai.backend.model_manager.metadata.fetch import MetadataFetch

metadata = MetadataFetch.from_civitai_url("https://civitai.com/models/58390/detail-tweaker-lora-lora")
print(metadata.description)
"""

import re
from pathlib import Path
from typing import Optional, Dict, Optional, Any
from datetime import datetime

import requests
from huggingface_hub import HfApi, configure_http_backend
from huggingface_hub.utils._errors import RepositoryNotFoundError
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from invokeai.app.services.model_records import UnknownModelException

from .base import (
    CivitaiMetadata,
    HuggingFaceMetadata,
    LicenseRestrictions,
    CommercialUsage,
)

HF_MODEL_RE = r"https?://huggingface.co/([\w\-.]+/[\w\-.]+)"
CIVITAI_MODEL_PAGE_RE = r"https?://civitai.com/models/(\d+)"
CIVITAI_VERSION_PAGE_RE = r"https?://civitai.com/models/(\d+)\?modelVersionId=(\d+)"
CIVITAI_DOWNLOAD_RE = r"https?://civitai.com/api/download/models/(\d+)"

CIVITAI_VERSION_ENDPOINT = "https://civitai.com/api/v1/model-versions/"
CIVITAI_MODEL_ENDPOINT = "https://civitai.com/api/v1/models/"

class MetadataFetch:
    """Fetch metadata from HuggingFace and Civitai URLs."""

    _requests: Session

    def __init__(self, session: Optional[Session]=None):
        """
        Initialize the fetcher with an optional requests.sessions.Session object.

        By providing a configurable Session object, we can support unit tests on
        this module without an internet connection.
        """
        self._requests = session or requests.Session()
        configure_http_backend(backend_factory = lambda: self._requests)

    def from_huggingface_repoid(self, repo_id: str) -> HuggingFaceMetadata:
        """Return a HuggingFaceMetadata object given the model's repo_id."""
        try:
            model_info = HfApi().model_info(repo_id=repo_id, files_metadata=True)
        except RepositoryNotFoundError as excp:
            raise UnknownModelException(f"'{repo_id}' not found. See trace for details.") from excp

        _, name = repo_id.split("/")
        return HuggingFaceMetadata(
            id = model_info.modelId,
            author = model_info.author,
            name = name,
            last_modified = model_info.lastModified,
            tags = model_info.tags,
            tag_dict = model_info.card_data.to_dict(),
            files = [Path(x.rfilename) for x in model_info.siblings]
        )

    def from_huggingface_url(self, url: AnyHttpUrl) -> HuggingFaceMetadata:
        """
        Return a HuggingFaceMetadata object given the model's web page URL.

        In the case of an invalid or missing URL, raises a ModelNotFound exception.
        """
        if match := re.match(HF_MODEL_RE, str(url)):
            repo_id = match.group(1)
            return self.from_huggingface_repoid(repo_id)
        else:
            raise UnknownModelException(f"'{url}' does not look like a HuggingFace model page")

    def from_civitai_versionid(self, version_id: int, model_metadata: Optional[Dict[str,Any]]=None) -> CivitaiMetadata:
        """Return Civitai metadata using a model's version id."""
        version_url = CIVITAI_VERSION_ENDPOINT + str(version_id)
        version = self._requests.get(version_url).json()

        model_url = CIVITAI_MODEL_ENDPOINT + str(version['modelId'])
        model = model_metadata or self._requests.get(model_url).json()
        safe_thumbnails = [x['url'] for x in version['images'] if x['nsfw']=='None']

        return CivitaiMetadata(
            id=version['modelId'],
            name=version['model']['name'],
            version_id=version['id'],
            version_name=version['name'],
            created=datetime.fromisoformat(re.sub(r"Z$", "+00:00", version['createdAt'])),
            base_model_trained_on=version['baseModel'],  # note - need a dictionary to turn into a BaseModelType
            download_url=version['downloadUrl'],
            thumbnail_url=safe_thumbnails[0] if safe_thumbnails else None,
            author=model['creator']['username'],
            description=model['description'],
            version_description=version['description'] or "",
            tags=model['tags'],
            trained_words=version['trainedWords'],
            nsfw=version['model']['nsfw'],
            restrictions=LicenseRestrictions(
                AllowNoCredit=model['allowNoCredit'],
                AllowCommercialUse=CommercialUsage(model['allowCommercialUse']),
                AllowDerivatives=model['allowDerivatives'],
                AllowDifferentLicense=model['allowDifferentLicense']
            ),
        )

    def from_civitai_modelid(self, model_id: int) -> CivitaiMetadata:
        """Return metadata from the default version of the indicated model."""
        model_url = CIVITAI_MODEL_ENDPOINT + str(model_id)
        model = self._requests.get(model_url).json()
        default_version = model['modelVersions'][0]['id']
        return self.from_civitai_versionid(default_version, model)

    def from_civitai_url(self, url: AnyHttpUrl) -> CivitaiMetadata:
        """Parse a Civitai URL that user is likely to pass and return its metadata."""
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



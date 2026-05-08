# Copyright (c) 2026 InvokeAI Development Team

"""Fetch model metadata from CivitAI."""

import json
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import requests
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from invokeai.backend.model_manager.metadata.fetch.fetch_base import ModelMetadataFetchBase
from invokeai.backend.model_manager.metadata.metadata_base import (
    AnyModelRepoMetadata,
    CivitaiMetadata,
    RemoteModelFile,
    UnknownMetadataException,
)
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant

CIVITAI_HOSTS = {"civitai.com", "www.civitai.com"}
CIVITAI_API_BASE_URL = "https://civitai.com/api/v1"


def is_civitai_url(url: str) -> bool:
    """Return whether the URL is hosted by CivitAI."""
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in CIVITAI_HOSTS


def is_civitai_model_version_url(url: str) -> bool:
    """Return whether the URL identifies a specific CivitAI model version."""
    if not is_civitai_url(url):
        return False

    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(path_parts) >= 3 and path_parts[:2] == ["api", "v1"] and path_parts[2] == "model-versions":
        return len(path_parts) >= 4 and path_parts[3] != "by-hash"
    if len(path_parts) >= 3 and path_parts[:2] == ["api", "download"] and path_parts[2] == "models":
        return len(path_parts) >= 4
    if len(path_parts) >= 2 and path_parts[0] in {"model-versions", "modelVersions"}:
        return True
    if len(path_parts) >= 2 and path_parts[0] == "models":
        return parse_qs(parsed.query).get("modelVersionId", [None])[0] is not None
    return False


class CivitaiMetadataFetch(ModelMetadataFetchBase):
    """Fetch model metadata from CivitAI."""

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the fetcher with an optional requests.sessions.Session object.

        By providing a configurable Session object, we can support unit tests on
        this module without an internet connection.
        """
        self._requests = session or requests.Session()

    @classmethod
    def from_json(cls, json_str: str) -> CivitaiMetadata:
        """Given the JSON representation of the metadata, return the corresponding Pydantic object."""
        return CivitaiMetadata.model_validate_json(json_str)

    def from_api_response(self, json_str: str) -> CivitaiMetadata:
        """Return metadata from a stored raw CivitAI model-version API response."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise UnknownMetadataException("Stored CivitAI metadata is not valid JSON") from e

        if not isinstance(data, dict):
            raise UnknownMetadataException("Stored CivitAI metadata is not an object")
        if data.get("type") == "civitai":
            return CivitaiMetadata.model_validate(data)
        return self._metadata_from_model_version(data)

    def from_id(self, id: str, variant: Optional[ModelRepoVariant] = None) -> AnyModelRepoMetadata:
        """Return a CivitaiMetadata object given a CivitAI model version id."""
        return self.from_model_version_id(self._parse_int(id, "model version id"))

    def from_url(self, url: AnyHttpUrl) -> AnyModelRepoMetadata:
        """
        Return a CivitaiMetadata object given a CivitAI URL.

        Supports web model pages with `modelVersionId`, model-version API URLs,
        model-version web pages, and CivitAI download URLs.
        """
        url_str = str(url)
        parsed = urlparse(url_str)
        if not is_civitai_url(url_str):
            raise UnknownMetadataException(f"'{url}' does not look like a CivitAI URL")

        path_parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(path_parts) >= 3 and path_parts[:2] == ["api", "v1"] and path_parts[2] == "model-versions":
            if len(path_parts) >= 4:
                return self.from_model_version_id(self._parse_int(path_parts[3], "model version id"))

        if len(path_parts) >= 3 and path_parts[:2] == ["api", "download"] and path_parts[2] == "models":
            if len(path_parts) >= 4:
                return self.from_model_version_id(self._parse_int(path_parts[3], "model version id"))

        if len(path_parts) >= 2 and path_parts[0] in {"model-versions", "modelVersions"}:
            return self.from_model_version_id(self._parse_int(path_parts[1], "model version id"))

        if len(path_parts) >= 2 and path_parts[0] == "models":
            model_id = self._parse_int(path_parts[1], "model id")
            model_version_id = self._parse_model_version_id(parsed.query)
            if model_version_id is None:
                raise UnknownMetadataException(f"'{url}' does not identify a CivitAI model version")
            return self.from_model_id(model_id, model_version_id)

        raise UnknownMetadataException(f"'{url}' does not look like a CivitAI model page")

    def from_model_id(self, model_id: int, model_version_id: Optional[int] = None) -> CivitaiMetadata:
        """Return metadata from a CivitAI model id, optionally selecting a specific version."""
        data = self._fetch_json(f"{CIVITAI_API_BASE_URL}/models/{model_id}")
        model_versions = data.get("modelVersions")
        if not isinstance(model_versions, list) or not model_versions:
            raise UnknownMetadataException(f"CivitAI model '{model_id}' has no model versions")

        version = None
        if model_version_id is not None:
            version = next((v for v in model_versions if v.get("id") == model_version_id), None)
            if version is None:
                raise UnknownMetadataException(
                    f"CivitAI model '{model_id}' does not include model version '{model_version_id}'"
                )
        else:
            version = model_versions[0]

        return self._metadata_from_model_version(version, data)

    def from_model_version_id(self, model_version_id: int) -> CivitaiMetadata:
        """Return metadata from a CivitAI model version id."""
        data = self._fetch_json(f"{CIVITAI_API_BASE_URL}/model-versions/{model_version_id}")
        return self._metadata_from_model_version(data)

    def from_hash(self, hash_value: str) -> CivitaiMetadata:
        """Return metadata from a CivitAI file hash."""
        data = self._fetch_json(f"{CIVITAI_API_BASE_URL}/model-versions/by-hash/{hash_value}")
        return self._metadata_from_model_version(data)

    def _fetch_json(self, url: str) -> dict[str, Any]:
        response = self._requests.get(url)
        if response.status_code == 404:
            raise UnknownMetadataException(f"'{url}' not found.")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise UnknownMetadataException(f"'{url}' did not return a CivitAI metadata object")
        return data

    def _metadata_from_model_version(
        self, version: dict[str, Any], model_response: Optional[dict[str, Any]] = None
    ) -> CivitaiMetadata:
        model_id = version.get("modelId") or (model_response or {}).get("id")
        model_version_id = version.get("id")
        if not isinstance(model_id, int) or not isinstance(model_version_id, int):
            raise UnknownMetadataException("Malformed CivitAI model version response")

        files = version.get("files")
        if not isinstance(files, list):
            raise UnknownMetadataException("Malformed CivitAI model version response: missing files")

        model_file = self._select_model_file(files)
        if model_file is None:
            raise UnknownMetadataException(f"CivitAI model version '{model_version_id}' has no model file")

        model_info = version.get("model")
        if not isinstance(model_info, dict):
            model_info = model_response or {}

        model_name = model_info.get("name") or version.get("name") or str(model_version_id)
        source_url = f"https://civitai.com/models/{model_id}?modelVersionId={model_version_id}"

        return CivitaiMetadata(
            name=str(model_name),
            model_id=model_id,
            model_version_id=model_version_id,
            trained_words=self._clean_trained_words(version.get("trainedWords")),
            files=[self._remote_file_from_civitai_file(model_file)],
            api_response=json.dumps(version, default=str),
            source_url=source_url,
        )

    def _select_model_file(self, files: list[Any]) -> Optional[dict[str, Any]]:
        model_files = [f for f in files if isinstance(f, dict) and f.get("type") == "Model"]
        if not model_files:
            return None
        return next((f for f in model_files if f.get("primary") is True), model_files[0])

    def _remote_file_from_civitai_file(self, file_data: dict[str, Any]) -> RemoteModelFile:
        download_url = file_data.get("downloadUrl")
        name = file_data.get("name")
        if not isinstance(download_url, str) or not isinstance(name, str):
            raise UnknownMetadataException("Malformed CivitAI model file response")

        size = file_data.get("sizeKB")
        size_bytes = int(float(size) * 1024) if size is not None else 0
        hashes = file_data.get("hashes") if isinstance(file_data.get("hashes"), dict) else {}
        sha256 = hashes.get("SHA256") if isinstance(hashes.get("SHA256"), str) else None

        return RemoteModelFile(url=download_url, path=Path(name), size=size_bytes, sha256=sha256)

    def _parse_model_version_id(self, query: str) -> Optional[int]:
        model_version_id = parse_qs(query).get("modelVersionId", [None])[0]
        return self._parse_int(model_version_id, "model version id") if model_version_id else None

    def _parse_int(self, value: str, label: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise UnknownMetadataException(f"Invalid CivitAI {label}: '{value}'")

    def _clean_trained_words(self, trained_words: Any) -> list[str]:
        if trained_words is None:
            return []
        if isinstance(trained_words, str):
            trained_words = [trained_words]
        if not isinstance(trained_words, list):
            return []

        seen = set()
        cleaned_words: list[str] = []
        for word in trained_words:
            if not isinstance(word, str):
                continue
            for cleaned_word in re.split(r"[,;\n]+", word):
                cleaned_word = cleaned_word.strip()
                if not cleaned_word or cleaned_word in seen:
                    continue
                cleaned_words.append(cleaned_word)
                seen.add(cleaned_word)
        return cleaned_words

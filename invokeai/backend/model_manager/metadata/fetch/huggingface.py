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
from types import SimpleNamespace
from typing import Optional
from urllib.parse import quote

import httpx
import requests
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from invokeai.backend.model_manager.metadata.fetch.fetch_base import ModelMetadataFetchBase
from invokeai.backend.model_manager.metadata.metadata_base import (
    HF_METADATA_REQUEST_TIMEOUT_SECONDS,
    AnyModelRepoMetadata,
    HuggingFaceMetadata,
    ModelMetadataUnavailableError,
    ModelRevisionUnavailableError,
    RemoteModelFile,
    UnknownMetadataException,
    is_transient_hf_http_error,
    revision_unavailable_message,
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
        self._has_custom_session = session is not None

    @classmethod
    def from_json(cls, json: str) -> HuggingFaceMetadata:
        """Given the JSON representation of the metadata, return the corresponding Pydantic object."""
        metadata = HuggingFaceMetadata.model_validate_json(json)
        return metadata

    def _model_info_via_session(self, repo_id: str, revision: Optional[str] = None) -> SimpleNamespace:
        """Fetch model info using the injected requests session (for testing/custom backends)."""
        params = {"blobs": "true"}
        url = f"https://huggingface.co/api/models/{repo_id}"
        if revision is not None:
            url += f"/revision/{quote(revision, safe='')}"
        resp = self._requests.get(url, params=params, timeout=HF_METADATA_REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 404:
            error_response = httpx.Response(
                status_code=resp.status_code,
                headers=resp.headers,
                request=httpx.Request("GET", str(resp.url)),
            )
            error_code = resp.headers.get("X-Error-Code", "")
            if error_code == "RevisionNotFound" or (revision is not None):
                raise RevisionNotFoundError(
                    f"Revision '{revision}' not found for repo '{repo_id}'.", response=error_response
                )
            raise RepositoryNotFoundError(f"Repository '{repo_id}' not found.", response=error_response)
        resp.raise_for_status()
        data = resp.json()
        # Convert siblings dicts to SimpleNamespace objects matching HfApi.model_info() shape
        siblings = []
        for s in data.get("siblings", []):
            siblings.append(
                SimpleNamespace(
                    rfilename=s.get("rfilename"),
                    size=s.get("size") or (s.get("lfs", {}) or {}).get("size"),
                    lfs=s.get("lfs"),
                )
            )
        return SimpleNamespace(id=data["id"], siblings=siblings, sha=data.get("sha"))

    def _model_info(self, repo_id: str, revision: Optional[str]) -> SimpleNamespace:
        try:
            # Use the injected session when provided (supports testing with mock adapters).
            # Otherwise use HfApi which uses httpx internally.
            if self._has_custom_session:
                return self._model_info_via_session(repo_id, revision)
            return HfApi().model_info(
                repo_id=repo_id,
                files_metadata=True,
                revision=revision,
                timeout=HF_METADATA_REQUEST_TIMEOUT_SECONDS,
            )
        # ConnectionError includes huggingface_hub's OfflineModeIsEnabled (HF_HUB_OFFLINE=1).
        except (requests.RequestException, httpx.TransportError, ConnectionError) as e:
            raise ModelMetadataUnavailableError(f"Could not fetch Hugging Face metadata for '{repo_id}'") from e
        except RepositoryNotFoundError as excp:
            raise UnknownMetadataException(f"'{repo_id}' not found. See trace for details.") from excp
        except RevisionNotFoundError:
            raise
        except HfHubHTTPError as e:
            if is_transient_hf_http_error(e):
                raise ModelMetadataUnavailableError(f"Could not fetch Hugging Face metadata for '{repo_id}'") from e
            raise

    def from_id(
        self,
        id: str,
        variant: Optional[ModelRepoVariant] = None,
        revision: Optional[str] = None,
        pin_to_commit: bool = False,
    ) -> AnyModelRepoMetadata:
        """Return a HuggingFaceMetadata object given the model's repo_id.

        Without `revision`, the files come from the branch named by `variant` if the repo has one, else from the
        default branch, and their URLs name that branch, as users and model records know them. `pin_to_commit`
        makes the URLs name the commit the branch points to now instead, so a download resumed later cannot mix in
        files from a newer commit. With `revision`, the files and URLs come from that revision.
        """
        # Handling for our special syntax - we only want the base HF `org/repo` here.
        repo_id = id.split("::")[0] or id

        if revision is not None:
            try:
                model_info = self._model_info(repo_id, revision)
            except RevisionNotFoundError as e:
                raise ModelRevisionUnavailableError(revision_unavailable_message(repo_id, revision)) from e
            url_revision = revision
        else:
            # Try the revision named by the selected variant; if the repo has none, fall back to the default branch.
            try:
                model_info = self._model_info(repo_id, variant)
            except RevisionNotFoundError:
                if variant is None:
                    raise
                variant = None
                model_info = self._model_info(repo_id, None)
            branch = variant or "main"
            url_revision = (getattr(model_info, "sha", None) or branch) if pin_to_commit else branch

        files: list[RemoteModelFile] = []

        _, name = repo_id.split("/")

        for s in model_info.siblings or []:
            assert s.rfilename is not None
            assert s.size is not None
            files.append(
                RemoteModelFile(
                    url=hf_hub_url(repo_id, s.rfilename, revision=url_revision),
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

        metadata = HuggingFaceMetadata(
            id=model_info.id,
            name=name,
            files=files,
            api_response=json.dumps(model_info.__dict__, default=str),
            is_diffusers=is_diffusers,
            ckpt_urls=ckpt_urls,
        )
        metadata._revision = url_revision
        return metadata

    def from_url(
        self, url: AnyHttpUrl, revision: Optional[str] = None, pin_to_commit: bool = False
    ) -> AnyModelRepoMetadata:
        """
        Return a HuggingFaceMetadata object given the model's web page URL.

        In the case of an invalid or missing URL, raises a ModelNotFound exception.
        """
        if match := re.match(HF_MODEL_RE, str(url), re.IGNORECASE):
            repo_id = match.group(1)
            return self.from_id(repo_id, revision=revision, pin_to_commit=pin_to_commit)
        else:
            raise UnknownMetadataException(f"'{url}' does not look like a HuggingFace model page")

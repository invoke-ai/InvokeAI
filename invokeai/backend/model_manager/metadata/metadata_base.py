"""This module defines core text-to-image model metadata fields.

Metadata comprises any descriptive information that is not essential
for getting the model to run. For example "author" is metadata, while
"type", "base" and "format" are not. The latter fields are part of the
model's config, as defined in invokeai.backend.model_manager.config.

Note that the "name" and "description" are also present in `config`
records. This is intentional. The config record fields are intended to
be editable by the user as a form of customization. The metadata
versions of these fields are intended to be kept in sync with the
remote repo.
"""

import re
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Union
from urllib.parse import unquote, urlparse

from huggingface_hub import hf_hub_url
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter
from pydantic.networks import AnyHttpUrl
from requests import HTTPError, RequestException
from requests.sessions import Session
from typing_extensions import Annotated

from invokeai.backend.model_manager.taxonomy import ModelRepoVariant
from invokeai.backend.model_manager.util.select_hf_files import filter_files

HF_METADATA_REQUEST_TIMEOUT_SECONDS = 15


class UnknownMetadataException(Exception):
    """Raised when no metadata is available for a model."""


class ModelMetadataUnavailableError(Exception):
    """Raised when a remote metadata request fails transiently."""


class ModelRevisionUnavailableError(Exception):
    """Raised when the repository revision an install is pinned to no longer exists."""


def is_commit_hash(revision: Optional[str]) -> bool:
    """Whether `revision` is a full Git commit hash, the only kind of revision whose files never change."""
    return revision is not None and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None


def revision_unavailable_message(repo_id: str, revision: str) -> str:
    return (
        f"The files of '{repo_id}' changed after this download started: revision '{revision}' is no longer available "
        "on Hugging Face, so the downloaded files cannot be completed. Start the install again to download the "
        "current version."
    )


def is_transient_hf_http_error(error: HfHubHTTPError | HTTPError) -> bool:
    """Return whether an HF HTTP response can reasonably succeed if retried later."""
    status_code = error.response.status_code if error.response is not None else None
    return status_code == 429 or (status_code is not None and 500 <= status_code < 600)


class RemoteModelFile(BaseModel):
    """Information about a downloadable file that forms part of a model."""

    url: AnyHttpUrl = Field(description="The url to download this model file")
    path: Path = Field(description="The path to the file, relative to the model root")
    size: Optional[int] = Field(description="The size of this file, in bytes", default=0)
    sha256: Optional[str] = Field(description="SHA256 hash of this model (not always available)", default=None)

    def __hash__(self) -> int:
        return hash(str(self))


class ModelMetadataBase(BaseModel):
    """Base class for model metadata information."""

    name: str = Field(description="model's name")


class BaseMetadata(ModelMetadataBase):
    """Adds typing data for discriminated union."""

    type: Literal["basemetadata"] = "basemetadata"


class ModelMetadataWithFiles(ModelMetadataBase):
    """Base class for metadata that contains a list of downloadable model file(s)."""

    files: List[RemoteModelFile] = Field(description="model files and their sizes", default_factory=list)

    def download_urls(
        self,
        variant: Optional[ModelRepoVariant] = None,
        subfolder: Optional[Path] = None,
        session: Optional[Session] = None,
    ) -> List[RemoteModelFile]:
        """
        Return a list of URLs needed to download the model.

        :param variant: Return files needed to reconstruct the indicated variant (e.g. ModelRepoVariant('fp16'))
        :param subfolder: Return files in the designated subfolder only
        :param session: A request.Session object for offline testing

        Note that the "variant" and "subfolder" concepts currently only apply to HuggingFace.
        However Civitai does have fields for the precision and format of its models, and may
        provide variant selection criteria in the future.
        """
        return self.files


# The revision segment of a Hugging Face file URL path: the first `/resolve/{revision}/` after an org and repo segment,
# behind any path prefix a mirror endpoint adds. An encoded revision (`refs%2Fpr%2F1`) is one segment.
_RESOLVE_PATH = re.compile(r"^(?:/.*?)??/[^/]+/[^/]+/resolve/([^/]+)/")


class HuggingFaceMetadata(ModelMetadataWithFiles):
    """Extended metadata fields provided by HuggingFace."""

    type: Literal["huggingface"] = "huggingface"
    id: str = Field(description="The HF model id")
    api_response: Optional[str] = Field(description="Response from the HF API as stringified JSON", default=None)
    is_diffusers: bool = Field(description="Whether the metadata is for a Diffusers format model", default=False)
    ckpt_urls: Optional[List[AnyHttpUrl]] = Field(
        description="URLs for all checkpoint format models in the metadata", default=None
    )
    # The revision the file URLs name: a branch, or the commit hash when the fetch pinned it. Private so the API
    # schema is unchanged.
    _revision: Optional[str] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        # A private attribute is not serialized, so metadata read back from JSON recovers the revision from its file
        # URLs (`.../{org}/{repo}/resolve/{revision}/...` on any endpoint); otherwise model_index.json would be read
        # from the default branch while the files come from the pinned commit. The URLs' repo is not compared with
        # `id`: they name the repo as it was asked for, `id` as the hub reports it, which can differ in case or after
        # a rename.
        super().model_post_init(context)
        revisions = {match.group(1) for f in self.files if (match := _RESOLVE_PATH.match(urlparse(str(f.url)).path))}
        if len(revisions) == 1:
            self._revision = unquote(revisions.pop())

    @property
    def revision(self) -> Optional[str]:
        """The revision the file URLs name, or None if this metadata was not fetched from the repository."""
        return self._revision

    def download_urls(
        self,
        variant: Optional[ModelRepoVariant] = None,
        subfolder: Optional[Path] = None,
        subfolders: Optional[List[Path]] = None,
        session: Optional[Session] = None,
        access_token_for: Optional[Callable[[str], Optional[str]]] = None,
    ) -> List[RemoteModelFile]:
        """
        Return list of downloadable files, filtering by variant and subfolder(s), if any.

        :param variant: Return model files needed to reconstruct the indicated variant
        :param subfolder: Return model files from the designated subfolder only (deprecated, use subfolders)
        :param subfolders: Return model files from the designated subfolders
        :param session: A request.Session object used for internet-free testing
        :param access_token_for: The bearer token to send with a request for a URL, or None for none. A gated
            repository serves its index files only to an authorized request.

        Note that there is special variant-filtering behavior here:
        When the fp16 variant is requested and not available, the
        full-precision model is returned.
        """
        session = session or Session()

        paths = filter_files([x.path for x in self.files], variant, subfolder, subfolders)  #  all files in the model

        # Determine prefix for model_index.json check - only applies for single subfolder
        prefix = ""
        if subfolder and not subfolders:
            prefix = f"{subfolder}/"

        # the next step reads model_index.json to determine which subdirectories belong
        # to the model (only for single subfolder case)
        if Path(f"{prefix}model_index.json") in paths:
            submodels = self._fetch_json(session, "model_index.json", subfolder, access_token_for)
            paths = [Path(subfolder or "", x) for x in paths if Path(x).parent.as_posix() in submodels]
            paths.insert(0, Path(f"{prefix}model_index.json"))
        elif not prefix and (repo_root := Path(self.name)) / "modular_model_index.json" in paths:
            # A Modular Diffusers repo names its components in modular_model_index.json instead. Each one
            # says which folder it loads from -- FLUX.3's tokenizer lives in `text_encoder/` -- and anything
            # else in the repo (FLUX.3 also ships BFL's own single-file package at the root) is not part of it.
            # File paths start with the repo name (HuggingFaceMetadataFetch.from_id), so folders compare below it.
            index = self._fetch_json(session, "modular_model_index.json", None, access_token_for)
            component_folders = _modular_component_folders(index)
            paths = [x for x in paths if x.parent.relative_to(repo_root).as_posix() in component_folders]
            paths.insert(0, repo_root / "modular_model_index.json")

        return [x for x in self.files if x.path in paths]

    def _fetch_json(
        self,
        session: Session,
        filename: str,
        subfolder: Optional[Path],
        access_token_for: Optional[Callable[[str], Optional[str]]],
    ) -> Any:
        """Read a JSON file of this repo at the revision its file URLs name.

        A transient HTTP or transport failure raises `ModelMetadataUnavailableError`; any other HTTP error re-raises.
        """
        url = hf_hub_url(
            self.id, filename=filename, subfolder=str(subfolder) if subfolder else None, revision=self._revision
        )
        token = access_token_for(url) if access_token_for else None
        headers = {"Authorization": f"Bearer {token}"} if token else None
        try:
            resp = session.get(url, headers=headers, timeout=HF_METADATA_REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except HTTPError as e:
            if is_transient_hf_http_error(e):
                raise ModelMetadataUnavailableError(f"Could not fetch {filename} for '{self.id}'") from e
            raise
        except RequestException as e:
            raise ModelMetadataUnavailableError(f"Could not fetch {filename} for '{self.id}'") from e
        return resp.json()


def _modular_component_folders(index: dict[str, Any]) -> set[str]:
    """The folders the components of a ``modular_model_index.json`` load from."""
    folders: set[str] = set()
    for name, spec in index.items():
        if name.startswith("_"):
            continue
        load_spec = spec[2] if isinstance(spec, list) and len(spec) > 2 and isinstance(spec[2], dict) else {}
        folders.add(str(load_spec.get("subfolder") or name))
    return folders


AnyModelRepoMetadata = Annotated[Union[BaseMetadata, HuggingFaceMetadata], Field(discriminator="type")]
AnyModelRepoMetadataValidator = TypeAdapter(AnyModelRepoMetadata)

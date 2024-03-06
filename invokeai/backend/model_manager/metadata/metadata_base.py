# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

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

from pathlib import Path
from typing import List, Literal, Optional, Union

from huggingface_hub import configure_http_backend, hf_hub_url
from pydantic import BaseModel, Field, TypeAdapter
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session
from typing_extensions import Annotated

from invokeai.backend.model_manager import ModelRepoVariant

from ..util import select_hf_files


class UnknownMetadataException(Exception):
    """Raised when no metadata is available for a model."""


class RemoteModelFile(BaseModel):
    """Information about a downloadable file that forms part of a model."""

    url: AnyHttpUrl = Field(description="The url to download this model file")
    path: Path = Field(description="The path to the file, relative to the model root")
    size: int = Field(description="The size of this file, in bytes")
    sha256: Optional[str] = Field(description="SHA256 hash of this model (not always available)", default=None)


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


class HuggingFaceMetadata(ModelMetadataWithFiles):
    """Extended metadata fields provided by HuggingFace."""

    type: Literal["huggingface"] = "huggingface"
    id: str = Field(description="The HF model id")
    api_response: Optional[str] = Field(description="Response from the HF API as stringified JSON", default=None)

    def download_urls(
        self,
        variant: Optional[ModelRepoVariant] = None,
        subfolder: Optional[Path] = None,
        session: Optional[Session] = None,
    ) -> List[RemoteModelFile]:
        """
        Return list of downloadable files, filtering by variant and subfolder, if any.

        :param variant: Return model files needed to reconstruct the indicated variant
        :param subfolder: Return model files from the designated subfolder only
        :param session: A request.Session object used for internet-free testing

        Note that there is special variant-filtering behavior here:
        When the fp16 variant is requested and not available, the
        full-precision model is returned.
        """
        session = session or Session()
        configure_http_backend(backend_factory=lambda: session)  # used in testing

        paths = select_hf_files.filter_files(
            [x.path for x in self.files], variant, subfolder
        )  #  all files in the model
        prefix = f"{subfolder}/" if subfolder else ""
        # the next step reads model_index.json to determine which subdirectories belong
        # to the model
        if Path(f"{prefix}model_index.json") in paths:
            url = hf_hub_url(self.id, filename="model_index.json", subfolder=str(subfolder) if subfolder else None)
            resp = session.get(url)
            resp.raise_for_status()
            submodels = resp.json()
            paths = [Path(subfolder or "", x) for x in paths if Path(x).parent.as_posix() in submodels]
            paths.insert(0, Path(f"{prefix}model_index.json"))

        return [x for x in self.files if x.path in paths]


AnyModelRepoMetadata = Annotated[Union[BaseMetadata, HuggingFaceMetadata], Field(discriminator="type")]
AnyModelRepoMetadataValidator = TypeAdapter(AnyModelRepoMetadata)

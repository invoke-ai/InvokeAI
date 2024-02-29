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

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from huggingface_hub import configure_http_backend, hf_hub_url
from pydantic import BaseModel, Field, TypeAdapter
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session
from typing_extensions import Annotated
from invokeai.app.invocations.constants import SCHEDULER_NAME_VALUES

from invokeai.backend.model_manager import ModelRepoVariant

from ..util import select_hf_files


class UnknownMetadataException(Exception):
    """Raised when no metadata is available for a model."""


class CommercialUsage(str, Enum):
    """Type of commercial usage allowed."""

    No = "None"
    Image = "Image"
    Rent = "Rent"
    RentCivit = "RentCivit"
    Sell = "Sell"


class LicenseRestrictions(BaseModel):
    """Broad categories of licensing restrictions."""

    AllowNoCredit: bool = Field(
        description="if true, model can be redistributed without crediting author", default=False
    )
    AllowDerivatives: bool = Field(description="if true, derivatives of this model can be redistributed", default=False)
    AllowDifferentLicense: bool = Field(
        description="if true, derivatives of this model be redistributed under a different license", default=False
    )
    AllowCommercialUse: Optional[CommercialUsage] = Field(
        description="Type of commercial use allowed or 'No' if no commercial use is allowed.", default=None
    )


class RemoteModelFile(BaseModel):
    """Information about a downloadable file that forms part of a model."""

    url: AnyHttpUrl = Field(description="The url to download this model file")
    path: Path = Field(description="The path to the file, relative to the model root")
    size: int = Field(description="The size of this file, in bytes")
    sha256: Optional[str] = Field(description="SHA256 hash of this model (not always available)", default=None)


class ModelDefaultSettings(BaseModel):
    vae: str | None
    vae_precision: str | None
    scheduler: SCHEDULER_NAME_VALUES | None
    steps: int | None
    cfg_scale: float | None
    cfg_rescale_multiplier: float | None

class ModelMetadataBase(BaseModel):
    """Base class for model metadata information."""

    name: str = Field(description="model's name")
    author: str = Field(description="model's author")
    tags: Optional[Set[str]] = Field(description="tags provided by model source", default=None)
    trigger_phrases: Optional[List[str]] = Field(description="trigger phrases for this model", default=None)
    default_settings: Optional[ModelDefaultSettings] = Field(description="default settings for this model", default=None)


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


class CivitaiMetadata(ModelMetadataWithFiles):
    """Extended metadata fields provided by Civitai."""

    type: Literal["civitai"] = "civitai"
    id: int = Field(description="Civitai version identifier")
    version_name: str = Field(description="Version identifier, such as 'V2-alpha'")
    version_id: int = Field(description="Civitai model version identifier")
    created: datetime = Field(description="date the model was created")
    updated: datetime = Field(description="date the model was last modified")
    published: datetime = Field(description="date the model was published to Civitai")
    description: str = Field(description="text description of model; may contain HTML")
    version_description: str = Field(
        description="text description of the model's reversion; usually change history; may contain HTML"
    )
    nsfw: bool = Field(description="whether the model tends to generate NSFW content", default=False)
    restrictions: LicenseRestrictions = Field(description="license terms", default_factory=LicenseRestrictions)
    trained_words: Set[str] = Field(description="words to trigger the model", default_factory=set)
    download_url: AnyHttpUrl = Field(description="download URL for this model")
    base_model_trained_on: str = Field(description="base model on which this model was trained (currently not an enum)")
    thumbnail_url: Optional[AnyHttpUrl] = Field(description="a thumbnail image for this model", default=None)
    weight_minmax: Tuple[float, float] = Field(
        description="minimum and maximum slider values for a LoRA or other secondary model", default=(-1.0, +2.0)
    )  # note: For future use

    @property
    def credit_required(self) -> bool:
        """Return True if you must give credit for derivatives of this model and images generated from it."""
        return not self.restrictions.AllowNoCredit

    @property
    def allow_commercial_use(self) -> bool:
        """Return True if commercial use is allowed."""
        if self.restrictions.AllowCommercialUse is None:
            return False
        else:
            return self.restrictions.AllowCommercialUse != CommercialUsage("None")

    @property
    def allow_derivatives(self) -> bool:
        """Return True if derivatives of this model can be redistributed."""
        return self.restrictions.AllowDerivatives

    @property
    def allow_different_license(self) -> bool:
        """Return true if derivatives of this model can use a different license."""
        return self.restrictions.AllowDifferentLicense


class HuggingFaceMetadata(ModelMetadataWithFiles):
    """Extended metadata fields provided by HuggingFace."""

    type: Literal["huggingface"] = "huggingface"
    id: str = Field(description="huggingface model id")
    tag_dict: Dict[str, Any]
    last_modified: datetime = Field(description="date of last commit to repo")

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
            url = hf_hub_url(self.id, filename="model_index.json", subfolder=subfolder)
            resp = session.get(url)
            resp.raise_for_status()
            submodels = resp.json()
            paths = [Path(subfolder or "", x) for x in paths if Path(x).parent.as_posix() in submodels]
            paths.insert(0, Path(f"{prefix}model_index.json"))

        return [x for x in self.files if x.path in paths]


AnyModelRepoMetadata = Annotated[Union[BaseMetadata, HuggingFaceMetadata, CivitaiMetadata], Field(discriminator="type")]
AnyModelRepoMetadataValidator = TypeAdapter(AnyModelRepoMetadata)

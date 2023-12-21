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
from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, TypeAdapter
from pydantic.networks import AnyHttpUrl
from typing_extensions import Annotated


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
    AllowCommercialUse: CommercialUsage = Field(
        description="Type of commercial use allowed or 'No' if no commercial use is allowed.", default_factory=set
    )


class ModelMetadataBase(BaseModel):
    """Base class for model metadata information."""

    name: str = Field(description="model's name")
    author: str = Field(description="model's author")
    tags: Set[str] = Field(description="tags provided by model source")


class HuggingFaceMetadata(ModelMetadataBase):
    """Extended metadata fields provided by HuggingFace."""

    type: Literal["huggingface"] = "huggingface"
    id: str = Field(description="huggingface model id")
    tag_dict: Dict[str, Any]
    last_modified: datetime = Field(description="date of last commit to repo")
    files: List[Path] = Field(description="sibling files that belong to this model", default_factory=list)


class CivitaiMetadata(ModelMetadataBase):
    """Extended metadata fields provided by Civitai."""

    type: Literal["civitai"] = "civitai"
    id: int = Field(description="Civitai model identifier")
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
    weight_min: float = Field(
        description="minimum suggested value for a LoRA or other secondary model", default=-1.0
    )  # note: For future use; not currently easily
    weight_max: float = Field(
        description="maximum suggested value for a LoRA or other secondary model", default=+2.0
    )  #       recoverable from metadata

    @property
    def credit_required(self) -> bool:
        """Return True if you must give credit for derivatives of this model and images generated from it."""
        return not self.restrictions.AllowNoCredit

    @property
    def allow_commercial_use(self) -> bool:
        """Return True if commercial use is allowed."""
        return self.restrictions.AllowCommercialUse != CommercialUsage("None")

    @property
    def allow_derivatives(self) -> bool:
        """Return True if derivatives of this model can be redistributed."""
        return self.restrictions.AllowDerivatives

    @property
    def allow_different_license(self) -> bool:
        """Return true if derivatives of this model can use a different license."""
        return self.restrictions.AllowDifferentLicense


AnyModelRepoMetadata = Annotated[Union[HuggingFaceMetadata, CivitaiMetadata], Field(discriminator="type")]
AnyModelRepoMetadataValidator = TypeAdapter(AnyModelRepoMetadata)

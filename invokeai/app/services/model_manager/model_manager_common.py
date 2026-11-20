from typing import Optional

from pydantic import BaseModel, Field

from invokeai.backend.model_manager.configs.factory import AnyModelConfig


class ModelIdentificationPreview(BaseModel):
    """What re-identifying a model record would do, without doing it."""

    proposed: Optional[AnyModelConfig] = Field(
        default=None,
        description="The record re-identification would write, with the user's fields kept. None when the files "
        "were not identified as any model.",
    )
    changed: bool = Field(
        description="Whether the proposed record differs from the stored one in type, format, base or variant."
    )
    refusal: Optional[str] = Field(
        default=None,
        description="Why a model class that recognised the files rejected them as unusable, if one did.",
    )

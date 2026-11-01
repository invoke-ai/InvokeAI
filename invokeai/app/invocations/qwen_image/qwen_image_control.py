"""Qwen Image ControlNet: alibaba-pai's Fun-ControlNet-Union for spatial control and inpainting, or InstantX's
ControlNet-Union for spatial control."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.util import validate_begin_end_step
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.qwen_image_vae import QWEN_IMAGE_VAE_MIN_TILE_SIZE

_CONTROL_SCALE_DESCRIPTION = (
    "Strength of the control signal. Recommended range: 0.70-0.95 for the Fun-ControlNet-Union; for the InstantX "
    "ControlNet-Union 0.80-1.00 on the original Qwen Image, about 1.5-2.0 on Qwen Image 2512."
)
_BEGIN_DESCRIPTION = "When the control is first applied (% of total steps)"
_END_DESCRIPTION = "When the control is last applied (% of total steps)"
_TILE_SIZE_DESCRIPTION = (
    f"{FieldDescriptions.vae_tile_size} Values between 1 and {QWEN_IMAGE_VAE_MIN_TILE_SIZE} are raised to "
    f"{QWEN_IMAGE_VAE_MIN_TILE_SIZE}."
)


class QwenImageControlField(BaseModel):
    """Qwen Image ControlNet conditioning: a control image, an inpaint image with its mask, or both (the InstantX
    ControlNet takes the control image alone)."""

    image_name: str | None = Field(default=None, description="The preprocessed control image (Canny, depth, pose, ...)")
    inpaint_image_name: str | None = Field(default=None, description="The image to inpaint")
    inpaint_mask_name: str | None = Field(default=None, description="The inpaint mask; white is regenerated")
    control_model: ModelIdentifierField = Field(description="The Qwen Image ControlNet model")
    control_context_scale: float = Field(default=0.85, ge=0.0, le=2.0, description=_CONTROL_SCALE_DESCRIPTION)
    begin_step_percent: float = Field(default=0.0, ge=0.0, le=1.0, description=_BEGIN_DESCRIPTION)
    end_step_percent: float = Field(default=1.0, ge=0.0, le=1.0, description=_END_DESCRIPTION)
    tiled: bool = Field(
        default=False,
        description="Encode the control images in tiles. The denoise encodes them at generation size, which for an "
        "upscale is the enlarged frame -- too large for an untiled encode on consumer hardware.",
    )
    tile_size: int = Field(default=0, ge=0, multiple_of=8, description=_TILE_SIZE_DESCRIPTION)


@invocation_output("qwen_image_control_output")
class QwenImageControlOutput(BaseInvocationOutput):
    """Qwen Image Control output containing control configuration."""

    control: QwenImageControlField = OutputField(description="Qwen Image control conditioning")


@invocation(
    "qwen_image_control",
    title="Qwen Image ControlNet",
    tags=["image", "qwen_image", "control", "controlnet", "inpaint"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImageControlInvocation(BaseInvocation):
    """Configure a Qwen Image ControlNet for spatial control, inpainting, or both.

    Takes a preprocessed control image (Canny, HED, depth, pose, MLSD, ...) and/or an image to inpaint with its mask
    (white is regenerated), plus the Qwen Image ControlNet model. The Fun-ControlNet-Union takes either or both
    (recommended scale 0.70-0.95); the InstantX ControlNet-Union takes the control image only (0.80-1.00 on the
    original Qwen Image, about 1.5-2.0 on Qwen Image 2512).
    """

    image: Optional[ImageField] = InputField(
        default=None, description="The preprocessed control image (Canny, HED, depth, pose, MLSD, ...)"
    )
    inpaint_image: Optional[ImageField] = InputField(
        default=None, description="The image to inpaint. Goes together with the inpaint mask."
    )
    inpaint_mask: Optional[ImageField] = InputField(
        default=None,
        description="The inpaint mask: white is regenerated, black is kept. Goes together with the inpaint image.",
    )
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model,
        title="Control Model",
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.ControlNet,
    )
    control_context_scale: float = InputField(
        default=0.85, ge=0.0, le=2.0, description=_CONTROL_SCALE_DESCRIPTION, title="Control Scale"
    )
    begin_step_percent: float = InputField(default=0.0, ge=0.0, le=1.0, description=_BEGIN_DESCRIPTION)
    end_step_percent: float = InputField(default=1.0, ge=0.0, le=1.0, description=_END_DESCRIPTION)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # NOTE: tile_size = 0 is a special value, as on the Qwen Image VAE nodes: the workflow UI cannot set None.
    tile_size: int = InputField(default=0, ge=0, multiple_of=8, description=_TILE_SIZE_DESCRIPTION)

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context: InvocationContext) -> QwenImageControlOutput:
        # Checked here rather than by a validator: connected inputs are filled in at execution time.
        if (self.inpaint_image is None) != (self.inpaint_mask is None):
            raise ValueError("The inpaint image and the inpaint mask go together: connect both or neither.")
        if self.image is None and self.inpaint_image is None:
            raise ValueError("Connect a control image, an inpaint image with its mask, or both.")
        return QwenImageControlOutput(
            control=QwenImageControlField(
                image_name=self.image.image_name if self.image is not None else None,
                inpaint_image_name=self.inpaint_image.image_name if self.inpaint_image is not None else None,
                inpaint_mask_name=self.inpaint_mask.image_name if self.inpaint_mask is not None else None,
                control_model=self.control_model,
                control_context_scale=self.control_context_scale,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                tiled=self.tiled,
                tile_size=self.tile_size,
            )
        )

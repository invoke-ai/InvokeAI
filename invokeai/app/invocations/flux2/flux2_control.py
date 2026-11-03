"""FLUX.2 [dev] ControlNet: the alibaba-pai Fun-ControlNet-Union for spatial control and inpainting."""

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
from invokeai.backend.util.vae_tiling_scope import MIN_TILE_SAMPLE_SIZE

_CONTROL_SCALE_DESCRIPTION = "Strength of the control signal. Recommended range: 0.65-0.80."
_BEGIN_DESCRIPTION = "When the control is first applied (% of total steps)"
_END_DESCRIPTION = "When the control is last applied (% of total steps)"
_TILE_SIZE_DESCRIPTION = (
    f"{FieldDescriptions.vae_tile_size} Values between 1 and {MIN_TILE_SAMPLE_SIZE} are raised to "
    f"{MIN_TILE_SAMPLE_SIZE}."
)


class Flux2ControlField(BaseModel):
    """FLUX.2 [dev] Fun-ControlNet conditioning: a control image, an inpaint image with its mask, or both."""

    image_name: str | None = Field(default=None, description="The preprocessed control image (Canny, depth, pose, ...)")
    inpaint_image_name: str | None = Field(default=None, description="The image to inpaint")
    inpaint_mask_name: str | None = Field(default=None, description="The inpaint mask; white is regenerated")
    control_model: ModelIdentifierField = Field(description="The FLUX.2 [dev] ControlNet model")
    control_context_scale: float = Field(default=0.75, ge=0.0, le=2.0, description=_CONTROL_SCALE_DESCRIPTION)
    begin_step_percent: float = Field(default=0.0, ge=0.0, le=1.0, description=_BEGIN_DESCRIPTION)
    end_step_percent: float = Field(default=1.0, ge=0.0, le=1.0, description=_END_DESCRIPTION)
    tiled: bool = Field(
        default=False,
        description="Encode the control images in tiles. The denoise encodes them at generation size, where an "
        "untiled FLUX.2 VAE encode takes several GB.",
    )
    tile_size: int = Field(default=0, ge=0, multiple_of=8, description=_TILE_SIZE_DESCRIPTION)


@invocation_output("flux2_control_output")
class Flux2ControlOutput(BaseInvocationOutput):
    """FLUX.2 [dev] ControlNet output containing control configuration."""

    control: Flux2ControlField = OutputField(description="FLUX.2 [dev] control conditioning")


@invocation(
    "flux2_control",
    title="FLUX.2 [dev] ControlNet",
    tags=["image", "flux", "flux2", "control", "controlnet", "inpaint"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2ControlInvocation(BaseInvocation):
    """Configure the FLUX.2 [dev] Fun-ControlNet-Union for spatial control, inpainting, or both.

    Takes a preprocessed control image (Canny, HED, depth, pose, MLSD, ...) and/or an image to inpaint with its mask
    (white is regenerated), plus the FLUX.2 [dev] ControlNet model. Applies to FLUX.2 [dev] only, not to Klein.
    Recommended control scale: 0.65-0.80.
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
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.ControlNet,
    )
    control_context_scale: float = InputField(
        default=0.75, ge=0.0, le=2.0, description=_CONTROL_SCALE_DESCRIPTION, title="Control Scale"
    )
    begin_step_percent: float = InputField(default=0.0, ge=0.0, le=1.0, description=_BEGIN_DESCRIPTION)
    end_step_percent: float = InputField(default=1.0, ge=0.0, le=1.0, description=_END_DESCRIPTION)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # NOTE: tile_size = 0 is a special value, as on the FLUX.2 VAE nodes: the workflow UI cannot set None.
    tile_size: int = InputField(default=0, ge=0, multiple_of=8, description=_TILE_SIZE_DESCRIPTION)

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context: InvocationContext) -> Flux2ControlOutput:
        # Checked here rather than by a validator: connected inputs are filled in at execution time.
        if (self.inpaint_image is None) != (self.inpaint_mask is None):
            raise ValueError("The inpaint image and the inpaint mask go together: connect both or neither.")
        if self.image is None and self.inpaint_image is None:
            raise ValueError("Connect a control image, an inpaint image with its mask, or both.")
        return Flux2ControlOutput(
            control=Flux2ControlField(
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

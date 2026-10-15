"""Z-Image Control invocation for spatial conditioning."""

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    InputField,
    OutputField,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.vae_tiling_scope import MIN_TILE_SAMPLE_SIZE


class ZImageControlField(BaseModel):
    """A Z-Image control conditioning field for spatial control (Canny, HED, Depth, Pose, MLSD)."""

    image_name: str = Field(description="The name of the preprocessed control image")
    control_model: ModelIdentifierField = Field(description="The Z-Image ControlNet adapter model")
    control_context_scale: float = Field(
        default=0.75,
        ge=0.0,
        le=2.0,
        description="The strength of the control signal. Recommended range: 0.65-0.80.",
    )
    begin_step_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="When the control is first applied (% of total steps)",
    )
    end_step_percent: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="When the control is last applied (% of total steps)",
    )
    tiled: bool = Field(
        default=False,
        description="Encode the control image in tiles. The denoise encodes it at generation size, which for an "
        "upscale is the enlarged frame -- too large for an untiled encode on consumer hardware.",
    )
    tile_size: int = Field(
        default=0,
        ge=0,
        multiple_of=8,
        description="Tile size in pixels for a tiled control encode; 0 uses the VAE's default.",
    )


@invocation_output("z_image_control_output")
class ZImageControlOutput(BaseInvocationOutput):
    """Z-Image Control output containing control configuration."""

    control: ZImageControlField = OutputField(description="Z-Image control conditioning")


@invocation(
    "z_image_control",
    title="Z-Image ControlNet",
    tags=["image", "z-image", "control", "controlnet"],
    category="conditioning",
    version="1.2.0",
    classification=Classification.Prototype,
)
class ZImageControlInvocation(BaseInvocation):
    """Configure Z-Image ControlNet for spatial conditioning.

    Takes a preprocessed control image (e.g., Canny edges, depth map, pose)
    and a Z-Image ControlNet adapter model to enable spatial control.

    Supports 5 control modes: Canny, HED, Depth, Pose, MLSD.
    Recommended control_context_scale: 0.65-0.80.
    """

    image: ImageField = InputField(
        description="The preprocessed control image (Canny, HED, Depth, Pose, or MLSD)",
    )
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model,
        title="Control Model",
        ui_model_base=BaseModelType.ZImage,
        ui_model_type=ModelType.ControlNet,
    )
    control_context_scale: float = InputField(
        default=0.75,
        ge=0.0,
        le=2.0,
        description="Strength of the control signal. Recommended range: 0.65-0.80.",
        title="Control Scale",
    )
    begin_step_percent: float = InputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="When the control is first applied (% of total steps)",
    )
    end_step_percent: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="When the control is last applied (% of total steps)",
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # NOTE: tile_size = 0 is a special value, as on the Z-Image VAE nodes: the workflow UI cannot set None.
    tile_size: int = InputField(
        default=0,
        ge=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{MIN_TILE_SAMPLE_SIZE} are raised to {MIN_TILE_SAMPLE_SIZE}.",
    )

    def invoke(self, context: InvocationContext) -> ZImageControlOutput:
        return ZImageControlOutput(
            control=ZImageControlField(
                image_name=self.image.image_name,
                control_model=self.control_model,
                control_context_scale=self.control_context_scale,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                tiled=self.tiled,
                tile_size=self.tile_size,
            )
        )

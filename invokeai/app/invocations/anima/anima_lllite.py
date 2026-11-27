"""Anima ControlNet-LLLite invocation for model-level inpaint conditioning."""

from typing import Optional

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


class AnimaLLLiteField(BaseModel):
    """An Anima ControlNet-LLLite conditioning field (e.g. inpaint adapter)."""

    image_name: str = Field(description="The name of the conditioning image (the initial/raster image)")
    mask_name: str | None = Field(
        default=None,
        description="The name of the inpaint mask image (white = inpaint area)",
    )
    control_model: ModelIdentifierField = Field(description="The Anima ControlNet-LLLite adapter model")
    weight: float = Field(
        default=1.0,
        ge=-10.0,
        le=10.0,
        description="The strength of the LLLite adapter",
    )
    begin_step_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="When the adapter is first applied (% of total steps)",
    )
    end_step_percent: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="When the adapter is last applied (% of total steps)",
    )


@invocation_output("anima_lllite_output")
class AnimaLLLiteOutput(BaseInvocationOutput):
    """Anima ControlNet-LLLite output containing adapter configuration."""

    control: AnimaLLLiteField = OutputField(description="Anima ControlNet-LLLite conditioning")


@invocation(
    "anima_lllite",
    title="Anima ControlNet-LLLite",
    tags=["image", "anima", "control", "controlnet", "inpaint"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class AnimaLLLiteInvocation(BaseInvocation):
    """Configure an Anima ControlNet-LLLite adapter for model-level conditioning.

    Takes a conditioning image (the initial/raster image), an optional inpaint
    mask (white = area to inpaint), and a LLLite adapter model. Inpainting
    adapters (4-channel conditioning) require a mask; other adapters ignore it.
    """

    image: ImageField = InputField(
        description="The conditioning image (the initial/raster image for inpainting)",
    )
    mask: Optional[ImageField] = InputField(
        default=None,
        description="The inpaint mask (white = area to inpaint). Required by inpainting adapters.",
    )
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model,
        title="Control Model",
        ui_model_base=BaseModelType.Anima,
        ui_model_type=ModelType.ControlNet,
    )
    weight: float = InputField(
        default=1.0,
        ge=-10.0,
        le=10.0,
        description="Strength of the LLLite adapter.",
    )
    begin_step_percent: float = InputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="When the adapter is first applied (% of total steps)",
    )
    end_step_percent: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="When the adapter is last applied (% of total steps)",
    )

    def invoke(self, context: InvocationContext) -> AnimaLLLiteOutput:
        return AnimaLLLiteOutput(
            control=AnimaLLLiteField(
                image_name=self.image.image_name,
                mask_name=self.mask.image_name if self.mask is not None else None,
                control_model=self.control_model,
                weight=self.weight,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
            )
        )

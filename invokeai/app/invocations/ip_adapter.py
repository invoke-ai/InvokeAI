from typing import Literal

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    UIType,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import ImageField
from invokeai.backend.model_management.models.base import BaseModelType

IP_ADAPTER_IMAGE_ENCODER_MODELS = Literal[
    "models/core/ip_adapters/sd-1/image_encoder/", "models/core/ip_adapters/sdxl/image_encoder"
]


class IPAdapterModelField(BaseModel):
    model_name: str = Field(description="Name of the IP-Adapter model")
    base_model: BaseModelType = Field(description="Base model")


class IPAdapterField(BaseModel):
    image: ImageField = Field(description="The IP-Adapter image prompt.")

    ip_adapter_model: IPAdapterModelField = Field(description="The IP-Adapter model to use.")

    # TODO(ryand): Create and use a `CLIPImageEncoderField` instead that is analogous to the `ClipField` used elsewhere.
    image_encoder_model: str = Field(description="The name of the CLIP image encoder model.")

    weight: float = Field(default=1.0, ge=0, description="The weight of the IP-Adapter.")


@invocation_output("ip_adapter_output")
class IPAdapterOutput(BaseInvocationOutput):
    # Outputs
    ip_adapter: IPAdapterField = OutputField(description=FieldDescriptions.ip_adapter, title="IP-Adapter")


@invocation("ip_adapter", title="IP-Adapter", tags=["ip_adapter", "control"], category="ip_adapter", version="1.0.0")
class IPAdapterInvocation(BaseInvocation):
    """Collects IP-Adapter info to pass to other nodes."""

    # Inputs
    image: ImageField = InputField(description="The IP-Adapter image prompt.")
    ip_adapter_model: IPAdapterModelField = InputField(
        description="The IP-Adapter model.",
        title="IP-Adapter Model",
        input=Input.Direct,
    )
    image_encoder_model: IP_ADAPTER_IMAGE_ENCODER_MODELS = InputField(
        default="models/core/ip_adapters/sd-1/image_encoder/", description="The name of the CLIP image encoder model."
    )
    weight: float = InputField(default=1.0, description="The weight of the IP-Adapter.", ui_type=UIType.Float)

    def invoke(self, context: InvocationContext) -> IPAdapterOutput:
        return IPAdapterOutput(
            ip_adapter=IPAdapterField(
                image=self.image,
                ip_adapter_model=self.ip_adapter_model,
                image_encoder_model=self.image_encoder_model,
                weight=self.weight,
            ),
        )

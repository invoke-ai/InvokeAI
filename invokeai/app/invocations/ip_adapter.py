from typing import Literal

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    InputField,
    InvocationContext,
    OutputField,
    UIType,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import ImageField

IP_ADAPTER_MODELS = Literal[
    "models/core/ip_adapters/sd-1/ip-adapter_sd15.bin",
    "models/core/ip_adapters/sd-1/ip-adapter-plus_sd15.bin",
    "models/core/ip_adapters/sd-1/ip-adapter-plus-face_sd15.bin",
    "models/core/ip_adapters/sdxl/ip-adapter_sdxl.bin",
]

IP_ADAPTER_IMAGE_ENCODER_MODELS = Literal[
    "models/core/ip_adapters/sd-1/image_encoder/", "models/core/ip_adapters/sdxl/image_encoder"
]


class IPAdapterField(BaseModel):
    image: ImageField = Field(description="The IP-Adapter image prompt.")

    # TODO(ryand): Create and use a custom `IpAdapterModelField`.
    ip_adapter_model: str = Field(description="The name of the IP-Adapter model.")

    # TODO(ryand): Create and use a `CLIPImageEncoderField` instead that is analogous to the `ClipField` used elsewhere.
    image_encoder_model: str = Field(description="The name of the CLIP image encoder model.")

    weight: float = Field(default=1.0, ge=0, description="The weight of the IP-Adapter.")


@invocation_output("ip_adapter_output")
class IPAdapterOutput(BaseInvocationOutput):
    # Outputs
    ip_adapter: IPAdapterField = OutputField(description=FieldDescriptions.ip_adapter)


@invocation("ip_adapter", title="IP-Adapter", tags=["ip_adapter", "control"], category="ip_adapter", version="1.0.0")
class IPAdapterInvocation(BaseInvocation):
    """Collects IP-Adapter info to pass to other nodes."""

    # Inputs
    image: ImageField = InputField(description="The IP-Adapter image prompt.")
    ip_adapter_model: IP_ADAPTER_MODELS = InputField(
        default="models/core/ip_adapters/sd-1/ip-adapter_sd15.bin", description="The name of the IP-Adapter model."
    )
    image_encoder_model: IP_ADAPTER_IMAGE_ENCODER_MODELS = InputField(
        default="models/core/ip_adapters/sd-1/image_encoder/", description="The name of the CLIP image encoder model."
    )
    weight: float = InputField(default=1.0, description="The weight of the IP-Adapter.", ui_type=UIType.Float)

    def invoke(self, context: InvocationContext) -> IPAdapterOutput:
        return IPAdapterOutput(
            ip_adapter=IPAdapterField(
                image=self.image,
                ip_adapter_model=(
                    context.services.configuration.get_config().root_dir / self.ip_adapter_model
                ).as_posix(),
                image_encoder_model=(
                    context.services.configuration.get_config().root_dir / self.image_encoder_model
                ).as_posix(),
                weight=self.weight,
            ),
        )

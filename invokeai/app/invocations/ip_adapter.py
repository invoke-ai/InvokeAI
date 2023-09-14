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
from invokeai.backend.model_management.models.base import BaseModelType, ModelType


class IPAdapterModelField(BaseModel):
    model_name: str = Field(description="Name of the IP-Adapter model")
    base_model: BaseModelType = Field(description="Base model")


class CLIPVisionModelField(BaseModel):
    model_name: str = Field(description="Name of the CLIP Vision image encoder model")
    base_model: BaseModelType = Field(description="Base model (usually 'Any')")


class IPAdapterField(BaseModel):
    image: ImageField = Field(description="The IP-Adapter image prompt.")
    ip_adapter_model: IPAdapterModelField = Field(description="The IP-Adapter model to use.")
    image_encoder_model: CLIPVisionModelField = Field(description="The name of the CLIP image encoder model.")
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
    weight: float = InputField(default=1.0, description="The weight of the IP-Adapter.", ui_type=UIType.Float)

    def invoke(self, context: InvocationContext) -> IPAdapterOutput:
        # Lookup the CLIP Vision encoder that is intended to be used with the IP-Adapter model.
        ip_adapter_info = context.services.model_manager.model_info(
            self.ip_adapter_model.model_name, self.ip_adapter_model.base_model, ModelType.IPAdapter
        )
        image_encoder_model_name = ip_adapter_info["image_encoder_model"].split("/")[-1].strip()
        image_encoder_model = CLIPVisionModelField(
            model_name=image_encoder_model_name,
            base_model=BaseModelType.Any,
        )

        return IPAdapterOutput(
            ip_adapter=IPAdapterField(
                image=self.image,
                ip_adapter_model=self.ip_adapter_model,
                image_encoder_model=image_encoder_model,
                weight=self.weight,
            ),
        )

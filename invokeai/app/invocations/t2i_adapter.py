from typing import Union

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


class T2IAdapterModelField(BaseModel):
    model_name: str = Field(description="Name of the T2I-Adapter model")
    base_model: BaseModelType = Field(description="Base model")


class T2IAdapterField(BaseModel):
    image: ImageField = Field(description="The T2I-Adapter image prompt.")
    t2i_adapter_model: T2IAdapterModelField = Field(description="The T2I-Adapter model to use.")
    weight: Union[float, list[float]] = Field(default=1, description="The weight given to the T2I-Adapter")


@invocation_output("t2i_adapter_output")
class T2IAdapterOutput(BaseInvocationOutput):
    t2i_adapter: T2IAdapterField = OutputField(description=FieldDescriptions.t2i_adapter, title="T2I Adapter")


@invocation(
    "t2i_adapter", title="T2I-Adapter", tags=["t2i_adapter", "control"], category="t2i_adapter", version="1.0.0"
)
class T2IAdapterInvocation(BaseInvocation):
    """Collects T2I-Adapter info to pass to other nodes."""

    # Inputs
    image: ImageField = InputField(description="The IP-Adapter image prompt.")
    ip_adapter_model: T2IAdapterModelField = InputField(
        description="The T2I-Adapter model.",
        title="T2I-Adapter Model",
        input=Input.Direct,
    )
    weight: Union[float, list[float]] = InputField(
        default=1, ge=0, description="The weight given to the T2I-Adapter", ui_type=UIType.Float, title="Weight"
    )

    def invoke(self, context: InvocationContext) -> T2IAdapterOutput:
        return T2IAdapterOutput(
            t2i_adapter=T2IAdapterField(
                image=self.image,
                t2i_adapter_model=self.ip_adapter_model,
                weight=self.weight,
            )
        )

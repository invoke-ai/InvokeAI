import os
from builtins import float
from typing import List, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.invocations.util import validate_begin_end_step, validate_weights
from invokeai.backend.model_management.models.base import BaseModelType, ModelType
from invokeai.backend.model_management.models.ip_adapter import get_ip_adapter_image_encoder_model_id


class IPAdapterModelField(BaseModel):
    model_name: str = Field(description="Name of the IP-Adapter model")
    base_model: BaseModelType = Field(description="Base model")

    model_config = ConfigDict(protected_namespaces=())


class CLIPVisionModelField(BaseModel):
    model_name: str = Field(description="Name of the CLIP Vision image encoder model")
    base_model: BaseModelType = Field(description="Base model (usually 'Any')")

    model_config = ConfigDict(protected_namespaces=())


class IPAdapterField(BaseModel):
    image: Union[ImageField, List[ImageField]] = Field(description="The IP-Adapter image prompt(s).")
    ip_adapter_model: IPAdapterModelField = Field(description="The IP-Adapter model to use.")
    image_encoder_model: CLIPVisionModelField = Field(description="The name of the CLIP image encoder model.")
    weight: Union[float, List[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the IP-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the IP-Adapter is last applied (% of total steps)"
    )

    @field_validator("weight")
    @classmethod
    def validate_ip_adapter_weight(cls, v):
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self


@invocation_output("ip_adapter_output")
class IPAdapterOutput(BaseInvocationOutput):
    # Outputs
    ip_adapter: IPAdapterField = OutputField(description=FieldDescriptions.ip_adapter, title="IP-Adapter")


@invocation("ip_adapter", title="IP-Adapter", tags=["ip_adapter", "control"], category="ip_adapter", version="1.1.2")
class IPAdapterInvocation(BaseInvocation):
    """Collects IP-Adapter info to pass to other nodes."""

    # Inputs
    image: Union[ImageField, List[ImageField]] = InputField(description="The IP-Adapter image prompt(s).")
    ip_adapter_model: IPAdapterModelField = InputField(
        description="The IP-Adapter model.", title="IP-Adapter Model", input=Input.Direct, ui_order=-1
    )

    weight: Union[float, List[float]] = InputField(
        default=1, description="The weight given to the IP-Adapter", title="Weight"
    )
    begin_step_percent: float = InputField(
        default=0, ge=0, le=1, description="When the IP-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1, ge=0, le=1, description="When the IP-Adapter is last applied (% of total steps)"
    )

    @field_validator("weight")
    @classmethod
    def validate_ip_adapter_weight(cls, v):
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context) -> IPAdapterOutput:
        # Lookup the CLIP Vision encoder that is intended to be used with the IP-Adapter model.
        ip_adapter_info = context.models.get_info(
            self.ip_adapter_model.model_name, self.ip_adapter_model.base_model, ModelType.IPAdapter
        )
        # HACK(ryand): This is bad for a couple of reasons: 1) we are bypassing the model manager to read the model
        # directly, and 2) we are reading from disk every time this invocation is called without caching the result.
        # A better solution would be to store the image encoder model reference in the IP-Adapter model info, but this
        # is currently messy due to differences between how the model info is generated when installing a model from
        # disk vs. downloading the model.
        image_encoder_model_id = get_ip_adapter_image_encoder_model_id(
            os.path.join(context.config.get().models_path, ip_adapter_info["path"])
        )
        image_encoder_model_name = image_encoder_model_id.split("/")[-1].strip()
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
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
            ),
        )

from builtins import float
from typing import List, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.invocations.util import validate_begin_end_step, validate_weights
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import AnyModelConfig, BaseModelType, IPAdapterConfig, ModelType


class IPAdapterField(BaseModel):
    image: Union[ImageField, List[ImageField]] = Field(description="The IP-Adapter image prompt(s).")
    ip_adapter_model: ModelIdentifierField = Field(description="The IP-Adapter model to use.")
    image_encoder_model: ModelIdentifierField = Field(description="The name of the CLIP image encoder model.")
    weight: Union[float, List[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the IP-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the IP-Adapter is last applied (% of total steps)"
    )

    @field_validator("weight")
    @classmethod
    def validate_ip_adapter_weight(cls, v: float) -> float:
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self) -> Self:
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self


@invocation_output("ip_adapter_output")
class IPAdapterOutput(BaseInvocationOutput):
    # Outputs
    ip_adapter: IPAdapterField = OutputField(description=FieldDescriptions.ip_adapter, title="IP-Adapter")


@invocation("ip_adapter", title="IP-Adapter", tags=["ip_adapter", "control"], category="ip_adapter", version="1.2.2")
class IPAdapterInvocation(BaseInvocation):
    """Collects IP-Adapter info to pass to other nodes."""

    # Inputs
    image: Union[ImageField, List[ImageField]] = InputField(description="The IP-Adapter image prompt(s).")
    ip_adapter_model: ModelIdentifierField = InputField(
        description="The IP-Adapter model.",
        title="IP-Adapter Model",
        input=Input.Direct,
        ui_order=-1,
        ui_type=UIType.IPAdapterModel,
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
    def validate_ip_adapter_weight(cls, v: float) -> float:
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self) -> Self:
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context: InvocationContext) -> IPAdapterOutput:
        # Lookup the CLIP Vision encoder that is intended to be used with the IP-Adapter model.
        ip_adapter_info = context.models.get_config(self.ip_adapter_model.key)
        assert isinstance(ip_adapter_info, IPAdapterConfig)
        image_encoder_model_id = ip_adapter_info.image_encoder_model_id
        image_encoder_model_name = image_encoder_model_id.split("/")[-1].strip()
        image_encoder_model = self._get_image_encoder(context, image_encoder_model_name)
        return IPAdapterOutput(
            ip_adapter=IPAdapterField(
                image=self.image,
                ip_adapter_model=self.ip_adapter_model,
                image_encoder_model=ModelIdentifierField.from_config(image_encoder_model),
                weight=self.weight,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
            ),
        )

    def _get_image_encoder(self, context: InvocationContext, image_encoder_model_name: str) -> AnyModelConfig:
        found = False
        while not found:
            image_encoder_models = context.models.search_by_attrs(
                name=image_encoder_model_name, base=BaseModelType.Any, type=ModelType.CLIPVision
            )
            found = len(image_encoder_models) > 0
            if not found:
                context.logger.warning(
                    f"The image encoder required by this IP Adapter ({image_encoder_model_name}) is not installed."
                )
                context.logger.warning("Downloading and installing now. This may take a while.")
                installer = context._services.model_manager.install
                job = installer.heuristic_import(f"InvokeAI/{image_encoder_model_name}")
                installer.wait_for_job(job, timeout=600)  # wait up to 10 minutes - then raise a TimeoutException
        assert len(image_encoder_models) == 1
        return image_encoder_models[0]

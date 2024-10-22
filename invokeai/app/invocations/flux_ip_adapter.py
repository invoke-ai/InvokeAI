from builtins import float
from typing import List, Literal, Union

from pydantic import field_validator, model_validator
from typing_extensions import Self

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, UIType
from invokeai.app.invocations.ip_adapter import (
    CLIP_VISION_MODEL_MAP,
    IPAdapterField,
    IPAdapterInvocation,
    IPAdapterOutput,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.invocations.util import validate_begin_end_step, validate_weights
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import (
    IPAdapterCheckpointConfig,
    IPAdapterInvokeAIConfig,
)


@invocation(
    "flux_ip_adapter",
    title="FLUX IP-Adapter",
    tags=["ip_adapter", "control"],
    category="ip_adapter",
    version="1.0.0",
    classification=Classification.Prototype,
)
class FluxIPAdapterInvocation(BaseInvocation):
    """Collects FLUX IP-Adapter info to pass to other nodes."""

    # FLUXIPAdapterInvocation is based closely on IPAdapterInvocation, but with some unsupported features removed.

    image: ImageField = InputField(description="The IP-Adapter image prompt(s).")
    ip_adapter_model: ModelIdentifierField = InputField(
        description="The IP-Adapter model.", title="IP-Adapter Model", ui_type=UIType.IPAdapterModel
    )
    # Currently, the only known ViT model used by FLUX IP-Adapters is ViT-L.
    clip_vision_model: Literal["ViT-L"] = InputField(description="CLIP Vision model to use.", default="ViT-L")
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
        assert isinstance(ip_adapter_info, (IPAdapterInvokeAIConfig, IPAdapterCheckpointConfig))

        # Note: There is a IPAdapterInvokeAIConfig.image_encoder_model_id field, but it isn't trustworthy.
        image_encoder_starter_model = CLIP_VISION_MODEL_MAP[self.clip_vision_model]
        image_encoder_model_id = image_encoder_starter_model.source
        image_encoder_model_name = image_encoder_starter_model.name
        image_encoder_model = IPAdapterInvocation.get_clip_image_encoder(
            context, image_encoder_model_id, image_encoder_model_name
        )

        return IPAdapterOutput(
            ip_adapter=IPAdapterField(
                image=self.image,
                ip_adapter_model=self.ip_adapter_model,
                image_encoder_model=ModelIdentifierField.from_config(image_encoder_model),
                weight=self.weight,
                target_blocks=[],  # target_blocks is currently unused for FLUX IP-Adapters.
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                mask=None,  # mask is currently unused for FLUX IP-Adapters.
            ),
        )

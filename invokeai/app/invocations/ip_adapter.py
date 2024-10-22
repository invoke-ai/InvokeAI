from builtins import float
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import FieldDescriptions, InputField, OutputField, TensorField, UIType
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.invocations.util import validate_begin_end_step, validate_weights
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    IPAdapterCheckpointConfig,
    IPAdapterInvokeAIConfig,
    ModelType,
)
from invokeai.backend.model_manager.starter_models import (
    StarterModel,
    clip_vit_l_image_encoder,
    ip_adapter_sd_image_encoder,
    ip_adapter_sdxl_image_encoder,
)


class IPAdapterField(BaseModel):
    image: Union[ImageField, List[ImageField]] = Field(description="The IP-Adapter image prompt(s).")
    ip_adapter_model: ModelIdentifierField = Field(description="The IP-Adapter model to use.")
    image_encoder_model: ModelIdentifierField = Field(description="The name of the CLIP image encoder model.")
    weight: Union[float, List[float]] = Field(default=1, description="The weight given to the IP-Adapter.")
    target_blocks: List[str] = Field(default=[], description="The IP Adapter blocks to apply")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the IP-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the IP-Adapter is last applied (% of total steps)"
    )
    mask: Optional[TensorField] = Field(
        default=None,
        description="The bool mask associated with this IP-Adapter. Excluded regions should be set to False, included "
        "regions should be set to True.",
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


CLIP_VISION_MODEL_MAP: dict[Literal["ViT-L", "ViT-H", "ViT-G"], StarterModel] = {
    "ViT-L": clip_vit_l_image_encoder,
    "ViT-H": ip_adapter_sd_image_encoder,
    "ViT-G": ip_adapter_sdxl_image_encoder,
}


@invocation("ip_adapter", title="IP-Adapter", tags=["ip_adapter", "control"], category="ip_adapter", version="1.5.0")
class IPAdapterInvocation(BaseInvocation):
    """Collects IP-Adapter info to pass to other nodes."""

    # Inputs
    image: Union[ImageField, List[ImageField]] = InputField(description="The IP-Adapter image prompt(s).", ui_order=1)
    ip_adapter_model: ModelIdentifierField = InputField(
        description="The IP-Adapter model.",
        title="IP-Adapter Model",
        ui_order=-1,
        ui_type=UIType.IPAdapterModel,
    )
    clip_vision_model: Literal["ViT-H", "ViT-G", "ViT-L"] = InputField(
        description="CLIP Vision model to use. Overrides model settings. Mandatory for checkpoint models.",
        default="ViT-H",
        ui_order=2,
    )
    weight: Union[float, List[float]] = InputField(
        default=1, description="The weight given to the IP-Adapter", title="Weight"
    )
    method: Literal["full", "style", "composition"] = InputField(
        default="full", description="The method to apply the IP-Adapter"
    )
    begin_step_percent: float = InputField(
        default=0, ge=0, le=1, description="When the IP-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1, ge=0, le=1, description="When the IP-Adapter is last applied (% of total steps)"
    )
    mask: Optional[TensorField] = InputField(
        default=None, description="A mask defining the region that this IP-Adapter applies to."
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

        if isinstance(ip_adapter_info, IPAdapterInvokeAIConfig):
            image_encoder_model_id = ip_adapter_info.image_encoder_model_id
            image_encoder_model_name = image_encoder_model_id.split("/")[-1].strip()
        else:
            image_encoder_starter_model = CLIP_VISION_MODEL_MAP[self.clip_vision_model]
            image_encoder_model_id = image_encoder_starter_model.source
            image_encoder_model_name = image_encoder_starter_model.name

        image_encoder_model = self.get_clip_image_encoder(context, image_encoder_model_id, image_encoder_model_name)

        if self.method == "style":
            if ip_adapter_info.base == "sd-1":
                target_blocks = ["up_blocks.1"]
            elif ip_adapter_info.base == "sdxl":
                target_blocks = ["up_blocks.0.attentions.1"]
            else:
                raise ValueError(f"Unsupported IP-Adapter base type: '{ip_adapter_info.base}'.")
        elif self.method == "composition":
            if ip_adapter_info.base == "sd-1":
                target_blocks = ["down_blocks.2", "mid_block"]
            elif ip_adapter_info.base == "sdxl":
                target_blocks = ["down_blocks.2.attentions.1"]
            else:
                raise ValueError(f"Unsupported IP-Adapter base type: '{ip_adapter_info.base}'.")
        elif self.method == "full":
            target_blocks = ["block"]
        else:
            raise ValueError(f"Unexpected IP-Adapter method: '{self.method}'.")

        return IPAdapterOutput(
            ip_adapter=IPAdapterField(
                image=self.image,
                ip_adapter_model=self.ip_adapter_model,
                image_encoder_model=ModelIdentifierField.from_config(image_encoder_model),
                weight=self.weight,
                target_blocks=target_blocks,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                mask=self.mask,
            ),
        )

    @classmethod
    def get_clip_image_encoder(
        cls, context: InvocationContext, image_encoder_model_id: str, image_encoder_model_name: str
    ) -> AnyModelConfig:
        image_encoder_models = context.models.search_by_attrs(
            name=image_encoder_model_name, base=BaseModelType.Any, type=ModelType.CLIPVision
        )

        if not len(image_encoder_models) > 0:
            context.logger.warning(
                f"The image encoder required by this IP Adapter ({image_encoder_model_name}) is not installed. \
                    Downloading and installing now. This may take a while."
            )

            installer = context._services.model_manager.install
            # Note: We hard-code the type to CLIPVision here because if the model contains both a CLIPVision and a
            # CLIPText model, the probe may treat it as a CLIPText model.
            job = installer.heuristic_import(
                image_encoder_model_id, ModelRecordChanges(name=image_encoder_model_name, type=ModelType.CLIPVision)
            )
            installer.wait_for_job(job, timeout=600)  # Wait for up to 10 minutes
            image_encoder_models = context.models.search_by_attrs(
                name=image_encoder_model_name, base=BaseModelType.Any, type=ModelType.CLIPVision
            )

            if len(image_encoder_models) == 0:
                context.logger.error("Error while fetching CLIP Vision Image Encoder")
                assert len(image_encoder_models) == 1

        return image_encoder_models[0]

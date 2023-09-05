from builtins import float
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator

from invokeai.app.invocations.primitives import ImageField

from ...backend.model_management import BaseModelType
from .baseinvocation import (
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

CONTROL_ADAPTER_TYPES = Literal["ControlNet", "IP-Adapter", "T2I-Adapter"]

CONTROLNET_MODE_VALUES = Literal["balanced", "more_prompt", "more_control", "unbalanced"]
CONTROLNET_RESIZE_VALUES = Literal[
    "just_resize",
    "crop_resize",
    "fill_resize",
    "just_resize_simple",
]


class ControlNetModelField(BaseModel):
    """ControlNet model field"""

    model_name: str = Field(description="Name of the ControlNet model")
    base_model: BaseModelType = Field(description="Base model")


class ControlField(BaseModel):
    control_type: CONTROL_ADAPTER_TYPES = Field(default="ControlNet", description="The type of control adapter")
    image: ImageField = Field(description="The control image")
    # control_model and ip_adapter_models are both optional
    #    but must be on the two present
    # if control_type == "ControlNet", then must be control_model
    # if control_type == "IP-Adapter", then must be ip_adapter_model
    control_model: Optional[ControlNetModelField] = Field(description="The ControlNet model to use")
    ip_adapter_model: Optional[str] = Field(description="The IP-Adapter model to use")
    image_encoder_model: Optional[str] = Field(description="The clip_image_encoder model to use")
    control_weight: Union[float, List[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode to use")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")

    @validator("control_weight")
    def validate_control_weight(cls, v):
        """Validate that all control weights in the valid range"""
        if isinstance(v, list):
            for i in v:
                if i < -1 or i > 2:
                    raise ValueError("Control weights must be within -1 to 2 range")
        else:
            if v < -1 or v > 2:
                raise ValueError("Control weights must be within -1 to 2 range")
        return v


@invocation_output("control_output")
class ControlOutput(BaseInvocationOutput):
    """node output for ControlNet info"""

    # Outputs
    control: ControlField = OutputField(description=FieldDescriptions.control)


@invocation("controlnet", title="ControlNet", tags=["controlnet"], category="controlnet", version="1.0.0")
class ControlNetInvocation(BaseInvocation):
    """Collects ControlNet info to pass to other nodes"""

    # Inputs
    image: ImageField = InputField(description="The control image")
    control_model: ControlNetModelField = InputField(
        default="lllyasviel/sd-controlnet-canny", description=FieldDescriptions.controlnet_model, input=Input.Direct
    )
    control_weight: Union[float, List[float]] = InputField(
        default=1.0, description="The weight given to the ControlNet", ui_type=UIType.Float
    )
    begin_step_percent: float = InputField(
        default=0, ge=-1, le=2, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = InputField(default="balanced", description="The control mode used")
    resize_mode: CONTROLNET_RESIZE_VALUES = InputField(default="just_resize", description="The resize mode used")

    def invoke(self, context: InvocationContext) -> ControlOutput:
        return ControlOutput(
            control=ControlField(
                control_type="ControlNet",
                image=self.image,
                control_model=self.control_model,
                # ip_adapter_model is currently optional
                #    must be either a control_model or ip_adapter_model
                # ip_adapter_model=None,
                control_weight=self.control_weight,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                control_mode=self.control_mode,
                resize_mode=self.resize_mode,
            ),
        )


IP_ADAPTER_MODELS = Literal[
    "models/core/ip_adapters/sd-1/ip-adapter_sd15.bin",
    "models/core/ip_adapters/sd-1/ip-adapter-plus_sd15.bin",
    "models/core/ip_adapters/sd-1/ip-adapter-plus-face_sd15.bin",
    "models/core/ip_adapters/sdxl/ip-adapter_sdxl.bin",
]

IP_ADAPTER_IMAGE_ENCODER_MODELS = Literal[
    "models/core/ip_adapters/sd-1/image_encoder/", "models/core/ip_adapters/sdxl/image_encoder"
]


@invocation("ipadapter", title="IP-Adapter", tags=["ipadapter"], category="ipadapter", version="1.0.0")
class IPAdapterInvocation(BaseInvocation):
    """Collects IP-Adapter info to pass to other nodes"""

    # Inputs
    image: ImageField = InputField(description="The control image")
    # control_model: ControlNetModelField = InputField(
    #    default="lllyasviel/sd-controlnet-canny", description=FieldDescriptions.controlnet_model, input=Input.Direct
    # )
    ip_adapter_model: IP_ADAPTER_MODELS = InputField(
        default="models/core/ip_adapters/sd-1/ip-adapter_sd15.bin", description="The IP-Adapter model"
    )
    image_encoder_model: IP_ADAPTER_IMAGE_ENCODER_MODELS = InputField(
        default="models/core/ip_adapters/sd-1/image_encoder/", description="The image encoder model"
    )
    control_weight: Union[float, List[float]] = InputField(
        default=1.0, description="The weight given to the ControlNet", ui_type=UIType.Float
    )
    # begin_step_percent: float = InputField(
    #     default=0, ge=-1, le=2, description="When the ControlNet is first applied (% of total steps)"
    # )
    # end_step_percent: float = InputField(
    #     default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    # )
    # control_mode: CONTROLNET_MODE_VALUES = InputField(default="balanced", description="The control mode used")
    # resize_mode: CONTROLNET_RESIZE_VALUES = InputField(default="just_resize", description="The resize mode used")

    def invoke(self, context: InvocationContext) -> ControlOutput:
        return ControlOutput(
            control=ControlField(
                control_type="IP-Adapter",
                image=self.image,
                # control_model is currently optional
                #    must be either a control_model or ip_adapter_model
                # control_model=None,
                ip_adapter_model=self.ip_adapter_model,
                image_encoder_model=self.image_encoder_model,
                control_weight=self.control_weight,
                # rest are currently ignored
                # begin_step_percent=self.begin_step_percent,
                # end_step_percent=self.end_step_percent,
                # control_mode=self.control_mode,
                # resize_mode=self.resize_mode,
            ),
        )

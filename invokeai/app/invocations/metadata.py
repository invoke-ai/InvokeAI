from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.fields import FieldDescriptions, InputField, MetadataField, OutputField, UIType
from invokeai.app.invocations.ip_adapter import IPAdapterModelField
from invokeai.app.invocations.model import LoRAModelField, MainModelField, VAEModelField
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.invocations.t2i_adapter import T2IAdapterField

from ...version import __version__


class MetadataItemField(BaseModel):
    label: str = Field(description=FieldDescriptions.metadata_item_label)
    value: Any = Field(description=FieldDescriptions.metadata_item_value)


class LoRAMetadataField(BaseModel):
    """LoRA Metadata Field"""

    lora: LoRAModelField = Field(description=FieldDescriptions.lora_model)
    weight: float = Field(description=FieldDescriptions.lora_weight)


class IPAdapterMetadataField(BaseModel):
    """IP Adapter Field, minus the CLIP Vision Encoder model"""

    image: ImageField = Field(description="The IP-Adapter image prompt.")
    ip_adapter_model: IPAdapterModelField = Field(
        description="The IP-Adapter model.",
    )
    weight: Union[float, list[float]] = Field(
        description="The weight given to the IP-Adapter",
    )
    begin_step_percent: float = Field(description="When the IP-Adapter is first applied (% of total steps)")
    end_step_percent: float = Field(description="When the IP-Adapter is last applied (% of total steps)")


@invocation_output("metadata_item_output")
class MetadataItemOutput(BaseInvocationOutput):
    """Metadata Item Output"""

    item: MetadataItemField = OutputField(description="Metadata Item")


@invocation("metadata_item", title="Metadata Item", tags=["metadata"], category="metadata", version="1.0.0")
class MetadataItemInvocation(BaseInvocation):
    """Used to create an arbitrary metadata item. Provide "label" and make a connection to "value" to store that data as the value."""

    label: str = InputField(description=FieldDescriptions.metadata_item_label)
    value: Any = InputField(description=FieldDescriptions.metadata_item_value, ui_type=UIType.Any)

    def invoke(self, context: InvocationContext) -> MetadataItemOutput:
        return MetadataItemOutput(item=MetadataItemField(label=self.label, value=self.value))


@invocation_output("metadata_output")
class MetadataOutput(BaseInvocationOutput):
    metadata: MetadataField = OutputField(description="Metadata Dict")


@invocation("metadata", title="Metadata", tags=["metadata"], category="metadata", version="1.0.0")
class MetadataInvocation(BaseInvocation):
    """Takes a MetadataItem or collection of MetadataItems and outputs a MetadataDict."""

    items: Union[list[MetadataItemField], MetadataItemField] = InputField(
        description=FieldDescriptions.metadata_item_polymorphic
    )

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        if isinstance(self.items, MetadataItemField):
            # single metadata item
            data = {self.items.label: self.items.value}
        else:
            # collection of metadata items
            data = {item.label: item.value for item in self.items}

        # add app version
        data.update({"app_version": __version__})
        return MetadataOutput(metadata=MetadataField.model_validate(data))


@invocation("merge_metadata", title="Metadata Merge", tags=["metadata"], category="metadata", version="1.0.0")
class MergeMetadataInvocation(BaseInvocation):
    """Merged a collection of MetadataDict into a single MetadataDict."""

    collection: list[MetadataField] = InputField(description=FieldDescriptions.metadata_collection)

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        data = {}
        for item in self.collection:
            data.update(item.model_dump())

        return MetadataOutput(metadata=MetadataField.model_validate(data))


GENERATION_MODES = Literal[
    "txt2img", "img2img", "inpaint", "outpaint", "sdxl_txt2img", "sdxl_img2img", "sdxl_inpaint", "sdxl_outpaint"
]


@invocation("core_metadata", title="Core Metadata", tags=["metadata"], category="metadata", version="1.0.1")
class CoreMetadataInvocation(BaseInvocation):
    """Collects core generation metadata into a MetadataField"""

    generation_mode: Optional[GENERATION_MODES] = InputField(
        default=None,
        description="The generation mode that output this image",
    )
    positive_prompt: Optional[str] = InputField(default=None, description="The positive prompt parameter")
    negative_prompt: Optional[str] = InputField(default=None, description="The negative prompt parameter")
    width: Optional[int] = InputField(default=None, description="The width parameter")
    height: Optional[int] = InputField(default=None, description="The height parameter")
    seed: Optional[int] = InputField(default=None, description="The seed used for noise generation")
    rand_device: Optional[str] = InputField(default=None, description="The device used for random number generation")
    cfg_scale: Optional[float] = InputField(default=None, description="The classifier-free guidance scale parameter")
    cfg_rescale_multiplier: Optional[float] = InputField(
        default=None, description=FieldDescriptions.cfg_rescale_multiplier
    )
    steps: Optional[int] = InputField(default=None, description="The number of steps used for inference")
    scheduler: Optional[str] = InputField(default=None, description="The scheduler used for inference")
    seamless_x: Optional[bool] = InputField(default=None, description="Whether seamless tiling was used on the X axis")
    seamless_y: Optional[bool] = InputField(default=None, description="Whether seamless tiling was used on the Y axis")
    clip_skip: Optional[int] = InputField(
        default=None,
        description="The number of skipped CLIP layers",
    )
    model: Optional[MainModelField] = InputField(default=None, description="The main model used for inference")
    controlnets: Optional[list[ControlField]] = InputField(
        default=None, description="The ControlNets used for inference"
    )
    ipAdapters: Optional[list[IPAdapterMetadataField]] = InputField(
        default=None, description="The IP Adapters used for inference"
    )
    t2iAdapters: Optional[list[T2IAdapterField]] = InputField(
        default=None, description="The IP Adapters used for inference"
    )
    loras: Optional[list[LoRAMetadataField]] = InputField(default=None, description="The LoRAs used for inference")
    strength: Optional[float] = InputField(
        default=None,
        description="The strength used for latents-to-latents",
    )
    init_image: Optional[str] = InputField(
        default=None,
        description="The name of the initial image",
    )
    vae: Optional[VAEModelField] = InputField(
        default=None,
        description="The VAE used for decoding, if the main model's default was not used",
    )

    # High resolution fix metadata.
    hrf_enabled: Optional[bool] = InputField(
        default=None,
        description="Whether or not high resolution fix was enabled.",
    )
    # TODO: should this be stricter or do we just let the UI handle it?
    hrf_method: Optional[str] = InputField(
        default=None,
        description="The high resolution fix upscale method.",
    )
    hrf_strength: Optional[float] = InputField(
        default=None,
        description="The high resolution fix img2img strength used in the upscale pass.",
    )

    # SDXL
    positive_style_prompt: Optional[str] = InputField(
        default=None,
        description="The positive style prompt parameter",
    )
    negative_style_prompt: Optional[str] = InputField(
        default=None,
        description="The negative style prompt parameter",
    )

    # SDXL Refiner
    refiner_model: Optional[MainModelField] = InputField(
        default=None,
        description="The SDXL Refiner model used",
    )
    refiner_cfg_scale: Optional[float] = InputField(
        default=None,
        description="The classifier-free guidance scale parameter used for the refiner",
    )
    refiner_steps: Optional[int] = InputField(
        default=None,
        description="The number of steps used for the refiner",
    )
    refiner_scheduler: Optional[str] = InputField(
        default=None,
        description="The scheduler used for the refiner",
    )
    refiner_positive_aesthetic_score: Optional[float] = InputField(
        default=None,
        description="The aesthetic score used for the refiner",
    )
    refiner_negative_aesthetic_score: Optional[float] = InputField(
        default=None,
        description="The aesthetic score used for the refiner",
    )
    refiner_start: Optional[float] = InputField(
        default=None,
        description="The start value used for refiner denoising",
    )

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        """Collects and outputs a CoreMetadata object"""

        return MetadataOutput(
            metadata=MetadataField.model_validate(
                self.model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
            )
        )

    model_config = ConfigDict(extra="allow")

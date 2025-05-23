from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    InputField,
    MetadataField,
    OutputField,
    UIType,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import CONTROLNET_MODE_VALUES, CONTROLNET_RESIZE_VALUES
from invokeai.version.invokeai_version import __version__


class MetadataItemField(BaseModel):
    label: str = Field(description=FieldDescriptions.metadata_item_label)
    value: Any = Field(description=FieldDescriptions.metadata_item_value)


class LoRAMetadataField(BaseModel):
    """LoRA Metadata Field"""

    model: ModelIdentifierField = Field(description=FieldDescriptions.lora_model)
    weight: float = Field(description=FieldDescriptions.lora_weight)


class IPAdapterMetadataField(BaseModel):
    """IP Adapter Field, minus the CLIP Vision Encoder model"""

    image: ImageField = Field(description="The IP-Adapter image prompt.")
    ip_adapter_model: ModelIdentifierField = Field(description="The IP-Adapter model.")
    clip_vision_model: Literal["ViT-L", "ViT-H", "ViT-G"] = Field(description="The CLIP Vision model")
    method: Literal["full", "style", "composition", "style_strong", "style_precise"] = Field(
        description="Method to apply IP Weights with"
    )
    weight: Union[float, list[float]] = Field(description="The weight given to the IP-Adapter")
    begin_step_percent: float = Field(description="When the IP-Adapter is first applied (% of total steps)")
    end_step_percent: float = Field(description="When the IP-Adapter is last applied (% of total steps)")


class T2IAdapterMetadataField(BaseModel):
    image: ImageField = Field(description="The control image.")
    processed_image: Optional[ImageField] = Field(default=None, description="The control image, after processing.")
    t2i_adapter_model: ModelIdentifierField = Field(description="The T2I-Adapter model to use.")
    weight: Union[float, list[float]] = Field(default=1, description="The weight given to the T2I-Adapter")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the T2I-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the T2I-Adapter is last applied (% of total steps)"
    )
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")


class ControlNetMetadataField(BaseModel):
    image: ImageField = Field(description="The control image")
    processed_image: Optional[ImageField] = Field(default=None, description="The control image, after processing.")
    control_model: ModelIdentifierField = Field(description="The ControlNet model to use")
    control_weight: Union[float, list[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode to use")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")


@invocation_output("metadata_item_output")
class MetadataItemOutput(BaseInvocationOutput):
    """Metadata Item Output"""

    item: MetadataItemField = OutputField(description="Metadata Item")


@invocation("metadata_item", title="Metadata Item", tags=["metadata"], category="metadata", version="1.0.1")
class MetadataItemInvocation(BaseInvocation):
    """Used to create an arbitrary metadata item. Provide "label" and make a connection to "value" to store that data as the value."""

    label: str = InputField(description=FieldDescriptions.metadata_item_label)
    value: Any = InputField(description=FieldDescriptions.metadata_item_value, ui_type=UIType.Any)

    def invoke(self, context: InvocationContext) -> MetadataItemOutput:
        return MetadataItemOutput(item=MetadataItemField(label=self.label, value=self.value))


@invocation_output("metadata_output")
class MetadataOutput(BaseInvocationOutput):
    metadata: MetadataField = OutputField(description="Metadata Dict")


@invocation("metadata", title="Metadata", tags=["metadata"], category="metadata", version="1.0.1")
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


@invocation("merge_metadata", title="Metadata Merge", tags=["metadata"], category="metadata", version="1.0.1")
class MergeMetadataInvocation(BaseInvocation):
    """Merged a collection of MetadataDict into a single MetadataDict."""

    collection: list[MetadataField] = InputField(description=FieldDescriptions.metadata_collection)

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        data = {}
        for item in self.collection:
            data.update(item.model_dump())

        return MetadataOutput(metadata=MetadataField.model_validate(data))


GENERATION_MODES = Literal[
    "txt2img",
    "img2img",
    "inpaint",
    "outpaint",
    "sdxl_txt2img",
    "sdxl_img2img",
    "sdxl_inpaint",
    "sdxl_outpaint",
    "flux_txt2img",
    "flux_img2img",
    "flux_inpaint",
    "flux_outpaint",
    "sd3_txt2img",
    "sd3_img2img",
    "sd3_inpaint",
    "sd3_outpaint",
    "cogview4_txt2img",
    "cogview4_img2img",
    "cogview4_inpaint",
    "cogview4_outpaint",
]


@invocation(
    "core_metadata",
    title="Core Metadata",
    tags=["metadata"],
    category="metadata",
    version="2.0.0",
    classification=Classification.Internal,
)
class CoreMetadataInvocation(BaseInvocation):
    """Used internally by Invoke to collect metadata for generations."""

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
    model: Optional[ModelIdentifierField] = InputField(default=None, description="The main model used for inference")
    controlnets: Optional[list[ControlNetMetadataField]] = InputField(
        default=None, description="The ControlNets used for inference"
    )
    ipAdapters: Optional[list[IPAdapterMetadataField]] = InputField(
        default=None, description="The IP Adapters used for inference"
    )
    t2iAdapters: Optional[list[T2IAdapterMetadataField]] = InputField(
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
    vae: Optional[ModelIdentifierField] = InputField(
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
    refiner_model: Optional[ModelIdentifierField] = InputField(
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

        as_dict = self.model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
        as_dict["app_version"] = __version__

        return MetadataOutput(metadata=MetadataField.model_validate(as_dict))

    model_config = ConfigDict(extra="allow")


@invocation(
    "metadata_field_extractor",
    title="Metadata Field Extractor",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Deprecated,
)
class MetadataFieldExtractorInvocation(BaseInvocation):
    """Extracts the text value from an image's metadata given a key.
    Raises an error if the image has no metadata or if the value is not a string (nesting not permitted)."""

    image: ImageField = InputField(description="The image to extract metadata from")
    key: str = InputField(description="The key in the image's metadata to extract the value from")

    def invoke(self, context: InvocationContext) -> StringOutput:
        image_name = self.image.image_name

        metadata = context.images.get_metadata(image_name=image_name)
        if not metadata:
            raise ValueError(f"No metadata found on image {image_name}")

        try:
            val = metadata.root[self.key]
            if not isinstance(val, str):
                raise ValueError(f"Metadata at key '{self.key}' must be a string")
            return StringOutput(value=val)
        except KeyError as e:
            raise ValueError(f"No key '{self.key}' found in the metadata for {image_name}") from e

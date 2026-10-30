# Adopted from @skunworkxdark's metadata nodes (MIT License)
# https://github.com/skunkworxdark/metadata-linked-nodes
# Thanks to @skunworkxdark for the original implementation!

import copy
from typing import Any, Dict, Literal, Optional, TypeVar, Union

from pydantic import model_validator

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
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
    Input,
    InputField,
    MetadataField,
    OutputField,
    UIType,
    WithMetadata,
)
from invokeai.app.invocations.flux.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.metadata import (
    LatentsMetaOutput,
    MetadataOutput,
    build_denoise_metadata,
)
from invokeai.app.invocations.model import (
    CLIPField,
    LoRAField,
    LoRALoaderOutput,
    ModelIdentifierField,
    Qwen3EncoderField,
    SDXLLoRALoaderOutput,
    UNetField,
    VAEField,
    VAEOutput,
)
from invokeai.app.invocations.primitives import (
    BooleanCollectionOutput,
    BooleanOutput,
    FloatCollectionOutput,
    FloatOutput,
    IntegerCollectionOutput,
    IntegerOutput,
    StringCollectionOutput,
    StringOutput,
)
from invokeai.app.invocations.scheduler import SchedulerOutput
from invokeai.app.invocations.sd.controlnet import ControlField, ControlNetInvocation
from invokeai.app.invocations.sd.denoise_latents import DenoiseLatentsInvocation
from invokeai.app.invocations.sd.ip_adapter import IPAdapterField, IPAdapterInvocation
from invokeai.app.invocations.sd.t2i_adapter import T2IAdapterField, T2IAdapterInvocation
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.version import __version__

CUSTOM_LABEL: str = "* CUSTOM LABEL *"

CORE_LABELS = Literal[
    f"{CUSTOM_LABEL}",
    "generation_mode",
    "positive_prompt",
    "positive_style_prompt",
    "negative_prompt",
    "negative_style_prompt",
    "width",
    "height",
    "seed",
    "rand_device",
    "cfg_scale",
    "cfg_rescale_multiplier",
    "steps",
    "scheduler",
    "clip_skip",
    "model",
    "vae",
    "seamless_x",
    "seamless_y",
    "guidance",
    "cfg_scale_start_step",
    "cfg_scale_end_step",
    "denoising_start",
    "denoising_end",
    "strength",
    "init_image",
    "add_noise",
    "shift",
    "dype_preset",
    "dype_scale",
    "dype_exponent",
    "freeu_s1",
    "freeu_s2",
    "freeu_b1",
    "freeu_b2",
    "hidiffusion",
    "hidiffusion_raunet",
    "hidiffusion_window_attn",
    "hidiffusion_t1_ratio",
    "hidiffusion_t2_ratio",
    "hrf_enabled",
    "hrf_method",
    "hrf_strength",
    "refiner_model",
    "refiner_cfg_scale",
    "refiner_steps",
    "refiner_scheduler",
    "refiner_positive_aesthetic_score",
    "refiner_negative_aesthetic_score",
    "refiner_start",
    "pid_mode",
    "pid_steps",
    "pid_decoder",
    "gemma2_encoder",
    "tile_width",
    "tile_height",
    "tile_overlap",
    "num_frames",
    "fps",
    "source_video_start_frame",
    "source_video_end_frame",
    "t5_encoder",
    "clip_embed_model",
    "clip_l_model",
    "clip_g_model",
    "mistral_encoder",
    "qwen3_encoder",
    "qwen3_source",
    "qwen3_5_encoder",
    "qwen3_vl_encoder",
    "ideogram4_caption",
    "ideogram4_color_palette",
    "ideogram4_guidance_scale",
    "ideogram4_mu",
    "ideogram4_sampler_preset",
    "ideogram4_steps",
    "ideogram4_unconditional_model",
    "krea2_rebalance_enabled",
    "krea2_rebalance_multiplier",
    "krea2_rebalance_weights",
    "krea2_seed_variance_enabled",
    "krea2_seed_variance_randomize_percent",
    "krea2_seed_variance_strength",
    "krea2_style_strength",
    "ltx2_audio_cfg_scale",
    "ltx2_base_height",
    "ltx2_base_width",
    "ltx2_component_source",
    "ltx2_conditioning_role",
    "ltx2_context_frames",
    "ltx2_guidance_rescale",
    "ltx2_modality_scale",
    "ltx2_noise_scale",
    "ltx2_schedule",
    "ltx2_stg_scale",
    "ltx2_text_encoder_model",
    "ltx2_two_stage",
    "minimax_h3_component_source",
    "minimax_h3_hybrid_base_model",
    "minimax_h3_hybrid_start_block",
    "minimax_h3_text_encoder_model",
    "minimax_h3_transformer_model",
    "qwen_image_component_source",
    "qwen_image_quantization",
    "qwen_image_qwen_vl_encoder",
    "qwen_image_shift",
    "qwen_image_vae",
    "wan_component_source",
    "wan_guidance_scale_low_noise",
    "wan_t5_encoder_model",
    "wan_transformer_low_noise",
    "z_image_seed_variance_enabled",
    "z_image_seed_variance_randomize_percent",
    "z_image_seed_variance_strength",
    "z_image_shift",
]

CORE_LABELS_STRING = Literal[
    f"{CUSTOM_LABEL}",
    "generation_mode",
    "positive_prompt",
    "positive_style_prompt",
    "negative_prompt",
    "negative_style_prompt",
    "rand_device",
    "init_image",
    "hrf_method",
    "pid_mode",
    "dype_preset",
    "ideogram4_caption",
    "ideogram4_color_palette",
    "ideogram4_sampler_preset",
    "krea2_rebalance_weights",
    "ltx2_conditioning_role",
    "ltx2_schedule",
    "qwen_image_quantization",
    "qwen_image_shift",
    "z_image_shift",
]

CORE_LABELS_INTEGER = Literal[
    f"{CUSTOM_LABEL}",
    "width",
    "height",
    "seed",
    "steps",
    "clip_skip",
    "cfg_scale_start_step",
    "cfg_scale_end_step",
    "refiner_steps",
    "pid_steps",
    "tile_width",
    "tile_height",
    "tile_overlap",
    "num_frames",
    "fps",
    "source_video_start_frame",
    "source_video_end_frame",
    "ideogram4_steps",
    "krea2_seed_variance_randomize_percent",
    "ltx2_base_height",
    "ltx2_base_width",
    "ltx2_context_frames",
    "minimax_h3_hybrid_start_block",
    "z_image_seed_variance_randomize_percent",
]

CORE_LABELS_FLOAT = Literal[
    f"{CUSTOM_LABEL}",
    "cfg_scale",
    "cfg_rescale_multiplier",
    "guidance",
    "denoising_start",
    "denoising_end",
    "strength",
    "shift",
    "dype_scale",
    "dype_exponent",
    "freeu_s1",
    "freeu_s2",
    "freeu_b1",
    "freeu_b2",
    "hidiffusion_t1_ratio",
    "hidiffusion_t2_ratio",
    "hrf_strength",
    "refiner_cfg_scale",
    "refiner_positive_aesthetic_score",
    "refiner_negative_aesthetic_score",
    "refiner_start",
    "fps",
    "ideogram4_guidance_scale",
    "ideogram4_mu",
    "krea2_rebalance_multiplier",
    "krea2_seed_variance_strength",
    "krea2_style_strength",
    "ltx2_audio_cfg_scale",
    "ltx2_guidance_rescale",
    "ltx2_modality_scale",
    "ltx2_noise_scale",
    "ltx2_stg_scale",
    "qwen_image_shift",
    "wan_guidance_scale_low_noise",
    "z_image_seed_variance_strength",
    "z_image_shift",
]

CORE_LABELS_BOOL = Literal[
    f"{CUSTOM_LABEL}",
    "seamless_x",
    "seamless_y",
    "add_noise",
    "hidiffusion",
    "hidiffusion_raunet",
    "hidiffusion_window_attn",
    "hrf_enabled",
    "krea2_rebalance_enabled",
    "krea2_seed_variance_enabled",
    "ltx2_two_stage",
    "z_image_seed_variance_enabled",
]

CORE_LABELS_SCHEDULER = Literal[
    f"{CUSTOM_LABEL}",
    "scheduler",
    "refiner_scheduler",
]

CORE_LABELS_MODEL = Literal[
    f"{CUSTOM_LABEL}",
    "model",
    "refiner_model",
    "pid_decoder",
    "gemma2_encoder",
    "t5_encoder",
    "clip_embed_model",
    "clip_l_model",
    "clip_g_model",
    "mistral_encoder",
    "qwen3_encoder",
    "qwen3_source",
    "qwen3_5_encoder",
    "qwen3_vl_encoder",
    "ideogram4_unconditional_model",
    "ltx2_component_source",
    "ltx2_text_encoder_model",
    "minimax_h3_component_source",
    "minimax_h3_hybrid_base_model",
    "minimax_h3_text_encoder_model",
    "minimax_h3_transformer_model",
    "qwen_image_component_source",
    "qwen_image_qwen_vl_encoder",
    "wan_component_source",
    "wan_t5_encoder_model",
    "wan_transformer_low_noise",
]

CORE_LABELS_VAE = Literal[
    f"{CUSTOM_LABEL}",
    "vae",
    "qwen_image_vae",
]

T = TypeVar("T")


def append_list(item_cls: type[T], new_item: T, items: Union[T, list[T], None] = None) -> list[T]:
    """Combines any number of items or lists into a single list,
    ensuring consistency in type.

    Args:
        item_cls: The expected type of elements in the list.
        items: An existing list or single item of type `item_cls`.
        new_items: Additional item(s) to append. (default=None)

    Returns:
        The updated list containing valid items.

    Raises:
        ValueError: If any item in the list or new_item is not of the expected type.
    """

    if not isinstance(new_item, item_cls):
        raise ValueError(f"Invalid new_item type in: {new_item},  expected {item_cls}")

    if items is None:
        return [new_item]

    result: list[T] = []

    if isinstance(items, item_cls):
        result.append(items)
    elif isinstance(items, list) and all(isinstance(i, item_cls) for i in items):
        result.extend(items)
    else:
        raise ValueError(f"Invalid items type in: {items},  expected {item_cls}")

    result.append(new_item)
    return result


def validate_custom_label(
    model: Union[
        "MetadataItemLinkedInvocation",
        "MetadataToStringInvocation",
        "MetadataToIntegerInvocation",
        "MetadataToFloatInvocation",
        "MetadataToBoolInvocation",
        "MetadataToSchedulerInvocation",
        "MetadataToModelInvocation",
        "MetadataToSDXLModelInvocation",
        "MetadataToVAEInvocation",
    ],
):
    if model.label == CUSTOM_LABEL:
        if model.custom_label is None or model.custom_label.strip() == "":
            raise ValueError("You must enter a Custom Label")
    return model


def extract_model_key(
    metadata: dict[str, Any],
    label: Union[str, None],
    default_key: str,
    model_type: ModelType,
    context: InvocationContext,
) -> str:
    """
    Extracts a model key from the metadata based on the given label.

    Args:
        metadata (dict): The metadata root dictionary.
        label (str): The label to search for.
        default_key (str): The default model key to return if not found.
        model_type (ModelType): model_type to use in the search if a model name_is found in the metadata
        context (object): The context object containing models.

    Returns:
        Model key
    """

    if label in metadata:
        if "key" in metadata[label]:
            if context.models.exists(metadata[label]["key"]):
                return metadata[label]["key"]
        if "name" in metadata[label]:
            search_model = context.models.search_by_attrs(name=metadata[label]["name"], type=model_type)
            if len(search_model) > 0:
                return search_model[0].key
        if "model_name" in metadata[label]:
            search_model = context.models.search_by_attrs(name=metadata[label]["model_name"], type=model_type)
            if len(search_model) > 0:
                return search_model[0].key

    return default_key


def get_model(
    model_key: str,
    context: InvocationContext,
) -> ModelIdentifierField:
    """
    Gets a model based upon a model_key

    Args:
        mode_key (str): The model key to get
        context (object): The context object containing models.

    Returns:
        ModelIdentifierField
    """
    if not context.models.exists(model_key):
        raise Exception(f"Unknown model: {model_key}")

    x = context.models.get_config(model_key)
    return ModelIdentifierField.from_config(x)


@invocation(
    "metadata_item_linked",
    title="Metadata Item Linked",
    tags=["metadata"],
    category="metadata",
    version="1.0.1",
    classification=Classification.Beta,
)
class MetadataItemLinkedInvocation(BaseInvocation, WithMetadata):
    """Used to Create/Add/Update a value into a metadata label"""

    label: CORE_LABELS = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    value: Any = InputField(description=FieldDescriptions.metadata_item_value, ui_type=UIType.Any)

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        k = self.custom_label if self.label == CUSTOM_LABEL else self.label
        v = self.value.vae if isinstance(self.value, VAEField) else self.value

        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        data.update({str(k): v})
        data.update({"app_version": __version__})

        return MetadataOutput(metadata=MetadataField.model_validate(data))


@invocation(
    "metadata_from_image",
    title="Metadata From Image",
    tags=["metadata"],
    category="metadata",
    version="1.0.1",
    classification=Classification.Beta,
)
class MetadataFromImageInvocation(BaseInvocation):
    """Used to create a core metadata item then Add/Update it to the provided metadata"""

    image: ImageField = InputField(description=FieldDescriptions.image)

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        data: Dict[str, Any] = {}
        image_metadata = context.images.get_metadata(self.image.image_name)
        if image_metadata is not None:
            data.update(image_metadata.root)

        return MetadataOutput(metadata=MetadataField.model_validate(data))


@invocation(
    "metadata_to_string",
    title="Metadata To String",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Beta,
)
class MetadataToStringInvocation(BaseInvocation, WithMetadata):
    """Extracts a string value of a label from metadata"""

    label: CORE_LABELS_STRING = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: str = InputField(description="The default string to use if not found in the metadata")

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> StringOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return StringOutput(value=str(output))


@invocation(
    "metadata_to_integer",
    title="Metadata To Integer",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Beta,
)
class MetadataToIntegerInvocation(BaseInvocation, WithMetadata):
    """Extracts an integer value of a label from metadata"""

    label: CORE_LABELS_INTEGER = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: int = InputField(description="The default integer to use if not found in the metadata")

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return IntegerOutput(value=int(output))


@invocation(
    "metadata_to_float",
    title="Metadata To Float",
    tags=["metadata"],
    category="metadata",
    version="1.1.0",
    classification=Classification.Beta,
)
class MetadataToFloatInvocation(BaseInvocation, WithMetadata):
    """Extracts a Float value of a label from metadata"""

    label: CORE_LABELS_FLOAT = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: float = InputField(description="The default float to use if not found in the metadata")

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> FloatOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return FloatOutput(value=float(output))


@invocation(
    "metadata_to_bool",
    title="Metadata To Bool",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Beta,
)
class MetadataToBoolInvocation(BaseInvocation, WithMetadata):
    """Extracts a Boolean value of a label from metadata"""

    label: CORE_LABELS_BOOL = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: bool = InputField(description="The default bool to use if not found in the metadata")

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return BooleanOutput(value=bool(output))


@invocation(
    "metadata_to_scheduler",
    title="Metadata To Scheduler",
    tags=["metadata"],
    category="metadata",
    version="1.0.1",
    classification=Classification.Beta,
)
class MetadataToSchedulerInvocation(BaseInvocation, WithMetadata):
    """Extracts a Scheduler value of a label from metadata"""

    label: CORE_LABELS_SCHEDULER = InputField(
        default="scheduler",
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="The default scheduler to use if not found in the metadata",
        ui_type=UIType.Scheduler,
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> SchedulerOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return SchedulerOutput(scheduler=output)


@invocation_output("metadata_to_model_output")
class MetadataToModelOutput(BaseInvocationOutput):
    """String to main model output"""

    model: ModelIdentifierField = OutputField(
        description=FieldDescriptions.main_model,
        title="Model",
    )
    name: str = OutputField(description="Model Name", title="Name")
    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    clip: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP")


@invocation_output("metadata_to_sdxl_model_output")
class MetadataToSDXLModelOutput(BaseInvocationOutput):
    """String to SDXL main model output"""

    model: ModelIdentifierField = OutputField(
        description=FieldDescriptions.main_model,
        title="Model",
    )
    name: str = OutputField(description="Model Name", title="Name")
    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")
    clip: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP 1")
    clip2: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP 2")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "metadata_to_model",
    title="Metadata To Model",
    tags=["metadata"],
    category="metadata",
    version="1.3.0",
    classification=Classification.Beta,
)
class MetadataToModelInvocation(BaseInvocation, WithMetadata):
    """Extracts a Model value of a label from metadata"""

    label: CORE_LABELS_MODEL = InputField(
        default="model",
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: ModelIdentifierField = InputField(
        description="The default model to use if not found in the metadata", ui_model_type=ModelType.Main
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> MetadataToModelOutput:
        data = {} if self.metadata is None else self.metadata.root
        label = self.custom_label if self.label == CUSTOM_LABEL else self.label

        model_key = extract_model_key(data, label, self.default_value.key, ModelType.Main, context)
        model = get_model(model_key, context)

        return MetadataToModelOutput(
            model=model,
            name=f"{model.base}: {model.name}",
            unet=UNetField(
                unet=model.model_copy(update={"submodel_type": SubModelType.UNet}),
                scheduler=model.model_copy(update={"submodel_type": SubModelType.Scheduler}),
                loras=[],
            ),
            clip=CLIPField(
                tokenizer=model.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
                text_encoder=model.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
                loras=[],
                skipped_layers=0,
            ),
            vae=VAEField(
                vae=model.model_copy(update={"submodel_type": SubModelType.VAE}),
            ),
        )


@invocation(
    "metadata_to_sdxl_model",
    title="Metadata To SDXL Model",
    tags=["metadata"],
    category="metadata",
    version="1.3.0",
    classification=Classification.Beta,
)
class MetadataToSDXLModelInvocation(BaseInvocation, WithMetadata):
    """Extracts a SDXL Model value of a label from metadata"""

    label: CORE_LABELS_MODEL = InputField(
        default="model",
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: ModelIdentifierField = InputField(
        description="The default SDXL Model to use if not found in the metadata",
        ui_model_type=ModelType.Main,
        ui_model_base=BaseModelType.StableDiffusionXL,
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> MetadataToSDXLModelOutput:
        data = {} if self.metadata is None else self.metadata.root
        label = self.custom_label if self.label == CUSTOM_LABEL else self.label

        model_key = extract_model_key(data, label, self.default_value.key, ModelType.Main, context)
        model = get_model(model_key, context)

        return MetadataToSDXLModelOutput(
            model=model,
            name=f"{model.base}: {model.name}",
            unet=UNetField(
                unet=model.model_copy(update={"submodel_type": SubModelType.UNet}),
                scheduler=model.model_copy(update={"submodel_type": SubModelType.Scheduler}),
                loras=[],
            ),
            clip=CLIPField(
                tokenizer=model.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
                text_encoder=model.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
                loras=[],
                skipped_layers=0,
            ),
            clip2=CLIPField(
                tokenizer=model.model_copy(update={"submodel_type": SubModelType.Tokenizer2}),
                text_encoder=model.model_copy(update={"submodel_type": SubModelType.TextEncoder2}),
                loras=[],
                skipped_layers=0,
            ),
            vae=VAEField(
                vae=model.model_copy(update={"submodel_type": SubModelType.VAE}),
            ),
        )


@invocation(
    "denoise_latents_meta",
    title=f"{DenoiseLatentsInvocation.UIConfig.title} + Metadata",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="metadata",
    version="1.6.0",
    classification=Classification.Deprecated,
)
class DenoiseLatentsMetaInvocation(DenoiseLatentsInvocation):
    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        obj = super().invoke(context)
        metadata = getattr(obj, "metadata", None) or self._build_metadata(obj.width, obj.height)
        return LatentsMetaOutput(
            latents=obj.latents,
            width=obj.width,
            height=obj.height,
            metadata=metadata,
        )


@invocation(
    "flux_denoise_meta",
    title=f"{FluxDenoiseInvocation.UIConfig.title} + Metadata",
    tags=["flux", "latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="metadata",
    version="1.2.0",
    classification=Classification.Deprecated,
)
class FluxDenoiseLatentsMetaInvocation(FluxDenoiseInvocation):
    """Run denoising process with a FLUX transformer model + metadata."""

    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        obj = super().invoke(context)
        metadata = getattr(obj, "metadata", None) or self._build_metadata()
        return LatentsMetaOutput(
            latents=obj.latents,
            width=obj.width,
            height=obj.height,
            metadata=metadata,
        )


@invocation(
    "z_image_denoise_meta",
    title=f"{ZImageDenoiseInvocation.UIConfig.title} + Metadata",
    tags=["z-image", "latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="metadata",
    version="1.2.0",
    classification=Classification.Deprecated,
)
class ZImageDenoiseMetaInvocation(ZImageDenoiseInvocation):
    """Run denoising process with a Z-Image transformer model + metadata."""

    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        obj = super().invoke(context)
        metadata = getattr(obj, "metadata", None) or self._build_metadata()
        return LatentsMetaOutput(
            latents=obj.latents,
            width=obj.width,
            height=obj.height,
            metadata=metadata,
        )


@invocation(
    "anima_denoise_meta",
    title=f"{AnimaDenoiseInvocation.UIConfig.title} + Metadata",
    tags=["anima", "latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="metadata",
    version="1.1.0",
    classification=Classification.Deprecated,
)
class AnimaDenoiseMetaInvocation(AnimaDenoiseInvocation):
    """Run denoising process with an Anima transformer model + metadata."""

    # Anima loads its VAE and text encoder as standalone models, so - unlike SD/FLUX, where they come out of
    # the main model - they cannot be derived from the transformer field. Accept them here so a workflow can
    # record the full component set the recall UI expects, without chaining extra Metadata Item Linked nodes.
    vae: Optional[VAEField] = InputField(
        default=None,
        description="Optional. The Anima VAE, recorded to metadata so recall can restore the VAE selection.",
        input=Input.Connection,
        title="VAE",
    )
    qwen3_encoder: Optional[Qwen3EncoderField] = InputField(
        default=None,
        description="Optional. The Anima Qwen3 encoder, recorded to metadata so recall can restore the encoder "
        "selection.",
        input=Input.Connection,
        title="Qwen3 Encoder",
    )

    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        obj = super().invoke(context)
        metadata = getattr(obj, "metadata", None) or self._build_metadata()
        extra: dict[str, Any] = {}
        if self.vae is not None:
            extra["vae"] = self.vae.vae
        if self.qwen3_encoder is not None:
            extra["qwen3_encoder"] = self.qwen3_encoder.text_encoder
        if extra:
            metadata = build_denoise_metadata(metadata, extra)
        return LatentsMetaOutput(
            latents=obj.latents,
            width=obj.width,
            height=obj.height,
            metadata=metadata,
        )


@invocation(
    "metadata_to_vae",
    title="Metadata To VAE",
    tags=["metadata"],
    category="metadata",
    version="1.2.1",
    classification=Classification.Beta,
)
class MetadataToVAEInvocation(BaseInvocation, WithMetadata):
    """Extracts a VAE value of a label from metadata"""

    label: CORE_LABELS_VAE = InputField(
        default="vae",
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: VAEField = InputField(
        description="The default VAE to use if not found in the metadata",
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> VAEOutput:
        data = {} if self.metadata is None else self.metadata.root
        label = self.custom_label if self.label == CUSTOM_LABEL else self.label

        model_key = extract_model_key(data, label, self.default_value.vae.key, ModelType.VAE, context)
        model = get_model(model_key, context)
        model.submodel_type = SubModelType.VAE

        return VAEOutput(vae=VAEField(vae=model))


@invocation_output("metadata_to_lora_collection_output")
class MetadataToLorasCollectionOutput(BaseInvocationOutput):
    """Model loader output"""

    lora: list[LoRAField] = OutputField(description="Collection of LoRA model and weights", title="LoRAs")


@invocation(
    "metadata_to_lora_collection",
    title="Metadata To LoRA Collection",
    tags=["metadata"],
    category="metadata",
    version="1.1.0",
    classification=Classification.Beta,
)
class MetadataToLorasCollectionInvocation(BaseInvocation, WithMetadata):
    """Extracts Lora(s) from metadata into a collection"""

    custom_label: str = InputField(
        default="loras",
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=[], description="LoRA models and weights. May be a single LoRA or collection.", title="LoRAs"
    )

    def invoke(self, context: InvocationContext) -> MetadataToLorasCollectionOutput:
        metadata = {} if self.metadata is None else self.metadata.root
        key: str = self.custom_label.strip()
        if not key:
            key = "loras"

        if key in metadata:
            loras = metadata[key]
        else:
            loras = []

        input_loras = self.loras if isinstance(self.loras, list) else [self.loras]
        output = MetadataToLorasCollectionOutput(lora=[])
        added_loras: list[str] = []

        for lora in input_loras:
            assert lora is LoRAField
            if lora.lora.key in added_loras:
                continue
            output.lora.append(lora)
            added_loras.append(lora.lora.key)

        for lora in loras:
            model_key = extract_model_key(lora, "model", "", ModelType.LoRA, context)
            if not model_key:
                model_key = extract_model_key(lora, "lora", "", ModelType.LoRA, context)
            if model_key:
                model = get_model(model_key, context)
                weight = float(lora["weight"])
                if model.key in added_loras:
                    continue
                output.lora.append(LoRAField(lora=model, weight=weight))

        return output


@invocation(
    "metadata_to_loras",
    title="Metadata To LoRAs",
    tags=["metadata"],
    category="metadata",
    version="1.1.1",
    classification=Classification.Beta,
)
class MetadataToLorasInvocation(BaseInvocation, WithMetadata):
    """Extracts a Loras value of a label from metadata"""

    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    clip: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP",
    )

    def invoke(self, context: InvocationContext) -> LoRALoaderOutput:
        data = {} if self.metadata is None else self.metadata.root
        key = "loras"
        if key in data:
            loras = data[key]
        else:
            loras = []

        output = LoRALoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)

        for lora in loras:
            model_key = extract_model_key(lora, "model", "", ModelType.LoRA, context)
            if model_key != "":
                model = get_model(model_key, context)
                weight = float(lora["weight"])

                if output.unet is not None:
                    if any(lora.lora.key == model_key for lora in output.unet.loras):
                        context.logger.info(f'LoRA "{model_key}" already applied to unet')
                    else:
                        output.unet.loras.append(
                            LoRAField(
                                lora=model,
                                weight=weight,
                            )
                        )

                if output.clip is not None:
                    if any(lora.lora.key == model_key for lora in output.clip.loras):
                        context.logger.info(f'LoRA "{model_key}" already applied to clip')
                    else:
                        output.clip.loras.append(
                            LoRAField(
                                lora=model,
                                weight=weight,
                            )
                        )

        return output


@invocation(
    "metadata_to_sdlx_loras",
    title="Metadata To SDXL LoRAs",
    tags=["metadata"],
    category="metadata",
    version="1.1.1",
    classification=Classification.Beta,
)
class MetadataToSDXLLorasInvocation(BaseInvocation, WithMetadata):
    """Extracts a SDXL Loras value of a label from metadata"""

    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    clip: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 1",
    )
    clip2: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 2",
    )

    def invoke(self, context: InvocationContext) -> SDXLLoRALoaderOutput:
        data = {} if self.metadata is None else self.metadata.root
        key = "loras"
        if key in data:
            loras = data[key]
        else:
            loras = []

        output = SDXLLoRALoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)

        if self.clip2 is not None:
            output.clip2 = copy.deepcopy(self.clip2)

        for lora in loras:
            model_key = extract_model_key(lora, "model", "", ModelType.LoRA, context)
            if model_key != "":
                model = get_model(model_key, context)
                weight = float(lora["weight"])

                if output.unet is not None:
                    if any(lora.lora.key == model_key for lora in output.unet.loras):
                        context.logger.info(f'LoRA "{model_key}" already applied to unet')
                    else:
                        output.unet.loras.append(
                            LoRAField(
                                lora=model,
                                weight=weight,
                            )
                        )

                if output.clip is not None:
                    if any(lora.lora.key == model_key for lora in output.clip.loras):
                        context.logger.info(f'LoRA "{model_key}" already applied to clip')
                    else:
                        output.clip.loras.append(
                            LoRAField(
                                lora=model,
                                weight=weight,
                            )
                        )

                if output.clip2 is not None:
                    if any(lora.lora.key == model_key for lora in output.clip2.loras):
                        context.logger.info(f'LoRA "{model_key}" already applied to clip')
                    else:
                        output.clip2.loras.append(
                            LoRAField(
                                lora=model,
                                weight=weight,
                            )
                        )

        return output


def _resolve_ref_image_field(img_container: Any) -> Optional[dict[str, Any]]:
    if isinstance(img_container, dict):
        crop = img_container.get("crop")
        if isinstance(crop, dict) and isinstance(crop.get("image"), dict) and crop["image"].get("image_name"):
            return {"image_name": crop["image"]["image_name"]}
        orig = img_container.get("original")
        if isinstance(orig, dict) and isinstance(orig.get("image"), dict) and orig["image"].get("image_name"):
            return {"image_name": orig["image"]["image_name"]}
        if img_container.get("image_name"):
            return {"image_name": img_container["image_name"]}
    return None


def _extract_canvas_control_layer_image(entity: dict[str, Any]) -> Optional[dict[str, Any]]:
    for obj in entity.get("objects") or []:
        if isinstance(obj, dict) and obj.get("type") == "image":
            obj_img = obj.get("image")
            if isinstance(obj_img, dict) and obj_img.get("image_name"):
                return {"image_name": obj_img["image_name"]}
            if obj.get("image_name"):
                return {"image_name": obj["image_name"]}
    return None


def _extract_controlnets_from_metadata(data: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in data.get("controlnets") or []:
        if not isinstance(item, dict):
            continue
        img = item.get("image") or ({"image_name": item["image_name"]} if item.get("image_name") else None)
        if img is None or ("control_model" not in item and "model" not in item):
            continue
        normalized = dict(item)
        normalized["image"] = img
        result.append(normalized)
    if not result:
        canvas_v2 = data.get("canvas_v2_metadata")
        control_layers = canvas_v2.get("controlLayers") if isinstance(canvas_v2, dict) else None
        entities = control_layers.get("entities", []) if isinstance(control_layers, dict) else []
        for entity in entities:
            if not isinstance(entity, dict) or entity.get("isEnabled", True) is False:
                continue
            adapter = entity.get("controlAdapter")
            if not isinstance(adapter, dict) or adapter.get("type") != "controlnet":
                continue
            model = adapter.get("model")
            img = _extract_canvas_control_layer_image(entity)
            if not model or img is None:
                continue
            begin_end = adapter.get("beginEndStepPct") or [0.0, 1.0]
            result.append(
                {
                    "image": img,
                    "control_model": model,
                    "control_weight": adapter.get("weight", 1.0),
                    "begin_step_percent": begin_end[0] if len(begin_end) > 0 else 0.0,
                    "end_step_percent": begin_end[1] if len(begin_end) > 1 else 1.0,
                    "control_mode": adapter.get("controlMode", "balanced"),
                    "resize_mode": "just_resize",
                }
            )
    return result


def _extract_ip_adapters_from_metadata(data: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ref_images = data.get("ref_images")
    if isinstance(ref_images, list):
        for entry in ref_images:
            if not isinstance(entry, dict) or entry.get("isEnabled", True) is False:
                continue
            cfg = entry.get("config")
            if not isinstance(cfg, dict) or cfg.get("type") != "ip_adapter":
                continue
            img = _resolve_ref_image_field(cfg.get("image"))
            model = cfg.get("model")
            if img is None or not model:
                continue
            begin_end = cfg.get("beginEndStepPct") or [0.0, 1.0]
            raw_weight = cfg.get("weight", 1.0)
            weight = float(raw_weight[0] if isinstance(raw_weight, list) and raw_weight else raw_weight)
            result.append(
                {
                    "image": img,
                    "ip_adapter_model": model,
                    "weight": weight,
                    "begin_step_percent": begin_end[0] if len(begin_end) > 0 else 0.0,
                    "end_step_percent": begin_end[1] if len(begin_end) > 1 else 1.0,
                    "clip_vision_model": cfg.get("clipVisionModel", "ViT-H"),
                    "method": cfg.get("method", "full"),
                }
            )
    if not result:
        canvas_v2 = data.get("canvas_v2_metadata")
        ref_imgs = canvas_v2.get("referenceImages") if isinstance(canvas_v2, dict) else None
        entities = ref_imgs.get("entities", []) if isinstance(ref_imgs, dict) else []
        for entity in entities:
            if not isinstance(entity, dict) or entity.get("isEnabled", True) is False:
                continue
            ipa_cfg = entity.get("ipAdapter") or entity.get("config")
            if not isinstance(ipa_cfg, dict):
                continue
            if "type" in ipa_cfg and ipa_cfg.get("type") != "ip_adapter":
                continue
            img = _resolve_ref_image_field(ipa_cfg.get("image"))
            model = ipa_cfg.get("model")
            if img is None or not model:
                continue
            begin_end = ipa_cfg.get("beginEndStepPct") or [0.0, 1.0]
            raw_weight = ipa_cfg.get("weight", 1.0)
            weight = float(raw_weight[0] if isinstance(raw_weight, list) and raw_weight else raw_weight)
            result.append(
                {
                    "image": img,
                    "ip_adapter_model": model,
                    "weight": weight,
                    "begin_step_percent": begin_end[0] if len(begin_end) > 0 else 0.0,
                    "end_step_percent": begin_end[1] if len(begin_end) > 1 else 1.0,
                    "clip_vision_model": ipa_cfg.get("clipVisionModel", "ViT-H"),
                    "method": ipa_cfg.get("method", "full"),
                }
            )
    if not result:
        for x in data.get("ipAdapters") or []:
            if not isinstance(x, dict):
                continue
            raw_img = x.get("image")
            if raw_img is None:
                continue
            img_list = raw_img if isinstance(raw_img, list) else [raw_img]
            cvm = x.get("clip_vision_model") or x.get("clipVisionModel")
            if cvm not in ("ViT-H", "ViT-G", "ViT-L"):
                enc = x.get("image_encoder_model")
                enc_name = str(enc.get("name", "") if isinstance(enc, dict) else getattr(enc, "name", "")).lower()
                if "vit-l" in enc_name or "large" in enc_name:
                    cvm = "ViT-L"
                elif "sdxl" in enc_name or "vit-g" in enc_name:
                    cvm = "ViT-G"
                else:
                    cvm = "ViT-H"
            raw_weight = x.get("weight", 1.0)
            weight = float(raw_weight[0] if isinstance(raw_weight, list) and raw_weight else raw_weight)
            for img in img_list:
                normalized = dict(x)
                normalized["image"] = img
                normalized["weight"] = weight
                normalized["clip_vision_model"] = cvm
                normalized["method"] = x.get("method") or "full"
                result.append(normalized)
    return result


def _extract_t2i_adapters_from_metadata(data: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in data.get("t2iAdapters") or []:
        if not isinstance(item, dict):
            continue
        img = item.get("image") or ({"image_name": item["image_name"]} if item.get("image_name") else None)
        if img is None or ("t2i_adapter_model" not in item and "model" not in item):
            continue
        normalized = dict(item)
        normalized["image"] = img
        result.append(normalized)
    if not result:
        canvas_v2 = data.get("canvas_v2_metadata")
        control_layers = canvas_v2.get("controlLayers") if isinstance(canvas_v2, dict) else None
        entities = control_layers.get("entities", []) if isinstance(control_layers, dict) else []
        for entity in entities:
            if not isinstance(entity, dict) or entity.get("isEnabled", True) is False:
                continue
            adapter = entity.get("controlAdapter")
            if not isinstance(adapter, dict) or adapter.get("type") != "t2i_adapter":
                continue
            model = adapter.get("model")
            img = _extract_canvas_control_layer_image(entity)
            if not model or img is None:
                continue
            begin_end = adapter.get("beginEndStepPct") or [0.0, 1.0]
            result.append(
                {
                    "image": img,
                    "t2i_adapter_model": model,
                    "weight": adapter.get("weight", 1.0),
                    "begin_step_percent": begin_end[0] if len(begin_end) > 0 else 0.0,
                    "end_step_percent": begin_end[1] if len(begin_end) > 1 else 1.0,
                    "resize_mode": "just_resize",
                }
            )
    return result


@invocation_output("md_control_list_output")
class MDControlListOutput(BaseInvocationOutput):
    # Outputs
    control_list: Optional[Union[ControlField, list[ControlField]]] = OutputField(
        description=FieldDescriptions.control,
        title="ControlNet-List",
    )


@invocation(
    "metadata_to_controlnets",
    title="Metadata To ControlNets",
    tags=["metadata"],
    category="metadata",
    version="1.2.0",
    classification=Classification.Beta,
)
class MetadataToControlnetsInvocation(BaseInvocation, WithMetadata):
    """Extracts a Controlnets value of a label from metadata"""

    control_list: Optional[Union[ControlField, list[ControlField]]] = InputField(
        default=None,
        title="ControlNet-List",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> MDControlListOutput:
        data = {} if self.metadata is None else self.metadata.root
        md_controls = _extract_controlnets_from_metadata(data)

        controls: Optional[Union[ControlField, list[ControlField]]]

        if self.control_list is not None:
            controls = self.control_list
        else:
            controls = []

        for x in md_controls:
            model_key = extract_model_key(x, "control_model", "", ModelType.ControlNet, context) or extract_model_key(
                x, "model", "", ModelType.ControlNet, context
            )
            if not model_key:
                continue
            model = get_model(model_key, context)
            if model.base not in (
                BaseModelType.StableDiffusion1,
                BaseModelType.StableDiffusion2,
                BaseModelType.StableDiffusionXL,
            ):
                continue

            cn = ControlNetInvocation(
                image=x["image"],
                control_model=model,
                control_weight=x.get("control_weight", x.get("weight", 1.0)),
                begin_step_percent=x.get("begin_step_percent", 0.0),
                end_step_percent=x.get("end_step_percent", 1.0),
                control_mode=x.get("control_mode", "balanced"),
                resize_mode=x.get("resize_mode", "just_resize"),
            )
            i = cn.invoke(context)

            controls = append_list(ControlField, i.control, controls)

        return MDControlListOutput(control_list=controls)


@invocation_output("md_ip_adapter_list_output")
class MDIPAdapterListOutput(BaseInvocationOutput):
    # Outputs
    ip_adapter_list: Optional[Union[IPAdapterField, list[IPAdapterField]]] = OutputField(
        description=FieldDescriptions.ip_adapter, title="IP-Adapter-List"
    )


@invocation(
    "metadata_to_ip_adapters",
    title="Metadata To IP-Adapters",
    tags=["metadata"],
    category="metadata",
    version="1.2.0",
    classification=Classification.Beta,
)
class MetadataToIPAdaptersInvocation(BaseInvocation, WithMetadata):
    """Extracts a IP-Adapters value of a label from metadata"""

    ip_adapter_list: Optional[Union[IPAdapterField, list[IPAdapterField]]] = InputField(
        description=FieldDescriptions.ip_adapter,
        title="IP-Adapter-List",
        default=None,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> MDIPAdapterListOutput:
        data = {} if self.metadata is None else self.metadata.root
        md_adapters = _extract_ip_adapters_from_metadata(data)

        adapters: Optional[Union[IPAdapterField, list[IPAdapterField]]]

        if self.ip_adapter_list is not None:
            adapters = self.ip_adapter_list
        else:
            adapters = []

        for x in md_adapters:
            model_key = extract_model_key(x, "ip_adapter_model", "", ModelType.IPAdapter, context) or extract_model_key(
                x, "model", "", ModelType.IPAdapter, context
            )
            if not model_key:
                continue
            model = get_model(model_key, context)
            if model.base not in (BaseModelType.StableDiffusion1, BaseModelType.StableDiffusionXL):
                continue

            ipa = IPAdapterInvocation(
                image=x["image"],
                ip_adapter_model=model,
                clip_vision_model=x.get("clip_vision_model", "ViT-H"),
                method=x.get("method", "full"),
                weight=x.get("weight", 1.0),
                begin_step_percent=x.get("begin_step_percent", 0.0),
                end_step_percent=x.get("end_step_percent", 1.0),
            )
            i = ipa.invoke(context)

            adapters = append_list(IPAdapterField, i.ip_adapter, adapters)

        return MDIPAdapterListOutput(ip_adapter_list=adapters)


@invocation_output("md_ip_adapters_output")
class MDT2IAdapterListOutput(BaseInvocationOutput):
    # Outputs
    t2i_adapter_list: Optional[Union[T2IAdapterField, list[T2IAdapterField]]] = OutputField(
        description=FieldDescriptions.t2i_adapter, title="T2I Adapter-List"
    )


@invocation(
    "metadata_to_t2i_adapters",
    title="Metadata To T2I-Adapters",
    tags=["metadata"],
    category="metadata",
    version="1.2.0",
    classification=Classification.Beta,
)
class MetadataToT2IAdaptersInvocation(BaseInvocation, WithMetadata):
    """Extracts a T2I-Adapters value of a label from metadata"""

    t2i_adapter_list: Optional[Union[T2IAdapterField, list[T2IAdapterField]]] = InputField(
        description=FieldDescriptions.ip_adapter,
        title="T2I-Adapter",
        default=None,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> MDT2IAdapterListOutput:
        data = {} if self.metadata is None else self.metadata.root
        md_adapters = _extract_t2i_adapters_from_metadata(data)

        adapters: Optional[Union[T2IAdapterField, list[T2IAdapterField]]]

        if self.t2i_adapter_list is not None:
            adapters = self.t2i_adapter_list
        else:
            adapters = []

        for x in md_adapters:
            model_key = extract_model_key(
                x, "t2i_adapter_model", "", ModelType.T2IAdapter, context
            ) or extract_model_key(x, "model", "", ModelType.T2IAdapter, context)
            if not model_key:
                continue
            model = get_model(model_key, context)
            if model.base not in (BaseModelType.StableDiffusion1, BaseModelType.StableDiffusionXL):
                continue

            t2i = T2IAdapterInvocation(
                image=x["image"],
                t2i_adapter_model=model,
                weight=x.get("weight", 1.0),
                begin_step_percent=x.get("begin_step_percent", 0.0),
                end_step_percent=x.get("end_step_percent", 1.0),
                resize_mode=x.get("resize_mode", "just_resize"),
            )
            i = t2i.invoke(context)

            adapters = append_list(T2IAdapterField, i.t2i_adapter, adapters)

        return MDT2IAdapterListOutput(t2i_adapter_list=adapters)


@invocation(
    "metadata_to_string_collection",
    title="Metadata To String Collection",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Beta,
)
class MetadataToStringCollectionInvocation(BaseInvocation, WithMetadata):
    """Extracts a string collection value of a label from metadata"""

    label: CORE_LABELS_STRING = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: list[str] = InputField(
        description="The default string collection to use if not found in the metadata"
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return StringCollectionOutput(collection=output)


@invocation(
    "metadata_to_integer_collection",
    title="Metadata To Integer Collection",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Beta,
)
class MetadataToIntegerCollectionInvocation(BaseInvocation, WithMetadata):
    """Extracts an integer value Collection of a label from metadata"""

    label: CORE_LABELS_INTEGER = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: list[int] = InputField(description="The default integer to use if not found in the metadata")

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return IntegerCollectionOutput(collection=output)


@invocation(
    "metadata_to_float_collection",
    title="Metadata To Float Collection",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Beta,
)
class MetadataToFloatCollectionInvocation(BaseInvocation, WithMetadata):
    """Extracts a Float value Collection of a label from metadata"""

    label: CORE_LABELS_FLOAT = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: list[float] = InputField(description="The default float to use if not found in the metadata")

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return FloatCollectionOutput(collection=output)


@invocation(
    "metadata_to_bool_collection",
    title="Metadata To Bool Collection",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Beta,
)
class MetadataToBoolCollectionInvocation(BaseInvocation, WithMetadata):
    """Extracts a Boolean value Collection of a label from metadata"""

    label: CORE_LABELS_BOOL = InputField(
        default=CUSTOM_LABEL,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    custom_label: Optional[str] = InputField(
        default=None,
        description=FieldDescriptions.metadata_item_label,
        input=Input.Direct,
    )
    default_value: list[bool] = InputField(description="The default bool to use if not found in the metadata")

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> BooleanCollectionOutput:
        data: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        output = data.get(str(self.custom_label if self.label == CUSTOM_LABEL else self.label), self.default_value)

        return BooleanCollectionOutput(collection=output)

# Adopted from @skunworkxdark's metadata nodes (MIT License)
# https://github.com/skunkworxdark/metadata-linked-nodes
# Thanks to @skunworkxdark for the original implementation!

import copy
from typing import Any, Dict, Literal, Optional, TypeVar, Union

from pydantic import model_validator

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.controlnet_image_processors import ControlField, ControlNetInvocation
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation
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
from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.ip_adapter import IPAdapterField, IPAdapterInvocation
from invokeai.app.invocations.metadata import LoRAMetadataField, MetadataOutput
from invokeai.app.invocations.model import (
    CLIPField,
    LoRAField,
    LoRALoaderOutput,
    ModelIdentifierField,
    SDXLLoRALoaderOutput,
    UNetField,
    VAEField,
    VAEOutput,
)
from invokeai.app.invocations.primitives import BooleanOutput, FloatOutput, IntegerOutput, LatentsOutput, StringOutput
from invokeai.app.invocations.scheduler import SchedulerOutput
from invokeai.app.invocations.t2i_adapter import T2IAdapterField, T2IAdapterInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import ModelType, SubModelType
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.version import __version__

CUSTOM_LABEL: str = "* CUSTOM LABEL *"

CORE_LABELS = Literal[
    f"{CUSTOM_LABEL}",
    "positive_prompt",
    "positive_style_prompt",
    "negative_prompt",
    "negative_style_prompt",
    "width",
    "height",
    "seed",
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
]

CORE_LABELS_STRING = Literal[
    f"{CUSTOM_LABEL}",
    "positive_prompt",
    "positive_style_prompt",
    "negative_prompt",
    "negative_style_prompt",
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
]

CORE_LABELS_FLOAT = Literal[
    f"{CUSTOM_LABEL}",
    "cfg_scale",
    "cfg_rescale_multiplier",
    "guidance",
]

CORE_LABELS_BOOL = Literal[
    f"{CUSTOM_LABEL}",
    "seamless_x",
    "seamless_y",
]

CORE_LABELS_SCHEDULER = Literal[
    f"{CUSTOM_LABEL}",
    "scheduler",
]

CORE_LABELS_MODEL = Literal[
    f"{CUSTOM_LABEL}",
    "model",
]

CORE_LABELS_VAE = Literal[
    f"{CUSTOM_LABEL}",
    "vae",
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
        ui_type=UIType.MainModel,
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
        ui_type=UIType.SDXLMainModel,
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
        description="The default model to use if not found in the metadata",
        ui_type=UIType.MainModel,
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
        ui_type=UIType.SDXLMainModel,
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


@invocation_output("latents_meta_output")
class LatentsMetaOutput(LatentsOutput, MetadataOutput):
    """Latents + metadata"""


@invocation(
    "denoise_latents_meta",
    title=f"{DenoiseLatentsInvocation.UIConfig.title} + Metadata",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.1.1",
)
class DenoiseLatentsMetaInvocation(DenoiseLatentsInvocation, WithMetadata):
    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        def _to_json(obj: Union[Any, list[Any]]):
            if not isinstance(obj, list):
                obj = [obj]

            return [
                item.model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
                for item in obj
            ]

        def _loras_to_json(obj: Union[Any, list[Any]]):
            if not isinstance(obj, list):
                obj = [obj]

            output: list[dict[str, Any]] = []
            for item in obj:
                output.append(
                    LoRAMetadataField(
                        model=item.lora,
                        weight=item.weight,
                    ).model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
                )
            return output

        obj = super().invoke(context)

        md: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        md.update({"width": obj.width})
        md.update({"height": obj.height})
        md.update({"steps": self.steps})
        md.update({"cfg_scale": self.cfg_scale})
        md.update({"cfg_rescale_multiplier": self.cfg_rescale_multiplier})
        md.update({"denoising_start": self.denoising_start})
        md.update({"denoising_end": self.denoising_end})
        md.update({"scheduler": self.scheduler})
        md.update({"model": self.unet.unet})
        if isinstance(self.control, ControlField) or (isinstance(self.control, list) and len(self.control) > 0):
            md.update({"controlnets": _to_json(self.control)})
        if isinstance(self.ip_adapter, IPAdapterField) or (
            isinstance(self.ip_adapter, list) and len(self.ip_adapter) > 0
        ):
            md.update({"ipAdapters": _to_json(self.ip_adapter)})
        if isinstance(self.t2i_adapter, T2IAdapterField) or (
            isinstance(self.t2i_adapter, list) and len(self.t2i_adapter) > 0
        ):
            md.update({"t2iAdapters": _to_json(self.t2i_adapter)})
        if len(self.unet.loras) > 0:
            md.update({"loras": _loras_to_json(self.unet.loras)})
        if self.noise is not None:
            md.update({"seed": self.noise.seed})

        params = obj.__dict__.copy()
        del params["type"]

        return LatentsMetaOutput(**params, metadata=MetadataField.model_validate(md))


@invocation(
    "flux_denoise_meta",
    title=f"{FluxDenoiseInvocation.UIConfig.title} + Metadata",
    tags=["flux", "latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.0.1",
)
class FluxDenoiseLatentsMetaInvocation(FluxDenoiseInvocation, WithMetadata):
    """Run denoising process with a FLUX transformer model + metadata."""

    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        def _loras_to_json(obj: Union[Any, list[Any]]):
            if not isinstance(obj, list):
                obj = [obj]

            output: list[dict[str, Any]] = []
            for item in obj:
                output.append(
                    LoRAMetadataField(
                        model=item.lora,
                        weight=item.weight,
                    ).model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
                )
            return output

        obj = super().invoke(context)

        md: Dict[str, Any] = {} if self.metadata is None else self.metadata.root
        md.update({"width": obj.width})
        md.update({"height": obj.height})
        md.update({"steps": self.num_steps})
        md.update({"guidance": self.guidance})
        md.update({"denoising_start": self.denoising_start})
        md.update({"denoising_end": self.denoising_end})
        md.update({"model": self.transformer.transformer})
        md.update({"seed": self.seed})
        md.update({"cfg_scale": self.cfg_scale})
        md.update({"cfg_scale_start_step": self.cfg_scale_start_step})
        md.update({"cfg_scale_end_step": self.cfg_scale_end_step})
        if len(self.transformer.loras) > 0:
            md.update({"loras": _loras_to_json(self.transformer.loras)})

        params = obj.__dict__.copy()
        del params["type"]

        return LatentsMetaOutput(**params, metadata=MetadataField.model_validate(md))


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
        key = "controlnets"
        if key in data:
            md_controls = data[key]
        else:
            md_controls = []

        controls: Optional[Union[ControlField, list[ControlField]]]

        if self.control_list is not None:
            controls = self.control_list
        else:
            controls = []

        for x in md_controls:
            model_key = extract_model_key(x, "control_model", "", ModelType.ControlNet, context)
            model = get_model(model_key, context)

            cn = ControlNetInvocation(
                image=x["image"],
                control_model=model,
                control_weight=x["control_weight"],
                begin_step_percent=x["begin_step_percent"],
                end_step_percent=x["end_step_percent"],
                control_mode=x["control_mode"],
                resize_mode=x["resize_mode"],
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
        key = "ipAdapters"
        if key in data:
            md_adapters = data[key]
        else:
            md_adapters = []

        adapters: Optional[Union[IPAdapterField, list[IPAdapterField]]]

        if self.ip_adapter_list is not None:
            adapters = self.ip_adapter_list
        else:
            adapters = []

        for x in md_adapters:
            model_key = extract_model_key(x, "ip_adapter_model", "", ModelType.IPAdapter, context)
            model = get_model(model_key, context)

            ipa = IPAdapterInvocation(
                image=x["image"],
                ip_adapter_model=model,
                weight=x["weight"],
                begin_step_percent=x["begin_step_percent"],
                end_step_percent=x["end_step_percent"],
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
        key = "t2iAdapters"
        if key in data:
            md_adapters = data[key]
        else:
            md_adapters = []

        adapters: Optional[Union[T2IAdapterField, list[T2IAdapterField]]]

        if self.t2i_adapter_list is not None:
            adapters = self.t2i_adapter_list
        else:
            adapters = []

        for x in md_adapters:
            model_key = extract_model_key(x, "t2i_adapter_model", "", ModelType.T2IAdapter, context)
            model = get_model(model_key, context)

            t2i = T2IAdapterInvocation(
                image=x["image"],
                t2i_adapter_model=model,
                weight=x["weight"],
                begin_step_percent=x["begin_step_percent"],
                end_step_percent=x["end_step_percent"],
                resize_mode=x["resize_mode"],
            )
            i = t2i.invoke(context)

            adapters = append_list(T2IAdapterField, i.t2i_adapter, adapters)

        return MDT2IAdapterListOutput(t2i_adapter_list=adapters)

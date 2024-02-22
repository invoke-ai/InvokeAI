import copy
from typing import Any, Literal, Optional, Union

from pydantic import model_validator

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    Input,
    InputField,
    InvocationContext,
    MetadataField,
    OutputField,
    UIType,
    WithMetadata,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.ip_adapter import (
    IPAdapterField,
    IPAdapterInvocation,
    IPAdapterModelField,
)
from invokeai.app.invocations.latent import SAMPLER_NAME_VALUES, DenoiseLatentsInvocation, SchedulerOutput
from invokeai.app.invocations.metadata import LoRAMetadataField, MetadataOutput
from invokeai.app.invocations.model import (
    ClipField,
    LoraInfo,
    LoraLoaderOutput,
    LoRAModelField,
    MainModelField,
    ModelInfo,
    SDXLLoraLoaderOutput,
    UNetField,
    VaeField,
    VAEModelField,
    VAEOutput,
)
from invokeai.app.invocations.primitives import (
    BooleanOutput,
    FloatOutput,
    ImageField,
    IntegerOutput,
    LatentsOutput,
    StringOutput,
)
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.shared.fields import FieldDescriptions
from invokeai.backend.model_management.models.base import ModelType, SubModelType
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
]

CORE_LABELS_FLOAT = Literal[
    f"{CUSTOM_LABEL}",
    "cfg_scale",
    "cfg_rescale_multiplier",
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


def append_list(new_item, items, item_cls):
    """Add an item to an exiting item or list of items then output as a list of items."""

    result = []
    if items is None or (isinstance(items, list) and len(items) == 0):
        pass
    elif isinstance(items, item_cls):
        result.append(items)
    elif isinstance(items, list) and all(isinstance(i, item_cls) for i in items):
        result.extend(items)
    else:
        raise ValueError(f"Invalid adapter list format: {items}")

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
        v = self.value.vae if isinstance(self.value, VaeField) else self.value

        data = {} if self.metadata is None else self.metadata.model_dump()
        data.update({k: v})
        data.update({"app_version": __version__})

        return MetadataOutput(metadata=MetadataField.model_validate(data))


@invocation(
    "metadata_from_image",
    title="Metadata From Image",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Beta,
)
class MetadataFromImageInvocation(BaseInvocation):
    """Used to create a core metadata item then Add/Update it to the provided metadata"""

    image: ImageField = InputField(description=FieldDescriptions.image)

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        data = {}
        image_metadata = context.services.images.get_metadata(self.image.image_name)
        if image_metadata is not None:
            data.update(image_metadata.model_dump())

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
        data = {} if self.metadata is None else self.metadata.model_dump()
        output = data.get(self.custom_label if self.label == CUSTOM_LABEL else self.label, self.default_value)

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
        data = {} if self.metadata is None else self.metadata.model_dump()
        output = data.get(self.custom_label if self.label == CUSTOM_LABEL else self.label, self.default_value)

        return IntegerOutput(value=int(output))


@invocation(
    "metadata_to_float",
    title="Metadata To Float",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
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
    default_value: int = InputField(description="The default float to use if not found in the metadata")

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> FloatOutput:
        data = {} if self.metadata is None else self.metadata.model_dump()
        output = data.get(self.custom_label if self.label == CUSTOM_LABEL else self.label, self.default_value)

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
        data = {} if self.metadata is None else self.metadata.model_dump()
        output = data.get(self.custom_label if self.label == CUSTOM_LABEL else self.label, self.default_value)

        return BooleanOutput(value=bool(output))


@invocation(
    "metadata_to_scheduler",
    title="Metadata To Scheduler",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
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
    default_value: SAMPLER_NAME_VALUES = InputField(
        default="euler",
        description="The default scheduler to use if not found in the metadata",
        ui_type=UIType.Scheduler,
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> SchedulerOutput:
        data = {} if self.metadata is None else self.metadata.model_dump()
        output = data.get(self.custom_label if self.label == CUSTOM_LABEL else self.label, self.default_value)

        return SchedulerOutput(scheduler=output)


@invocation_output("metadata_to_model_output")
class MetadataToModelOutput(BaseInvocationOutput):
    """String to main model output"""

    model: MainModelField = OutputField(description=FieldDescriptions.main_model, title="Model")
    name: str = OutputField(description="Model Name", title="Name")
    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")
    vae: VaeField = OutputField(description=FieldDescriptions.vae, title="VAE")
    clip: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP")


@invocation_output("metadata_to_sdxl_model_output")
class MetadataToSDXLModelOutput(BaseInvocationOutput):
    """String to SDXL main model output"""

    model: MainModelField = OutputField(
        description=FieldDescriptions.main_model, title="Model", ui_type=UIType.SDXLMainModel
    )
    name: str = OutputField(description="Model Name", title="Name")
    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")
    clip: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP 1")
    clip2: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP 2")
    vae: VaeField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "metadata_to_model",
    title="Metadata To Model",
    tags=["metadata"],
    category="metadata",
    version="1.1.0",
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
    default_value: MainModelField = InputField(
        description="The default model to use if not found in the metadata",
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> MetadataToModelOutput:
        data = {} if self.metadata is None else self.metadata.model_dump()
        model = MainModelField(
            **data.get(self.custom_label if self.label == CUSTOM_LABEL else self.label, self.default_value)
        )

        base_model = model.base_model
        model_name = model.model_name
        model_type = ModelType.Main

        # TODO: not found exceptions
        if not context.services.model_manager.model_exists(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
        ):
            raise Exception(f"Unknown {base_model} {model_type} model: {model_name}")

        return MetadataToModelOutput(
            model=model,
            name=f"{base_model}: {model_name}",
            unet=UNetField(
                unet=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.UNet,
                ),
                scheduler=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Scheduler,
                ),
                loras=[],
            ),
            clip=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder,
                ),
                loras=[],
                skipped_layers=0,
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Vae,
                ),
            ),
        )


@invocation(
    "metadata_to_sdxl_model",
    title="Metadata To SDXL Model",
    tags=["metadata"],
    category="metadata",
    version="1.1.0",
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
    default_value: MainModelField = InputField(
        description="The default SDXL Model to use if not found in the metadata", ui_type=UIType.SDXLMainModel
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> MetadataToSDXLModelOutput:
        data = {} if self.metadata is None else self.metadata.model_dump()
        model = MainModelField(
            **data.get(self.custom_label if self.label == CUSTOM_LABEL else self.label, self.default_value)
        )

        base_model = model.base_model
        model_name = model.model_name
        model_type = ModelType.Main

        if not context.services.model_manager.model_exists(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
        ):
            raise Exception(f"Unknown {base_model} {model_type} model: {model_name}")

        return MetadataToSDXLModelOutput(
            model=model,
            name=f"{base_model}: {model_name}",
            unet=UNetField(
                unet=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.UNet,
                ),
                scheduler=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Scheduler,
                ),
                loras=[],
            ),
            clip=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder,
                ),
                loras=[],
                skipped_layers=0,
            ),
            clip2=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer2,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder2,
                ),
                loras=[],
                skipped_layers=0,
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Vae,
                ),
            ),
        )


@invocation_output("latents_meta_output")
class LatentsMetaOutput(LatentsOutput, MetadataOutput):
    """Latents + metadata"""


@invocation(
    "denoise_latents_meta",
    title="Denoise Latents + metadata",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.0.1",
)
class DenoiseLatentsMetaInvocation(DenoiseLatentsInvocation, WithMetadata):
    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        def _to_json(obj):
            if not isinstance(obj, list):
                obj = [obj]

            return [
                item.model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
                for item in obj
            ]

        def _loras_to_json(obj):
            if not isinstance(obj, list):
                obj = [obj]

            return [
                LoRAMetadataField(
                    lora=LoRAModelField(model_name=item.model_name, base_model=item.base_model), weight=item.weight
                ).model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
                for item in obj
            ]

        obj = super().invoke(context)

        md = {} if self.metadata is None else self.metadata.model_dump()
        md.update({"width": obj.width})
        md.update({"height": obj.height})
        md.update({"steps": self.steps})
        md.update({"cfg_scale": self.cfg_scale})
        md.update({"cfg_rescale_multiplier": self.cfg_rescale_multiplier})
        md.update({"denoising_start": self.denoising_start})
        md.update({"denoising_end": self.denoising_end})
        md.update({"scheduler": self.scheduler})
        md.update({"model": self.unet.unet})
        if self.control is not None and isinstance(self.control, list) and len(self.control) > 0:
            md.update({"controlnets": _to_json(self.control)})
        if self.ip_adapter is not None and isinstance(self.ip_adapter, list) and len(self.ip_adapter) > 0:
            md.update({"ipAdapters": _to_json(self.ip_adapter)})
        if self.t2i_adapter is not None and isinstance(self.t2i_adapter, list) and len(self.t2i_adapter) > 0:
            md.update({"t2iAdapters": _to_json(self.t2i_adapter)})
        if len(self.unet.loras) > 0:
            md.update({"loras": _loras_to_json(self.unet.loras)})
        if self.noise is not None:
            md.update({"seed": self.noise.seed})

        params = obj.__dict__.copy()
        del params["type"]

        return LatentsMetaOutput(**params, metadata=MetadataField.model_validate(md))


@invocation(
    "metadata_to_vae",
    title="Metadata To VAE",
    tags=["metadata"],
    category="metadata",
    version="1.1.0",
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
    default_vae: VaeField = InputField(
        description="The default VAE to use if not found in the metadata",
    )

    _validate_custom_label = model_validator(mode="after")(validate_custom_label)

    def invoke(self, context: InvocationContext) -> VAEOutput:
        data = {} if self.metadata is None else self.metadata.model_dump()
        key = self.custom_label if self.label == CUSTOM_LABEL else self.label
        if key in data:
            model = VAEModelField(**data.get(key, ""))
            vae = VaeField(
                vae=ModelInfo(
                    model_name=model.model_name,
                    base_model=model.base_model,
                    model_type=ModelType.Vae,
                ),
            )
        else:
            vae = self.default_vae

        if not context.services.model_manager.model_exists(
            base_model=vae.vae.base_model,
            model_name=vae.vae.model_name,
            model_type=vae.vae.model_type,
        ):
            raise Exception(f"Unknown vae name: {vae.vae.model_name}!")

        return VAEOutput(vae=vae)


@invocation(
    "metadata_to_loras",
    title="Metadata To LoRAs",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
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
    clip: Optional[ClipField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP",
    )

    def invoke(self, context: InvocationContext) -> LoraLoaderOutput:
        data = {} if self.metadata is None else self.metadata.model_dump()
        key = "loras"
        if key in data:
            loras = data.get(key, "")
        else:
            loras = []

        output = LoraLoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)

        for x in loras:
            lora = LoRAMetadataField(**x)
            if not context.services.model_manager.model_exists(
                base_model=lora.lora.base_model,
                model_name=lora.lora.model_name,
                model_type=ModelType.Lora,
            ):
                raise Exception(f"Unknown lora {lora.lora.base_model}:{lora.lora.model_name}")

            if self.unet is not None:
                output.unet.loras.append(
                    LoraInfo(
                        base_model=lora.lora.base_model,
                        model_name=lora.lora.model_name,
                        model_type=ModelType.Lora,
                        submodel=None,
                        weight=lora.weight,
                    )
                )

            if self.clip is not None:
                output.clip.loras.append(
                    LoraInfo(
                        base_model=lora.lora.base_model,
                        model_name=lora.lora.model_name,
                        model_type=ModelType.Lora,
                        submodel=None,
                        weight=lora.weight,
                    )
                )

        return output


@invocation(
    "metadata_to_sdlx_loras",
    title="Metadata To SDXL LoRAs",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
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
    clip: Optional[ClipField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 1",
    )
    clip2: Optional[ClipField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 2",
    )

    def invoke(self, context: InvocationContext) -> SDXLLoraLoaderOutput:
        data = {} if self.metadata is None else self.metadata.model_dump()
        key = "loras"
        if key in data:
            loras = data.get(key, "")
        else:
            loras = []

        output = SDXLLoraLoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)

        if self.clip2 is not None:
            output.clip2 = copy.deepcopy(self.clip2)

        for x in loras:
            lora = LoRAMetadataField(**x)
            if not context.services.model_manager.model_exists(
                base_model=lora.lora.base_model,
                model_name=lora.lora.model_name,
                model_type=ModelType.Lora,
            ):
                raise Exception(f"Unknown LoRA {lora.lora.base_model}:{lora.lora.model_name}")

            if self.unet is not None:
                output.unet.loras.append(
                    LoraInfo(
                        base_model=lora.lora.base_model,
                        model_name=lora.lora.model_name,
                        model_type=ModelType.Lora,
                        submodel=None,
                        weight=lora.weight,
                    )
                )

            if self.clip is not None:
                output.clip.loras.append(
                    LoraInfo(
                        base_model=lora.lora.base_model,
                        model_name=lora.lora.model_name,
                        model_type=ModelType.Lora,
                        submodel=None,
                        weight=lora.weight,
                    )
                )

            if self.clip2 is not None:
                output.clip2.loras.append(
                    LoraInfo(
                        base_model=lora.lora.base_model,
                        model_name=lora.lora.model_name,
                        model_type=ModelType.Lora,
                        submodel=None,
                        weight=lora.weight,
                    )
                )

        return output


@invocation_output("md_control_list_output")
class MDControlListOutput(BaseInvocationOutput):
    # Outputs
    control_list: list[ControlField] = OutputField(
        description=FieldDescriptions.control,
        title="ControlNet-List",
    )


@invocation(
    "metadata_to_controlnets",
    title="Metadata To ControlNets",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
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
        data = {} if self.metadata is None else self.metadata.model_dump()
        key = "controlnets"
        if key in data:
            md_controls = data.get(key, "")
        else:
            md_controls = []

        controls = []

        if self.control_list is not None:
            controls = self.control_list

        for x in md_controls:
            c = ControlField(**x)
            controls = append_list(c, controls, ControlField)

        return MDControlListOutput(control_list=controls)


@invocation_output("md_ip_adapter_list_output")
class MDIPAdapterListOutput(BaseInvocationOutput):
    # Outputs
    ip_adapter_list: list[IPAdapterField] = OutputField(
        description=FieldDescriptions.ip_adapter, title="IP-Adapter-List"
    )


@invocation(
    "metadata_to_ip_adapters",
    title="Metadata To IP-Adapters",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
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
        data = {} if self.metadata is None else self.metadata.model_dump()
        key = "ipAdapters"
        if key in data:
            md_adapters = data.get(key, "")
        else:
            md_adapters = []

        adapters = []

        if self.ip_adapter_list is not None:
            adapters = self.ip_adapter_list

        for x in md_adapters:
            ipa = IPAdapterInvocation(
                image=x["image"],
                ip_adapter_model=IPAdapterModelField(**x["ip_adapter_model"]),
                weight=x["weight"],
                begin_step_percent=x["begin_step_percent"],
                end_step_percent=x["end_step_percent"],
            )
            a = ipa.invoke(context)

            adapters = append_list(a.ip_adapter, adapters, IPAdapterField)

        return MDIPAdapterListOutput(ip_adapter_list=adapters)


@invocation_output("md_ip_adapters_output")
class MDT2IAdapterListOutput(BaseInvocationOutput):
    # Outputs
    t2i_adapter_list: list[T2IAdapterField] = OutputField(
        description=FieldDescriptions.t2i_adapter, title="T2I Adapter-List"
    )


@invocation(
    "metadata_to_t2i_adapters",
    title="Metadata To T2I-Adapters",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
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
        data = {} if self.metadata is None else self.metadata.model_dump()
        key = "t2iAdapters"
        if key in data:
            md_adapters = data.get(key, "")
        else:
            md_adapters = []

        adapters = []

        if self.t2i_adapter_list is not None:
            adapters = self.t2i_adapter_list

        for x in md_adapters:
            a = T2IAdapterField(**x)
            adapters = append_list(a, adapters, T2IAdapterField)

        return MDT2IAdapterListOutput(t2i_adapter_list=adapters)

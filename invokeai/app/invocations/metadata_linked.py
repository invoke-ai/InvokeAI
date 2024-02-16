import json
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
from invokeai.app.invocations.latent import SAMPLER_NAME_VALUES, DenoiseLatentsInvocation, SchedulerOutput
from invokeai.app.invocations.metadata import LoRAMetadataField, MetadataOutput
from invokeai.app.invocations.model import LoRAModelField, MainModelField, VaeField
from invokeai.app.invocations.primitives import (
    BooleanOutput,
    FloatOutput,
    ImageField,
    IntegerOutput,
    LatentsOutput,
    StringOutput,
)
from invokeai.app.shared.fields import FieldDescriptions
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


@invocation_output("metadata_to_sdxl_model_output")
class MetadataToSDXLModelOutput(BaseInvocationOutput):
    """String to SDXL main model output"""

    model: MainModelField = OutputField(
        description=FieldDescriptions.main_model, title="Model", ui_type=UIType.SDXLMainModel
    )
    name: str = OutputField(description="Model Name", title="Name")


@invocation(
    "metadata_to_model",
    title="Metadata To Model",
    tags=["metadata"],
    category="metadata",
    version="1.0.1",
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

        return MetadataToModelOutput(model=model, name=f"{model.base_model}: {model.model_name}")


@invocation(
    "metadata_to_sdxl_model",
    title="Metadata To SDXL Model",
    tags=["metadata"],
    category="metadata",
    version="1.0.1",
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

        return MetadataToSDXLModelOutput(model=model, name=f"{model.base_model}: {model.model_name}")


@invocation_output("latents_meta_output")
class LatentsMetaOutput(LatentsOutput, MetadataOutput):
    """Latents + metadata"""


@invocation(
    "denoise_latents_meta",
    title="Denoise Latents + metadata",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.0.0",
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
        if self.control is not None:
            md.update({"controlnets": _to_json(self.control)})
        if self.ip_adapter is not None:
            md.update({"ipAdapters": _to_json(self.ip_adapter)})
        if self.t2i_adapter is not None:
            md.update({"t2iAdapters": _to_json(self.t2i_adapter)})
        if len(self.unet.loras) > 0:
            md.update({"loras": _loras_to_json(self.unet.loras)})
        if self.noise is not None:
            md.update({"seed": self.noise.seed})

        params = obj.__dict__.copy()
        del params["type"]

        return LatentsMetaOutput(**params, metadata=MetadataField.model_validate(md))

from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.metadata import LoRAMetadataField
from invokeai.app.invocations.model import CLIPField, LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("sd3_lora_loader_output")
class SD3LoRALoaderOutput(BaseInvocationOutput):
    """SD3 LoRA loader output."""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )
    clip_l: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP L")
    clip_g: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP G")
    # The model fields carry LoRAField entries, which Core Metadata cannot take. This echoes the applied LoRA in the
    # shape `core_metadata.loras` wants, so a workflow can record it without retyping the model key.
    lora_metadata: LoRAMetadataField = OutputField(
        description="The applied LoRA, in the shape Core Metadata's `loras` field takes.",
        title="LoRA Metadata",
    )


@invocation_output("sd3_lora_collection_loader_output")
class SD3LoRACollectionLoaderOutput(BaseInvocationOutput):
    """SD3 LoRA collection loader output."""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )
    clip_l: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP L")
    clip_g: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP G")
    lora_metadata: list[LoRAMetadataField] = OutputField(
        default=[],
        description="The applied LoRAs, in the shape Core Metadata's `loras` field takes.",
        title="LoRA Metadata",
    )


def _raise_unless_sd3_lora(context: InvocationContext, lora_key: str) -> None:
    """Refuse a key that does not resolve to an installed SD 3 LoRA.

    The identifier's own base/type are client-supplied: a hand-written workflow can label any model key as an SD 3
    LoRA and so reach model patching. Only the stored config is authoritative.
    """
    if not context.models.exists(lora_key):
        raise ValueError(f"Unknown lora: {lora_key}!")
    stored_config = context.models.get_config(lora_key)
    if stored_config.type is not ModelType.LoRA or stored_config.base is not BaseModelType.StableDiffusion3:
        raise ValueError(
            f"Model '{lora_key}' is not an SD 3 LoRA (resolved to "
            f"type={getattr(stored_config.type, 'value', stored_config.type)}, "
            f"base={getattr(stored_config.base, 'value', stored_config.base)})."
        )


def _with_lora(field, lora: LoRAField):
    """A copy of a model field with ``lora`` appended, or ``None`` for an unconnected input."""
    if field is None:
        return None
    copied = field.model_copy(deep=True)
    copied.loras.append(lora)
    return copied


@invocation(
    "sd3_lora_loader",
    title="Apply LoRA - SD3",
    tags=["lora", "model", "sd3"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class SD3LoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to an SD3 transformer and its CLIP encoders.

    A LoRA patches only what it trained; T5 layers in a LoRA are dropped, since the T5 encoder takes no LoRA.
    """

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.StableDiffusion3,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
        default=None, description=FieldDescriptions.transformer, input=Input.Connection, title="Transformer"
    )
    clip_l: CLIPField | None = InputField(
        default=None, description=FieldDescriptions.clip, input=Input.Connection, title="CLIP L"
    )
    clip_g: CLIPField | None = InputField(
        default=None, description=FieldDescriptions.clip, input=Input.Connection, title="CLIP G"
    )

    def invoke(self, context: InvocationContext) -> SD3LoRALoaderOutput:
        lora_key = self.lora.key
        _raise_unless_sd3_lora(context, lora_key)

        for role, field in (("transformer", self.transformer), ("CLIP L", self.clip_l), ("CLIP G", self.clip_g)):
            if field is not None and any(lora.lora.key == lora_key for lora in field.loras):
                raise ValueError(f'LoRA "{lora_key}" already applied to {role}.')

        lora = LoRAField(lora=self.lora, weight=self.weight)
        return SD3LoRALoaderOutput(
            transformer=_with_lora(self.transformer, lora),
            clip_l=_with_lora(self.clip_l, lora),
            clip_g=_with_lora(self.clip_g, lora),
            lora_metadata=LoRAMetadataField(model=self.lora, weight=self.weight),
        )


@invocation(
    "sd3_lora_collection_loader",
    title="Apply LoRA Collection - SD3",
    tags=["lora", "model", "sd3"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class SD3LoRACollectionLoader(BaseInvocation):
    """Apply a collection of LoRAs to an SD3 transformer and its CLIP encoders."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRAs to apply. May be a single LoRA or a collection.",
        title="LoRAs",
        ui_model_base=[BaseModelType.StableDiffusion3],
        ui_model_type=ModelType.LoRA,
    )
    transformer: Optional[TransformerField] = InputField(
        default=None, description=FieldDescriptions.transformer, input=Input.Connection, title="Transformer"
    )
    clip_l: CLIPField | None = InputField(
        default=None, description=FieldDescriptions.clip, input=Input.Connection, title="CLIP L"
    )
    clip_g: CLIPField | None = InputField(
        default=None, description=FieldDescriptions.clip, input=Input.Connection, title="CLIP G"
    )

    def invoke(self, context: InvocationContext) -> SD3LoRACollectionLoaderOutput:
        output = SD3LoRACollectionLoaderOutput(
            transformer=self.transformer.model_copy(deep=True) if self.transformer is not None else None,
            clip_l=self.clip_l.model_copy(deep=True) if self.clip_l is not None else None,
            clip_g=self.clip_g.model_copy(deep=True) if self.clip_g is not None else None,
        )
        if self.loras is None:
            return output

        for lora in self.loras if isinstance(self.loras, list) else [self.loras]:
            lora_key = lora.lora.key
            _raise_unless_sd3_lora(context, lora_key)
            # A LoRA already on a field (from an earlier loader, or twice in the collection) is not applied again.
            fields = [f for f in (output.transformer, output.clip_l, output.clip_g) if f is not None]
            missing = [f for f in fields if not any(applied.lora.key == lora_key for applied in f.loras)]
            for field in missing:
                field.loras.append(lora)
            if missing:
                output.lora_metadata.append(LoRAMetadataField(model=lora.lora, weight=lora.weight))
        return output

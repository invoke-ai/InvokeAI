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
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("ernie_image_lora_loader_output")
class ErnieImageLoRALoaderOutput(BaseInvocationOutput):
    """ERNIE-Image LoRA loader output."""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )
    # The transformer carries LoRAField entries, which Core Metadata cannot take. This echoes the applied LoRA
    # in the shape `core_metadata.loras` wants, so a workflow can record it without retyping the model key.
    lora_metadata: LoRAMetadataField = OutputField(
        description="The applied LoRA, in the shape Core Metadata's `loras` field takes.",
        title="LoRA Metadata",
    )


@invocation_output("ernie_image_lora_collection_loader_output")
class ErnieImageLoRACollectionLoaderOutput(BaseInvocationOutput):
    """ERNIE-Image LoRA collection loader output."""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )
    lora_metadata: list[LoRAMetadataField] = OutputField(
        default=[],
        description="The applied LoRAs, in the shape Core Metadata's `loras` field takes.",
        title="LoRA Metadata",
    )


def _raise_unless_ernie_image_lora(context: InvocationContext, lora_key: str) -> None:
    """Refuse a key that does not resolve to an installed ERNIE-Image LoRA.

    The identifier's own base/type are client-supplied: a hand-written workflow can label any model key as an
    ERNIE-Image LoRA and so reach model patching. Only the stored config is authoritative.
    """
    if not context.models.exists(lora_key):
        raise ValueError(f"Unknown lora: {lora_key}!")
    stored_config = context.models.get_config(lora_key)
    if stored_config.type is not ModelType.LoRA or stored_config.base is not BaseModelType.ErnieImage:
        raise ValueError(
            f"Model '{lora_key}' is not an ERNIE-Image LoRA (resolved to "
            f"type={getattr(stored_config.type, 'value', stored_config.type)}, "
            f"base={getattr(stored_config.base, 'value', stored_config.base)})."
        )


@invocation(
    "ernie_image_lora_loader",
    title="Apply LoRA - ERNIE-Image",
    tags=["lora", "model", "ernie-image"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ErnieImageLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to an ERNIE-Image transformer."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.ErnieImage,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> ErnieImageLoRALoaderOutput:
        lora_key = self.lora.key
        _raise_unless_ernie_image_lora(context, lora_key)

        if any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')

        transformer = self.transformer.model_copy(deep=True)
        transformer.loras.append(LoRAField(lora=self.lora, weight=self.weight))

        return ErnieImageLoRALoaderOutput(
            transformer=transformer,
            lora_metadata=LoRAMetadataField(model=self.lora, weight=self.weight),
        )


@invocation(
    "ernie_image_lora_collection_loader",
    title="Apply LoRA Collection - ERNIE-Image",
    tags=["lora", "model", "ernie-image"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ErnieImageLoRACollectionLoader(BaseInvocation):
    """Apply a collection of LoRAs to an ERNIE-Image transformer."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRAs to apply. May be a single LoRA or a collection.",
        title="LoRAs",
        ui_model_base=[BaseModelType.ErnieImage],
        ui_model_type=ModelType.LoRA,
    )
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> ErnieImageLoRACollectionLoaderOutput:
        if self.transformer is None:
            return ErnieImageLoRACollectionLoaderOutput()

        transformer = self.transformer.model_copy(deep=True)
        if self.loras is None:
            return ErnieImageLoRACollectionLoaderOutput(transformer=transformer)

        lora_metadata: list[LoRAMetadataField] = []
        for lora in self.loras if isinstance(self.loras, list) else [self.loras]:
            lora_key = lora.lora.key
            _raise_unless_ernie_image_lora(context, lora_key)
            if any(applied.lora.key == lora_key for applied in transformer.loras):
                continue
            transformer.loras.append(lora)
            lora_metadata.append(LoRAMetadataField(model=lora.lora, weight=lora.weight))

        return ErnieImageLoRACollectionLoaderOutput(transformer=transformer, lora_metadata=lora_metadata)

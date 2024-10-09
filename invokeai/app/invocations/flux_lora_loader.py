from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.invocations.model import CLIPField, LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import BaseModelType


@invocation_output("flux_lora_loader_output")
class FluxLoRALoaderOutput(BaseInvocationOutput):
    """FLUX LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="FLUX Transformer"
    )
    clip: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP")


@invocation(
    "flux_lora_loader",
    title="FLUX LoRA",
    tags=["lora", "model", "flux"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class FluxLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a FLUX transformer and/or text encoder."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model, title="LoRA", ui_type=UIType.LoRAModel
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="FLUX Transformer",
    )
    clip: CLIPField | None = InputField(
        default=None,
        title="CLIP",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> FluxLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        # Check for existing LoRAs with the same key.
        if self.transformer and any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')
        if self.clip and any(lora.lora.key == lora_key for lora in self.clip.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to CLIP encoder.')

        output = FluxLoRALoaderOutput()

        # Attach LoRA layers to the models.
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )
        if self.clip is not None:
            output.clip = self.clip.model_copy(deep=True)
            output.clip.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        return output


@invocation(
    "flux_lora_collection_loader",
    title="FLUX LoRA Collection Loader",
    tags=["lora", "model", "flux"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class FLUXLoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to a FLUX transformer."""

    loras: LoRAField | list[LoRAField] = InputField(
        description="LoRA models and weights. May be a single LoRA or collection.", title="LoRAs"
    )

    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )
    clip: CLIPField | None = InputField(
        default=None,
        title="CLIP",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> FluxLoRALoaderOutput:
        output = FluxLoRALoaderOutput()
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        for lora in loras:
            if lora.lora.key in added_loras:
                continue

            if not context.models.exists(lora.lora.key):
                raise Exception(f"Unknown lora: {lora.lora.key}!")

            assert lora.lora.base is BaseModelType.Flux

            added_loras.append(lora.lora.key)

            if self.transformer is not None:
                if output.transformer is None:
                    output.transformer = self.transformer.model_copy(deep=True)
                output.transformer.loras.append(lora)

            if self.clip is not None:
                if output.clip is None:
                    output.clip = self.clip.model_copy(deep=True)
                output.clip.loras.append(lora)

        return output

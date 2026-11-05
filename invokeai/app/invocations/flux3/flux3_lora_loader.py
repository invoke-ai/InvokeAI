from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("flux3_lora_loader_output")
class Flux3LoRALoaderOutput(BaseInvocationOutput):
    """FLUX.3 LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )


def _require_flux3_lora(context: InvocationContext, lora: ModelIdentifierField) -> None:
    """Refuse a LoRA that is not installed as a FLUX.3 LoRA: its keys would match nothing, or the wrong layers, in this
    transformer. FLUX.1 and FLUX.2 LoRAs share module names with it, but not its layers."""
    if not context.models.exists(lora.key):
        raise ValueError(f"Unknown LoRA: {lora.key}!")
    config = context.models.get_config(lora.key)
    if config.type is not ModelType.LoRA:
        raise ValueError(f"'{config.name}' is a {config.type.value} model, not a LoRA.")
    if config.base is not BaseModelType.Flux3:
        raise ValueError(f"LoRA '{config.name}' is for {config.base.value} models, not FLUX.3.")


@invocation(
    "flux3_lora_loader",
    title="Apply LoRA - FLUX.3",
    tags=["lora", "model", "flux3", "flux"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux3LoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a FLUX.3 transformer. A LoRA trained on Klein Base also applies to Klein."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.Flux3,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=1.0, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> Flux3LoRALoaderOutput:
        _require_flux3_lora(context, self.lora)
        if self.transformer and any(lora.lora.key == self.lora.key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{self.lora.key}" already applied to transformer.')

        output = Flux3LoRALoaderOutput()
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(LoRAField(lora=self.lora, weight=self.weight))
        return output


@invocation(
    "flux3_lora_collection_loader",
    title="Apply LoRA Collection - FLUX.3",
    tags=["lora", "model", "flux3", "flux"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux3LoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to a FLUX.3 transformer."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRA models and weights. May be a single LoRA or collection.",
        title="LoRAs",
        ui_model_base=[BaseModelType.Flux3],
        ui_model_type=ModelType.LoRA,
    )
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> Flux3LoRALoaderOutput:
        output = Flux3LoRALoaderOutput()
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)

        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        # Including those an upstream loader applied: a LoRA applied twice doubles its weight.
        added = {lora.lora.key for lora in output.transformer.loras} if output.transformer is not None else set()
        for lora in loras:
            if lora is None or lora.lora.key in added:
                continue
            _require_flux3_lora(context, lora.lora)
            added.add(lora.lora.key)
            if output.transformer is not None:
                output.transformer.loras.append(lora)
        return output

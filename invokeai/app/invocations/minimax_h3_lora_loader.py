from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import LoRAField, MiniMaxH3TransformerField, ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("minimax_h3_lora_loader_output")
class MiniMaxH3LoRALoaderOutput(BaseInvocationOutput):
    """MiniMax H3 LoRA loader output."""

    transformer: Optional[MiniMaxH3TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="MiniMax H3 Transformer"
    )


@invocation(
    "minimax_h3_lora_loader",
    title="Apply LoRA - MiniMax H3",
    tags=["lora", "model", "minimax"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3LoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a MiniMax H3 transformer (e.g. the Turbo step-distillation LoRA)."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.MiniMaxH3,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(
        default=1.0,
        description="Strength of the LoRA. The Turbo LoRA is trained for 1.0; adjust only to trade off "
        "motion artifacts (raise slightly) against over-sharpening (lower slightly).",
    )
    transformer: MiniMaxH3TransformerField = InputField(
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> MiniMaxH3LoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        # The identifier's own base/type fields are client-supplied and cannot be trusted:
        # a hand-authored workflow can label any model key as an H3 LoRA and reach model
        # patching. Only the config the key actually resolves to is authoritative.
        stored_config = context.models.get_config(lora_key)
        if stored_config.type is not ModelType.LoRA or stored_config.base is not BaseModelType.MiniMaxH3:
            raise ValueError(
                f"Model '{lora_key}' is not a MiniMax H3 LoRA (resolved to "
                f"type={getattr(stored_config.type, 'value', stored_config.type)}, "
                f"base={getattr(stored_config.base, 'value', stored_config.base)})."
            )

        if any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')

        transformer = self.transformer.model_copy(deep=True)
        transformer.loras.append(LoRAField(lora=self.lora, weight=self.weight))

        return MiniMaxH3LoRALoaderOutput(transformer=transformer)


@invocation(
    "minimax_h3_lora_collection_loader",
    title="Apply LoRA Collection - MiniMax H3",
    tags=["lora", "model", "minimax"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3LoRACollectionLoader(BaseInvocation):
    """Apply a collection of LoRAs to a MiniMax H3 transformer."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRAs to apply. May be a single LoRA or a collection.",
        title="LoRAs",
        ui_model_base=[BaseModelType.MiniMaxH3],
        ui_model_type=ModelType.LoRA,
    )
    transformer: Optional[MiniMaxH3TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> MiniMaxH3LoRALoaderOutput:
        output = MiniMaxH3LoRALoaderOutput()

        if self.transformer is None:
            return output

        output.transformer = self.transformer.model_copy(deep=True)

        if self.loras is None:
            return output

        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        for lora in loras:
            lora_key = lora.lora.key
            if not context.models.exists(lora_key):
                raise ValueError(f"Unknown lora: {lora_key}!")

            # Same trust boundary as the single loader: only the resolved config is authoritative.
            stored_config = context.models.get_config(lora_key)
            if stored_config.type is not ModelType.LoRA or stored_config.base is not BaseModelType.MiniMaxH3:
                raise ValueError(
                    f"Model '{lora_key}' is not a MiniMax H3 LoRA (resolved to "
                    f"type={getattr(stored_config.type, 'value', stored_config.type)}, "
                    f"base={getattr(stored_config.base, 'value', stored_config.base)})."
                )

            if any(item.lora.key == lora_key for item in output.transformer.loras):
                continue
            output.transformer.loras.append(lora)

        return output

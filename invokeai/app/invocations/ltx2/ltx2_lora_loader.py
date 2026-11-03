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
from invokeai.app.invocations.model import LoRAField, LTX2TransformerField, ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("ltx2_lora_loader_output")
class LTX2LoRALoaderOutput(BaseInvocationOutput):
    """LTX-2 LoRA loader output."""

    transformer: Optional[LTX2TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="LTX-2 Transformer"
    )
    # The transformer field carries LoRAField entries, which metadata cannot consume (different
    # shape, different key names). This echoes the same LoRA in the shape core_metadata.loras
    # wants, so a workflow can record what it applied without retyping the model key.
    lora_metadata: LoRAMetadataField = OutputField(
        description="The applied LoRA, in the shape Core Metadata's `loras` field takes.",
        title="LoRA Metadata",
    )


@invocation_output("ltx2_lora_collection_loader_output")
class LTX2LoRACollectionLoaderOutput(BaseInvocationOutput):
    """LTX-2 LoRA collection loader output.

    Separate from the single loader's output because a collection applies zero or more LoRAs, so
    its metadata echo is a list — which is also what `core_metadata.loras` takes directly, with no
    intervening collect node.
    """

    transformer: Optional[LTX2TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="LTX-2 Transformer"
    )
    lora_metadata: list[LoRAMetadataField] = OutputField(
        default=[],
        description="The applied LoRAs, in the shape Core Metadata's `loras` field takes.",
        title="LoRA Metadata",
    )


@invocation(
    "ltx2_lora_loader",
    title="Apply LoRA - LTX-2",
    tags=["lora", "model", "ltx", "ltx2"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class LTX2LoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to an LTX-2 transformer (e.g. the distilled step accelerator).

    A step-distillation LoRA changes what schedule the run should use, not just how many steps it
    takes. The denoise node reads its schedule from the transformer's *variant*, which names the
    checkpoint rather than the patch, so a distilled LoRA on a dev checkpoint still has to be told
    to take the distilled schedule — set `schedule` on the denoise node rather than leaving it on
    `auto`.
    """

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.LTX2,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(
        default=1.0,
        description="Strength of the LoRA. The distilled accelerator is trained for 1.0; it carries no "
        "alpha tensors, so this is the only scaling applied.",
    )
    transformer: LTX2TransformerField = InputField(
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> LTX2LoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        # The identifier's own base/type fields are client-supplied and cannot be trusted: a
        # hand-authored workflow can label any model key as an LTX-2 LoRA and reach model
        # patching. Only the config the key actually resolves to is authoritative.
        stored_config = context.models.get_config(lora_key)
        if stored_config.type is not ModelType.LoRA or stored_config.base is not BaseModelType.LTX2:
            raise ValueError(
                f"Model '{lora_key}' is not an LTX-2 LoRA (resolved to "
                f"type={getattr(stored_config.type, 'value', stored_config.type)}, "
                f"base={getattr(stored_config.base, 'value', stored_config.base)})."
            )

        if any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')

        transformer = self.transformer.model_copy(deep=True)
        transformer.loras.append(LoRAField(lora=self.lora, weight=self.weight))

        return LTX2LoRALoaderOutput(
            transformer=transformer,
            lora_metadata=LoRAMetadataField(model=self.lora, weight=self.weight),
        )


@invocation(
    "ltx2_lora_collection_loader",
    title="Apply LoRA Collection - LTX-2",
    tags=["lora", "model", "ltx", "ltx2"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class LTX2LoRACollectionLoader(BaseInvocation):
    """Apply a collection of LoRAs to an LTX-2 transformer."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRAs to apply. May be a single LoRA or a collection.",
        title="LoRAs",
        ui_model_base=[BaseModelType.LTX2],
        ui_model_type=ModelType.LoRA,
    )
    transformer: Optional[LTX2TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> LTX2LoRACollectionLoaderOutput:
        if self.transformer is None:
            return LTX2LoRACollectionLoaderOutput()

        transformer = self.transformer.model_copy(deep=True)

        if self.loras is None:
            return LTX2LoRACollectionLoaderOutput(transformer=transformer)

        lora_metadata: list[LoRAMetadataField] = []
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        for lora in loras:
            lora_key = lora.lora.key
            if not context.models.exists(lora_key):
                raise ValueError(f"Unknown lora: {lora_key}!")

            # Same trust boundary as the single loader: only the resolved config is authoritative.
            stored_config = context.models.get_config(lora_key)
            if stored_config.type is not ModelType.LoRA or stored_config.base is not BaseModelType.LTX2:
                raise ValueError(
                    f"Model '{lora_key}' is not an LTX-2 LoRA (resolved to "
                    f"type={getattr(stored_config.type, 'value', stored_config.type)}, "
                    f"base={getattr(stored_config.base, 'value', stored_config.base)})."
                )

            if any(item.lora.key == lora_key for item in transformer.loras):
                continue
            transformer.loras.append(lora)
            lora_metadata.append(LoRAMetadataField(model=lora.lora, weight=lora.weight))

        return LTX2LoRACollectionLoaderOutput(transformer=transformer, lora_metadata=lora_metadata)

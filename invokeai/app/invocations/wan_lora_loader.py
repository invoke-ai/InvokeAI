from typing import Literal, Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, WanTransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

# Target option for routing a LoRA to one or both Wan A14B expert lists.
#
# - ``auto``: read the LoRA config's ``expert`` field (set by the probe / from
#   filename). ``"high"`` -> primary list only, ``"low"`` -> low-noise list
#   only, ``None`` -> both lists.
# - ``both``: append to both lists regardless of the config.
# - ``high``: append only to the primary list (high-noise expert).
# - ``low``: append only to the low-noise list (low-noise expert).
WanLoRATarget = Literal["auto", "both", "high", "low"]


def _resolve_target(target: WanLoRATarget, lora_expert: str | None) -> tuple[bool, bool]:
    """Return (apply_to_primary, apply_to_low_noise) based on the requested
    target and the LoRA's recorded expert tag."""
    if target == "both":
        return True, True
    if target == "high":
        return True, False
    if target == "low":
        return False, True
    # auto
    if lora_expert == "high":
        return True, False
    if lora_expert == "low":
        return False, True
    return True, True


@invocation_output("wan_lora_loader_output")
class WanLoRALoaderOutput(BaseInvocationOutput):
    """Wan 2.2 LoRA loader output."""

    transformer: Optional[WanTransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Wan Transformer"
    )


@invocation(
    "wan_lora_loader",
    title="Apply LoRA - Wan 2.2",
    tags=["lora", "model", "wan"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA to the Wan 2.2 transformer(s).

    For A14B (dual expert) the LoRA's recorded ``expert`` field determines
    which expert list it lands in: ``"high"`` -> primary list, ``"low"`` ->
    low-noise list, ``None`` (untagged) -> both lists. Use the ``target``
    field to override.

    For TI2V-5B (single transformer) only the primary list is used at denoise
    time; the low-noise routing is harmless but ignored.
    """

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.Wan,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    target: WanLoRATarget = InputField(
        default="auto",
        description="Which expert(s) to apply this LoRA to. 'auto' uses the LoRA's "
        "recorded expert tag (or both if untagged); 'both'/'high'/'low' override it.",
    )
    transformer: WanTransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Wan Transformer",
    )

    def invoke(self, context: InvocationContext) -> WanLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        output = WanLoRALoaderOutput()
        if self.transformer is None:
            return output

        lora_config = context.models.get_config(self.lora)
        lora_expert = getattr(lora_config, "expert", None)
        to_primary, to_low_noise = _resolve_target(self.target, lora_expert)

        # Reject duplicates on whichever list(s) we're about to append to.
        if to_primary and any(item.lora.key == lora_key for item in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to primary transformer list.')
        if to_low_noise and any(item.lora.key == lora_key for item in self.transformer.loras_low_noise):
            raise ValueError(f'LoRA "{lora_key}" already applied to low-noise transformer list.')

        output.transformer = self.transformer.model_copy(deep=True)
        new_lora = LoRAField(lora=self.lora, weight=self.weight)
        if to_primary:
            output.transformer.loras.append(new_lora)
        if to_low_noise:
            output.transformer.loras_low_noise.append(new_lora)

        return output


@invocation(
    "wan_lora_collection_loader",
    title="Apply LoRA Collection - Wan 2.2",
    tags=["lora", "model", "wan"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanLoRACollectionLoader(BaseInvocation):
    """Apply a collection of LoRAs to the Wan 2.2 transformer(s).

    Each LoRA is routed to the primary and/or low-noise list based on its
    recorded ``expert`` tag (set by the probe from the filename). Untagged
    LoRAs go to both lists.
    """

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRAs to apply. May be a single LoRA or a collection.",
        title="LoRAs",
    )
    transformer: Optional[WanTransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Wan Transformer",
    )

    def invoke(self, context: InvocationContext) -> WanLoRALoaderOutput:
        output = WanLoRALoaderOutput()

        if self.transformer is None:
            return output

        output.transformer = self.transformer.model_copy(deep=True)

        if self.loras is None:
            return output

        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added: set[str] = set()

        for lora in loras:
            if lora is None or lora.lora.key in added:
                continue

            if not context.models.exists(lora.lora.key):
                raise ValueError(f"Unknown lora: {lora.lora.key}!")

            if lora.lora.base is not BaseModelType.Wan:
                raise ValueError(
                    f"LoRA '{lora.lora.key}' is for "
                    f"{lora.lora.base.value if lora.lora.base else 'unknown'} models, "
                    "not Wan 2.2."
                )

            lora_config = context.models.get_config(lora.lora)
            lora_expert = getattr(lora_config, "expert", None)
            to_primary, to_low_noise = _resolve_target("auto", lora_expert)

            added.add(lora.lora.key)

            if to_primary:
                output.transformer.loras.append(lora)
            if to_low_noise:
                output.transformer.loras_low_noise.append(lora)

        return output

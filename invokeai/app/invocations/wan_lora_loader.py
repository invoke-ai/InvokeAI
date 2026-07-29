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
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
    WanLoRAVariantType,
    WanVariantType,
)

# Target option for routing a LoRA to one or both Wan A14B expert lists.
#
# - ``auto``: read the LoRA config's ``expert`` field (set by the probe / from
#   filename). ``"high"`` -> primary list only, ``"low"`` -> low-noise list
#   only, ``None`` -> both lists.
# - ``both``: append to both lists regardless of the config.
# - ``high``: append only to the primary list (high-noise expert).
# - ``low``: append only to the low-noise list (low-noise expert).
WanLoRATarget = Literal["auto", "both", "high", "low"]


def _assert_is_wan_lora(lora_config: object, lora_key: str) -> None:
    """Reject an identifier whose *resolved* config is not a Wan LoRA.

    The identifier's own ``base``/``type`` fields are client-supplied and cannot be
    trusted: a hand-authored workflow can label any existing model key as a Wan LoRA
    and reach model patching (or fail minutes in, after expensive loading). Only the
    config the key actually resolves to is authoritative.
    """
    config_type = getattr(lora_config, "type", None)
    config_base = getattr(lora_config, "base", None)
    if config_type is not ModelType.LoRA or config_base is not BaseModelType.Wan:
        raise ValueError(
            f"Model '{lora_key}' is not a Wan LoRA (resolved to "
            f"type={getattr(config_type, 'value', config_type)}, "
            f"base={getattr(config_base, 'value', config_base)})."
        )


def _assert_lora_variant_matches_main(lora_config: object, main_config: object, lora_key: str) -> None:
    """Reject an A14B LoRA wired against a 5B main (and vice versa).

    A mismatch otherwise crashes deep in the layer patcher mid-denoise with an opaque
    tensor-shape error, after minutes of model loading. Skips silently when either
    variant is unrecorded (e.g. a LoRA whose targeted layers don't pin the inner dim).
    """
    lora_variant = getattr(lora_config, "variant", None)
    main_variant = getattr(main_config, "variant", None)
    if lora_variant is None or main_variant is None:
        return
    lora_is_5b = lora_variant == WanLoRAVariantType.Wan5B
    main_is_5b = main_variant == WanVariantType.TI2V_5B
    if lora_is_5b != main_is_5b:
        raise ValueError(
            f"LoRA '{lora_key}' targets Wan {lora_variant.value.upper()} models, but the "
            f"transformer is a {main_variant.value} model. A14B and 5B LoRAs are not interchangeable."
        )


def _warn_if_low_routing_is_inert(
    context: InvocationContext, main_config: object, lora_key: str, to_primary: bool, to_low_noise: bool
) -> None:
    """Warn when a LoRA is routed only to the low-noise list of a TI2V-5B main.

    The single-transformer TI2V-5B denoise path consumes only the primary list, so
    such a LoRA silently has no effect — the node would otherwise report success
    while doing nothing.
    """
    if to_primary or not to_low_noise:
        return
    if getattr(main_config, "variant", None) == WanVariantType.TI2V_5B:
        context.logger.warning(
            f"LoRA '{lora_key}' is routed only to the low-noise expert, which the single-transformer "
            "TI2V-5B variant never uses — the LoRA will have no effect."
        )


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
    version="1.0.1",
    classification=Classification.Prototype,
)
class WanLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA to the Wan 2.2 transformer(s).

    For A14B (dual expert) the LoRA's recorded ``expert`` field determines
    which expert list it lands in: ``"high"`` -> primary list, ``"low"`` ->
    low-noise list, ``None`` (untagged) -> both lists. Use the ``target``
    field to override.

    For TI2V-5B (single transformer) only the primary list is used at denoise
    time; a LoRA routed only to the low-noise list would be inert, so that
    routing logs a warning.
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

        lora_config = context.models.get_config(self.lora)
        _assert_is_wan_lora(lora_config, lora_key)

        output = WanLoRALoaderOutput()
        if self.transformer is None:
            return output

        main_config = context.models.get_config(self.transformer.transformer)
        _assert_lora_variant_matches_main(lora_config, main_config, lora_key)

        lora_expert = getattr(lora_config, "expert", None)
        to_primary, to_low_noise = _resolve_target(self.target, lora_expert)
        _warn_if_low_routing_is_inert(context, main_config, lora_key, to_primary, to_low_noise)

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
    version="1.0.1",
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
        ui_model_base=[BaseModelType.Wan],
        ui_model_type=ModelType.LoRA,
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
        main_config = context.models.get_config(self.transformer.transformer)

        for lora in loras:
            if lora is None or lora.lora.key in added:
                continue

            lora_key = lora.lora.key
            if not context.models.exists(lora_key):
                raise ValueError(f"Unknown lora: {lora_key}!")

            lora_config = context.models.get_config(lora.lora)
            _assert_is_wan_lora(lora_config, lora_key)
            _assert_lora_variant_matches_main(lora_config, main_config, lora_key)

            lora_expert = getattr(lora_config, "expert", None)
            to_primary, to_low_noise = _resolve_target("auto", lora_expert)
            _warn_if_low_routing_is_inert(context, main_config, lora_key, to_primary, to_low_noise)

            # Reject LoRAs already applied upstream (same invariant the single loader
            # enforces) — re-appending would silently double the effective weight.
            # Intra-collection duplicates are skipped via `added` before reaching here.
            if to_primary and any(item.lora.key == lora_key for item in output.transformer.loras):
                raise ValueError(f'LoRA "{lora_key}" already applied to primary transformer list.')
            if to_low_noise and any(item.lora.key == lora_key for item in output.transformer.loras_low_noise):
                raise ValueError(f'LoRA "{lora_key}" already applied to low-noise transformer list.')

            added.add(lora_key)

            if to_primary:
                output.transformer.loras.append(lora)
            if to_low_noise:
                output.transformer.loras_low_noise.append(lora)

        return output

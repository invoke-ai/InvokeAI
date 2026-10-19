from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import (
    MiniMaxH3AdaLNOverlayField,
    MiniMaxH3TransformerField,
    ModelIdentifierField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_MiniMaxH3_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

# The released H3 transformers have 50 blocks (0..49). The denoise node re-checks the range
# against the overlay file's real block count.
MINIMAX_H3_LAST_BLOCK = 49
MINIMAX_H3_HYBRID_DEFAULT_START_BLOCK = 25


@invocation_output("minimax_h3_hybrid_overlay_output")
class MiniMaxH3HybridOverlayOutput(BaseInvocationOutput):
    """MiniMax H3 hybrid AdaLN overlay output."""

    transformer: MiniMaxH3TransformerField = OutputField(
        description=FieldDescriptions.transformer, title="MiniMax H3 Transformer"
    )
    # Echoed so a graph can record the overlay it ran with; the input is Input.Direct, so a
    # duplicate literal typed into a metadata node would silently go stale.
    overlay_model: ModelIdentifierField = OutputField(
        description="The transformer checkpoint whose AdaLN projections were overlaid.", title="Overlay Model"
    )


@invocation(
    "minimax_h3_hybrid_overlay",
    title="Hybrid AdaLN Overlay - MiniMax H3",
    tags=["model", "minimax", "video"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3HybridOverlayInvocation(BaseInvocation):
    """Overlay the AdaLN modulation projections of a second MiniMax H3 task transformer onto the
    loaded one, blending the two shipped checkpoints.

    FL2VA has the better output quality; Ref2VA understands references but carries a training
    defect. The two differ almost only in their per-block AdaLN projections, so running FL2VA
    with Ref2VA's AdaLN projections for the upper blocks keeps FL2VA's quality while restoring
    reference conditioning. Load the FL2VA transformer in the model loader, select the Ref2VA
    checkpoint here; the result counts as a Ref2VA transformer and takes reference conditioning.
    """

    transformer: MiniMaxH3TransformerField = InputField(
        description="The base transformer (FL2VA for the recommended hybrid).",
        input=Input.Connection,
        title="Transformer",
    )
    overlay_model: ModelIdentifierField = InputField(
        description="Single-file MiniMax H3 transformer whose AdaLN projections are overlaid (Ref2VA for the "
        "recommended hybrid). Must match the base transformer's kind: both AdaLN-pruned, or both full.",
        input=Input.Direct,
        ui_model_base=BaseModelType.MiniMaxH3,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Checkpoint,
        title="Overlay Model",
    )
    start_block: int = InputField(
        default=MINIMAX_H3_HYBRID_DEFAULT_START_BLOCK,
        ge=0,
        le=MINIMAX_H3_LAST_BLOCK,
        description="First transformer block (inclusive, 0-49) whose AdaLN projection comes from the overlay. "
        "Lower values hand more of the model to the overlay: stronger reference adherence, more of its quality "
        "loss. 25 is the recommended balance.",
        title="Start Block",
    )
    end_block: int = InputField(
        default=MINIMAX_H3_LAST_BLOCK,
        ge=0,
        le=MINIMAX_H3_LAST_BLOCK,
        description="Last transformer block (inclusive, 0-49) whose AdaLN projection comes from the overlay.",
        title="End Block",
    )
    include_final_layer: bool = InputField(
        default=False,
        description="Also take the final-layer AdaLN projection from the overlay. It is the single most divergent "
        "tensor between the two checkpoints, so this maximizes reference behaviour at a visible quality cost.",
        title="Include Final Layer",
    )

    def invoke(self, context: InvocationContext) -> MiniMaxH3HybridOverlayOutput:
        if self.start_block > self.end_block and not self.include_final_layer:
            raise ValueError(
                f"The overlay selects nothing: start block {self.start_block} is after end block {self.end_block} "
                "and the final layer is excluded."
            )
        if self.transformer.adaln_overlay is not None:
            raise ValueError("This transformer already carries a hybrid AdaLN overlay; only one can be applied.")

        overlay_key = self.overlay_model.key
        if not context.models.exists(overlay_key):
            raise ValueError(f"Unknown overlay model: {overlay_key}")
        # The identifier's own base/type/format are client-supplied; only the resolved config is
        # authoritative.
        overlay_config = context.models.get_config(overlay_key)
        if not isinstance(overlay_config, Main_Checkpoint_MiniMaxH3_Config):
            raise ValueError(
                f"Model '{overlay_key}' is not a MiniMax H3 single-file transformer checkpoint (resolved to "
                f"type={getattr(overlay_config.type, 'value', overlay_config.type)}, "
                f"base={getattr(overlay_config.base, 'value', overlay_config.base)}, "
                f"format={getattr(overlay_config.format, 'value', overlay_config.format)})."
            )

        base_key = self.transformer.transformer.key
        if not context.models.exists(base_key):
            raise ValueError(f"Unknown transformer model: {base_key}")
        base_config = context.models.get_config(base_key)
        # A diffusers-folder transformer is always the full model; only single files come pruned.
        base_pruned = bool(getattr(base_config, "pruned", False))
        if base_pruned != overlay_config.pruned:
            kind = {True: "AdaLN-pruned", False: "full (non-pruned)"}
            raise ValueError(
                f"The overlay '{overlay_config.name}' is {kind[overlay_config.pruned]} but the transformer "
                f"'{base_config.name}' is {kind[base_pruned]}; their AdaLN projections have different shapes. "
                "Pair a pruned base with a pruned overlay, or a full base with a full overlay."
            )
        if overlay_key == base_key:
            raise ValueError("The overlay model is the transformer itself; the overlay would change nothing.")

        transformer = self.transformer.model_copy(deep=True)
        transformer.adaln_overlay = MiniMaxH3AdaLNOverlayField(
            overlay=self.overlay_model,
            start_block=self.start_block,
            end_block=self.end_block,
            include_final_layer=self.include_final_layer,
        )
        # The AdaLN projections route the modalities, so they decide the task the hybrid serves:
        # the denoise node's task/conditioning guard keys off this stamp.
        transformer.variant = overlay_config.variant.value

        return MiniMaxH3HybridOverlayOutput(transformer=transformer, overlay_model=self.overlay_model)

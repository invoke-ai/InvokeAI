import math

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, Krea2ConditioningField
from invokeai.app.invocations.primitives import Krea2ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.sampling_utils import KREA2_SELECT_LAYERS
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Krea2ConditioningInfo,
)

_NUM_TEXT_LAYERS = len(KREA2_SELECT_LAYERS)  # 12


@invocation(
    "krea2_conditioning_rebalance",
    title="Conditioning Rebalance - Krea-2",
    tags=["conditioning", "krea2", "krea-2"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Krea2ConditioningRebalanceInvocation(BaseInvocation):
    """Per-layer rebalancing of Krea-2 text conditioning (improves prompt adherence).

    Krea-2 conditioning stacks 12 Qwen3-VL hidden-state layers per token. Weighting those layers
    individually (and applying an overall multiplier) lets you push the model harder toward the prompt,
    counteracting the quality-dilution from distillation. Ported from the ComfyUI
    `ConditioningKrea2Rebalance` node. This is an optional pass between the text encoder and denoise.
    """

    conditioning: Krea2ConditioningField = InputField(
        description=FieldDescriptions.cond, input=Input.Connection, title="Conditioning"
    )
    per_layer_weights: str = InputField(
        default="1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0",
        description=f"Comma-separated gains for the {_NUM_TEXT_LAYERS} tapped encoder layers (exactly "
        f"{_NUM_TEXT_LAYERS} values).",
    )
    multiplier: float = InputField(
        default=4.0,
        allow_inf_nan=False,
        description="Overall multiplier applied to the conditioning after per-layer weighting.",
    )

    def _parse_weights(self) -> list[float]:
        try:
            weights = [float(x.strip()) for x in self.per_layer_weights.split(",") if x.strip() != ""]
        except ValueError as e:
            raise ValueError(f"per_layer_weights must be comma-separated numbers: {e}") from e
        if len(weights) != _NUM_TEXT_LAYERS:
            raise ValueError(f"per_layer_weights must have exactly {_NUM_TEXT_LAYERS} values, got {len(weights)}.")
        if not all(math.isfinite(weight) for weight in weights):
            raise ValueError("per_layer_weights must contain only finite values.")
        return weights

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Krea2ConditioningOutput:
        weights = self._parse_weights()

        cond_data = context.conditioning.load(self.conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        conditioning = cond_data.conditionings[0]
        assert isinstance(conditioning, Krea2ConditioningInfo)

        embeds = conditioning.prompt_embeds  # (B, seq, 12, hidden)
        gains = torch.tensor(weights, dtype=embeds.dtype, device=embeds.device).view(1, 1, _NUM_TEXT_LAYERS, 1)
        embeds = embeds * gains * self.multiplier

        new_data = ConditioningFieldData(
            conditionings=[
                Krea2ConditioningInfo(prompt_embeds=embeds, prompt_embeds_mask=conditioning.prompt_embeds_mask)
            ]
        )
        conditioning_name = context.conditioning.save(new_data)
        return Krea2ConditioningOutput.build(conditioning_name)

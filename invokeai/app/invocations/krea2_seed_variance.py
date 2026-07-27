import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, Krea2ConditioningField
from invokeai.app.invocations.primitives import Krea2ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Krea2ConditioningInfo,
)


@invocation(
    "krea2_seed_variance",
    title="Seed Variance - Krea-2",
    tags=["conditioning", "krea2", "krea-2", "variance"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Krea2SeedVarianceInvocation(BaseInvocation):
    """Inject per-seed diversity into Krea-2 text conditioning.

    Distilled few-step models (like Krea-2-Turbo) suffer from low seed variance — different seeds give
    near-identical images. This adds seeded uniform noise to a random subset of the text-embedding
    values, trading some prompt adherence for variety (the same idea as the Z-Image-Turbo
    `SeedVarianceEnhancer`). Optional pass between the text encoder and denoise.

    The noise magnitude is auto-calibrated relative to the embedding's standard deviation, so a given
    `strength` behaves consistently regardless of the embedding scale — in particular it stays sane
    whether or not the upstream Conditioning Rebalance node has scaled the embeddings up.
    """

    conditioning: Krea2ConditioningField = InputField(
        description=FieldDescriptions.cond, input=Input.Connection, title="Conditioning"
    )
    strength: float = InputField(
        default=0.1,
        ge=0.0,
        le=2.0,
        allow_inf_nan=False,
        description="Noise strength as a multiplier of the embedding std (0=off, 0.1=subtle, 0.5=strong).",
    )
    randomize_percent: float = InputField(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Percentage of embedding values that get perturbed (Bernoulli mask); 0 disables.",
    )
    variance_seed: int = InputField(default=0, description="Seed for the variance noise (vary this to get variety).")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Krea2ConditioningOutput:
        cond_data = context.conditioning.load(self.conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        conditioning = cond_data.conditionings[0]
        assert isinstance(conditioning, Krea2ConditioningInfo)

        embeds = conditioning.prompt_embeds  # (B, seq, 12, hidden)

        # No-op when the effect is disabled, so the node can stay wired in the graph without altering output.
        if self.strength == 0.0 or self.randomize_percent == 0.0:
            return Krea2ConditioningOutput.build(self.conditioning.conditioning_name)

        # Auto-calibrate the noise magnitude to the embedding scale (same approach as the Z-Image enhancer).
        # This keeps the perceptual effect of `strength` stable across prompts and, crucially, whether or not
        # the upstream rebalance has multiplied the embeddings up.
        embed_std = torch.std(embeds.to(torch.float32)).item()
        actual_strength = self.strength * embed_std

        generator = torch.Generator(device=embeds.device).manual_seed(self.variance_seed)
        noise = torch.rand(embeds.shape, generator=generator, dtype=torch.float32, device=embeds.device) * 2.0 - 1.0
        noise = noise * actual_strength
        mask = torch.bernoulli(
            torch.full(embeds.shape, self.randomize_percent / 100.0, dtype=torch.float32, device=embeds.device),
            generator=generator,
        )
        embeds = (embeds.to(torch.float32) + noise * mask).to(embeds.dtype)

        new_data = ConditioningFieldData(
            conditionings=[
                Krea2ConditioningInfo(prompt_embeds=embeds, prompt_embeds_mask=conditioning.prompt_embeds_mask)
            ]
        )
        conditioning_name = context.conditioning.save(new_data)
        return Krea2ConditioningOutput.build(conditioning_name)

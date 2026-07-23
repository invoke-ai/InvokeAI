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
    `SeedVarianceEnhancer`). Optional pass between the text encoder and denoise; the defaults are
    aggressive and may need tuning for Krea-2.
    """

    conditioning: Krea2ConditioningField = InputField(
        description=FieldDescriptions.cond, input=Input.Connection, title="Conditioning"
    )
    strength: float = InputField(
        default=20.0,
        allow_inf_nan=False,
        description="Magnitude of the uniform noise added to the embeddings (noise in [-strength, +strength]).",
    )
    randomize_percent: float = InputField(
        default=50.0,
        ge=1.0,
        le=100.0,
        description="Percentage of embedding values that get perturbed (Bernoulli mask).",
    )
    variance_seed: int = InputField(default=0, description="Seed for the variance noise (vary this to get variety).")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Krea2ConditioningOutput:
        cond_data = context.conditioning.load(self.conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        conditioning = cond_data.conditionings[0]
        assert isinstance(conditioning, Krea2ConditioningInfo)

        embeds = conditioning.prompt_embeds  # (B, seq, 12, hidden)
        generator = torch.Generator(device=embeds.device).manual_seed(self.variance_seed)
        noise = torch.rand(embeds.shape, generator=generator, dtype=torch.float32, device=embeds.device) * 2.0 - 1.0
        noise = noise * self.strength
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

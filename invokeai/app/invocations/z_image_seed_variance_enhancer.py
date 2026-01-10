import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    ZImageConditioningField,
)
from invokeai.app.invocations.primitives import ZImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    ZImageConditioningInfo,
)


@invocation(
    "z_image_seed_variance_enhancer",
    title="Seed Variance Enhancer - Z-Image",
    tags=["conditioning", "z-image", "variance", "seed"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ZImageSeedVarianceEnhancerInvocation(BaseInvocation):
    """Adds seed-based noise to Z-Image conditioning to increase variance between seeds.

    Z-Image-Turbo can produce relatively similar images with different seeds,
    making it harder to explore variations of a prompt. This node implements
    reproducible, seed-based noise injection into text embeddings to increase
    visual variation while maintaining reproducibility.

    The noise strength is auto-calibrated relative to the embedding's standard
    deviation, ensuring consistent results across different prompts.
    """

    conditioning: ZImageConditioningField = InputField(
        description=FieldDescriptions.cond,
        input=Input.Connection,
        title="Conditioning",
    )
    seed: int = InputField(
        default=0,
        ge=0,
        description="Seed for reproducible noise generation. Different seeds produce different noise patterns.",
    )
    strength: float = InputField(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Noise strength as multiplier of embedding std. 0=off, 0.1=subtle, 0.5=strong.",
    )
    randomize_percent: float = InputField(
        default=50.0,
        ge=1.0,
        le=100.0,
        description="Percentage of embedding values to add noise to (1-100). Lower values create more selective noise patterns.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ZImageConditioningOutput:
        # Load conditioning data
        cond_data = context.conditioning.load(self.conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1, "Expected exactly one conditioning tensor"
        z_image_conditioning = cond_data.conditionings[0]
        assert isinstance(z_image_conditioning, ZImageConditioningInfo), "Expected ZImageConditioningInfo"

        # Early return if strength is zero (no modification needed)
        if self.strength == 0:
            return ZImageConditioningOutput(conditioning=self.conditioning)

        # Clone embeddings to avoid modifying the original
        prompt_embeds = z_image_conditioning.prompt_embeds.clone()

        # Calculate actual noise strength based on embedding statistics
        # This auto-calibration ensures consistent results across different prompts
        embed_std = torch.std(prompt_embeds).item()
        actual_strength = self.strength * embed_std

        # Generate deterministic noise using the seed
        generator = torch.Generator(device=prompt_embeds.device)
        generator.manual_seed(self.seed)
        noise = torch.rand(prompt_embeds.shape, generator=generator, device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        noise = noise * 2 - 1  # Scale to [-1, 1]
        noise = noise * actual_strength

        # Create selective mask for noise application
        generator.manual_seed(self.seed + 1)
        noise_mask = torch.bernoulli(
            torch.ones_like(prompt_embeds) * (self.randomize_percent / 100.0),
            generator=generator,
        ).bool()

        # Apply noise only to masked positions
        prompt_embeds = prompt_embeds + (noise * noise_mask)

        # Save modified conditioning
        new_conditioning = ZImageConditioningInfo(prompt_embeds=prompt_embeds)
        conditioning_data = ConditioningFieldData(conditionings=[new_conditioning])
        conditioning_name = context.conditioning.save(conditioning_data)

        return ZImageConditioningOutput(
            conditioning=ZImageConditioningField(
                conditioning_name=conditioning_name,
                mask=self.conditioning.mask,
            )
        )

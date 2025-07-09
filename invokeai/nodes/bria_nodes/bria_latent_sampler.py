import torch

from invokeai.app.invocations.fields import Input, InputField
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import (
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    LatentsField,
    OutputField,
)
from invokeai.backend.model_manager.config import MainDiffusersConfig
from invokeai.invocation_api import (
    BaseInvocation,
    Classification,
    InputField,
    InvocationContext,
    invocation,
    invocation_output,
)


@invocation_output("bria_latent_sampler_output")
class BriaLatentSamplerInvocationOutput(BaseInvocationOutput):
    """Base class for nodes that output a CogView text conditioning tensor."""

    latents: LatentsField = OutputField(description=FieldDescriptions.cond)
    latent_image_ids: LatentsField = OutputField(description=FieldDescriptions.cond)


@invocation(
    "bria_latent_sampler",
    title="Latent Sampler - Bria",
    tags=["image", "bria"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class BriaLatentSamplerInvocation(BaseInvocation):
    seed: int = InputField(
        default=42,
        title="Seed",
        description="The seed to use for the latent sampler",
    )
    transformer: TransformerField = InputField(
        description="Bria model (Transformer) to load",
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> BriaLatentSamplerInvocationOutput:
        device = torch.device("cuda")
        transformer_config = context.models.get_config(self.transformer.transformer)
        if not isinstance(transformer_config, MainDiffusersConfig):
            raise ValueError("Transformer config is not a MainDiffusersConfig")
        # TODO: get latent channels from transformer config
        latent_channels = 16
        latent_height, latent_width = 128, 128
        shrunk = latent_channels // 4
        gen = torch.Generator(device=device).manual_seed(self.seed)

        noise4d = torch.randn((1, shrunk, latent_height, latent_width), device=device, generator=gen)
        latents = noise4d.view(1, shrunk, latent_height // 2, 2, latent_width // 2, 2).permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(1, (latent_height // 2) * (latent_width // 2), shrunk * 4)

        latent_image_ids = torch.zeros((latent_height // 2, latent_width // 2, 3), device=device, dtype=torch.long)
        latent_image_ids[..., 1] = torch.arange(latent_height // 2, device=device)[:, None]
        latent_image_ids[..., 2] = torch.arange(latent_width // 2, device=device)[None, :]
        latent_image_ids = latent_image_ids.view(-1, 3)

        saved_latents_tensor = context.tensors.save(latents)
        saved_latent_image_ids_tensor = context.tensors.save(latent_image_ids)
        latents_output = LatentsField(latents_name=saved_latents_tensor)
        latent_image_ids_output = LatentsField(latents_name=saved_latent_image_ids_tensor)

        return BriaLatentSamplerInvocationOutput(
            latents=latents_output,
            latent_image_ids=latent_image_ids_output,
        )

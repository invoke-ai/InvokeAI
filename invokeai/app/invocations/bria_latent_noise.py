import torch
from pydantic import BaseModel, Field 

from invokeai.app.invocations.fields import Input, InputField, OutputField
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import (
    BaseInvocationOutput,
    FieldDescriptions,
    LatentsField,
)
from invokeai.backend.bria.pipeline_bria_controlnet import prepare_latents
from invokeai.invocation_api import (
    BaseInvocation,
    Classification,
    InvocationContext,
    invocation,
    invocation_output,
)


@invocation_output("bria_latent_noise_output")
class BriaLatentNoiseInvocationOutput(BaseInvocationOutput):
    """Base class for nodes that output Bria latent tensors."""
    latents: LatentsField = OutputField(description="The latent noise")
    latent_image_ids: LatentsField = OutputField(description="The latent image ids.")
    height: int = OutputField(description="The height of the output image")
    width: int = OutputField(description="The width of the output image")

@invocation(
    "bria_latent_noise",
    title="Latent Noise - Bria",
    tags=["image", "bria"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class BriaLatentNoiseInvocation(BaseInvocation):
    """ Generate latent noise for Bria. """

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
    height: int = InputField(
        default=1024,
        title="Height",
        description="The height of the output image",
    )
    width: int = InputField(
        default=1024,
        title="Width",
        description="The width of the output image",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> BriaLatentNoiseInvocationOutput:
        with context.models.load(self.transformer.transformer) as transformer:
            device = transformer.device
            dtype = transformer.dtype

        generator = torch.Generator(device=device).manual_seed(self.seed)

        num_channels_latents = 4
        latents, latent_image_ids = prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=self.height,
            width=self.width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        saved_latents_tensor = context.tensors.save(latents)
        saved_latent_image_ids_tensor = context.tensors.save(latent_image_ids)
        latents_output = LatentsField(latents_name=saved_latents_tensor)
        latent_image_ids_output = LatentsField(latents_name=saved_latent_image_ids_tensor)

        return BriaLatentNoiseInvocationOutput(
            latents=latents_output,
            latent_image_ids=latent_image_ids_output,
            height=self.height,
            width=self.width,
        )

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
from invokeai.backend.bria.pipeline_bria_controlnet import prepare_latents
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
        height, width = 1024, 1024
        generator = torch.Generator(device=device).manual_seed(self.seed)
        
        num_channels_latents = 4  # due to patch=2, we devide by 4
        latents, latent_image_ids = prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
            )

        saved_latents_tensor = context.tensors.save(latents)
        saved_latent_image_ids_tensor = context.tensors.save(latent_image_ids)
        latents_output = LatentsField(latents_name=saved_latents_tensor)
        latent_image_ids_output = LatentsField(latents_name=saved_latent_image_ids_tensor)

        return BriaLatentSamplerInvocationOutput(
            latents=latents_output,
            latent_image_ids=latent_image_ids_output,
        )

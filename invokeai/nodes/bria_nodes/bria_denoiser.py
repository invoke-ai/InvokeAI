import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from invokeai.app.invocations.fields import Input, InputField
from invokeai.app.invocations.model import SubModelType, TransformerField
from invokeai.app.invocations.primitives import (
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    OutputField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.invocation_api import BaseInvocation, Classification, InputField, invocation, invocation_output

from invokeai.backend.bria.pipeline import get_original_sigmas, retrieve_timesteps
from invokeai.backend.bria.transformer_bria import BriaTransformer2DModel

@invocation_output("bria_denoise_output")
class BriaDenoiseInvocationOutput(BaseInvocationOutput):
    latents: LatentsField = OutputField(description=FieldDescriptions.latents)


@invocation(
    "bria_denoise",
    title="Denoise - Bria",
    tags=["image", "bria"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class BriaDenoiseInvocation(BaseInvocation):
    num_steps: int = InputField(
        default=30, title="Number of Steps", description="The number of steps to use for the denoiser"
    )
    guidance_scale: float = InputField(
        default=5.0, title="Guidance Scale", description="The guidance scale to use for the denoiser"
    )

    transformer: TransformerField = InputField(
        description="Bria model (Transformer) to load",
        input=Input.Connection,
        title="Transformer",
    )
    latents: LatentsField = InputField(
        description="Latents to denoise",
        input=Input.Connection,
        title="Latents",
    )
    latent_image_ids: LatentsField = InputField(
        description="Latent Image IDs to denoise",
        input=Input.Connection,
        title="Latent Image IDs",
    )
    pos_embeds: LatentsField = InputField(
        description="Positive Prompt Embeds",
        input=Input.Connection,
        title="Positive Prompt Embeds",
    )
    neg_embeds: LatentsField = InputField(
        description="Negative Prompt Embeds",
        input=Input.Connection,
        title="Negative Prompt Embeds",
    )
    text_ids: LatentsField = InputField(
        description="Text IDs",
        input=Input.Connection,
        title="Text IDs",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> BriaDenoiseInvocationOutput:
        latents = context.tensors.load(self.latents.latents_name)
        pos_embeds = context.tensors.load(self.pos_embeds.latents_name)
        neg_embeds = context.tensors.load(self.neg_embeds.latents_name)
        text_ids = context.tensors.load(self.text_ids.latents_name)
        latent_image_ids = context.tensors.load(self.latent_image_ids.latents_name)
        scheduler_identifier = self.transformer.transformer.model_copy(update={"submodel_type": SubModelType.Scheduler})

        device = None
        dtype = None
        with (
            context.models.load(self.transformer.transformer) as transformer,
            context.models.load(scheduler_identifier) as scheduler,
        ):
            assert isinstance(transformer, BriaTransformer2DModel)
            assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
            dtype = transformer.dtype
            device = transformer.device
            latents, pos_embeds, neg_embeds = map(lambda x: x.to(device, dtype), (latents, pos_embeds, neg_embeds))
            prompt_embeds = torch.cat([neg_embeds, pos_embeds]) if self.guidance_scale > 1 else pos_embeds

            sigmas = get_original_sigmas(1000, self.num_steps)
            timesteps, _ = retrieve_timesteps(scheduler, self.num_steps, device, None, sigmas, mu=0.0)

            for t in timesteps:
                # Prepare model input efficiently
                if self.guidance_scale > 1:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents
                
                # Prepare timestep tensor efficiently
                if isinstance(t, torch.Tensor):
                    timestep_tensor = t.expand(latent_model_input.shape[0])
                else:
                    timestep_tensor = torch.tensor([t] * latent_model_input.shape[0], device=device, dtype=torch.float32)

                noise_pred = transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep_tensor,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=None,
                        return_dict=False,
                    )[0]

                if self.guidance_scale > 1:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

                # Convert timestep for scheduler
                t_step = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
                
                # Use scheduler step with proper dtypes
                latents = scheduler.step(noise_pred, t_step, latents, return_dict=False)[0]

        assert isinstance(latents, torch.Tensor)
        saved_input_latents_tensor = context.tensors.save(latents)
        latents_output = LatentsField(latents_name=saved_input_latents_tensor)
        return BriaDenoiseInvocationOutput(latents=latents_output)

from typing import List, Tuple
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from invokeai.backend.bria.controlnet_bria import BriaControlModes, BriaMultiControlNetModel
from invokeai.backend.bria.controlnet_utils import prepare_control_images
from invokeai.nodes.bria_nodes.bria_controlnet import BriaControlNetField

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from invokeai.app.invocations.fields import Input, InputField, LatentsField, OutputField
from invokeai.app.invocations.model import SubModelType, TransformerField, VAEField
from invokeai.app.invocations.primitives import BaseInvocationOutput, FieldDescriptions
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
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
        title="VAE",
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
    control: BriaControlNetField | list[BriaControlNetField] | None = InputField(
        description="ControlNet",
        input=Input.Connection,
        title="ControlNet",
        default = None,
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
            context.models.load(self.vae.vae) as vae,
        ):
            assert isinstance(transformer, BriaTransformer2DModel)
            assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
            assert isinstance(vae, AutoencoderKL)
            dtype = transformer.dtype
            device = transformer.device
            latents, pos_embeds, neg_embeds = map(lambda x: x.to(device, dtype), (latents, pos_embeds, neg_embeds))
            prompt_embeds = torch.cat([neg_embeds, pos_embeds]) if self.guidance_scale > 1 else pos_embeds

            sigmas = get_original_sigmas(1000, self.num_steps)
            timesteps, _ = retrieve_timesteps(scheduler, self.num_steps, device, None, sigmas, mu=0.0)
            width, height = 1024, 1024
            if self.control is not None:
                control_model, control_images, control_modes, control_scales = self._prepare_multi_control(
                        context=context,
                        vae=vae,
                        width=width,
                        height=height,
                        device=device,
                        
                    )

            for t in timesteps:
                # Prepare model input efficiently
                if self.guidance_scale > 1:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents
                
                timestep_tensor = t.expand(latent_model_input.shape[0])

                controlnet_block_samples, controlnet_single_block_samples = None, None
                if self.control is not None:
                    controlnet_block_samples, controlnet_single_block_samples = control_model(
                        hidden_states=latents,
                        controlnet_cond=control_images, # type: ignore
                        controlnet_mode=control_modes, # type: ignore
                        conditioning_scale=control_scales, # type: ignore
                        timestep=timestep_tensor,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )

                noise_pred = transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep_tensor,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=None,
                        return_dict=False,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
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



    def _prepare_multi_control(
        self,
        context: InvocationContext,
        vae: AutoencoderKL,
        width: int,
        height: int,
        device: torch.device
    ) -> Tuple[BriaMultiControlNetModel, List[torch.Tensor], List[torch.Tensor], List[float]]:

        control = self.control if isinstance(self.control, list) else [self.control]
        control_images, control_models, control_modes, control_scales = [], [], [], []
        for controlnet in control:
            if controlnet is not None:
                control_models.append(context.models.load(controlnet.model).model)
                control_images.append(context.images.get_pil(controlnet.image.image_name))
                control_modes.append(BriaControlModes[controlnet.mode].value)   
                control_scales.append(controlnet.conditioning_scale)
        
        control_model = BriaMultiControlNetModel(control_models).to(device)
        tensored_control_images, tensored_control_modes = prepare_control_images(
            vae=vae,
            control_images=control_images, 
            control_modes=control_modes, 
            width=width,
            height=height,
            device=device, 
            )
        return control_model, tensored_control_images, tensored_control_modes, control_scales
        
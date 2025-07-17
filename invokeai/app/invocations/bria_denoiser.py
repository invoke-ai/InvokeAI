from typing import List, Tuple
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from invokeai.backend.bria.controlnet_bria import BriaControlModes, BriaMultiControlNetModel
from invokeai.backend.bria.controlnet_utils import prepare_control_images
from invokeai.backend.bria.pipeline_bria_controlnet import BriaControlNetPipeline
from invokeai.app.invocations.bria_controlnet import BriaControlNetField

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from invokeai.app.invocations.fields import Input, InputField, LatentsField, OutputField
from invokeai.app.invocations.model import SubModelType, T5EncoderField, TransformerField, VAEField
from invokeai.app.invocations.primitives import BaseInvocationOutput, FieldDescriptions
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.invocation_api import BaseInvocation, Classification, invocation, invocation_output
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
    t5_encoder: T5EncoderField = InputField(
        title="T5Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
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
            context.models.load(self.t5_encoder.text_encoder) as t5_encoder,
            context.models.load(self.t5_encoder.tokenizer) as t5_tokenizer,
        ):
            assert isinstance(transformer, BriaTransformer2DModel)
            assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
            assert isinstance(vae, AutoencoderKL)
            dtype = transformer.dtype
            device = transformer.device
            latents, pos_embeds, neg_embeds = map(lambda x: x.to(device, dtype), (latents, pos_embeds, neg_embeds))

            control_model, control_images, control_modes, control_scales = None, None, None, None
            if self.control is not None:
                control_model, control_images, control_modes, control_scales = self._prepare_multi_control(
                    context=context,
                    vae=vae,
                    width=1024,
                    height=1024,
                    device=vae.device,
                )

            pipeline = BriaControlNetPipeline(
                transformer=transformer,
                scheduler=scheduler,
                vae=vae,
                text_encoder=t5_encoder,
                tokenizer=t5_tokenizer,
                controlnet=control_model,
            )
            pipeline.to(device=transformer.device, dtype=transformer.dtype)

            latents = pipeline(
                control_image=control_images,
                control_mode=control_modes,
                width=1024,
                height=1024,
                controlnet_conditioning_scale=control_scales,
                num_inference_steps=self.num_steps, 
                max_sequence_length=128,
                guidance_scale=self.guidance_scale,
                latents=latents,
                latent_image_ids=latent_image_ids,
                text_ids=text_ids,
                prompt_embeds=pos_embeds,
                negative_prompt_embeds=neg_embeds,
                output_type="latent",
            )[0]

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
                control_modes.append(BriaControlModes[controlnet.mode].value)   
                control_scales.append(controlnet.conditioning_scale)
                try:
                    control_images.append(context.images.get_pil(controlnet.image.image_name))
                except:
                    raise FileNotFoundError(f"Control image {controlnet.image.image_name} not found. Make sure not to delete the preprocessed image before finishing the pipeline.")

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
        
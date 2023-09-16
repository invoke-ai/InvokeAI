from typing import List, Union

import torch
from diffusers import StableDiffusionUpscalePipeline

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    UIType,
    invocation,
)
from invokeai.app.invocations.image import ImageOutput
from invokeai.app.invocations.latent import SAMPLER_NAME_VALUES, get_scheduler
from invokeai.app.invocations.metadata import CoreMetadata
from invokeai.app.invocations.model import UNetField, VaeField
from invokeai.app.invocations.primitives import ConditioningField, ImageField
from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.util.step_callback import stable_diffusion_step_callback
from invokeai.backend import BaseModelType
from invokeai.backend.stable_diffusion import ConditioningData, PipelineIntermediateState, PostprocessingSettings


@invocation("upscale_sdx4", title="Upscale (Stable Diffusion x4)", tags=["upscale"], version="0.1.0")
class UpscaleLatentsInvocation(BaseInvocation):
    """Upscales an image using an upscaling diffusion model.

    https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

    The upscaling model is its own thing, independent of other Stable Diffusion text-to-image
    models. We don't have ControlNet or LoRA support for it. It has its own VAE.
    """

    # Inputs
    image: ImageField = InputField(description="The image to upscale")

    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    steps: int = InputField(default=10, gt=0, description=FieldDescriptions.steps)
    cfg_scale: Union[float, List[float]] = InputField(
        default=7.5, ge=1, description=FieldDescriptions.cfg_scale, ui_type=UIType.Float
    )
    scheduler: SAMPLER_NAME_VALUES = InputField(default="euler", description=FieldDescriptions.scheduler)
    seed: int = InputField(default=0, description=FieldDescriptions.seed)

    unet: UNetField = InputField(description=FieldDescriptions.unet, input=Input.Connection)
    vae: VaeField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    metadata: CoreMetadata = InputField(default=None, description=FieldDescriptions.core_metadata)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)

    @torch.inference_mode()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        model_manager = context.services.model_manager
        unet_info = model_manager.get_model(**self.unet.unet.dict(), context=context)
        vae_info = model_manager.get_model(**self.vae.vae.dict(), context=context)

        with unet_info as unet, vae_info as vae:
            # don't re-use the same scheduler instance for both fields
            low_res_scheduler = get_scheduler(context, self.unet.scheduler, self.scheduler, self.seed ^ 0xFFFFFFFF)
            scheduler = get_scheduler(context, self.unet.scheduler, self.scheduler, self.seed ^ 0xF7F7F7F7)

            conditioning_data = self.get_conditioning_data(context, scheduler, unet, self.seed)

            pipeline = StableDiffusionUpscalePipeline(
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                unet=unet,
                low_res_scheduler=low_res_scheduler,
                scheduler=scheduler,
            )

            if self.tiled or context.services.configuration.tiled_decode:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            generator = torch.Generator().manual_seed(self.seed)

            output = pipeline(
                image=image,
                # latents=noise,
                num_inference_steps=self.steps,
                guidance_scale=self.cfg_scale,
                # noise_level =
                generator=generator,
                prompt_embeds=conditioning_data.text_embeddings.embeds.data,
                negative_prompt_embeds=conditioning_data.unconditioned_embeddings.embeds.data,
                output_type="pil",
                callback=lambda *args: self.dispatch_upscale_progress(context, *args),
            )
            result_image = output.images[0]

        image_dto = context.services.images.create(
            image=result_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata.dict() if self.metadata else None,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )

    def get_conditioning_data(
        self,
        context: InvocationContext,
        scheduler,
        unet,
        seed,
    ) -> ConditioningData:
        # FIXME: duplicated from DenoiseLatentsInvocation.get_conditoning_data
        positive_cond_data = context.services.latents.get(self.positive_conditioning.conditioning_name)
        c = positive_cond_data.conditionings[0].to(device=unet.device, dtype=unet.dtype)
        extra_conditioning_info = c.extra_conditioning

        negative_cond_data = context.services.latents.get(self.negative_conditioning.conditioning_name)
        uc = negative_cond_data.conditionings[0].to(device=unet.device, dtype=unet.dtype)

        conditioning_data = ConditioningData(
            unconditioned_embeddings=uc,
            text_embeddings=c,
            guidance_scale=self.cfg_scale,
            extra=extra_conditioning_info,
            postprocessing_settings=PostprocessingSettings(
                threshold=0.0,  # threshold,
                warmup=0.2,  # warmup,
                h_symmetry_time_pct=None,  # h_symmetry_time_pct,
                v_symmetry_time_pct=None,  # v_symmetry_time_pct,
            ),
        )

        conditioning_data = conditioning_data.add_scheduler_args_if_applicable(
            scheduler,
            # for ddim scheduler
            eta=0.0,  # ddim_eta
            # for ancestral and sde schedulers
            # FIXME: why do we need both a generator here and a seed argument to get_scheduler?
            generator=torch.Generator(device=unet.device).manual_seed(seed ^ 0xFFFFFFFF),
        )
        return conditioning_data

    def dispatch_upscale_progress(self, context, step, timestep, latents):
        graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
        source_node_id = graph_execution_state.prepared_source_mapping[self.id]
        intermediate_state = PipelineIntermediateState(
            step=step,
            order=1,  # FIXME: fudging this, but why does it need both order and total-steps anyway?
            total_steps=self.steps,
            timestep=timestep,
            latents=latents,
        )
        stable_diffusion_step_callback(
            context=context,
            intermediate_state=intermediate_state,
            node=self.dict(),
            source_node_id=source_node_id,
            base_model=BaseModelType.StableDiffusionXLRefiner,  # FIXME: this upscaler needs its own model type
        )

from typing import Literal, Optional

import torch
from diffusers import StableDiffusionUpscalePipeline
from pydantic import Field

from invokeai.app.invocations.baseinvocation import InvocationConfig, InvocationContext
from invokeai.app.invocations.image import ImageOutput
from invokeai.app.invocations.latent import TextToLatentsInvocation, get_scheduler
from invokeai.app.invocations.metadata import CoreMetadata
from invokeai.app.invocations.model import VaeField
from invokeai.app.models.image import ImageField, ResourceOrigin, ImageCategory
from invokeai.app.util.step_callback import stable_diffusion_step_callback
from invokeai.backend.stable_diffusion import PipelineIntermediateState


class UpscaleLatentsInvocation(TextToLatentsInvocation):
    """Upscales an image using an upscaling diffusion model.

    https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

    The upscaling model is its own thing, independent of other Stable Diffusion text-to-image
    models. We don't have ControlNet or LoRA support for it. It has its own VAE.
    """

    type: Literal["upscale_sdx4"] = "upscale_sdx4"

    # Inputs
    image: Optional[ImageField] = Field(description="The image to upscale")
    vae: VaeField = Field(default=None, description="VAE submodel")
    metadata: Optional[CoreMetadata] = Field(default=None, description="Optional core metadata to be written to the image")
    tiled: bool = Field(
        default=False,
        description="Decode latents by overlapping tiles(less memory consumption)")
    # TODO: fp32: bool = Field(DEFAULT_PRECISION=='float32', description="Decode in full precision")
    # FIXME: We inherited the `control` field from the superclass, but don't support it.

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Upscale (Stable Diffusion x4)",
                "tags": ["scale"],
                "type_hints": {
                    "model": "model",
                    "cfg_scale": "number",
                }
            }
        }

    @torch.inference_mode()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        model_manager = context.services.model_manager
        unet_info = model_manager.get_model(**self.unet.unet.dict(), context=context)
        vae_info = model_manager.get_model(**self.vae.vae.dict(), context=context)

        with unet_info as unet, vae_info as vae:
            # don't re-use the same scheduler instance for both fields
            low_res_scheduler = get_scheduler(context, self.unet.scheduler, self.scheduler)
            scheduler = get_scheduler(context, self.unet.scheduler, self.scheduler)

            conditioning_data = self.get_conditioning_data(context, scheduler, unet)

            # TODO: https://github.com/huggingface/diffusers/issues/4349
            class FakeEncoder:
                class FakeEncoderConfig:
                    pass

                config = FakeEncoderConfig()
                dtype = unet.dtype

            pipeline = StableDiffusionUpscalePipeline(
                vae=vae,
                text_encoder=FakeEncoder(),
                tokenizer=None,
                unet=unet,
                low_res_scheduler=low_res_scheduler,
                scheduler=scheduler
            )

            if self.tiled or context.services.configuration.tiled_decode:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            output = pipeline(
                image=image,
                # latents=noise,
                num_inference_steps = self.steps,
                guidance_scale = self.cfg_scale,
                # noise_level =
                # generator =
                prompt_embeds=conditioning_data.text_embeddings,
                negative_prompt_embeds=conditioning_data.unconditioned_embeddings,
                output_type="pil",
                callback = lambda *args: self.dispatch_upscale_progress(context, *args)
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
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )

    def dispatch_upscale_progress(self, context, step, timestep, latents):
        graph_execution_state = context.services.graph_execution_manager.get(
            context.graph_execution_state_id
        )
        source_node_id = graph_execution_state.prepared_source_mapping[self.id]
        intermediate_state = PipelineIntermediateState(None, step, timestep, latents)
        stable_diffusion_step_callback(
            context=context,
            intermediate_state=intermediate_state,
            node=self.dict(),
            source_node_id=source_node_id,
        )

from pathlib import Path
from typing import Literal
from pydantic import Field

import accelerate
import torch
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from invokeai.app.invocations.model import TransformerField, VAEField
from optimum.quanto import qfloat8
from PIL import Image
from safetensors.torch import load_file
from transformers.models.auto import AutoModelForTextEncoding

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    WithBoard,
    WithMetadata,
    UIType,
)
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4
from invokeai.backend.quantization.fast_quantized_diffusion_model import FastQuantizedDiffusersModel
from invokeai.backend.quantization.fast_quantized_transformers_model import FastQuantizedTransformersModel
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice

TFluxModelKeys = Literal["flux-schnell"]
FLUX_MODELS: dict[TFluxModelKeys, str] = {"flux-schnell": "black-forest-labs/FLUX.1-schnell"}


class QuantizedFluxTransformer2DModel(FastQuantizedDiffusersModel):
    base_class = FluxTransformer2DModel


class QuantizedModelForTextEncoding(FastQuantizedTransformersModel):
    auto_class = AutoModelForTextEncoding


@invocation(
    "flux_text_to_image",
    title="FLUX Text to Image",
    tags=["image"],
    category="image",
    version="1.0.0",
)
class FluxTextToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Text-to-image generation using a FLUX model."""

    transformer: TransformerField = InputField(
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="Transformer",
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    positive_text_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_steps: int = InputField(default=4, description="Number of diffusion steps.")
    guidance: float = InputField(
        default=4.0,
        description="The guidance strength. Higher values adhere more strictly to the prompt, and will produce less diverse images.",
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:

        # Load the conditioning data.
        cond_data = context.conditioning.load(self.positive_text_conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        flux_conditioning = cond_data.conditionings[0]
        assert isinstance(flux_conditioning, FLUXConditioningInfo)

        latents = self._run_diffusion(context, flux_conditioning.clip_embeds, flux_conditioning.t5_embeds)
        image = self._run_vae_decoding(context, latents)
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)

    def _run_diffusion(
        self,
        context: InvocationContext,
        clip_embeddings: torch.Tensor,
        t5_embeddings: torch.Tensor,
    ):
        scheduler_info = context.models.load(self.transformer.scheduler)
        transformer_info = context.models.load(self.transformer.transformer)

        # HACK(ryand): Manually empty the cache. Currently we don't check the size of the model before loading it from
        # disk. Since the transformer model is large (24GB), there's a good chance that it will OOM on 32GB RAM systems
        # if the cache is not empty.
        # context.models._services.model_manager.load.ram_cache.make_room(24 * 2**30)

        with transformer_info as transformer, scheduler_info as scheduler:
            assert isinstance(transformer, FluxTransformer2DModel)
            assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)

            x = denoise(
                model=transformer,
                img=img,
                img_ids=img_ids,
                txt=t5_embeddings,
                txt_ids=txt_ids,
                vec=clip_embeddings,
                timesteps=timesteps,
                guidance=self.guidance,
            )

        x = unpack(x.float(), self.height, self.width)

        return x

    def _prepare_latent_img_patches(self, latent_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert an input image in latent space to patches for diffusion.

        This implementation was extracted from:
        https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/sampling.py#L32

        Returns:
            tuple[Tensor, Tensor]: (img, img_ids), as defined in the original flux repo.
        """
        bs, c, h, w = latent_img.shape

        # Pixel unshuffle with a scale of 2, and flatten the height/width dimensions to get an array of patches.
        img = rearrange(latent_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)

        # Generate patch position ids.
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        return img, img_ids

    def _run_vae_decoding(
        self,
        context: InvocationContext,
        latents: torch.Tensor,
    ) -> Image.Image:
        vae_info = context.models.load(self.vae.vae)
        with vae_info as vae:
            assert isinstance(vae, AutoencoderKL)

        img.clamp(-1, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

            latents = flux_pipeline_with_vae._unpack_latents(
                latents, self.height, self.width, flux_pipeline_with_vae.vae_scale_factor
            )
            latents = (
                latents / flux_pipeline_with_vae.vae.config.scaling_factor
            ) + flux_pipeline_with_vae.vae.config.shift_factor
            latents = latents.to(dtype=vae.dtype)
            image = flux_pipeline_with_vae.vae.decode(latents, return_dict=False)[0]
            image = flux_pipeline_with_vae.image_processor.postprocess(image, output_type="pil")[0]

        assert isinstance(image, Image.Image)
        return image

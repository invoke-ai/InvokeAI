import torch

from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.app.invocations.latent import DenoiseLatentsInvocation, ImageToLatentsInvocation, LatentsToImageInvocation
from invokeai.app.invocations.model import UNetField, VaeField
from invokeai.app.invocations.noise import get_noise
from invokeai.app.invocations.primitives import ConditioningField
from invokeai.backend.tiled_refinement.refiners.base_refiner import BaseRefiner
from invokeai.backend.util.devices import choose_torch_device


class ImageToImageRefiner(BaseRefiner):
    def __init__(
        self,
        context: InvocationContext,
        positive_conditioning: ConditioningField,
        negative_conditioning: ConditioningField,
        vae: VaeField,
        unet: UNetField,
        denoising_start: float,
        denoising_end: float,
    ):
        super().__init__()
        self._context = context
        self._positive_conditioning = positive_conditioning
        self._negative_conditioning = negative_conditioning
        self._vae = vae
        self._unet = unet
        self._denoising_start = denoising_start
        self._denoising_end = denoising_end

    def refine(self, image_tile: torch.Tensor) -> torch.Tensor:
        # VAE Encode
        image_to_latents = ImageToLatentsInvocation(vae=self._vae)
        vae_info = self._context.services.model_manager.get_model(
            **self._vae.vae.model_dump(),
            context=self._context,
        )
        latents = image_to_latents.vae_encode(vae_info=vae_info, upcast=False, tiled=False, image_tensor=image_tile)

        # UNet Denoise
        denoise_latents = DenoiseLatentsInvocation(
            positive_conditioning=self._positive_conditioning,
            negative_conditioning=self._negative_conditioning,
            steps=20,
            denoising_start=self._denoising_start,
            enoising_end=self._denoising_end,
            # scheduler="euler",
            unet=self._unet,
        )
        noise = get_noise(
            width=latents.shape[-1],
            height=latents.shape[-2],
            device=choose_torch_device(),
            seed=0,
            downsampling_factor=1,
        )
        refined_latents = denoise_latents.denoise(
            context=self._context, latents=latents, noise=noise, seed=0, step_callback=lambda x: None
        )

        # VAE Decode
        latents_to_image = LatentsToImageInvocation(vae=self._vae)
        refined_image_tile = latents_to_image.vae_decode(context=self._context, latents=refined_latents)

        return refined_image_tile

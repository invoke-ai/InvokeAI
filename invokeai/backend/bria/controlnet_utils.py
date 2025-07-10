from typing import List, Tuple
from PIL import Image
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from diffusers.image_processor import VaeImageProcessor

import torch



@torch.no_grad()
def prepare_control_images(
    vae: AutoencoderKL,
    control_images: list[Image.Image],
    control_modes: list[int],
    width: int,
    height: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    
    tensored_control_images = []
    tensored_control_modes = []
    for idx, control_image_ in enumerate(control_images):
        tensored_control_image = _prepare_image(
            image=control_image_,
            width=width,
            height=height,
            device=device,
            dtype=vae.dtype,
        )
        height, width = tensored_control_image.shape[-2:]

        # vae encode
        tensored_control_image = vae.encode(tensored_control_image).latent_dist.sample()
        tensored_control_image = (tensored_control_image) * vae.config.scaling_factor

        # pack
        height_control_image, width_control_image = tensored_control_image.shape[2:]
        tensored_control_image = _pack_latents(
            tensored_control_image,
            height_control_image,
            width_control_image,
        )
        tensored_control_images.append(tensored_control_image)
        tensored_control_modes.append(torch.tensor(control_modes[idx]).expand(
            tensored_control_image.shape[0]).to(device, dtype=torch.long))

    return tensored_control_images, tensored_control_modes

def _prepare_image(
    image: Image.Image,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    image = image.convert("RGB")
    image = VaeImageProcessor(vae_scale_factor=16).preprocess(image, height=height, width=width)
    image = image.repeat_interleave(1, dim=0)
    image = image.to(device=device, dtype=dtype)
    return image

def _pack_latents(latents, height, width):
    latents = latents.view(1, 4, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(1, (height // 2) * (width // 2), 16)

    return latents


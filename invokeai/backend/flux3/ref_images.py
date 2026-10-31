"""FLUX.3 reference images: fitted to the token budget, VAE-encoded, normalized and packed.

Unlike FLUX.2, a reference is VAE-encoded untiled wherever the encoder's attention runs fused, as both reference
pipelines encode it: tiling moves every token (see ``encode_reference_image``).

The size budget follows the reference implementation (``src/flux3/pipeline.py``, Apache-2.0): a single
reference may have up to 4 MP, each of several up to 1 MP, and every side lies between 64 and 4096 px. The
reference reflect-pads a single reference to the 16 px grid and crops the output back afterwards; InvokeAI
sets the output size itself, so every reference is center-cropped to the grid instead, as diffusers does.

References are clean conditioning content: they enter the transformer as a token stream of their own at
timestep 0, told apart from each other and from the generated image on the ``l`` position axis
(-1, -2, ...).
"""

import math

import torch
from PIL import Image, ImageOps

from invokeai.backend.flux2.latent_norm import bn_normalize
from invokeai.backend.flux2.ref_image_extension import encode_reference_image
from invokeai.backend.flux2.sampling_utils import pack_flux2
from invokeai.backend.flux3.sampling import image_position_ids
from invokeai.backend.model_manager.load.load_base import LoadedModel

FLUX3_MAX_REFERENCE_IMAGES = 10
SINGLE_REFERENCE_MAX_PIXELS = 4 * 1024 * 1024
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024
REFERENCE_MIN_SIDE = 64
REFERENCE_MAX_SIDE = 4096
_GRID = 16  # VAE downsampling (8) times the 2x2 packing


def fit_reference_image(image: Image.Image, num_references: int) -> Image.Image:
    """``image`` as RGB, scaled into the size budget, center-cropped to a multiple of 16 px per side."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if max(width, height) / min(width, height) > REFERENCE_MAX_SIDE / REFERENCE_MIN_SIDE:
        raise ValueError(
            f"A {width}x{height} reference is too elongated: FLUX.3 takes references with sides between "
            f"{REFERENCE_MIN_SIDE} and {REFERENCE_MAX_SIDE} px, an aspect ratio of at most "
            f"{REFERENCE_MAX_SIDE // REFERENCE_MIN_SIDE}:1."
        )
    max_pixels = SINGLE_REFERENCE_MAX_PIXELS if num_references == 1 else MULTI_REFERENCE_MAX_PIXELS
    scale = min(1.0, math.sqrt(max_pixels / (width * height)), REFERENCE_MAX_SIDE / max(width, height))
    # A tiny reference is scaled up instead: below 64 px a side has fewer than four tokens to carry it.
    scale = max(scale, REFERENCE_MIN_SIDE / min(width, height))
    if scale != 1.0:
        # Round up when scaling up, so a side scaled to the 64 px minimum cannot land on 63 and crop to 48.
        to_int = math.ceil if scale > 1.0 else int
        width, height = max(1, to_int(width * scale)), max(1, to_int(height * scale))
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    crop_w, crop_h = width // _GRID * _GRID, height // _GRID * _GRID
    if (crop_w, crop_h) != (width, height):
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        image = image.crop((left, top, left + crop_w, top + crop_h))
    return image


def encode_flux3_references(
    images: list[Image.Image],
    vae_info: LoadedModel,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference token stream and its position ids: ``(1, S, 128)`` in ``dtype`` and ``(S, 4)`` float32."""
    if len(images) > FLUX3_MAX_REFERENCE_IMAGES:
        raise ValueError(f"FLUX.3 takes at most {FLUX3_MAX_REFERENCE_IMAGES} reference images, got {len(images)}.")

    tokens: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    for index, image in enumerate(images):
        fitted = fit_reference_image(image, len(images))
        latents = encode_reference_image(vae_info, fitted, tiling="if_attention_materializes").float()
        packed = bn_normalize(pack_flux2(latents).to(device), bn_mean, bn_std)
        tokens.append(packed.to(dtype))
        ids.append(
            image_position_ids(fitted.height // _GRID, fitted.width // _GRID, l_coord=-(1 + index), device=device)
        )
    return torch.cat(tokens, dim=1), torch.cat(ids, dim=0)

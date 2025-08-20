from typing import Literal

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.flux.modules.autoencoder import AutoEncoder


def estimate_vae_working_memory_sd15_sdxl(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKL | AutoencoderTiny,
    tile_size: int | None,
    fp32: bool,
) -> int:
    """Estimate the working memory required to encode or decode the given tensor."""
    # It was found experimentally that the peak working memory scales linearly with the number of pixels and the
    # element size (precision). This estimate is accurate for both SD1 and SDXL.
    element_size = 4 if fp32 else 2

    # This constant is determined experimentally and takes into consideration both allocated and reserved memory. See #8414
    # Encoding uses ~45% the working memory as decoding.
    scaling_constant = 2200 if operation == "decode" else 1100

    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1

    if tile_size is not None:
        if tile_size == 0:
            tile_size = vae.tile_sample_min_size
            assert isinstance(tile_size, int)
        h = tile_size
        w = tile_size
        working_memory = h * w * element_size * scaling_constant

        # We add 25% to the working memory estimate when tiling is enabled to account for factors like tile overlap
        # and number of tiles. We could make this more precise in the future, but this should be good enough for
        # most use cases.
        working_memory = working_memory * 1.25
    else:
        h = latent_scale_factor_for_operation * image_tensor.shape[-2]
        w = latent_scale_factor_for_operation * image_tensor.shape[-1]
        working_memory = h * w * element_size * scaling_constant

    if fp32:
        # If we are running in FP32, then we should account for the likely increase in model size (~250MB).
        working_memory += 250 * 2**20

    print(f"estimate_vae_working_memory_sd15_sdxl: {int(working_memory)}")

    return int(working_memory)


def estimate_vae_working_memory_cogview4(
    operation: Literal["encode", "decode"], image_tensor: torch.Tensor, vae: AutoencoderKL
) -> int:
    """Estimate the working memory required by the invocation in bytes."""
    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1

    h = latent_scale_factor_for_operation * image_tensor.shape[-2]
    w = latent_scale_factor_for_operation * image_tensor.shape[-1]
    element_size = next(vae.parameters()).element_size()

    # This constant is determined experimentally and takes into consideration both allocated and reserved memory. See #8414
    # Encoding uses ~45% the working memory as decoding.
    scaling_constant = 2200 if operation == "decode" else 1100
    working_memory = h * w * element_size * scaling_constant

    print(f"estimate_vae_working_memory_cogview4: {int(working_memory)}")

    return int(working_memory)


def estimate_vae_working_memory_flux(
    operation: Literal["encode", "decode"], image_tensor: torch.Tensor, vae: AutoEncoder
) -> int:
    """Estimate the working memory required by the invocation in bytes."""

    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1

    out_h = latent_scale_factor_for_operation * image_tensor.shape[-2]
    out_w = latent_scale_factor_for_operation * image_tensor.shape[-1]
    element_size = next(vae.parameters()).element_size()

    # This constant is determined experimentally and takes into consideration both allocated and reserved memory. See #8414
    # Encoding uses ~45% the working memory as decoding.
    scaling_constant = 2200 if operation == "decode" else 1100

    working_memory = out_h * out_w * element_size * scaling_constant

    print(f"estimate_vae_working_memory_flux: {int(working_memory)}")

    return int(working_memory)


def estimate_vae_working_memory_sd3(
    operation: Literal["encode", "decode"], image_tensor: torch.Tensor, vae: AutoencoderKL
) -> int:
    """Estimate the working memory required by the invocation in bytes."""
    # Encode operations use approximately 50% of the memory required for decode operations

    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1

    h = latent_scale_factor_for_operation * image_tensor.shape[-2]
    w = latent_scale_factor_for_operation * image_tensor.shape[-1]
    element_size = next(vae.parameters()).element_size()

    # This constant is determined experimentally and takes into consideration both allocated and reserved memory. See #8414
    # Encoding uses ~45% the working memory as decoding.
    scaling_constant = 2200 if operation == "decode" else 1100

    working_memory = h * w * element_size * scaling_constant

    print(f"estimate_vae_working_memory_sd3: {int(working_memory)}")

    return int(working_memory)

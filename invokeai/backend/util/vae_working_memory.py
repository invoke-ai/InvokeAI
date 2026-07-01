from typing import Literal

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
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


def estimate_vae_working_memory_qwen_image(
    operation: Literal["encode", "decode"], image_tensor: torch.Tensor, vae: AutoencoderKLQwenImage
) -> int:
    """Estimate the working memory required by the invocation in bytes.

    The Qwen Image VAE is a video-style autoencoder that operates on 5D tensors of shape
    (B, C, num_frames, H, W). Tiling is not used, so peak working memory scales with the full
    spatial output. The two trailing dimensions are the spatial H/W in latent space (decode) or
    pixel space (encode), matching the convention used by the other estimators here.
    """
    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1

    h = latent_scale_factor_for_operation * image_tensor.shape[-2]
    w = latent_scale_factor_for_operation * image_tensor.shape[-1]
    element_size = next(vae.parameters()).element_size()

    # The Qwen Image VAE is much heavier than the SD/SDXL VAE and needs correspondingly larger
    # constants. These were calibrated by measuring peak *reserved* memory growth (not just allocated
    # -- reserved is what the cache's `free >= estimate` check compares against) across a resolution
    # grid in fp16, on both an AMD W7900 (ROCm) and an NVIDIA card (CUDA). See
    # scripts/calibrate_qwen_vae_working_memory.py.
    #
    # Implied constant = reserved_bytes / (h * w * element_size). Per-point maxima (fp16):
    #              512^2  768^2  1024^2  1536^2  1792^2  2048^2    -> ship (max observed + ~8% headroom)
    #   ROCm decode  5132   4596   4570    3273    3735    4813    -> 5500
    #   ROCm encode  5864   5858   5858    3532    4364   (OOM)    -> 6300
    #   CUDA decode  2660   2519   2690    2671    2281   (OOM)    -> 2900
    #   CUDA encode  1456   1451   1458    1456    1455    1455    -> 1600
    #
    # Why this branches on backend (the only estimator here that does):
    #  - The Qwen VAE is attention-heavy. With Flash/efficient attention (CUDA) the attention memory
    #    is O(area) and the curve is flat/linear; the ROCm build falls back to math attention, which
    #    is O(area^2), so ROCm reserves ~2x (decode) to ~4x (encode) more and goes super-linear above
    #    ~1792^2. The two backends differ far more than any headroom, so a single constant would
    #    either under-estimate on ROCm (OOM) or massively over-budget on CUDA (needless eviction).
    #  - "Encoding is half of decoding" (as the sibling estimators assume) is only true on CUDA. On
    #    ROCm encode reserves >= decode, so the ROCm encode constant is sized accordingly -- this is
    #    the path Qwen Image Edit exercises.
    #  - On ROCm the linear model under-estimates for decodes well above 2048^2, but those OOM on a
    #    48GB card regardless; on CUDA the curve stays linear so no extra term is needed.
    is_rocm = torch.version.hip is not None
    if operation == "decode":
        scaling_constant = 5500 if is_rocm else 2900
    else:  # encode
        scaling_constant = 6300 if is_rocm else 1600

    working_memory = h * w * element_size * scaling_constant

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

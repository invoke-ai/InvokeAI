from typing import Literal

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.flux.modules.autoencoder import AutoEncoder

_WAN_VAE_SINGLE_FRAME_DECODE_SCALING_CONSTANT = 2900
_WAN_VAE_VIDEO_DECODE_SCALING_CONSTANT_A14B = 6500
_WAN_VAE_VIDEO_DECODE_SCALING_CONSTANT_TI2V = 7000


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


def estimate_vae_working_memory_anima(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKLWan,
    tile_size: int | None,
) -> int:
    """Estimate the working memory required to encode or decode with the Wan 2.1 VAE (Anima).

    The Wan VAE uses 3D convolutions and needs noticeably more working memory per output
    pixel than the 2D VAEs estimated above. Calibrated empirically on a 1024x1024 fp16
    decode: peak reserved memory was ~5.95GB for a full decode and ~1.73GB with 512px
    tiles (384px stride), i.e. ~2900 bytes per output pixel per element byte. Encoding
    follows the house ratio of ~50% of decode.
    """
    element_size = next(vae.parameters()).element_size()
    scaling_constant = 2900 if operation == "decode" else 1450

    if tile_size is not None:
        h = tile_size
        w = tile_size
        # Add 25% to account for tile overlap.
        working_memory = h * w * element_size * scaling_constant * 1.25
    else:
        latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1
        h = latent_scale_factor_for_operation * image_tensor.shape[-2]
        w = latent_scale_factor_for_operation * image_tensor.shape[-1]
        working_memory = h * w * element_size * scaling_constant

    return int(working_memory)


# Bytes of chunk working set per output pixel per element byte, measured at ~12 on a W7900 (a
# 768x1344 28-frame chunk peaks at 1.31 GiB allocated in fp32) and rounded up for other canvases.
MINIMAX_H3_CHUNK_BYTES_PER_PIXEL = 14

# The caching allocator holds more than is live — measured 1.3-1.7x across tile-heavy decodes — and
# the reservation has to cover what it holds.
MINIMAX_H3_ALLOCATOR_HEADROOM = 1.6


def estimate_vae_working_memory_minimax_h3(
    operation: Literal["encode", "decode"],
    vae: "torch.nn.Module",
    pixel_height: int,
    pixel_width: int,
    pixel_frames: int,
) -> int:
    """Estimate the working memory to encode/decode with the MiniMax H3 video VAE.

    Two terms, both measured rather than borrowed from Wan (whose VAE is a pixel-resolution conv
    stack decoded one frame at a time; H3's decoder is a ViT over the 16x16 latent grid, so Wan's
    per-pixel constant is orders of magnitude too large here):

    - **Chunk term.** The VAE processes one temporal chunk at a time, spatially tiled
      (``use_tiling`` defaults to True with 256px tiles; the released frames are the blended-tile
      ones). ``_decode_clip`` materializes ``tokens_chunk_size + token_overlap`` latent frames — 28
      pixel frames with the released geometry — as tile activations, the accumulated tile rows and
      the stitched chunk; ``_encode_clip`` does the same over ``clip_length`` frames. It therefore
      scales with chunk frames x canvas, not with clip length.
    - **Clip term.** ``_decode`` accumulates every chunk and then concatenates, so two copies of
      the whole RGB clip are live at the peak. Encode keeps one.

    The sum is scaled by :data:`MINIMAX_H3_ALLOCATOR_HEADROOM` because the reservation has to cover
    what the caching allocator *holds*, not what is live: across the hundreds of tile calls in a
    long clip, reserved runs ~1.3-1.7x allocated from block rounding and fragmentation. Ignoring
    that gap is what made a 243-frame 768x1344 decode fail — it needed 7.92 GiB reserved against a
    6.49 GiB reservation, and since partial loading packs VRAM up to exactly this number, the
    shortfall was a hard failure rather than a near miss (on ROCm it surfaces from hipBLAS as
    HIPBLAS_STATUS_INTERNAL_ERROR, so neither the caller nor the cache recognizes it as an OOM).

    Calibrated 2026-08-09 on a W7900 (gfx1100, fp32, real released config), peak reserved:
    one 256px tile at 28 frames 0.56 GiB; one 768x1344 chunk 1.81 GiB; a full 768x1344 decode
    4.37 GiB at 90 frames and 7.92 GiB at 243 frames. This formula returns ~1.3-1.4x those.
    """
    element_size = next(vae.parameters()).element_size()

    if operation == "decode":
        tokens_chunk_size = int(getattr(vae, "tokens_chunk_size", 5))
        token_overlap = int(getattr(vae, "token_overlap", 2))
        temporal_ratio = int(getattr(vae, "temporal_compression_ratio", 4))
        chunk_frames = (tokens_chunk_size + token_overlap) * temporal_ratio
    else:
        chunk_frames = int(getattr(getattr(vae, "config", None), "clip_length", 17))
    # A clip with fewer frames than a chunk cannot fill one.
    chunk_frames = max(1, min(chunk_frames, pixel_frames))

    chunk_bytes = chunk_frames * pixel_height * pixel_width * element_size * MINIMAX_H3_CHUNK_BYTES_PER_PIXEL

    clip_copies = 2 if operation == "decode" else 1
    clip_bytes = clip_copies * 3 * pixel_frames * pixel_height * pixel_width * element_size

    return int((chunk_bytes + clip_bytes) * MINIMAX_H3_ALLOCATOR_HEADROOM)


def estimate_vae_working_memory_wan(
    operation: Literal["encode", "decode"],
    vae: AutoencoderKLWan,
    pixel_height: int,
    pixel_width: int,
    pixel_frames: int,
    tile_size: int | None = None,
    streaming: bool = False,
) -> int:
    """Estimate the working memory required to encode or decode with a Wan VAE.

    Callers pass pixel-space dimensions, so the VAE's spatial scale factor is already
    applied. Single-frame decode and encode use the original Wan 2.1 calibration;
    multi-frame decode uses conservative, VAE-variant-specific calibrations because
    causal-convolution state makes the single-frame value unsafe at video resolutions.
    The Wan VAE processes the clip causally, one latent frame at a time with cached
    features. In streaming mode, only one temporal-upscale chunk of the RGB output is
    kept on the execution device; otherwise the full output clip and its transient copy
    are budgeted.
    """
    element_size = next(vae.parameters()).element_size()

    # The original 2900-byte calibration covers a single Wan 2.1 frame. Multi-frame video
    # decodes retain causal-convolution state that makes that constant unsafe at video
    # resolutions. These conservative constants are based on measured allocated-memory
    # peaks with allocator headroom: 6500 for the z_dim=16 A14B VAE and 7000 for the
    # larger z_dim=48 TI2V VAE. Keep the single-frame value for image decode and the
    # existing encode calibration.
    if operation == "decode" and pixel_frames > 1:
        try:
            z_dim = int(getattr(vae.config, "z_dim", 16))
        except (TypeError, ValueError):
            z_dim = 48
        scaling_constant = (
            _WAN_VAE_VIDEO_DECODE_SCALING_CONSTANT_TI2V if z_dim >= 32 else _WAN_VAE_VIDEO_DECODE_SCALING_CONSTANT_A14B
        )
    else:
        scaling_constant = _WAN_VAE_SINGLE_FRAME_DECODE_SCALING_CONSTANT if operation == "decode" else 1450
    if tile_size is not None:
        # Add 25% for tile overlap.
        per_frame = tile_size * tile_size * element_size * scaling_constant * 1.25
    else:
        per_frame = pixel_height * pixel_width * element_size * scaling_constant

    # Streaming decode moves each causal decoder chunk to CPU immediately. Only one
    # temporal-upscale chunk remains on the execution device, instead of the full RGB
    # clip plus the transient copy created by torch.cat.
    if operation == "decode" and streaming:
        temporal_scale = int(getattr(vae.config, "scale_factor_temporal", None) or 4)
        resident_frames = min(pixel_frames, temporal_scale)
        clip_copies = 1
    else:
        resident_frames = pixel_frames
        clip_copies = 2 if operation == "decode" else 1
    clip_bytes = clip_copies * 3 * resident_frames * pixel_height * pixel_width * element_size

    return int(per_frame + clip_bytes)


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
    #  - XPU (Intel Arc) takes the CUDA constants deliberately, not by omission. Measured on
    #    Arc Pro B70 / torch 2.13+xpu: SDPA peak memory doubles when the sequence length doubles
    #    (2.00x at 2048 -> 4096 -> 8192 -> 16384, 2.0 MB at seq=16384 against 512 MB for a
    #    materialised seq^2 score matrix), i.e. XPU gets an efficient kernel and is in the same
    #    O(area) regime as CUDA. If a future driver regresses to math attention, this branch --
    #    not the constants -- is what needs to change.
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

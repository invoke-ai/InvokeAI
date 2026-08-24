from typing import Literal

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.devices import TorchDevice

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


# The FLUX.2 VAE runs one attention block at the bottom of the encoder and one at the top of the
# decoder, on the 8x-downsampled grid. Both are single-head, with the head dim set to the block
# width: 512 for the stock VAE and 384 for the small-decoder variant. The distinction does not
# matter here -- what matters is that both sit far above the 128 head dim ROCm's fused SDPA kernels
# accept, so only the value's side of that limit is load-bearing.
_FLUX2_VAE_MID_BLOCK_HEADS = 1
_FLUX2_VAE_MID_BLOCK_HEAD_DIM = 512
_FLUX2_VAE_SPATIAL_COMPRESSION = 8


def estimate_vae_working_memory_flux2(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKLFlux2,
    tile_size: int | None = None,
    device: torch.device | None = None,
) -> int:
    """Estimate the working memory required to encode or decode with the FLUX.2 (32-channel) VAE.

    Peak memory scales linearly with pixel area and element size, as it does for the FLUX.1 VAE.
    Measured on CUDA/bf16 as peak *reserved* memory (the conservative quantity, including allocator
    overhead), the implied constants are ~2170 (decode) and ~1070 (encode) bytes per pixel per
    element byte, flat across 512-1536px; the constants below round those up and match the FLUX.1
    ones. For reference, decoding 1024x1024 peaks at ~4.3GB and 1536x1536 at ~9.6GB -- far above the
    default ``device_working_mem_gb``, which is why this estimate must be passed to the model cache.

    That linear term holds only while ``AutoencoderKLFlux2``'s mid-block attention runs through a
    fused SDPA kernel, which is what CUDA does (verified: the memory-efficient kernel takes the
    512-wide head, and measured peak stays linear from 512 to 1536px). A build whose fused kernels
    reject the head dim -- ROCm caps it at 128 -- drops to SDPA's ``math`` fallback and materializes
    a (pixels/8)^2 score matrix on top of the linear term: ~3.5GB at 1024px and ~17GB at 1536px. We
    ask torch which path applies rather than assuming, so the estimate is right on both.

    When tiling is enabled the peak is bounded by a single tile instead of the full image (measured
    ~0.55GB flat at a 512px tile, from 1024px up to the 2024px reference-image cap), and the score
    matrix, if one is materialized at all, is bounded by the tile too.
    """
    param = next(vae.parameters())
    element_size = param.element_size()

    # Encoding uses ~50% the working memory of decoding.
    scaling_constant = 2200 if operation == "decode" else 1100

    if tile_size is not None:
        # Add 25% for tile overlap and the blending buffers, mirroring the SD1/SDXL estimate.
        working_memory = tile_size * tile_size * element_size * scaling_constant * 1.25
        mid_block_seq_len = (tile_size // _FLUX2_VAE_SPATIAL_COMPRESSION) ** 2
    else:
        latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1
        out_h = latent_scale_factor_for_operation * image_tensor.shape[-2]
        out_w = latent_scale_factor_for_operation * image_tensor.shape[-1]
        working_memory = out_h * out_w * element_size * scaling_constant
        mid_block_seq_len = (out_h // _FLUX2_VAE_SPATIAL_COMPRESSION) * (out_w // _FLUX2_VAE_SPATIAL_COMPRESSION)

    working_memory += sdpa_score_matrix_bytes(
        device=device if device is not None else TorchDevice.choose_torch_device(),
        dtype=param.dtype,
        num_heads=_FLUX2_VAE_MID_BLOCK_HEADS,
        head_dim=_FLUX2_VAE_MID_BLOCK_HEAD_DIM,
        seq_len=mid_block_seq_len,
    )

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
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKLQwenImage,
    tile_size: int | None = None,
) -> int:
    """Estimate the working memory required by the invocation in bytes.

    The Qwen Image VAE is a video-style autoencoder that operates on 5D tensors of shape
    (B, C, num_frames, H, W). The two trailing dimensions are the spatial H/W in latent space
    (decode) or pixel space (encode), matching the convention used by the other estimators here.

    Without tiling, peak working memory scales with the full spatial extent. With tiling it is
    bounded by a single tile instead, so the estimate must follow suit — otherwise the cache keeps
    reserving the full-frame figure (~11.8 GB for a 2560x1440 encode on CUDA) and tiling buys
    nothing. Mirrors ``estimate_vae_working_memory_wan``: one tile plus 25% for the tile overlap,
    plus the pixel-space buffers, which stay resident on the execution device either way.

    ``tile_size`` is the resolved tile size (the nodes' 0 sentinel already substituted), and assumes
    the 4:3 tile-to-stride ratio applied by ``patch_qwen_image_vae_tiling``.
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

    if tile_size is not None and tile_size > 0:
        # Bounded by one tile (plus overlap) rather than the full frame.
        working_memory = tile_size * tile_size * element_size * scaling_constant * 1.25
        # The full RGB image is the encode input / decode output and stays resident regardless. Unlike
        # the per-tile term this scales with the output area, so it is the term that decides whether the
        # estimate still holds at the resolutions tiling exists for.
        #
        # `tiled_decode` holds several pixel-space copies at once: every decoded tile in `rows`
        # ((tile_min / tile_stride)^2 ~ 1.8 frames at the 4:3 ratio the nodes set), the blended and
        # cropped `result_rows` (~1 frame) and the final `torch.cat` output (~1 frame). Measured at
        # ~5 frames on a 2560x1440 fp16 decode. Encode consumes its input image without duplicating it,
        # and accumulates only latents (16 channels at 1/64 the area — negligible).
        image_copies = 5 if operation == "decode" else 1
        working_memory += image_copies * 3 * h * w * element_size
    else:
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

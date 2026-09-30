from typing import Literal

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_tiling_scope import resolve_tile_size

# The diffusers AutoencoderKL (SD1/SDXL, SD3, CogView4) and the FLUX.1 AutoEncoder run the same
# mid-block self-attention as the FLUX.2 VAE: one head over the 512-channel width, on the
# 8x-downsampled grid. Where that attention runs on the math kernel (see `sdpa_score_matrix_bytes`)
# its score matrix is the dominant term for a large untiled image, exactly as it is for FLUX.2;
# `_vae_mid_block_score_matrix_bytes` prices it for those estimators the same way. The video VAEs
# (Wan, Qwen-Image) need no such term: their ROCm constants below were measured with math attention.
_CLASSIC_VAE_MID_BLOCK_HEADS = 1
_CLASSIC_VAE_MID_BLOCK_HEAD_DIM = 512

# Fixed cost of the LTX-2 VAE's own buffers, and the marginal cost of a tile pixel-element.
# See `estimate_vae_working_memory_ltx2` for the measurements these are fitted to.
_LTX2_VAE_BASE_BYTES = 512 * 2**20
_LTX2_AUDIO_BASE_BYTES = 128 * 2**20
_LTX2_AUDIO_BYTES_PER_LATENT = 6.5 * 2**20
_LTX2_VAE_DECODE_BYTES_PER_TILE_ELEMENT = 466
_LTX2_DECODE_CLIP_COPIES = 3
_LTX2_VAE_ENCODE_BYTES_PER_TILE_ELEMENT = 700
_LTX2_VAE_TILED_ENCODE_BYTES_PER_TILE_ELEMENT = 360

_WAN_VAE_SINGLE_FRAME_DECODE_SCALING_CONSTANT = 2900
_WAN_VAE_VIDEO_DECODE_SCALING_CONSTANT_A14B = 6500
_WAN_VAE_VIDEO_DECODE_SCALING_CONSTANT_TI2V = 7000


def _vae_mid_block_score_matrix_bytes(
    out_h: int, out_w: int, dtype: torch.dtype, batch_size: int = 1, device: torch.device | None = None
) -> int:
    """Score-matrix bytes for a classic VAE's mid-block attention over an `out_h` x `out_w` output."""
    return sdpa_score_matrix_bytes(
        device=device if device is not None else TorchDevice.choose_torch_device(),
        dtype=dtype,
        num_heads=_CLASSIC_VAE_MID_BLOCK_HEADS * batch_size,
        head_dim=_CLASSIC_VAE_MID_BLOCK_HEAD_DIM,
        seq_len=(out_h // LATENT_SCALE_FACTOR) * (out_w // LATENT_SCALE_FACTOR),
    )


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

    if isinstance(vae, AutoencoderKL):
        # max, not sum: the mid-block attention and the full-resolution convolutions peak in
        # different phases of the forward (see the FLUX.2 estimator for the measurements).
        score_dtype = torch.float32 if fp32 else next(vae.parameters()).dtype
        working_memory = max(working_memory, _vae_mid_block_score_matrix_bytes(h, w, score_dtype))

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

    # max, not sum: the mid-block attention and the full-resolution convolutions peak in different
    # phases of the forward (see the FLUX.2 estimator for the measurements).
    return int(max(working_memory, _vae_mid_block_score_matrix_bytes(h, w, next(vae.parameters()).dtype)))


# What a tiled decode does *not* bound: the assembled image, several times over, per output pixel.
#
# `AutoencoderKL.tiled_decode` assembles entirely on the device: every decoded tile stays live in
# `rows` while the cropped rows and then the concatenated result are built. On top of that, a decode
# node runs `clamp(-1, 1)`, `+ 1.0` and `* 127.5` over the result and then `.byte()` it: more
# element-size RGB buffers plus the 8-bit one.
#
# The measurement below was taken when this estimator also served InvokeAI's own port of the BFL
# autoencoder, which merged on the *host* and therefore had a far smaller slope. That class is gone;
# the constant is kept at the value that covers the diffusers slope, which is the one that remains.
#
# Measured on this repo's fixtures (RTX 4090, fp32), device peak over baseline against one full
# image, at 1024px and 2048px:
#
#   InvokeAI AutoEncoder   196.0 -> 200.0 MiB   image 12 -> 48 MiB   marginal slope 0.11
#   diffusers AutoencoderKL 178.7 -> 235.1 MiB  image 12 -> 48 MiB   marginal slope 1.57
#
# The BFL class merges on the *host*, so almost nothing image-sized is on the device until the
# final `stack(...).to(device)`, by which point the tile activations have been freed; the diffusers
# class assembles on-device and grows at ~1.6 copies. Both peaks are dominated by the tile term
# above, which is why the earlier value of 7 -- derived rather than measured -- over-reserved by
# roughly 7x on the node it was named for. Four covers the measured slope plus the node's in-window
# post-processing, and one constant still covers both classes.
_FLUX_VAE_TILED_IMAGE_COPIES = 4
_IMAGE_CHANNELS = 3

# An encoder emits mean and logvar before `DiagonalGaussianDistribution` splits them, so the tensor
# a tiled encode assembles is twice the latent width.
_VAE_MOMENT_CHANNELS_PER_LATENT = 2


def estimate_vae_working_memory_flux(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKL,
    tile_size: int | None = None,
) -> int:
    """Estimate the working memory required by the invocation in bytes.

    `tile_size` is in output pixels and defaults to None, i.e. a single-pass decode -- the existing
    call sites that pass no tile depend on that signature. When set, the decode term is bounded by
    one tile instead of the whole image, and the full-resolution assembly and post-processing that
    tiling does *not* bound are added on top. `tile_size <= 0` is the nodes' "use the default"
    sentinel; see `resolve_tile_size`.

    Both operations tile. The tile-bounded term is the same for either; what tiling does *not*
    bound differs, so the residual is chosen by `operation` -- see `_FLUX_VAE_TILED_IMAGE_COPIES`
    for the decode side.
    """

    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1
    element_size = next(vae.parameters()).element_size()

    # This constant is determined experimentally and takes into consideration both allocated and reserved memory. See #8414
    # Encoding uses ~45% the working memory as decoding.
    scaling_constant = 2200 if operation == "decode" else 1100

    out_h = latent_scale_factor_for_operation * image_tensor.shape[-2]
    out_w = latent_scale_factor_for_operation * image_tensor.shape[-1]

    # Resolved against a module-level constant, never by reading `vae.tile_sample_min_size`: the VAE
    # belongs to the model cache, so that attribute reflects whatever the previous invocation set
    # rather than the default this node is asking for.
    tile = resolve_tile_size(tile_size) if tile_size is not None else None

    # `_tiled_decode` short-circuits to a single pass once the tile covers the image on both axes,
    # so a tile that large has to be priced as the untiled decode it will actually run. The node
    # field carries no upper bound: a 2048px tile on a 1024px image would otherwise reserve ~23GB
    # for a decode that needs ~4.6GB, evicting the transformer from the cache for nothing.
    if tile is not None and (tile < out_h or tile < out_w):
        # `calc_tiles_min_overlap` clamps the tile to the image *per axis*, so the largest tile the
        # decode builds is this, not `tile` squared. Without the clamp an 8192px tile on a
        # 1024x8200 image -- tiled, because one axis exceeds it -- prices 8192x8192.
        tile_h = min(tile, out_h)
        tile_w = min(tile, out_w)
        # A 25% margin for tile overlap and the number of tiles, mirroring the SD1/SDXL estimator.
        working_memory = tile_h * tile_w * element_size * scaling_constant * 1.25
        if operation == "decode":
            working_memory += out_h * out_w * _IMAGE_CHANNELS * (_FLUX_VAE_TILED_IMAGE_COPIES * element_size + 1)
        else:
            # A tiled encode leaves two things un-bounded: the full-resolution image it slices tiles
            # from, and the moments it assembles.
            #
            # The moments are *not* one copy. `AutoencoderKL._tiled_encode` appends every encoded
            # tile to `rows`, then builds `result_rows` from them -- blending against `rows[i-1][j]`,
            # so `rows` stays live throughout -- and finally concatenates `enc` on top of both. The
            # tiles overlap, so their total area is `1 / (1 - overlap_factor)**2` times the output;
            # `result_rows` and `enc` are one output each. Hence the factor below, which is 3.78 at
            # the 0.25 every `AutoencoderKL` ships.
            #
            # Read from the VAE rather than hard-coded: an overlap factor of 0.5 would make the
            # tiles alone four times the output, and a fixed 3.78 would then under-reserve. An
            # estimate consumed as `free >= estimate` has to err upwards.
            config = getattr(vae, "config", None)
            latent_channels = getattr(config, "latent_channels", 16)
            overlap_factor = getattr(vae, "tile_overlap_factor", 0.25)
            live_moment_copies = 1.0 / (1.0 - overlap_factor) ** 2 + 2.0
            moments = (
                (out_h // LATENT_SCALE_FACTOR)
                * (out_w // LATENT_SCALE_FACTOR)
                * _VAE_MOMENT_CHANNELS_PER_LATENT
                * latent_channels
                * element_size
            )
            working_memory += out_h * out_w * _IMAGE_CHANNELS * element_size
            working_memory += moments * live_moment_copies
        score_h, score_w = tile_h, tile_w
    else:
        working_memory = out_h * out_w * element_size * scaling_constant
        score_h, score_w = out_h, out_w

    # max, not sum: the mid-block attention and the full-resolution convolutions peak in different
    # phases of the forward (see the FLUX.2 estimator for the measurements).
    dtype = next(vae.parameters()).dtype
    return int(max(working_memory, _vae_mid_block_score_matrix_bytes(score_h, score_w, dtype)))


# The FLUX.2 VAE runs one attention block at the bottom of the encoder and one at the top of the
# decoder, on the 8x-downsampled grid. Both are single-head, with the head dim set to the block
# width: 512 for the stock VAE and 384 for the small-decoder variant. Either way it is far past the
# 128 head dim some builds cap their fused SDPA kernels at, so whether the score matrix is
# materialized is a per-build question -- `sdpa_score_matrix_bytes` asks rather than assumes.
_FLUX2_VAE_MID_BLOCK_HEADS = 1
_FLUX2_VAE_MID_BLOCK_HEAD_DIM = 512
_FLUX2_VAE_SPATIAL_COMPRESSION = 8

# Peak reserved bytes per output pixel per element byte, per conv backend and operation. Fitted with
# `scripts/calibrate_flux2_working_memory.py --only vae`, which reproduces this table on any build:
#
#                                     decode   encode   encode/decode
#   cuDNN   RTX 4090, torch 2.7.1       2185     1072       0.49
#   MIOpen  RX 9070 XT, torch 2.10      3453     2688       0.78
#   MIOpen  PRO W7900, torch 2.10       3525     2688       0.76
#
# Two AMD generations (RDNA3 and RDNA4), the same numbers: the encode column agrees to the byte and
# the decode column to 2%. So this is MIOpen, not a per-card quirk, and the column below is fitted
# to the larger with ~2% headroom.
#
# MIOpen's convolution workspaces are simply larger than cuDNN's. This is not the attention term --
# it shows up identically on the fused path, and the implied constants are flat across resolution on
# both backends, so the linear model itself holds. And the "encoding costs half of decoding" ratio
# the other estimators in this module use turns out to be a cuDNN property rather than an
# architectural one, which is why the two operations carry their own numbers here instead of a ratio.
#
# Shipping the MIOpen numbers everywhere would add ~60% to every cuDNN decode for nothing, so the
# constant follows the backend.
#
# Caveat for AMD users: `MIOPEN_FIND_MODE=2` selects convolution algorithms heuristically instead of
# by benchmark, and measured a uniform 1.28x more memory at every resolution. It is not the default
# and is not budgeted for here -- raise `device_working_mem_gb` if you set it.
_FLUX2_VAE_SCALING_CONSTANTS: dict[str, dict[str, int]] = {
    "cudnn": {"decode": 2200, "encode": 1100},
    "miopen": {"decode": 3600, "encode": 2750},
}


def _flux2_vae_scaling_constant(operation: Literal["encode", "decode"], device: torch.device) -> int:
    """Pick the pixel-area constant for the convolution backend this device will actually use.

    A HIP build reports ``device.type == "cuda"``, so the torch build -- not the device string -- is
    what separates MIOpen from cuDNN. MPS and CPU are unmeasured and take the cuDNN column; on MPS
    that pairs with a score-matrix term the probe always charges there, so the total is not thin.
    """
    is_rocm = device.type == "cuda" and torch.version.hip is not None
    return _FLUX2_VAE_SCALING_CONSTANTS["miopen" if is_rocm else "cudnn"][operation]


def estimate_vae_working_memory_flux2(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKLFlux2,
    tile_size: int | None = None,
    device: torch.device | None = None,
) -> int:
    """Estimate the working memory required to encode or decode with the FLUX.2 (32-channel) VAE.

    Peak memory scales linearly with pixel area and element size, as it does for the FLUX.1 VAE, and
    the implied constant is flat across 512-1536px on every backend measured. What is *not* constant
    is the constant itself: MIOpen's convolution workspaces cost ~1.6x cuDNN's for a decode and ~2.4x
    for an encode, so it is looked up per backend -- see ``_FLUX2_VAE_SCALING_CONSTANTS`` for the
    fitted table and the caveats. Peak *reserved* memory is what is measured throughout, the
    conservative quantity that includes allocator overhead.

    For reference, decoding 1024x1024 peaks at ~4.3GB on cuDNN and ~6.6GB on MIOpen, and 1536x1536 at
    ~9.6GB -- far above the default ``device_working_mem_gb``, which is why this estimate must be
    passed to the model cache.

    That linear term holds only while ``AutoencoderKLFlux2``'s mid-block attention runs through a
    fused SDPA kernel, which is what CUDA does (verified: the memory-efficient kernel takes the
    512-wide head, and measured peak stays linear from 512 to 1536px). Where the math kernel runs
    instead -- every ROCm build, by policy, because its fused kernels return wrong output for this
    head width (see ``rocm_sdpa_uses_math_kernel``), and MPS, which has no fused kernel at all -- it
    materializes a (pixels/8)^2 score matrix, which grows quadratically and overtakes the linear
    term somewhere past 1280px. ``sdpa_score_matrix_bytes`` applies the ROCm rule and asks torch
    for everything else, so the estimate is right on all of them.

    The two terms are independent: a build can have a fused kernel and still need the larger
    convolution constant (gfx1201 did, before the ROCm rule above). They also do not add -- see the
    ``max`` at the end of this function for why, and for the measurements behind it. (Unlike the transformer, this attention does not go through
    diffusers' attention dispatcher -- ``AttnProcessor2_0`` calls ``F.scaled_dot_product_attention``
    itself -- so torch's own answer is the whole answer here.)

    When tiling is enabled the peak is bounded by a single tile instead of the full image (measured
    ~0.55GB flat at a 512px tile, from 1024px up to the 2024px reference-image cap), and the score
    matrix, if one is materialized at all, is bounded by the tile too.

    Both terms are per sample. `vae.decode` takes whatever batch the latents carry, and a
    ``LatentsField`` is not pinned to one, so the batch has to multiply through: measured at 1024px
    decode, peak reserved is 4.23GB at batch 1, 7.96GB at batch 2 and 11.89GB at batch 3 -- linear,
    and slightly sub-linear per sample, so multiplying the single-sample estimate stays an upper
    bound. The score matrix is shaped (batch, heads, S, S), so it scales the same way. The encode
    call sites all pass batch 1 today; the shared estimator does not assume it.
    """
    param = next(vae.parameters())
    element_size = param.element_size()

    device = device if device is not None else TorchDevice.choose_torch_device()
    scaling_constant = _flux2_vae_scaling_constant(operation, device)
    batch_size = image_tensor.shape[0] if image_tensor.dim() >= 4 else 1

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

    working_memory *= batch_size
    score_matrix_bytes = sdpa_score_matrix_bytes(
        device=device,
        dtype=param.dtype,
        # The score matrix is (batch, heads, S, S); one head per sample prices the whole batch.
        num_heads=_FLUX2_VAE_MID_BLOCK_HEADS * batch_size,
        head_dim=_FLUX2_VAE_MID_BLOCK_HEAD_DIM,
        seq_len=mid_block_seq_len,
    )

    # max, not sum: the two terms peak in different phases of the same forward. The mid-block sits at
    # the 8x-downsampled bottleneck -- first in the decoder, last in the encoder -- so the full-
    # resolution convolution feature maps that drive the linear term are not live while the score
    # matrix is, and peak *reserved* is a high-water mark, not a running total. Measured (see the
    # constants table above for the method): on gfx1100 and gfx1201 forcing `math` moves the measured
    # peak by nothing at all up to 1024px, and the totals stay flat-linear in area either way. On
    # CUDA the score matrix only pokes above the convolution peak at 1536px, and then by 2.6GB
    # against the 21.5GB this term prices standalone, because the attention phase reuses blocks the
    # allocator is already holding. Summing them reserved 11.1GB for a 1024px gfx1100 decode that
    # measures 6.7GB; taking the max reserves 6.9GB.
    #
    # Where a max model is weakest is the crossover, where the two terms are near-equal and whatever
    # overlap exists is no longer hidden. There is exactly one measured point like that: a 768px
    # encode with cuDNN's linear constant and a materializing kernel measures 1.80GB against a
    # 1.35GB max. It is not reachable as a shortfall, because the cache floors every reservation at
    # `device_working_mem_gb` (3GB by default, see `ModelCache._get_vram_available`) and the whole
    # crossover region sits under that floor. On the builds that really do materialize -- gfx1100 and
    # gfx1201 -- the MIOpen constant keeps the linear term above the score term across the measured
    # range, so the crossover does not arise there at all.
    return int(max(working_memory, score_matrix_bytes))


def estimate_vae_working_memory_anima(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKLWan | AutoencoderKLQwenImage,
    tile_size: int | None,
) -> int:
    """Estimate the working memory required to encode or decode with the Wan 2.1 VAE (Anima).

    Anima reaches that VAE as either class: AutoencoderKLWan (original layout) or AutoencoderKLQwenImage
    (the diffusers-layout Qwen-Image export), which run the same network on the same weights.

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
        if operation == "encode":
            # ...plus what tiling does not bound on the encode: the full-resolution frame the tiles
            # are sliced from, and the moments assembled out of them. Without these the estimate is
            # flat while the measurement grows about 4 bytes per pixel -- measured on a 4090 at 1.07x
            # headroom at 1024px falling to 0.69x at 4096px, i.e. an under-reservation, and this node
            # has no tiled retry to fall back on the way its decode sibling does.
            #
            # The decode has the same kind of un-bounded assembly and does not price it either. That
            # is pre-existing -- `anima_l2i` has tiled since it was written -- and is left alone here
            # rather than changed in a diff about the encode.
            frame_h, frame_w = image_tensor.shape[-2], image_tensor.shape[-1]
            latent_area = (frame_h // LATENT_SCALE_FACTOR) * (frame_w // LATENT_SCALE_FACTOR)
            working_memory += frame_h * frame_w * _IMAGE_CHANNELS * element_size
            working_memory += latent_area * _VAE_MOMENT_CHANNELS_PER_LATENT * vae.config.z_dim * element_size
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


def estimate_vae_working_memory_ltx2(
    operation: Literal["encode", "decode"],
    vae: "torch.nn.Module",
    pixel_height: int,
    pixel_width: int,
    pixel_frames: int,
    tile_size: int | None = None,
    temporal_tile: int | None = None,
    tiled: bool = False,
) -> int:
    """Estimate the working memory to encode or decode with the LTX-2 video VAE.

    The decode is always tiled in space and time (see
    :func:`invokeai.backend.ltx2.video_decoding.scoped_ltx2_tiling`), so its activation term scales
    with one tile's pixel volume and is almost flat in the clip's own size. Measured on a W7900
    (gfx1100, bf16 weights, released 2.5 VAE), as peak *reserved* minus the weights, decoding
    1248x704:

    | tile | temporal tile | frames | activation |
    |------|---------------|--------|------------|
    | 256  | 16            | 33     | 1.12 GiB   |
    | 512  | 16            | 33     | 3.85 GiB   |
    | 512  | 16            | 121    | 3.60 GiB   |
    | 512  | 32            | 33     | 5.57 GiB   |
    | 768  | 16            | 33     | 6.45 GiB   |

    A tile costs less than its volume suggests as it grows (the fixed cost of the decoder's own
    buffers dominates a small tile), which is why the fit is affine rather than proportional; the
    constants below bracket every row above by 11-55%. The clip term is added on top: the temporal
    tiler holds both the decoded rows and the clip it concatenates them into, and the rows are not
    one clip's worth: at the default 16-frame tile the latent stride is one, so a row is kept for
    every latent frame and the accumulation runs to about two clips. Three is what that sums to,
    and it is the term that grows with the clip -- the tile term does not, so a long clip would
    otherwise eat the margin the measured rows show.

    Encode has the same two shapes with one clip copy. ``tiled=False`` is the first-frame encode,
    where the whole "tile" is the one frame and the fixed per-tile cost dominates: 1248x704x1
    measured 1.26 GiB against 1.65 GiB predicted. ``tiled=True`` is a whole conditioning clip,
    which cannot run any other way -- untiled, the activation grows with the clip and a
    1248x704x121 encode needs about 65 GiB. Tiled at 512/16, measured the same way:

    | canvas     | frames | activation | of which the tile term |
    |------------|--------|------------|------------------------|
    | 768x512    | 33     | 2.82 GiB   | 2.75 GiB               |
    | 768x512    | 121    | 2.87 GiB   | 2.60 GiB               |
    | 1248x704   | 121    | 3.20 GiB   | 2.61 GiB               |
    | 1248x704   | 241    | 3.78 GiB   | 2.60 GiB               |
    | 1920x1088  | 121    | 4.02 GiB   | 2.61 GiB               |

    The tile term is flat across a 16x range of clip volume, as it should be, and works out at
    333-352 bytes per tile element; the constant rounds up from there. Reading the encode constant
    for a tiled run would over-reserve it twofold, which is cache the rest of the graph loses.
    """
    element_size = next(vae.parameters()).element_size()
    spatial_ratio = int(getattr(vae, "spatial_compression_ratio", 32))
    temporal_ratio = int(getattr(vae, "temporal_compression_ratio", 8))

    if operation == "decode":
        tile_height = max(
            spatial_ratio, min(tile_size or int(getattr(vae, "tile_sample_min_height", 512)), pixel_height)
        )
        tile_width = max(spatial_ratio, min(tile_size or int(getattr(vae, "tile_sample_min_width", 512)), pixel_width))
        # Not clamped to the clip's own frames: the temporal tile is two *latent* frames, which
        # decode to a full tile of pixel frames even when the clip is shorter than one.
        tile_frames = max(temporal_ratio, temporal_tile or int(getattr(vae, "tile_sample_min_num_frames", 16)))
        bytes_per_element = _LTX2_VAE_DECODE_BYTES_PER_TILE_ELEMENT
        clip_copies = _LTX2_DECODE_CLIP_COPIES
    elif tiled:
        tile_height = max(
            spatial_ratio, min(tile_size or int(getattr(vae, "tile_sample_min_height", 512)), pixel_height)
        )
        tile_width = max(spatial_ratio, min(tile_size or int(getattr(vae, "tile_sample_min_width", 512)), pixel_width))
        tile_frames = max(temporal_ratio, temporal_tile or int(getattr(vae, "tile_sample_min_num_frames", 16)))
        bytes_per_element = _LTX2_VAE_TILED_ENCODE_BYTES_PER_TILE_ELEMENT
        clip_copies = 1
    else:
        tile_height, tile_width, tile_frames = pixel_height, pixel_width, pixel_frames
        bytes_per_element = _LTX2_VAE_ENCODE_BYTES_PER_TILE_ELEMENT
        clip_copies = 1

    activation_bytes = tile_frames * tile_height * tile_width * element_size * bytes_per_element
    clip_bytes = clip_copies * 3 * pixel_frames * pixel_height * pixel_width * element_size
    return int(_LTX2_VAE_BASE_BYTES + activation_bytes + clip_bytes)


def estimate_audio_working_memory_ltx2(num_audio_latents: int) -> int:
    """Estimate the working memory to turn LTX-2 audio latents into a waveform.

    Three stages run back to back over the whole soundtrack, none of them tiled: the audio VAE
    decodes the latents to a log-mel spectrogram, the vocoder synthesizes a 16 kHz waveform from
    it, and a bandwidth extender resynthesizes that at 48 kHz. Every intermediate is proportional
    to the clip's length, so the estimate is a line in the latent count (25 latents per second).

    Measured on a W7900 (gfx1100, released 2.5 audio VAE and vocoder, fp32 weights), as peak
    *reserved* minus the two models:

    | audio latents | duration | working set |
    |---------------|----------|-------------|
    | 126           | 5 s      | 0.71 GiB    |
    | 251           | 10 s     | 1.30 GiB    |
    | 501           | 20 s     | 2.18 GiB    |
    | 1251          | 50 s     | 6.93 GiB    |

    The constants bracket every row by 16-30%. They also make an absurd request fail at the
    reservation rather than inside a forward: a 481-frame clip at 1 fps is 481 seconds of audio,
    and asking the cache for the ~76 GiB that needs is the honest answer.
    """
    return int(_LTX2_AUDIO_BASE_BYTES + num_audio_latents * _LTX2_AUDIO_BYTES_PER_LATENT)


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

    # max, not sum: the mid-block attention and the full-resolution convolutions peak in different
    # phases of the forward (see the FLUX.2 estimator for the measurements).
    return int(max(working_memory, _vae_mid_block_score_matrix_bytes(h, w, next(vae.parameters()).dtype)))

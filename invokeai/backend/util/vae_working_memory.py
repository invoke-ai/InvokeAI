from collections.abc import Mapping
from dataclasses import replace
from typing import Literal, cast

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.util import wddm
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.vae_tiling_scope import diffusers_tile, scoped_vae_tile
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import convolution_line
from invokeai.backend.working_memory.platform import platform_for
from invokeai.backend.working_memory.spec import EstimatorSpec, register

# Share of the device's memory an untiled decode may claim before the decode nodes tile it up front (see
# `should_pretile_vae_decode`). High enough that common resolutions on common cards stay untiled -- tiled output is
# not pixel-identical -- and low enough that a decode no card of that size can hold is tiled before it starts.
VAE_PRETILE_VRAM_FRACTION = 0.9


def should_pretile_vae_decode(
    device: torch.device, full_decode_bytes: int, vram_fraction: float = VAE_PRETILE_VRAM_FRACTION
) -> bool:
    """Whether a decode should be tiled up front because its untiled working memory would claim more than
    ``vram_fraction`` of the memory ``device`` will keep resident.

    Waiting for an out-of-memory error does not work everywhere: on Windows, drivers page an allocation that does not
    fit into system memory instead of failing it (always for ROCm, by default for NVIDIA's sysmem fallback), and the
    decode just runs very slowly. ``device`` is where the VAE runs: a ``cpu_only`` VAE (and MPS, sharing system
    memory) is never tiled on these grounds.

    On Windows the ceiling is the video-memory budget rather than the card's nameplate total -- the budget is the point
    past which Windows pages this process out. With one correction: a budget *below* what this process keeps resident is
    Windows asking it to trim itself, and the cache does exactly that to honour the reservation, so the ceiling is then
    that residency. Measured on an RX 9060 XT (16 GB): with Invoke holding 12 GiB and nothing else on the card the
    budget read 7.62 GiB, and comparing against that tiled a 7.0 GiB decode that fits, changing output that had been
    pixel-identical.

    Residency, not ``CurrentUsage``: that also counts what Windows has already paged out, which a foreign process forces
    exactly when the budget carries its pressure (next to an 8 GiB holder the budget stepped down to 7.58 GiB while our
    ``CurrentUsage`` kept growing past our 7.70 GiB in VRAM). So the override is ``CurrentUsage`` minus this process's
    paged bytes, and where either is unknown -- ``CurrentUsage`` beyond the card, or no counter reading -- the budget
    alone decides (so a transient counter failure can tile a decode that the next run leaves untiled).

    Known limit: the budget reacts to a foreign process only once this process grows into the memory it holds -- it
    stays at 15.09 GiB next to an 8 GiB holder while Invoke holds little -- torch's free figure ignores other processes,
    and the adapter-wide performance counter contradicts itself between processes (15.66 GiB read from the holding
    process against 0.18 GiB from another, same instance, same instant). A decode sized before that point can still
    page, and the end-of-session paging warning is what surfaces it.
    """
    if device.type == "cuda":
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        local = wddm.local_video_memory(device)
        if local is not None:
            budget_bytes, usage_bytes = local
            ceiling_bytes = budget_bytes
            paged = wddm.paged_bytes(device) if usage_bytes is not None else None
            if paged is not None:
                ceiling_bytes = max(budget_bytes, usage_bytes - paged)
            total_bytes = min(total_bytes, ceiling_bytes)
    elif device.type == "xpu":
        total_bytes = torch.xpu.get_device_properties(device).total_memory
    else:
        return False
    if full_decode_bytes <= vram_fraction * total_bytes:
        return False
    InvokeAILogger.get_logger(__name__).debug(
        f"Decoding in tiles: an untiled decode would need ~{full_decode_bytes / 2**30:.1f} GiB of working memory, more "
        f"than {vram_fraction:.0%} of the {total_bytes / 2**30:.1f} GiB the GPU keeps resident."
    )
    return True


# The diffusers AutoencoderKL (SD1/SDXL, SD3) and the FLUX.1 AutoEncoder run the same
# mid-block self-attention as the FLUX.2 VAE: one head over the 512-channel width, on the
# 8x-downsampled grid. Where that attention runs on the math kernel (see `sdpa_score_matrix_bytes`)
# its score matrix is the dominant term for a large untiled image, exactly as it is for FLUX.2;
# `_vae_mid_block_score_matrix_bytes` prices it for those estimators the same way. The video VAEs
# (Wan, Qwen-Image) need no such term: the Qwen-Image ROCm constants below were measured with math
# attention. The Wan video-decode constant was measured on CUDA and on an RX 9060 XT (see there).
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
_WAN_VAE_VIDEO_DECODE_SCALING_CONSTANT = 10000


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


# The SD1/SDXL, SD3, FLUX.1 and FLUX.2 VAEs run the same diffusers AutoencoderKL convolution stack, and pay the same
# for it: a line in the output's pixel count per operation and the dtype it runs in, `k_<operation>_<dtype>`
# bytes per pixel plus `b_<operation>_<dtype>` bytes the process pays whatever the size, shipped per convolution
# backend and allocator (`platform_for`). The values ship in `backend/working_memory/data/vae.json` with the points
# they are fitted on (`backend/working_memory/fit.py`): calibration runs in a running server measure them on a card,
# and `scripts/working_memory/build_shipped.py` turns their exports into shipped points. Every spec here declares that
# line (`fit.convolution_line`) and opts in to user calibration (`user_calibrated`), so a calibration run's decode
# windows refit these lines per device.
#
# None of the three scales out of the line. The demand does not follow the element size: on MIOpen fp16 needs about a
# third of what bf16 does, and on both backends fp32 needs less per value byte than bf16. The allocator matters as
# much: `cudaMallocAsync` needs 25-35% less than the native allocator on cuDNN. And MIOpen's bf16 workspaces are
# larger than cuDNN's, ~1.5x for a decode and ~2.4x for an encode -- not the attention term, which shows up the same
# on the fused path. The base is why the per-pixel figure rises at small sizes; on HIP it includes the ~300 MiB the
# driver maps beyond the allocator's own figure on the first operation of a process.
#
# Each estimator declares the lines of the dtypes it can run, and the estimate prices the dtype the operation will run
# in, which is not always the dtype the cached VAE holds (see `sd_vae_run_dtype`). CogView4's VAE is an AutoencoderKL
# too, but a wider one (block widths up to 1024, three layers per block, no mid-block attention): it carries its own
# lines, about 1.9x the family's for a decode.
#
# A tiled operation of the SD1/SDXL, FLUX.1 or FLUX.2 VAE pays the untiled line at its largest tile and, on top of it,
# the estimator's tiled line in the image's pixels; a tiled decode reserves the larger of that and its assembly line,
# also in the image's pixels (`_tiled_convolutions`). The tiled and assembly lines are per estimator: what tiling adds
# differs per family, so each family's entry carries its own beside the untiled lines it borrows.
#
# Caveat for AMD users: `MIOPEN_FIND_MODE=2` (FAST) selects convolution algorithms heuristically instead of by
# benchmark, and measured a uniform 1.28x more memory at every resolution when the MIOpen values were first fitted.
# Invoke has since made FAST its default (`run_app.py`); the RX 9060 XT points were taken under it. If you see decodes
# exceed their reservation on another card, raise `device_working_mem_gb`.
_PLATFORM_KEYS = ("conv_backend", "allocator")
_DTYPE_NAMES: dict[torch.dtype, str] = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}


def _convolution_coefficients(*dtypes: str, tiled: bool = False) -> frozenset[str]:
    """The coefficients of the convolution lines for both operations in these dtypes, and with `tiled` those of the
    tiled lines and of the tiled decode's assembly line (`_tiled_convolutions`) too."""
    operations = ("decode", "encode")
    lines = [f"{operation}_{dtype}" for operation in operations for dtype in dtypes]
    if tiled:
        lines += [f"tiled_{operation}_{dtype}" for operation in operations for dtype in dtypes]
        lines += [f"assembled_decode_{dtype}" for dtype in dtypes]
    return frozenset(f"{part}_{line}" for part in ("k", "b") for line in lines)


def dtype_name(dtype: torch.dtype) -> str:
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise ValueError(f"No VAE working-memory coefficients are measured for {dtype}.")
    return name


def _compute_dtype(vae: torch.nn.Module) -> torch.dtype:
    """The dtype a VAE no node casts computes in.

    An SDNQ VAE computes in its packed weights' `compute_dtype`: they dequantize to it, and the layer casts every other
    operand to it, an fp32 bias included. Otherwise it is that of the first floating-point parameter wider than a byte:
    narrower weights (fp8 storage) are what a layer dequantizes from, not what its activations are.

    A packed parameter's `compute_dtype` wins over any floating-point one, so an SDNQ VAE that kept whole layers
    unquantized in fp32 would be priced below its real cost; that case is not verified for the shipped SDNQ VAEs.
    """
    first_wide_float: torch.dtype | None = None
    for param in vae.parameters():
        packed_compute_dtype = getattr(param, "compute_dtype", None)
        if isinstance(packed_compute_dtype, torch.dtype):
            return packed_compute_dtype
        if first_wide_float is None and param.is_floating_point() and param.element_size() > 1:
            first_wide_float = param.dtype
    if first_wide_float is not None:
        return first_wide_float
    raise ValueError(f"{type(vae).__name__} has no floating-point parameters to compute in.")


def _estimate(spec: EstimatorSpec, features: Mapping[str, FeatureValue], device: torch.device) -> WorkingMemoryEstimate:
    """`spec.estimate`, refusing a dtype the spec has no line for with the estimator, dtype and platform named, rather
    than a bare `KeyError` from inside the terms."""
    line = f"k_{features['operation']}_{features['dtype']}"
    if line not in spec.coefficients:
        raise ValueError(
            f"Working-memory estimator '{spec.id}' has no {features['dtype']} {features['operation']} line on "
            f"{platform_for(device)}: the VAE computes in a dtype it is not measured for."
        )
    return spec.estimate(features, device)


def _convolutions(
    features: Mapping[str, FeatureValue], values: Mapping[str, float], pixels: float, samples: int = 1
) -> tuple[WorkingMemoryTerm, ...]:
    """The convolution line of this operation and dtype over `pixels` output pixels, for `samples` samples.

    The base scales with the samples too. A process pays it once, so with a base of at least zero, as an untiled or a
    tiled line's always is, that is an upper bound rather than a fit, and it keeps a batch at exactly that many
    single-sample estimates. An assembly line's base may be negative, and is taken once then (`_tiled_convolutions`).
    """
    line = f"{features['operation']}_{features['dtype']}"
    return (
        WorkingMemoryTerm.fitted("convolutions", f"k_{line}", pixels * samples, values),
        WorkingMemoryTerm.fitted("convolution_base", f"b_{line}", samples, values),
    )


def _tiled_convolutions(
    features: Mapping[str, FeatureValue], values: Mapping[str, float], samples: int = 1
) -> tuple[WorkingMemoryTerm, ...]:
    """A tiled operation's convolutions, for `samples` samples: the untiled line at its largest tile, and the tiled
    line `tiled_<operation>_<dtype>` over the image's pixels for what tiling adds on top. A decode reserves the larger
    of that and its assembly line `assembled_decode_<dtype>` over the image's pixels (`fit.Line.assembly`). Its base is
    per sample like the others where it is at least zero, and taken once where it is negative, ``k * P + max(b, n *
    b)`` for ``n`` samples: ``n`` of a negative base would price a batch's assembly below one sample's line over the
    same pixels.

    The tile is clamped to the image per axis, as every tiled path here slices it, and the sizes are the ones
    `fit.convolution_line` places the operation at, so the estimate prices what the fit fitted. Measured fresh-process
    (`scripts/calibrate_vae_tiling_working_memory.py`, the points in `data/vae.json`) at image sides 1024 to 12288 px
    with 512 and 1024 px tiles, 2048 to 12288 px with a 1536 px tile on the RTX 4090, and 2048 to 8192 px with 256 and
    768 px tiles, the 256 px ones bounding their estimates from below only (`fit.SMALLEST_APP_SIDE`):

    - A tiled decode demands about what the untiled decode of one tile does, flat in the image, but for a jump some
      allocators take once the image spans several tiles (~0.25 GiB under `cudaMallocAsync` and ~0.36 GiB under
      MIOpen's native allocator on Windows, at a 512 px tile). Up to about 4096 px the decoded tiles diffusers keeps fit
      in the allocator's slack around the tile's own peak. Above it the assembly takes over: every decoded tile, the
      blended rows and the concatenated image, about 3.8 copies of the image, and the node's post-processing, on which
      the native allocators fragment. At 8192 px a 512 px tile measured up to 3x its untiled demand and a 1024 px tile
      up to 1.4x; at 12288 px up to 7x and 2.4x (SD fp16 on MIOpen's native allocator, 7.25 against 2.99 GiB).
    - A tiled encode needs more than its tile does, for the input image it slices and the moments it assembles: by up
      to ~42 bytes per image pixel over the tile's own demand at 4096 px, ~29 at 8192 px and ~20 at 12288 px. Its
      tiled line prices that.

    The tiled line and the assembly line are fitted together on what the points measured, the tiled line above the
    untiled demand measured at their tile (`fit.two_phase_envelope`, `fit.FitPoint.reference`): what tiling adds to that
    demand, which carries to a tile no point measured whose growth is no larger than the measured tiles'.
    """
    line = convolution_line(features)
    assert line is not None and line.sits_on is not None, "a tiled operation's features place it on its tiled line"
    _, tile_area = line.sits_on
    tile_phase = (
        *_convolutions(features, values, tile_area, samples),
        WorkingMemoryTerm.fitted("tiled", f"k_{line.suffix}", line.size, values),
        WorkingMemoryTerm.fitted("tiled_base", f"b_{line.suffix}", samples, values),
    )
    if line.assembly is None:
        return tile_phase
    base_samples = samples if values[f"b_{line.assembly}"] >= 0 else 1
    assembly_phase = (
        WorkingMemoryTerm.fitted("assembled", f"k_{line.assembly}", line.size, values),
        WorkingMemoryTerm.fitted("assembled_base", f"b_{line.assembly}", base_samples, values),
    )
    return _larger_phase(tile_phase, assembly_phase)


def _larger_phase(
    first: tuple[WorkingMemoryTerm, ...], second: tuple[WorkingMemoryTerm, ...]
) -> tuple[WorkingMemoryTerm, ...]:
    """Two phases of one operation that peak apart, of which only the larger is reserved: the terms of both, the
    smaller phase's not included. A phase reserves the bytes of its included terms, so it can itself be the larger of
    two. On a tie the first is reserved."""
    reserved: list[float] = [0, 0]
    for index, phase in enumerate((first, second)):
        for term in phase:
            if term.included:
                reserved[index] += term.bytes
    second_wins = reserved[1] > reserved[0]
    return (
        *(replace(term, included=term.included and not second_wins) for term in first),
        *(replace(term, included=term.included and second_wins) for term in second),
    )


def _peak_of_two_phases(
    convolution_phase: tuple[WorkingMemoryTerm, ...], score_matrix_bytes: int
) -> tuple[WorkingMemoryTerm, ...]:
    """The convolution phase and the mid-block score matrix, of which only the larger is reserved.

    max, not sum: the two peak in different phases of the same forward. The mid-block sits at the 8x-downsampled
    bottleneck -- first in the decoder, last in the encoder -- so the full-resolution convolution feature maps that drive
    the linear term are not live while the score matrix is, and peak *reserved* is a high-water mark, not a running
    total. Measured on the FLUX.2 VAE (the points and their method are in `data/vae.json`): on gfx1100 and gfx1201
    forcing `math` moves the measured peak by nothing at all up to 1024px, and the totals stay flat-linear in area
    either way. On CUDA the score matrix only pokes above the convolution peak at 1536px, and then by 2.6GB against the
    21.5GB this term prices standalone, because the attention phase reuses blocks the allocator is already holding.
    Summing them reserved 11.1GB for a 1024px gfx1100 decode that measures 6.7GB; taking the max reserves 6.9GB.

    Where a max model is weakest is the crossover, where the two terms are near-equal and whatever overlap exists is
    no longer hidden. There is exactly one measured point like that: a 768px encode with cuDNN's linear coefficient and
    a materializing kernel measures 1.80GB against a 1.35GB max. It is not reachable as a shortfall, because the cache
    floors every reservation at `device_working_mem_gb` (3GB by default, see `ModelCache._get_vram_available`) and the
    whole crossover region sits under that floor. On the builds that really do materialize -- gfx1100 and gfx1201 --
    the MIOpen coefficient keeps the linear term above the score term across the measured range, so the crossover does
    not arise there at all.

    On a tie the convolutions are the ones reserved. A tiled decode's convolution phase is itself the larger of its
    tile and its assembly (`_tiled_convolutions`); the score matrix is the tile's, so the reservation is the largest of
    the three.
    """
    return _larger_phase(convolution_phase, (WorkingMemoryTerm("mid_block_scores", score_matrix_bytes, True),))


def _sd_terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    if features["tile"] is not None:
        terms = _tiled_convolutions(features, values)
    else:
        terms = _convolutions(features, values, cast(int, features["height"]) * cast(int, features["width"]))

    # The tiny autoencoder has no mid-block attention, so it is priced without a score matrix.
    if features["vae_class"] == "AutoencoderKL":
        terms = _peak_of_two_phases(terms, cast(int, features["score_matrix_bytes"]))
    upcast_weight_bytes = cast(int, features["upcast_weight_bytes"])
    if upcast_weight_bytes:
        terms += (WorkingMemoryTerm("upcast_weights", upcast_weight_bytes, True),)
    return terms


VAE_SD_SPEC = register(
    EstimatorSpec(
        id="vae.sd",
        version=3,
        coefficients=_convolution_coefficients("fp16", "fp32", tiled=True),
        platform_keys=_PLATFORM_KEYS,
        terms=_sd_terms,
        line=convolution_line,
        user_calibrated=True,
    )
)


SD_VAE_TILE_OVERLAP_FACTOR = 0.25
"""The overlap the SD1/SDXL nodes patch into the shared VAE for a tile field above 0: `l2i` and `i2l` patch with it
(`patch_vae_tiling_params`), and `estimate_vae_working_memory_sd15_sdxl` prices the tile it gives, so the two cannot
drift."""


def sd_vae_run_dtype(operation: Literal["encode", "decode"], fp32: bool, device: torch.device) -> torch.dtype:
    """The dtype the SD1/SDXL nodes cast the shared VAE to before they run it: `l2i` and `i2l` cast with this, and
    `estimate_vae_working_memory_sd15_sdxl` prices it, so the two cannot drift.

    fp32 when the node asks for it, and for a decode on the CPU; fp16 otherwise, on every GPU and for every
    `precision`, so bf16 never runs. The cached VAE's own dtype says nothing about it: it is the one the VAE was loaded
    in, which the model cache returns it to when the node's lock ends.
    """
    if fp32 or (operation == "decode" and device.type == "cpu"):
        return torch.float32
    return torch.float16


def _upcast_weight_bytes(vae: torch.nn.Module, run_dtype: torch.dtype) -> int:
    """What the node's in-place cast to `run_dtype` adds to the VAE's resident weights.

    The cast runs once the cache has placed the VAE, so the growth comes out of the reservation: 160 MiB for the 83.7M
    parameters of the SD1/SDXL VAE going from fp16 to fp32, nothing for a VAE loaded in fp32. Read before the lock,
    the VAE holds the dtype it was loaded in: the model cache returns it there when the previous node's lock ended.
    Measured points exclude resident weights by construction (they record 0 here), so the fitted line does not cover
    it. An upcast encode casts the decoder's mid-block and post-quant conv back, so for it this counts a little more
    than grows: conservative.
    """
    return sum(p.numel() * max(0, run_dtype.itemsize - p.element_size()) for p in vae.parameters())


def estimate_vae_working_memory_sd15_sdxl(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKL | AutoencoderTiny,
    tile_size: int | None,
    fp32: bool,
    device: torch.device | None = None,
) -> WorkingMemoryEstimate:
    """Estimate the working memory required to encode or decode the given tensor.

    It was found experimentally that the peak working memory is a line in the number of pixels, per dtype. This
    estimate is accurate for both SD1 and SDXL. It prices the dtype the node will cast the VAE to (`sd_vae_run_dtype`),
    not the one the cached VAE holds. `device` is where the VAE will run; it picks the platform's coefficients and
    defaults to the session device.

    `tile_size` is the node's field in output pixels, None for a single pass, and 0 or below for the VAE's own tile,
    which the node leaves in place then. A field above 0 is priced at the tile `patch_vae_tiling_params` gives the VAE
    for it at `SD_VAE_TILE_OVERLAP_FACTOR` (`diffusers_tile`), not at the field: a 24 px field runs a 32 px tile, a
    200 px one 192 px. A tiled operation is priced on its largest tile, clamped to the image per axis as the VAE slices
    it, plus the tiled line, and a tiled decode on the larger of that and its assembly line (`_tiled_convolutions`); a
    tile covering the image on both axes is priced as the single pass the VAE runs then (`AutoencoderKL` tiles only when
    a side is strictly larger than the tile; for the tiny autoencoder, which tiles anyway, see below).
    """
    device = device if device is not None else TorchDevice.choose_torch_device()
    run_dtype = sd_vae_run_dtype(operation, fp32, device)
    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1
    height = latent_scale_factor_for_operation * image_tensor.shape[-2]
    width = latent_scale_factor_for_operation * image_tensor.shape[-1]

    # The nodes patch the VAE's tile for a field above 0 only, and leave its own in place otherwise.
    if tile_size is not None and tile_size <= 0:
        tile_size = vae.tile_sample_min_size
        assert isinstance(tile_size, int)
    elif tile_size is not None:
        tile_size, _ = diffusers_tile(vae, tile_size, SD_VAE_TILE_OVERLAP_FACTOR)
    # The tiny autoencoder runs `_tiled_decode` even for a covering tile: one tile over the whole image, with an output
    # buffer and blend masks, a few bytes per pixel. The single pass this prices instead is on the AutoencoderKL lines,
    # which cost many times what its convolutions do (64 channels wide against 512), so it stays an upper bound.
    if tile_size is not None and tile_size >= height and tile_size >= width:
        tile_size = None

    features: dict[str, FeatureValue] = {
        "operation": operation,
        "height": height,
        "width": width,
        "tile": tile_size,
        "dtype": dtype_name(run_dtype),
        "vae_class": "AutoencoderKL" if isinstance(vae, AutoencoderKL) else "AutoencoderTiny",
        "upcast_weight_bytes": _upcast_weight_bytes(vae, run_dtype),
    }
    if isinstance(vae, AutoencoderKL):
        score_h, score_w = (min(tile_size, height), min(tile_size, width)) if tile_size is not None else (height, width)
        features["score_matrix_bytes"] = _vae_mid_block_score_matrix_bytes(score_h, score_w, run_dtype, device=device)

    return _estimate(VAE_SD_SPEC, features, device)


def _single_pass_terms(
    features: Mapping[str, FeatureValue], values: Mapping[str, float]
) -> tuple[WorkingMemoryTerm, ...]:
    """An untiled AutoencoderKL pass: the convolutions over the whole image, or the mid-block score matrix."""
    convolutions = _convolutions(features, values, cast(int, features["height"]) * cast(int, features["width"]))
    return _peak_of_two_phases(convolutions, cast(int, features["score_matrix_bytes"]))


def _single_pass_features(
    operation: Literal["encode", "decode"], image_tensor: torch.Tensor, vae: AutoencoderKL, device: torch.device
) -> dict[str, FeatureValue]:
    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1
    height = latent_scale_factor_for_operation * image_tensor.shape[-2]
    width = latent_scale_factor_for_operation * image_tensor.shape[-1]
    dtype = _compute_dtype(vae)
    return {
        "operation": operation,
        "height": height,
        "width": width,
        "dtype": dtype_name(dtype),
        "score_matrix_bytes": _vae_mid_block_score_matrix_bytes(height, width, dtype, device=device),
    }


def _cogview4_terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    """An untiled CogView4 VAE pass: its convolutions over the whole image. It has no mid-block attention
    (`mid_block_add_attention: false`), so there is no score matrix to price."""
    return _convolutions(features, values, cast(int, features["height"]) * cast(int, features["width"]))


# The CogView4 loader forces bf16: the VAE produces black images in fp16.
VAE_COGVIEW4_SPEC = register(
    EstimatorSpec(
        id="vae.cogview4",
        version=2,
        coefficients=_convolution_coefficients("bf16"),
        platform_keys=_PLATFORM_KEYS,
        terms=_cogview4_terms,
        line=convolution_line,
        user_calibrated=True,
    )
)


def estimate_vae_working_memory_cogview4(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKL,
    device: torch.device | None = None,
) -> WorkingMemoryEstimate:
    """Estimate the working memory required by the invocation in bytes.

    `device` is where the VAE will run; it picks the convolution backend's coefficients and defaults to the session
    device.
    """
    device = device if device is not None else TorchDevice.choose_torch_device()
    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1
    features: dict[str, FeatureValue] = {
        "operation": operation,
        "height": latent_scale_factor_for_operation * image_tensor.shape[-2],
        "width": latent_scale_factor_for_operation * image_tensor.shape[-1],
        "dtype": dtype_name(_compute_dtype(vae)),
    }
    return _estimate(VAE_COGVIEW4_SPEC, features, device)


# What the Anima estimator adds for a tiled encode: the full-resolution RGB image, and the moments -- mean and
# logvar, twice the latent width, before `DiagonalGaussianDistribution` splits them.
_IMAGE_CHANNELS = 3
_VAE_MOMENT_CHANNELS_PER_LATENT = 2


def _flux_terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    if features["tile"] is not None:
        return _peak_of_two_phases(_tiled_convolutions(features, values), cast(int, features["score_matrix_bytes"]))
    return _single_pass_terms(features, values)


# The FLUX.1 loaders avoid fp16, which this autoencoder is broken in, except one: an SDNQ VAE folder loads through the
# generic VAE loader and computes in the configured precision, which `auto` makes fp16 on CUDA.
VAE_FLUX_SPEC = register(
    EstimatorSpec(
        id="vae.flux",
        version=3,
        coefficients=_convolution_coefficients("fp16", "bf16", "fp32", tiled=True),
        platform_keys=_PLATFORM_KEYS,
        terms=_flux_terms,
        line=convolution_line,
        user_calibrated=True,
    )
)


def estimate_vae_working_memory_flux(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKL,
    tile_size: int | None = None,
    device: torch.device | None = None,
) -> WorkingMemoryEstimate:
    """Estimate the working memory required by the invocation in bytes.

    `tile_size` is in output pixels and defaults to None, i.e. a single pass -- the existing call
    sites that pass no tile depend on that signature. When set, the operation is priced on its
    largest tile, clamped to the image per axis, plus the tiled line for what tiling adds: the
    allocator's jump, and the encode's input image and moments. A decode reserves the larger of
    that and its assembly line, for the assembled output and the node's post-processing
    (`_tiled_convolutions`). `tile_size <= 0` is the nodes' "use the default" sentinel; see
    `resolve_tile_size`. The tile priced is the one `scoped_vae_tiling` runs for the field
    (`scoped_vae_tile`), which can be smaller than the field. Both operations tile, each with its
    own tiled line.

    `device` is where the VAE will run (a `cpu_only` VAE runs on the CPU); it picks the convolution
    backend's coefficients. Defaults to the session device.
    """
    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1
    dtype = _compute_dtype(vae)
    device = device if device is not None else TorchDevice.choose_torch_device()

    out_h = latent_scale_factor_for_operation * image_tensor.shape[-2]
    out_w = latent_scale_factor_for_operation * image_tensor.shape[-1]

    # Resolved against a module-level constant, never by reading `vae.tile_sample_min_size`: the VAE
    # belongs to the model cache, so that attribute reflects whatever the previous invocation set
    # rather than the default this node is asking for.
    tile = scoped_vae_tile(vae, tile_size)[0] if tile_size is not None else None

    # `_tiled_decode` short-circuits to a single pass once the tile covers the image on both axes,
    # so a tile that large has to be priced as the untiled decode it will actually run. The node
    # field carries no upper bound: a 2048px tile on a 1024px image would otherwise reserve ~23GB
    # for a decode that needs ~4.6GB, evicting the transformer from the cache for nothing.
    if tile is not None and (tile < out_h or tile < out_w):
        score_h, score_w = min(tile, out_h), min(tile, out_w)
    else:
        tile = None
        score_h, score_w = out_h, out_w

    features: dict[str, FeatureValue] = {
        "operation": operation,
        "height": out_h,
        "width": out_w,
        "dtype": dtype_name(dtype),
        "tile": tile,
        "score_matrix_bytes": _vae_mid_block_score_matrix_bytes(score_h, score_w, dtype, device=device),
    }
    return _estimate(VAE_FLUX_SPEC, features, device)


# The FLUX.2 VAE runs one attention block at the bottom of the encoder and one at the top of the
# decoder, on the 8x-downsampled grid. Both are single-head, with the head dim set to the block
# width: 512 for the stock VAE and 384 for the small-decoder variant. Either way it is far past the
# 128 head dim some builds cap their fused SDPA kernels at, so whether the score matrix is
# materialized is a per-build question -- `sdpa_score_matrix_bytes` asks rather than assumes.
_FLUX2_VAE_MID_BLOCK_HEADS = 1
_FLUX2_VAE_MID_BLOCK_HEAD_DIM = 512
_FLUX2_VAE_SPATIAL_COMPRESSION = 8


def _flux2_terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    batch_size = cast(int, features["batch_size"])
    if features["tile"] is not None:
        # The largest slice `AutoencoderKLFlux2` takes is the tile, clamped to the image per axis.
        convolutions = _tiled_convolutions(features, values, samples=batch_size)
    else:
        height, width = cast(int, features["height"]), cast(int, features["width"])
        convolutions = _convolutions(features, values, height * width, samples=batch_size)
    # Both terms are per sample; the larger is reserved (see `_peak_of_two_phases` for why, and the measurements).
    return _peak_of_two_phases(convolutions, cast(int, features["score_matrix_bytes"]))


# The FLUX.2 loaders avoid fp16 (as FLUX.1's do), and none of them goes through the generic VAE loader.
VAE_FLUX2_SPEC = register(
    EstimatorSpec(
        id="vae.flux2",
        version=3,
        coefficients=_convolution_coefficients("bf16", "fp32", tiled=True),
        platform_keys=_PLATFORM_KEYS,
        terms=_flux2_terms,
        line=convolution_line,
        user_calibrated=True,
    )
)


def estimate_vae_working_memory_flux2(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKLFlux2,
    tile_size: int | None = None,
    device: torch.device | None = None,
) -> WorkingMemoryEstimate:
    """Estimate the working memory required to encode or decode with the FLUX.2 (32-channel) VAE.

    Peak memory is a line in pixel area per dtype, as it is for the FLUX.1 VAE, from 512 to 2048px on
    every backend measured. The line itself is not one: MIOpen's bf16 convolution workspaces cost ~1.5x
    cuDNN's for a decode and ~2.4x for an encode, and the allocator and the dtype move it as much, so it
    is shipped per platform and dtype -- see `_convolution_coefficients` for the caveats and
    `backend/working_memory/data/vae.json` for the fitted values and their points. The points measure
    peak *reserved* memory or more, the conservative quantity that includes allocator overhead.

    For reference, decoding 1024x1024 in bf16 peaks at ~4.3GB on cuDNN and ~6.6GB on MIOpen, and
    1536x1536 at ~9.6GB -- far above the default ``device_working_mem_gb``, which is why this estimate
    must be passed to the model cache.

    That linear term holds only while ``AutoencoderKLFlux2``'s mid-block attention runs through a
    fused SDPA kernel, which is what CUDA does (verified: the memory-efficient kernel takes the
    512-wide head, and measured peak stays linear from 512 to 1536px). Where the math kernel runs
    instead -- every ROCm build, by policy, because its fused kernels return wrong output for this
    head width (see ``rocm_sdpa_uses_math_kernel``), and MPS, which has no fused kernel at all -- it
    materializes a (pixels/8)^2 score matrix, which grows quadratically and overtakes the linear
    term somewhere past 1280px. ``sdpa_score_matrix_bytes`` applies the ROCm rule and asks torch
    for everything else, so the estimate is right on all of them.

    The two terms are independent: a build can have a fused kernel and still need the larger
    convolution coefficient (gfx1201 did, before the ROCm rule above). They also do not add -- see
    `_peak_of_two_phases` for why, and for the measurements behind it. (Unlike the transformer, this
    attention does not go through diffusers' attention dispatcher -- ``AttnProcessor2_0`` calls
    ``F.scaled_dot_product_attention`` itself -- so torch's own answer is the whole answer here.)

    `tile_size` is in output pixels; None prices a single pass, and `tile_size <= 0` is the nodes'
    "use the default" sentinel (see `resolve_tile_size`). The tile priced is the one
    `scoped_vae_tiling` runs for the field (`scoped_vae_tile`), which can be smaller than the field.
    A tiled operation is priced on the largest tile instead of the full image, plus the tiled line
    for what tiling adds on top, a decode at the larger of that and its assembly line
    (`_tiled_convolutions`), and the score matrix is priced on the tile too, if one is materialized
    at all. That tile is clamped to the image per axis, as the slices `AutoencoderKLFlux2` takes
    are, and a tile covering the image on both axes is priced as the single pass it runs: the VAE
    tiles only when a side is strictly larger than the tile.

    Both terms are per sample. `vae.decode` takes whatever batch the latents carry, and a
    ``LatentsField`` is not pinned to one, so the batch has to multiply through: measured at 1024px
    decode, peak reserved is 4.23GB at batch 1, 7.96GB at batch 2 and 11.89GB at batch 3 -- linear,
    and slightly sub-linear per sample, so multiplying the single-sample estimate stays an upper
    bound. The score matrix is shaped (batch, heads, S, S), so it scales the same way. The encode
    call sites all pass batch 1 today; the shared estimator does not assume it.
    """
    dtype = _compute_dtype(vae)
    device = device if device is not None else TorchDevice.choose_torch_device()
    batch_size = image_tensor.shape[0] if image_tensor.dim() >= 4 else 1
    latent_scale_factor_for_operation = LATENT_SCALE_FACTOR if operation == "decode" else 1
    out_h = latent_scale_factor_for_operation * image_tensor.shape[-2]
    out_w = latent_scale_factor_for_operation * image_tensor.shape[-1]

    # Resolved against the module-level default, never read back from the cached VAE, whose tile
    # attributes are whatever the previous invocation left there. A tile covering the image on both
    # axes runs the single pass, so it is priced as one (`tile` None).
    tile = scoped_vae_tile(vae, tile_size)[0] if tile_size is not None else None
    if tile is not None and (tile < out_h or tile < out_w):
        tile_h, tile_w = min(tile, out_h), min(tile, out_w)
    else:
        tile = None
        tile_h, tile_w = out_h, out_w
    mid_block_seq_len = (tile_h // _FLUX2_VAE_SPATIAL_COMPRESSION) * (tile_w // _FLUX2_VAE_SPATIAL_COMPRESSION)

    features: dict[str, FeatureValue] = {
        "operation": operation,
        "height": out_h,
        "width": out_w,
        "tile": tile,
        "batch_size": batch_size,
        "dtype": dtype_name(dtype),
        "score_matrix_bytes": sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            # The score matrix is (batch, heads, S, S); one head per sample prices the whole batch.
            num_heads=_FLUX2_VAE_MID_BLOCK_HEADS * batch_size,
            head_dim=_FLUX2_VAE_MID_BLOCK_HEAD_DIM,
            seq_len=mid_block_seq_len,
        ),
    }
    return _estimate(VAE_FLUX2_SPEC, features, device)


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
    multi-frame decode uses a calibration of peak reserved memory because causal-convolution
    state makes the single-frame value unsafe at video resolutions.
    The Wan VAE processes the clip causally, one latent frame at a time with cached
    features. In streaming mode, only one temporal-upscale chunk of the RGB output is
    kept on the execution device; otherwise the full output clip and its transient copy
    are budgeted.
    """
    element_size = next(vae.parameters()).element_size()

    # The original 2900-byte calibration covers a single Wan 2.1 frame. Multi-frame video
    # decodes retain causal-convolution state that makes that constant unsafe at video
    # resolutions. The video constant is calibrated on peak *reserved* growth, which is what the
    # cache's `free >= estimate` check has to cover: the decode's live tensors peak at only about
    # two thirds of what the allocator reserves. The earlier constants (6500 A14B, 7000 TI2V)
    # covered the live tensors; in a server faking a 16 GB card the 832x480x49 decode then overran
    # the card by 1.3-1.9 GB, and with sysmem fallback such a decode stalled for minutes.
    #
    # Implied constant = (reserved - clip bytes) / (h * w * element_size), with the clip bytes this
    # function budgets for the mode (`wan_vae_clip_bytes`: two full clips, or one 4-frame chunk when
    # streaming); bf16,
    # RTX 4090, `scripts/calibrate_wan_vae_working_memory.py --dtype bfloat16 [--no-streaming]`, and
    # "server" = torch's peak stats around the decode inside a running server:
    #                       640x352x49  832x480x49  832x480x81  1280x704x49  1280x704x81  1280x704x121
    #   A14B full              7512        8457        7751         7557                     8073
    #   A14B streaming                     7773                                 8601
    #   TI2V full              8903        8457        8557         8943                     8511
    #   TI2V full, server                  9505                     8944
    #   TI2V streaming                     9004                                 9230
    #   TI2V streaming, server             9081                     9235
    # The reserved peak barely grows with the frame count (TI2V 1280x704 reserved 15876 MiB at 49
    # and at 121 frames), so the clip bytes are budget on top of it rather than part of it. Both
    # VAEs reserved exactly the same 6666 MiB at 832x480x49 standalone, so one constant covers
    # both: the highest measurement, the server's 480p full decode, plus ~5%.
    # ROCm (RX 9060 XT, gfx1200, torch 2.13+rocm10, with the app's conv3d decomposition, SDPA guard
    # and expandable segments): 7973-8813 standalone across both VAEs, full and streaming up to
    # 832x480x81, and 7719-8297 for TI2V 832x480x49 in a server -- covered as well.
    #
    # The same estimate decides auto-tiling (`should_pretile_vae_decode`): with it, 832x480 clips
    # tile on 8 GB cards and 720p clips on 16 GB cards, where their untiled decodes reserve about as
    # much as the card holds or more. A14B's 1280x720x81 sits right at the line on 20 GB cards: one
    # constant for both VAEs gives A14B 16-24% over its own standalone maxima at 720p, kept because
    # its in-server 480p peak was not measured.
    # Keep the single-frame value for image decode and the existing encode calibration.
    if operation == "decode" and pixel_frames > 1:
        scaling_constant = _WAN_VAE_VIDEO_DECODE_SCALING_CONSTANT
    else:
        scaling_constant = _WAN_VAE_SINGLE_FRAME_DECODE_SCALING_CONSTANT if operation == "decode" else 1450
    if tile_size is not None:
        # Add 25% for tile overlap.
        per_frame = tile_size * tile_size * element_size * scaling_constant * 1.25
    else:
        per_frame = pixel_height * pixel_width * element_size * scaling_constant

    return int(
        per_frame + wan_vae_clip_bytes(operation, vae, pixel_height, pixel_width, pixel_frames, tile_size, streaming)
    )


def wan_vae_clip_bytes(
    operation: Literal["encode", "decode"],
    vae: AutoencoderKLWan,
    pixel_height: int,
    pixel_width: int,
    pixel_frames: int,
    tile_size: int | None = None,
    streaming: bool = False,
) -> int:
    """The RGB clip bytes `estimate_vae_working_memory_wan` budgets on top of the VAE's own working set.

    Shared with `scripts/calibrate_wan_vae_working_memory.py`, which subtracts exactly this from a measurement to
    report the implied constant.
    """
    element_size = next(vae.parameters()).element_size()
    # Streaming decode moves each causal decoder chunk to CPU immediately. Only one
    # temporal-upscale chunk remains on the execution device, instead of the full RGB
    # clip plus the transient copy created by torch.cat.
    if operation == "decode" and streaming:
        temporal_scale = int(getattr(vae.config, "scale_factor_temporal", None) or 4)
        resident_frames = min(pixel_frames, temporal_scale)
        clip_copies = 1
    elif operation == "decode" and tile_size is not None:
        # diffusers' tiled_decode keeps every row of decoded tiles, the blended rows, their concatenation and the
        # clamped result alive together, and the tile's own working set is small next to them: reserved memory grew
        # by 5.2 (A14B) and 5.6 (TI2V) clip copies per frame (1280x704 at 81 vs 121 frames, bf16).
        resident_frames = pixel_frames
        clip_copies = 6
    else:
        resident_frames = pixel_frames
        clip_copies = 2 if operation == "decode" else 1
    return clip_copies * 3 * resident_frames * pixel_height * pixel_width * element_size


# What a full-frame Qwen-Image VAE decode was actually measured to peak at, per output pixel-byte, against the output
# area it was measured at -- the per-point table in `estimate_vae_working_memory_qwen_image`, taking the larger of the
# measured ROCm cards at each point (the W7900 column; the RX 9060 XT measures 2650-2772 throughout).
#
# This exists because the shipped constants are one flat figure fitted to the worst point of a curve that is not
# flat, across cards that differ by up to 1.9x. That is the right figure to *reserve* -- a reservation that is too
# small is an out-of-memory error or, on Windows, a decode paged into system memory. It is the wrong figure to decide
# tiling with: a tiled decode is not pixel-identical, so a 1536^2 decode whose real peak is 14.4 GiB must not be
# tiled on a 24 GiB card because a flat 5500 prices it at 24.2 GiB.
#
# Neither column is extrapolated. Past its last measured area the decision falls back to the reservation constant,
# which is what "we have not measured this, assume the conservative figure" has to mean here: CUDA's grid ends at
# 1792^2 because 2048^2 ran out of memory on the card it was measured on, and carrying a flat maximum past that point
# priced a 2048^2 decode 0.6 GiB under a 24 GiB card's line while its own reservation exceeded the whole card.
# What the estimator reserves per output pixel-byte: the worst measured point of each backend's column plus ~8%
# headroom (the per-point grid is in `estimate_vae_working_memory_qwen_image`).
_QWEN_VAE_RESERVATION_CONSTANTS: dict[str, dict[str, int]] = {
    "rocm": {"decode": 5500, "encode": 6300},
    "cuda": {"decode": 2900, "encode": 1600},
}

_QWEN_VAE_MEASURED_DECODE_PEAKS: dict[str, tuple[tuple[int, int], ...]] = {
    "rocm": (
        (512 * 512, 5132),
        (768 * 768, 4596),
        (1024 * 1024, 4570),
        (1536 * 1536, 3273),
        (1792 * 1792, 3735),
        (2048 * 2048, 4813),
    ),
    "cuda": (
        (512 * 512, 2660),
        (768 * 768, 2519),
        (1024 * 1024, 2690),
        (1536 * 1536, 2671),
        (1792 * 1792, 2281),
    ),
}


def qwen_image_untiled_decode_peak_bytes(
    image_tensor: torch.Tensor, vae: AutoencoderKLQwenImage, device: torch.device
) -> int:
    """What an untiled decode of this latent is expected to peak at, for the up-front tiling decision only.

    Reservations keep using `estimate_vae_working_memory_qwen_image`, which is deliberately conservative. Between two
    measured points this is the chord between them in bytes -- so an interior area can price below either neighbour's
    per-pixel constant, which is the point: the constant dips while the bytes rise, and taking the larger neighbour
    priced a smaller image above a larger one. Past the last measured area it takes the reservation figure, since a
    decode larger than anything measured is not one to guess low about.
    """
    h = LATENT_SCALE_FACTOR * image_tensor.shape[-2]
    w = LATENT_SCALE_FACTOR * image_tensor.shape[-1]
    element_size = next(vae.parameters()).element_size()
    is_rocm = device.type == "cuda" and torch.version.hip is not None
    points = _QWEN_VAE_MEASURED_DECODE_PEAKS["rocm" if is_rocm else "cuda"]

    area = h * w
    if area > points[-1][0]:
        constant = _QWEN_VAE_RESERVATION_CONSTANTS["rocm" if is_rocm else "cuda"]["decode"]
        return h * w * element_size * constant
    # Interpolate the bytes, not the constant: the per-pixel figure dips (ROCm 3273 at 1536^2, CUDA 2281 at 1792^2)
    # while the bytes themselves rise with area, so taking the larger neighbouring constant priced a smaller image
    # above a larger one -- a 1728x1856 CUDA decode at 15.96 GiB against 1792^2 at 13.64 GiB. Measured points keep
    # their measured value.
    previous_area, previous_bytes = 0, 0
    for measured_area, peak in points:
        measured_bytes = measured_area * peak
        if measured_area >= area:
            if measured_area == area or previous_area == 0:
                return int(area * element_size * peak)
            span = (measured_bytes - previous_bytes) / (measured_area - previous_area)
            return int(element_size * (previous_bytes + span * (area - previous_area)))
        previous_area, previous_bytes = measured_area, measured_bytes
    raise AssertionError("areas past the measured range are handled above")


def estimate_vae_working_memory_qwen_image(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKLQwenImage,
    tile_size: int | None = None,
    device: torch.device | None = None,
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
    the 4:3 tile-to-stride ratio applied by ``patch_qwen_image_vae_tiling``. ``device`` is where the
    VAE will run; it selects the constants below and defaults to the session device.
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
    #  - A second ROCm card measures far below the shipped pair: on an RX 9060 XT (gfx1200, torch 2.12+rocm7.14,
    #    fp16, 2026-09-23, the same script with the attention guard installed) the implied constant is flat at
    #    2650-2772 for decode and 1541-1552 for encode, all the way to 2048^2. The shipped figures stay at the
    #    W7900's, which is the card that needs them; what the spread means is that the reservation carries
    #    cross-card conservatism, so it must not double as the up-front tiling decision -- see
    #    `qwen_image_untiled_decode_peak_bytes`, which prices that decision from the measured curve instead.
    # A named device answers for that device -- a `cpu_only` VAE is not on MIOpen even on a ROCm build. Without one
    # the answer stays the build-wide one these constants were always selected by, so callers that pass no device
    # (and tests that pin none) keep the figure they had.
    is_rocm = torch.version.hip is not None and (device is None or device.type == "cuda")
    scaling_constant = _QWEN_VAE_RESERVATION_CONSTANTS["rocm" if is_rocm else "cuda"][operation]

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


# The SD3 VAE loads in the configured precision and runs uncast: `auto` gives fp16 on CUDA, and bf16 and fp32 are
# honoured. It carries the family's values in `data/vae.json`, as the one member that runs all three.
VAE_SD3_SPEC = register(
    EstimatorSpec(
        id="vae.sd3",
        version=2,
        coefficients=_convolution_coefficients("fp16", "bf16", "fp32"),
        platform_keys=_PLATFORM_KEYS,
        terms=_single_pass_terms,
        line=convolution_line,
        user_calibrated=True,
    )
)


def estimate_vae_working_memory_sd3(
    operation: Literal["encode", "decode"],
    image_tensor: torch.Tensor,
    vae: AutoencoderKL,
    device: torch.device | None = None,
) -> WorkingMemoryEstimate:
    """Estimate the working memory required by the invocation in bytes.

    `device` is where the VAE will run; it picks the convolution backend's coefficients and defaults to the session
    device.
    """
    device = device if device is not None else TorchDevice.choose_torch_device()
    return _estimate(VAE_SD3_SPEC, _single_pass_features(operation, image_tensor, vae, device), device)


def estimate_untiled_vae_decode(
    estimator: str, height: int, width: int, dtype: str, device: torch.device
) -> WorkingMemoryEstimate:
    """The estimate an untiled single-image decode of `height` x `width` output pixels in `dtype` gets from the decode
    node that prices with `estimator`, on `device`: the features each estimator builds from its VAE, without one.

    For planning sizes before anything is loaded (a calibration plan). The SD1/SDXL VAE is priced as one already in
    `dtype`, so without the growth of an in-place cast, which a first fp32 decode after an fp16 one adds on top.
    """
    torch_dtype = {name: candidate for candidate, name in _DTYPE_NAMES.items()}.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"No VAE working-memory coefficients are measured for {dtype}.")
    features: dict[str, FeatureValue] = {"operation": "decode", "height": height, "width": width, "dtype": dtype}
    if estimator == VAE_COGVIEW4_SPEC.id:
        return _estimate(VAE_COGVIEW4_SPEC, features, device)
    features["score_matrix_bytes"] = _vae_mid_block_score_matrix_bytes(height, width, torch_dtype, device=device)
    if estimator == VAE_SD3_SPEC.id:
        return _estimate(VAE_SD3_SPEC, features, device)
    if estimator == VAE_FLUX_SPEC.id:
        return _estimate(VAE_FLUX_SPEC, features | {"tile": None}, device)
    if estimator == VAE_FLUX2_SPEC.id:
        return _estimate(VAE_FLUX2_SPEC, features | {"tile": None, "batch_size": 1}, device)
    if estimator == VAE_SD_SPEC.id:
        sd_features = features | {"tile": None, "vae_class": "AutoencoderKL", "upcast_weight_bytes": 0}
        return _estimate(VAE_SD_SPEC, sd_features, device)
    raise ValueError(f"No untiled decode is priced by working-memory estimator '{estimator}'.")

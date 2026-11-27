"""Measure Wan VAE decode memory and compare it with the shipped estimate.

Run on a CUDA or ROCm device with either a local diffusers VAE directory or a
single Wan ``.safetensors`` checkpoint:

    python scripts/calibrate_wan_vae_working_memory.py --vae /path/to/vae-or-checkpoint

The default shape matches the 12 GiB-card calibration point. Use ``--no-streaming``
to measure the full-frame decode path, or ``--tiling`` to measure the spatially
tiled path used as a low-VRAM fallback. Tiling overrides streaming. The script
reports allocated and reserved deltas; the implied scaling constant uses reserved
memory, which the shipped estimator is calibrated on: the model cache keeps the estimate
free for the decode, and what the decode takes from the device is what the allocator
reserves, about 1.5x its peak of live tensors. The table behind the shipped constant
(in ``estimate_vae_working_memory_wan``) was measured with ``--dtype bfloat16``, both
with and without ``--no-streaming``; its highest point came from a running server, so
the constant sits above what this script reports.

The allocator is configured from the environment only: the script does not read
``pytorch_cuda_alloc_conf`` from ``invokeai.yaml``. To measure the allocator a server
configures there, export the same value as ``PYTORCH_CUDA_ALLOC_CONF`` first.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Direct script execution puts ``scripts/`` on sys.path, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invokeai.app.util.torch_cuda_allocator import apply_rocm_windows_allocator_default  # noqa: E402

# The server's ROCm runtime, so a ROCm measurement is what Invoke would see: both must be set before torch is
# imported (see `invokeai.app.run_app` and `apply_rocm_windows_allocator_default`). Inert on CUDA. Only when run as a
# script, so importing the module, as its tests do, leaves the process environment alone.
if __name__ == "__main__":
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    apply_rocm_windows_allocator_default(logging.getLogger(__name__))

import torch  # noqa: E402
from diffusers.models.autoencoders import AutoencoderKLWan  # noqa: E402

from invokeai.backend.model_manager.load.model_loaders.vae import _wan_vae_init_kwargs_for  # noqa: E402
from invokeai.backend.util.attention import install_rocm_sdpa_guard  # noqa: E402
from invokeai.backend.util.qwen_image_vae import (  # noqa: E402
    patch_qwen_image_vae_tiling,
    resolve_qwen_image_vae_tile_size,
)
from invokeai.backend.util.vae_working_memory import (  # noqa: E402
    estimate_vae_working_memory_wan,
    wan_vae_clip_bytes,
)
from invokeai.backend.wan.rocm_causal_conv3d import patch_wan_causal_conv3d_for_rocm  # noqa: E402
from invokeai.backend.wan.vae_decode import iter_wan_vae_decode_chunks  # noqa: E402

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _load_vae(path: Path, dtype: torch.dtype) -> AutoencoderKLWan:
    if path.is_dir():
        vae = AutoencoderKLWan.from_pretrained(path, local_files_only=True, torch_dtype=dtype)
        vae.eval()
        return vae
    if not path.is_file():
        raise ValueError("--vae must point to a diffusers directory or a Wan .safetensors file")

    import accelerate
    from safetensors.torch import load_file

    state_dict = load_file(str(path), device="cpu")
    try:
        latent_channels = int(state_dict["decoder.conv_in.weight"].shape[1])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Wan checkpoint is missing decoder.conv_in.weight or has an invalid shape") from exc

    with accelerate.init_empty_weights():
        vae = AutoencoderKLWan(**_wan_vae_init_kwargs_for(latent_channels))
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            state_dict[key] = tensor.to(dtype=dtype)
    vae.load_state_dict(state_dict, strict=True, assign=True)
    vae.eval()
    return vae


@torch.inference_mode()
def _measure(
    vae: AutoencoderKLWan,
    pixel_height: int,
    pixel_width: int,
    pixel_frames: int,
    streaming: bool,
    tiling: bool = False,
    tile_size: int | None = None,
) -> dict[str, int | float | bool | str | None]:
    temporal_scale = int(getattr(vae.config, "scale_factor_temporal", None) or 4)
    spatial_scale = int(getattr(vae.config, "scale_factor_spatial", None) or 8)
    if pixel_frames < 1 or (pixel_frames - 1) % temporal_scale != 0:
        raise ValueError(f"pixel_frames must satisfy (frames - 1) % {temporal_scale} == 0")
    if pixel_height < 1 or pixel_width < 1 or pixel_height % spatial_scale or pixel_width % spatial_scale:
        raise ValueError(f"height and width must be positive multiples of {spatial_scale}")

    device = torch.device("cuda")
    vae.to(device=device)
    if tiling:
        streaming = False
        if tile_size is None:
            tile_size = resolve_qwen_image_vae_tile_size(0)  # the video node's tile
        # The node pairs a tile with a stride of 3/4 of it, rounded down to 8 px. A tile on this grid keeps that stride
        # on the latent grid (16 px for TI2V), so the decode steps through tiles the way it crops them.
        tile_grid = 4 * spatial_scale
        if tile_size < tile_grid or tile_size % tile_grid:
            raise ValueError(f"tile_size must be a positive multiple of {tile_grid}")
    else:
        tile_size = None
    element_size = next(vae.parameters()).element_size()
    latent_frames = (pixel_frames - 1) // temporal_scale + 1
    latent_height = pixel_height // spatial_scale
    latent_width = pixel_width // spatial_scale
    latents = torch.randn(
        1,
        int(getattr(vae.config, "z_dim", 16)),
        latent_frames,
        latent_height,
        latent_width,
        device=device,
        dtype=next(vae.parameters()).dtype,
    )

    estimate = estimate_vae_working_memory_wan(
        operation="decode",
        vae=vae,
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        pixel_frames=pixel_frames,
        tile_size=tile_size,
        streaming=streaming,
    )
    clip_bytes = wan_vae_clip_bytes(
        operation="decode",
        vae=vae,
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        pixel_frames=pixel_frames,
        tile_size=tile_size,
        streaming=streaming,
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    if tiling:
        assert tile_size is not None
        scaling_basis_bytes = tile_size**2 * element_size * 1.25
    else:
        scaling_basis_bytes = pixel_height * pixel_width * element_size
    # The node's tiling geometry: a stride that follows the tile size, which `enable_tiling` would otherwise leave at
    # the default 192 px, dropping bands of the frame under a smaller tile.
    with patch_qwen_image_vae_tiling(vae, tile_size):
        if streaming:
            for chunk in iter_wan_vae_decode_chunks(vae, latents):
                chunk = chunk[0].cpu()
        else:
            vae.decode(latents, return_dict=False)[0].cpu()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    measured_allocated_delta = peak_allocated - baseline_allocated
    measured_reserved_delta = peak_reserved - baseline_reserved
    implied_constant = (measured_reserved_delta - clip_bytes) / scaling_basis_bytes
    return {
        "device": torch.cuda.get_device_name(device),
        "backend": "ROCm" if torch.version.hip is not None else "CUDA",
        "dtype": str(next(vae.parameters()).dtype),
        "streaming": streaming,
        "tiling": tiling,
        "tile_size": tile_size,
        "pixel_height": pixel_height,
        "pixel_width": pixel_width,
        "pixel_frames": pixel_frames,
        "estimate_bytes": estimate,
        "measured_allocated_delta_bytes": measured_allocated_delta,
        "measured_reserved_delta_bytes": measured_reserved_delta,
        "implied_scaling_constant": implied_constant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--vae", type=Path, required=True, help="Diffusers AutoencoderKLWan directory or .safetensors checkpoint."
    )
    parser.add_argument("--height", type=int, default=704, help="Pixel height. Default: 704.")
    parser.add_argument("--width", type=int, default=1280, help="Pixel width. Default: 1280.")
    parser.add_argument("--frames", type=int, default=81, help="Pixel frame count. Default: 81.")
    parser.add_argument("--dtype", choices=list(DTYPES), default="float16", help="VAE dtype. Default: float16.")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure chunked streaming decode. Use --no-streaming for full decode.",
    )
    parser.add_argument(
        "--tiling",
        action="store_true",
        help="Measure spatially tiled full decode. Overrides --streaming; use --tile-size to override the tile size.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Spatial tile size in pixels. Requires --tiling; defaults to the video node's 256.",
    )
    args = parser.parse_args()

    if args.tile_size is not None and not args.tiling:
        parser.error("--tile-size requires --tiling")
    if args.tile_size is not None and args.tile_size <= 0:
        parser.error("--tile-size must be positive")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA or ROCm device required")
    # What the Wan VAE loader and the app's startup apply on ROCm; no-ops on CUDA.
    patch_wan_causal_conv3d_for_rocm()
    install_rocm_sdpa_guard()
    vae = _load_vae(args.vae, DTYPES[args.dtype])
    try:
        result = _measure(
            vae,
            args.height,
            args.width,
            args.frames,
            args.streaming,
            tiling=args.tiling,
            tile_size=args.tile_size,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("VAE decode ran out of device memory; reduce --height, --width, or --frames") from exc

    gib = 2**30
    print(f"device: {result['device']} ({result['backend']})")
    print(f"dtype: {result['dtype']}; streaming: {result['streaming']}; tiling: {result['tiling']}")
    if result["tiling"]:
        print(f"tile size: {result['tile_size']}px")
    print(f"shape: {result['pixel_height']}x{result['pixel_width']}x{result['pixel_frames']}")
    print(f"estimate: {result['estimate_bytes'] / gib:.3f} GiB")
    print(f"measured allocated delta: {result['measured_allocated_delta_bytes'] / gib:.3f} GiB")
    print(f"measured reserved delta: {result['measured_reserved_delta_bytes'] / gib:.3f} GiB")
    print(f"implied scaling constant (reserved): {result['implied_scaling_constant']:.1f}")
    # With --tiling the clip copies can exceed the whole measurement, leaving a negative constant: the coverage is the
    # figure that says whether the estimate holds.
    print(f"estimate covers: {result['estimate_bytes'] / result['measured_reserved_delta_bytes']:.2f}x of reserved")


if __name__ == "__main__":
    main()

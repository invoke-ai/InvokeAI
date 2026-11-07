"""Measure the working memory of tiled VAE decodes and encodes, as shipped-data points for `vae.flux`, `vae.flux2` and
`vae.sd`.

What a point measures
---------------------
The demand of one VAE operation, decode or encode, run as its node runs it inside its `model_on_device` window, and
measured as a fresh window (`invokeai/backend/working_memory/measurement.py`, whose definitions the in-server probe's
windows share). With the VAE resident and the node's input on the host, where the node has it: synchronize, release
the allocator's cached blocks, read the baseline and reset the peaks; run the node's in-window work; the demand is then
the largest growth of three counters over the baseline: the driver's figure for this process, torch's reserved peak
and its allocated peak. An operation whose allocations failed, or only succeeded after the allocator released its
cache and retried (a squeezed peak), or after which Windows keeps more than `measurement.SPILL_TOLERANCE_BYTES` of the
process in shared memory, is invalid: it is recorded as such, not as a point.

Every point runs in a fresh process (`scripts/working_memory/fresh_points.py`, which the calibration scripts share),
which first applies the server's runtime (`apply_server_runtime` in `invokeai/app/run_app.py`): the `MIOPEN_FIND_MODE`
default, the integrated-GPU check, the allocator, the AOTriton decision and the monkeypatches, the ROCm SDPA guard among
them, in the server's order. Its settings are those of a server without a config file, except the allocator
configuration: `--alloc-conf`, or without one the platform default the server uses (expandable segments for ROCm on
Windows, torch's native allocator elsewhere). This script's own process runs the integrated-GPU check once before the
first point: the measuring processes inherit the `HIP_VISIBLE_DEVICES` it sets, which their own check keeps, as the
server keeps an exported one.

The operation then runs twice, cold and warm, and the point takes the larger demand (`measure_cold_and_warm` and
`measured_bytes_and_method` in `measurement.py`): a `demand_peak` where both runs read the driver's figure, else a
`reserved_peak` of torch's counters alone.

The window
----------
What each node does between entering `model_on_device` and leaving it, which is what its reservation has to cover and
what a point runs (`_OPERATIONS`). The tiling state is set with the node's own helper, so the tiled loop that runs is
the node's.

- `vae.flux`, decode: `z_image_l2i` (`invocations/vae/z_image_latents_to_image.py`), the larger of the two decodes this
  estimator prices: the latents moved to the device in the VAE's dtype, `TorchDevice.empty_cache()`, the scaling and
  shift, `vae.decode` under `scoped_vae_tiling`, then the clamp, the `127.5 * (x + 1)` and the `byte()` that make the
  image, and its copy to the host. `flux_vae_decode` does the post-processing after its window.
- `vae.flux`, encode: `flux_vae_encode`: the image moved to the device in the VAE's dtype, `vae.encode` under
  `scoped_vae_tiling`, a sample of the latent distribution with the node's seeded generator, the shift and the scaling.
  The image is made into a tensor on the host before the window (`image_resized_to_grid_as_tensor`), as the node does.
- `vae.flux2`, decode: `flux2_vae_decode`: the latents moved to the device, `vae.decode` under `scoped_vae_tiling`.
  The node's post-processing runs after its window.
- `vae.flux2`, encode: `flux2_vae_encode`: the image moved to the device, `vae.encode` under `scoped_vae_tiling`, the
  distribution's mode.
- `vae.sd`, decode: `l2i`: the latents moved to the device and cast to `sd_vae_run_dtype`, the tiling switched on or
  off, `patch_vae_tiling_params` for a tile field above 0, `TorchDevice.empty_cache()`, the scaling, `vae.decode`, the
  denormalization and clamp, the copy to the host and the PIL image.
- `vae.sd`, encode: `i2l`: the tiling switch and patch, the image moved to the device in the VAE's dtype,
  `ImageToLatentsInvocation._encode_to_tensor` (the node's own: encode and sample), the scaling.

The VAE is built in the dtype `sd_vae_run_dtype` gives, so the SD nodes' casts of the VAE (to that dtype, and an fp32
encode's decoder parts back) would change nothing and are left out: a point measures the operation, not an in-place cast
of the weights, and its `upcast_weight_bytes` is 0, as on every shipped point.

The VAEs
--------
Random weights in the measured dtype at the real geometry, because memory depends on shapes, not on weight values:

- FLUX.1 (and Z-Image, which shares it): `AutoencoderKL(**get_flux_vae_diffusers_config())`, as its loader builds it.
- FLUX.2: `AutoencoderKLFlux2(block_out_channels=(128, 256, 512, 512))`, as `model_loaders/flux.py` builds the stock
  VAE.
- SDXL: `SDXL_VAE_CONFIG`, the released SDXL VAE's config (`stabilityai/sdxl-vae`, the `vae/config.json` of the SDXL
  base and refiner repos: block widths 128-512, 2 layers per block, 4 latent channels, `sample_size` 1024); the SD1 VAE
  differs only in `sample_size` 512 and its scaling factor.

Each point carries the features of the node's own estimator call (`estimate_vae_working_memory_flux`, `_flux2`,
`_sd15_sdxl` with the node's arguments: the host tensor, the VAE, the tile field or None, the fp32 switch), taken in
the measuring process after the runtime setup, so a score matrix is priced for the attention kernel this build runs.
Each point is priced before it runs, the estimate plus the weights; it is skipped when that exceeds free device memory,
unless `--force`. An oversized allocation does not always fail: where the driver backs it with system RAM (NVIDIA's
sysmem fallback, AMD on Windows) it pages, which can stall the whole machine.

The grid
--------
Per family in its dtypes (FLUX.1 and FLUX.2 in bf16, which their loaders run; SD in fp16 and fp32, the two
`sd_vae_run_dtype` runs on a GPU), per operation:

- an untiled reference at each tile's side, so a run compares tiled and untiled demand on its own;
- each `--tiles` field (512, 1024 and the node default 0) at each `--sizes` side (1024 to 8192 px) above the tile the
  VAE then runs. A side at or below it runs the single pass, which the reference measures. The 0 sentinel resolves to
  512 px for FLUX.1 and FLUX.2 (`resolve_tile_size`) but to the VAE's own `tile_sample_min_size` for SD, 1024 px for
  SDXL, so it re-measures the 512 px tile there and the 1024 px one here: its points should equal those.

The sides run to 8192 px because a tiled decode has two phases, and the second only shows above 4096 px: up to there
the decoded tiles diffusers keeps fit in the allocator's slack around the tile's own peak, and above it assembling
them into the image takes over (`fit.Line.assembly`). Sides that stop at 4096 px measure the tile phase alone and fit
no assembly line; the shipped one would be extrapolated.

Every point is a fit point (`"fit": true`, `in_fit` true): an untiled reference of its family's untiled line, a tiled
point of its estimator's tiled line, which sits on the untiled one at the tile and is fitted above the reference's
demand there, and of a decode's assembly line (`fit.convolution_line`). A tile, or an untiled reference's image,
below `fit.SMALLEST_APP_SIDE` (512 px) is covered but caps nothing. A 0 sentinel point shares its features with the
explicit tile's, so the builder merges the two as one measurement and keeps the larger reading: a second cold run of
the same operation, which ships no point of its own. In the shipped runs 3 of the 208 pairs differed, by up to
+512 MiB (the RTX 4090 under `cudaMallocAsync`, the SD fp16 decode at 3072 px with a 1024 px tile).

Output
------
Every measured point is appended to `--out` as one JSON object per line, the point file
`scripts/working_memory/build_shipped.py --points` reads. It carries the fields of a `MeasuredPoint`
(`invokeai/backend/working_memory/coefficients.py`): `estimator`, `version` and `features`, the estimate's own;
`platform`, as `platform_for` gives it; `measured_bytes`, `method`, `device` and `in_fit`; and `provenance`. The other
keys are the script's, which the builder ignores: `fit`, `point` (the grid entry) and `raw` (each run's counter deltas,
the prediction, the tile the VAE ran and the runtime facts). An invalid, skipped or failed operation is no point: it
shows in the table on stdout, with its reasons, and stays out of the file.

After the table, a summary per family, operation, dtype and tile: per side the measured demand against the estimate
it was priced at, against the untiled demand at the tile's side, and the growth per output pixel above that; the growth
per pixel across the sides; and for the 0 sentinel its difference to the tile it resolves to.

Usage
-----
One run per platform measures the default grid. The shipped points (`data/vae.json`, 2026-10-19) came from these runs
per platform, which the builder merges into one set: the default grid in two runs, the sides 1024 to 4096 px first and
`--sizes 6144,8192` after; `--sizes 12288 --tiles 512,1024`; `--sizes 2048,4096,8192 --tiles 256,768`, for tiles the
grid does not have (the 256 px ones bound their estimates from below only); and on the RTX 4090
`--sizes 2048,4096,8192,12288 --tiles 1536`, the largest tile the upscale tab sets.

RTX 4090 (CUDA): the native allocator, then `cudaMallocAsync`.

    python scripts/calibrate_vae_tiling_working_memory.py --out 4090-native-tiled.jsonl
    python scripts/calibrate_vae_tiling_working_memory.py --out 4090-async-tiled.jsonl \
        --alloc-conf backend:cudaMallocAsync

RX 9060 XT (ROCm on Windows): the server's default expandable segments, then the native allocator.

    python scripts/calibrate_vae_tiling_working_memory.py --out 9060xt-expandable-tiled.jsonl
    python scripts/calibrate_vae_tiling_working_memory.py --out 9060xt-native-tiled.jsonl \
        --alloc-conf expandable_segments:False

`--families`, `--tiles` and `--sizes` narrow a run. `--dry-run` prints the planned points without touching a device,
each priced as a CPU prices it: with the shipped values a CPU borrows, and an SD decode in fp32, which is what a CPU
runs (the dtype column says which line priced it).
"""

import argparse
import dataclasses
import datetime
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

# None of these imports torch: a measuring process has to configure the allocator before torch is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR  # noqa: E402
from invokeai.backend.util.vae_tiling_scope import diffusers_latent_tile, resolve_tile_size  # noqa: E402
from scripts.working_memory import fresh_points  # noqa: E402

if TYPE_CHECKING:
    import torch

    from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate

GIB = 2**30
MIB = 2**20

Family = Literal["flux", "flux2", "sd"]
Operation = Literal["decode", "encode"]
FAMILIES: tuple[Family, ...] = get_args(Family)
OPERATIONS: tuple[Operation, ...] = get_args(Operation)

DTYPES: Mapping[Family, tuple[str, ...]] = {
    "flux": ("bfloat16",),
    "flux2": ("bfloat16",),
    "sd": ("float16", "float32"),
}
"""The dtypes each family runs in on a GPU: the FLUX.1 and FLUX.2 loaders' bf16, and SD's `sd_vae_run_dtype`, fp16
unless the node asks for fp32."""

FLUX2_BLOCK_OUT_CHANNELS = (128, 256, 512, 512)
"""The stock FLUX.2 VAE's encoder and decoder widths, as `model_loaders/flux.py` reads them from the checkpoint."""

SDXL_VAE_CONFIG: Mapping[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "act_fn": "silu",
    "latent_channels": 4,
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": 0.13025,
    "force_upcast": True,
}
"""The released SDXL VAE's config (`stabilityai/sdxl-vae`, the SDXL base and refiner repos' `vae/config.json`), as
`AutoencoderKL` keyword arguments; diffusers' defaults supply the quant convs it has."""

TILE_OVERLAP_FACTOR = 0.25
"""The overlap every one of these VAEs ships and the SD nodes patch in."""

LOGGER_NAME = "calibrate_vae_tiling_working_memory"

_NODES: Mapping[tuple[Family, Operation], str] = {
    ("flux", "decode"): "z_image_l2i",
    ("flux", "encode"): "flux_vae_encode",
    ("flux2", "decode"): "flux2_vae_decode",
    ("flux2", "encode"): "flux2_vae_encode",
    ("sd", "decode"): "l2i",
    ("sd", "encode"): "i2l",
}


@dataclass(frozen=True)
class Point:
    """One VAE operation to measure, in a process of its own."""

    family: Family
    operation: Operation
    dtype: str
    """A torch dtype name, one of the family's `DTYPES`."""
    side: int
    """The square image's side in output pixels."""
    tile: int | None
    """The node's `tile_size` field with tiling on, 0 its "default" sentinel; None for an untiled operation."""

    @property
    def fit(self) -> bool:
        """Every point fits a line: an untiled reference the untiled one, a tiled point its estimator's tiled line."""
        return True


def applied_tile(family: Family, tile: int) -> int:
    """The tile, in output pixels, the VAE runs for a node's tile field.

    FLUX.1 and FLUX.2 go through `scoped_vae_tiling`, which resolves the field (`resolve_tile_size`: 0 is the shipped
    512 px) and snaps it to a latent tile the diffusers loop assembles. SD goes through `patch_vae_tiling_params`,
    which snaps the same way, for a field above 0 only: the 0 sentinel leaves the VAE's own `tile_sample_min_size`,
    its `sample_size`.
    """
    if family == "sd":
        if tile <= 0:
            return int(SDXL_VAE_CONFIG["sample_size"])
        requested = tile
    else:
        requested = resolve_tile_size(tile)
    return diffusers_latent_tile(requested, LATENT_SCALE_FACTOR, TILE_OVERLAP_FACTOR) * LATENT_SCALE_FACTOR


def plan(*, families: Sequence[Family], sizes: Sequence[int], tiles: Sequence[int]) -> list[Point]:
    """The operations a run measures, in order: per family, dtype and operation, the untiled reference at each tile's
    side, then each tile at the sides above the tile it runs."""
    points: list[Point] = []
    for family in families:
        references = sorted({applied_tile(family, tile) for tile in tiles})
        for dtype in DTYPES[family]:
            for operation in OPERATIONS:
                points += [Point(family, operation, dtype, side, None) for side in references]
                points += [
                    Point(family, operation, dtype, side, tile)
                    for tile in tiles
                    for side in sizes
                    if side > applied_tile(family, tile)
                ]
    return points


def provenance(point: Point, facts: Mapping[str, Any], date: str) -> str:
    """A measured point's `provenance`: where and how it was measured."""
    vae = {
        "flux": "FLUX.1 AutoencoderKL with its loader's geometry",
        "flux2": f"FLUX.2 AutoencoderKLFlux2 with block widths {FLUX2_BLOCK_OUT_CHANNELS}",
        "sd": "SDXL AutoencoderKL from the released config",
    }[point.family]
    tiling = (
        "untiled"
        if point.tile is None
        else f"tile field {point.tile}, a {applied_tile(point.family, point.tile)}px tile"
    )
    return (
        f"scripts/calibrate_vae_tiling_working_memory.py ({date}): {fresh_points.runtime_provenance(facts)}. "
        f"Random-init {vae}, {point.operation} {tiling}, run as `{_NODES[(point.family, point.operation)]}` runs it "
        "inside its model_on_device window; memory does not depend on weight values. A fresh process with the "
        "server's runtime, the larger demand of a cold and a warm run; no failed allocation, allocator retry or spill."
    )


# Everything below imports torch. In a measuring process it runs only after `fresh_points.apply_server_runtime`.


def build_vae(family: Family) -> "torch.nn.Module":
    """The family's VAE at its real geometry on the meta device."""
    import torch
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    from invokeai.backend.flux.util import get_flux_vae_diffusers_config

    with torch.device("meta"):
        if family == "flux":
            return AutoencoderKL(**get_flux_vae_diffusers_config())
        if family == "flux2":
            return AutoencoderKLFlux2(block_out_channels=FLUX2_BLOCK_OUT_CHANNELS)
        return AutoencoderKL(**SDXL_VAE_CONFIG)


def _materialize(module: "torch.nn.Module", device: "torch.device", dtype: "torch.dtype") -> "torch.nn.Module":
    """A meta-device module allocated directly in `dtype` on `device`, randomly initialized."""
    import torch

    module = module.to(dtype).to_empty(device=device)
    for param in module.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return module.eval()


def latent_channels(vae: "torch.nn.Module") -> int:
    return int(vae.config.latent_channels)


def host_input(point: Point, vae: "torch.nn.Module") -> "torch.Tensor":
    """What the node holds on the host when it enters its window: the latents a decode loads, or the image tensor an
    encode makes of its image with `image_resized_to_grid_as_tensor`, batch 1."""
    import numpy as np
    import torch
    from PIL import Image

    from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor

    generator = torch.Generator().manual_seed(0)
    if point.operation == "decode":
        latent = point.side // LATENT_SCALE_FACTOR
        return torch.randn(1, latent_channels(vae), latent, latent, generator=generator)
    pixels = np.random.default_rng(0).integers(0, 256, (point.side, point.side, 3), dtype=np.uint8)
    return image_resized_to_grid_as_tensor(Image.fromarray(pixels))[None]


def estimate(
    point: Point, vae: "torch.nn.Module", tensor: "torch.Tensor", device: "torch.device"
) -> "WorkingMemoryEstimate":
    """The node's own estimate for the point: its estimator with the arguments it passes, the host tensor, the VAE and
    the tile field (None untiled)."""
    from invokeai.backend.util.vae_working_memory import (
        estimate_vae_working_memory_flux,
        estimate_vae_working_memory_flux2,
        estimate_vae_working_memory_sd15_sdxl,
    )

    if point.family == "flux":
        return estimate_vae_working_memory_flux(point.operation, tensor, vae, tile_size=point.tile, device=device)
    if point.family == "flux2":
        return estimate_vae_working_memory_flux2(point.operation, tensor, vae, tile_size=point.tile, device=device)
    return estimate_vae_working_memory_sd15_sdxl(
        point.operation, tensor, vae, tile_size=point.tile, fp32=point.dtype == "float32", device=device
    )


def _flux_decode(point: Point, vae: Any, latents: "torch.Tensor", device: "torch.device") -> Callable[[], object]:
    """`z_image_l2i` inside its window: post-processing included, unlike `flux_vae_decode`."""
    import torch
    from einops import rearrange
    from PIL import Image

    from invokeai.backend.util.devices import TorchDevice
    from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling

    def run() -> object:
        vae_dtype = next(iter(vae.parameters())).dtype
        z = latents.to(device=device, dtype=vae_dtype)
        TorchDevice.empty_cache()
        z = z / vae.config.scaling_factor
        shift_factor = getattr(vae.config, "shift_factor", None)
        if shift_factor is not None:
            z = z + shift_factor
        with torch.inference_mode():
            with scoped_vae_tiling(vae, point.tile):
                img = vae.decode(z, return_dict=False)[0]
        img = img.clamp(-1, 1)
        img = rearrange(img[0], "c h w -> h w c")
        return Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

    return run


def _flux_encode(point: Point, vae: Any, image: "torch.Tensor", device: "torch.device") -> Callable[[], object]:
    """`flux_vae_encode` inside its window. Its generator is made before the window, as the node makes it."""
    import torch

    from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling

    generator = torch.Generator(device=device).manual_seed(0)

    def run() -> object:
        vae_dtype = next(iter(vae.parameters())).dtype
        x = image.to(device=device, dtype=vae_dtype)
        with scoped_vae_tiling(vae, point.tile):
            latents = vae.encode(x).latent_dist.sample(generator)
        shift_factor = getattr(vae.config, "shift_factor", None)
        if shift_factor is not None:
            latents = latents - shift_factor
        return latents * vae.config.scaling_factor

    return run


def _flux2_decode(point: Point, vae: Any, latents: "torch.Tensor", device: "torch.device") -> Callable[[], object]:
    """`flux2_vae_decode` inside its window; it post-processes after it."""
    from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling

    def run() -> object:
        z = latents.to(device=device, dtype=next(iter(vae.parameters())).dtype)
        with scoped_vae_tiling(vae, point.tile):
            return vae.decode(z, return_dict=False)[0]

    return run


def _flux2_encode(point: Point, vae: Any, image: "torch.Tensor", device: "torch.device") -> Callable[[], object]:
    """`flux2_vae_encode` inside its window."""
    from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling

    def run() -> object:
        x = image.to(device=device, dtype=next(iter(vae.parameters())).dtype)
        with scoped_vae_tiling(vae, point.tile):
            latent_dist = vae.encode(x, return_dict=False)[0]
        return latent_dist.mode()

    return run


def _sd_tiling(point: Point, vae: Any):
    """The SD nodes' tiling setup: the switch, and `patch_vae_tiling_params` for a tile field above 0."""
    from invokeai.backend.stable_diffusion.vae_tiling import patch_vae_tiling_params

    if point.tile is not None:
        vae.enable_tiling()
    else:
        vae.disable_tiling()
    tile_size = point.tile or 0
    if tile_size > 0:
        return patch_vae_tiling_params(vae, tile_sample_min_size=tile_size, tile_overlap_factor=TILE_OVERLAP_FACTOR)
    return nullcontext()


def _sd_decode(point: Point, vae: Any, latents: "torch.Tensor", device: "torch.device") -> Callable[[], object]:
    """`l2i` inside its window."""
    import torch
    from diffusers.image_processor import VaeImageProcessor

    from invokeai.backend.util.devices import TorchDevice
    from invokeai.backend.util.vae_working_memory import sd_vae_run_dtype

    def run() -> object:
        z = latents.to(device)
        z = z.to(dtype=sd_vae_run_dtype("decode", point.dtype == "float32", device))
        tiling_context = _sd_tiling(point, vae)
        TorchDevice.empty_cache()
        with torch.inference_mode(), tiling_context:
            z = z / vae.config.scaling_factor
            image = vae.decode(z, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            np_image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return VaeImageProcessor.numpy_to_pil(np_image)[0]

    return run


def _sd_encode(point: Point, vae: Any, image: "torch.Tensor", device: "torch.device") -> Callable[[], object]:
    """`i2l` inside its window (`ImageToLatentsInvocation.vae_encode`), the node's sampling included."""
    import torch

    from invokeai.app.invocations.vae.image_to_latents import ImageToLatentsInvocation

    def run() -> object:
        tiling_context = _sd_tiling(point, vae)
        x = image.to(device=device, dtype=vae.dtype)
        with torch.inference_mode(), tiling_context:
            latents = ImageToLatentsInvocation._encode_to_tensor(vae, x)
        return vae.config.scaling_factor * latents

    return run


_OPERATIONS: Mapping[tuple[Family, Operation], Callable[..., Callable[[], object]]] = {
    ("flux", "decode"): _flux_decode,
    ("flux", "encode"): _flux_encode,
    ("flux2", "decode"): _flux2_decode,
    ("flux2", "encode"): _flux2_encode,
    ("sd", "decode"): _sd_decode,
    ("sd", "encode"): _sd_encode,
}


def operation(point: Point, vae: Any, tensor: "torch.Tensor", device: "torch.device") -> Callable[[], object]:
    """The work the point's node does inside its `model_on_device` window, on the host `tensor` it enters it with, as a
    call a fresh window can measure (see the module docstring for what each node does there)."""
    return _OPERATIONS[(point.family, point.operation)](point, vae, tensor, device)


def _measure(point: Point, force: bool) -> dict[str, Any]:
    """The output line of `point`, measured in this process after `fresh_points.apply_server_runtime`."""
    import torch

    from invokeai.backend.util import vae_working_memory
    from invokeai.backend.util.devices import TorchDevice
    from invokeai.backend.working_memory.measurement import measure_cold_and_warm
    from invokeai.backend.working_memory.platform import platform_for

    # The device the node itself takes.
    device = TorchDevice.choose_torch_device()
    if device.type != "cuda":
        raise SystemExit(f"Needs a CUDA or ROCm device; this process would run on {device}.")
    dtype = getattr(torch, point.dtype)

    # The nodes' `invoke` runs under `no_grad`; their own `inference_mode` blocks are in the operations.
    with torch.no_grad():
        vae = build_vae(point.family).to(dtype)
        tensor = host_input(point, vae)
        wm = estimate(point, vae, tensor, device)
        platform = platform_for(device)
        facts = fresh_points.runtime_facts(device, platform.os, vae_working_memory.__file__)
        common = {
            "estimator": wm.estimator,
            "version": wm.version,
            "platform": dataclasses.asdict(platform),
            "features": dict(wm.features),
            "device": facts["device"],
        }
        weights = sum(param.numel() for param in vae.parameters()) * dtype.itemsize
        free, _ = TorchDevice.cuda_mem_get_info(device)
        raw = {
            "reservation_bytes": wm.reservation_bytes,
            "weight_bytes": weights,
            "predicted_bytes": wm.reservation_bytes + weights,
            "free_bytes": free,
            "applied_tile": None if point.tile is None else applied_tile(point.family, point.tile),
            "runtime": facts,
        }
        if raw["predicted_bytes"] > free and not force:
            return fresh_points.skipped_line(point, common, raw)

        vae = _materialize(vae, device, dtype)
        windows = measure_cold_and_warm(device, operation(point, vae, tensor, device))
    date = datetime.date.today().isoformat()
    return fresh_points.point_line(
        point, common=common, provenance=provenance(point, facts, date), windows=windows, raw=raw
    )


_COLUMNS = f"{'family':>6} {'op':>6} {'dtype':>8} {'side':>5} {'tile':>5} {'runs':>5}"


def _grid(point: Mapping[str, Any]) -> str:
    tile = point["tile"]
    runs = "" if tile is None else applied_tile(point["family"], tile)
    return (
        f"{point['family']:>6} {point['operation']:>6} {point['dtype']:>8} {point['side']:>5} "
        f"{'-' if tile is None else tile:>5} {runs:>5}"
    )


def _head(line: Mapping[str, Any]) -> str:
    """A line's grid columns in the table."""
    return _grid(line["point"])


def _summarize(lines: Sequence[Mapping[str, Any]]) -> None:
    """Per family, operation, dtype and tile field: the numbers a refit of the tiled estimate needs."""
    measured = [line for line in lines if line["status"] == "point"]
    groups = dict.fromkeys((p["family"], p["operation"], p["dtype"]) for p in (line["point"] for line in measured))
    for family, op, dtype in groups:
        of_group = [line for line in measured if (line["point"]["family"], line["point"]["operation"]) == (family, op)]
        of_group = [line for line in of_group if line["point"]["dtype"] == dtype]
        untiled = {line["point"]["side"]: line for line in of_group if line["point"]["tile"] is None}
        for side, line in sorted(untiled.items()):
            print(
                f"{family} {op} {dtype}, untiled {side}px: {line['measured_bytes'] / GIB:.3f} GiB, the estimate "
                f"{line['raw']['reservation_bytes'] / line['measured_bytes']:.2f}x"
            )
        tiled: dict[int, list[Mapping[str, Any]]] = {}
        for line in sorted(of_group, key=lambda line: line["point"]["side"]):
            if line["point"]["tile"] is not None:
                tiled.setdefault(line["point"]["tile"], []).append(line)
        for tile, rows in tiled.items():
            runs = applied_tile(family, tile)
            reference = untiled.get(runs)
            print(
                f"{family} {op} {dtype}, tile {tile} ({runs}px), untiled {runs}px "
                + (f"{reference['measured_bytes'] / GIB:.3f} GiB:" if reference else "not measured:")
            )
            for line in rows:
                side, demand = line["point"]["side"], line["measured_bytes"]
                text = (
                    f"  {side}px: {demand / GIB:.3f} GiB, the estimate {line['raw']['reservation_bytes'] / GIB:.3f} "
                    f"GiB ({line['raw']['reservation_bytes'] / demand:.2f}x)"
                )
                if reference:
                    above = demand - reference["measured_bytes"]
                    text += (
                        f"; {demand / reference['measured_bytes']:.2f}x the untiled tile, "
                        f"{above / (side * side):+.2f} bytes per output pixel above it"
                    )
                print(text)
            if len(rows) >= 2:
                lo, hi = rows[0], rows[-1]
                pixels = hi["point"]["side"] ** 2 - lo["point"]["side"] ** 2
                slope = (hi["measured_bytes"] - lo["measured_bytes"]) / pixels
                print(f"  {lo['point']['side']}-{hi['point']['side']}px: {slope:+.3f} bytes per output pixel")
            # The sentinel against the explicit field it resolves to: the same tile, so the same demand.
            explicit = [
                lines_ for t, lines_ in tiled.items() if tile == 0 and t > 0 and applied_tile(family, t) == runs
            ]
            if explicit:
                by_side = {line["point"]["side"]: line["measured_bytes"] for line in explicit[0]}
                differences = [
                    f"{(line['measured_bytes'] - by_side[line['point']['side']]) / MIB:+.0f} MiB at "
                    f"{line['point']['side']}px"
                    for line in rows
                    if line["point"]["side"] in by_side
                ]
                print(f"  tile {tile} against the {runs}px tile field: {', '.join(differences) or 'no common side'}")


def _print_plan(points: Sequence[Point]) -> None:
    """The planned points, each priced as a CPU prices it (see the module docstring), plus the weights."""
    import torch

    cpu = torch.device("cpu")
    print(f"{_COLUMNS} {'priced':>6} {'estimate GiB':>12} {'weights GiB':>11} {'total GiB':>9}")
    vaes: dict[tuple[Family, str], Any] = {}
    for point in points:
        key = (point.family, point.dtype)
        if key not in vaes:
            vaes[key] = build_vae(point.family).to(getattr(torch, point.dtype))
        vae = vaes[key]
        if point.operation == "decode":
            latent = point.side // LATENT_SCALE_FACTOR
            shape: tuple[int, ...] = (1, latent_channels(vae), latent, latent)
        else:
            shape = (1, 3, point.side, point.side)
        wm = estimate(point, vae, torch.empty(shape, device="meta"), cpu)
        weights = sum(param.numel() for param in vae.parameters()) * vae.dtype.itemsize
        print(
            f"{_grid(dataclasses.asdict(point))} {wm.features['dtype']:>6} {wm.reservation_bytes / GIB:>12.2f} "
            f"{weights / GIB:>11.2f} {(wm.reservation_bytes + weights) / GIB:>9.2f}"
        )
    print(f"{len(points)} operations, one process each")


def _ints(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of integers: {text!r}")


def _sizes(text: str) -> list[int]:
    sizes = _ints(text)
    multiple = 2 * LATENT_SCALE_FACTOR
    if not sizes or any(px <= 0 or px % multiple for px in sizes):
        raise argparse.ArgumentTypeError(f"sizes must be positive multiples of {multiple} px, got {text!r}")
    return sizes


def _tiles(text: str) -> list[int]:
    """Tile fields: 0, the nodes' sentinel, or a multiple of 32 px from 128, which every one of these VAEs assembles
    at exactly that size (`diffusers_latent_tile`)."""
    tiles = _ints(text)
    if not tiles or any(tile != 0 and (tile < 128 or tile % 32) for tile in tiles):
        raise argparse.ArgumentTypeError(f"tiles must be 0 or multiples of 32 px from 128, got {text!r}")
    return list(dict.fromkeys(tiles))


def _families(text: str) -> list[Family]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names or not set(names) <= set(FAMILIES):
        raise argparse.ArgumentTypeError(f"families must be among {', '.join(FAMILIES)}, got {text!r}")
    return [family for family in FAMILIES if family in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="The JSON-lines file the lines are appended to. Required to measure.")
    parser.add_argument("--families", type=_families, default=",".join(FAMILIES), help="Default: flux,flux2,sd.")
    parser.add_argument(
        "--sizes", type=_sizes, default="1024,1536,2048,3072,4096,6144,8192", help="Tiled image sides, px."
    )
    parser.add_argument("--tiles", type=_tiles, default="512,1024,0", help="Tile fields; 0 is the node default.")
    parser.add_argument(
        "--alloc-conf",
        default="",
        help="The allocator configuration, as the server's pytorch_cuda_alloc_conf. Empty: the platform's default.",
    )
    parser.add_argument("--force", action="store_true", help="Run points predicted not to fit free device memory.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned points without touching a device.")
    parser.add_argument("--single", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single is not None:
        fresh_points.measure_in_this_process(args.single, point_type=Point, measure=_measure, logger_name=LOGGER_NAME)
    points = plan(families=args.families, sizes=args.sizes, tiles=args.tiles)
    if args.dry_run:
        _print_plan(points)
        return
    if args.out is None:
        parser.error("--out is required to measure")
    fresh_points.check_alloc_conf(parser, args.alloc_conf)
    lines = fresh_points.measure_points(
        points,
        script=Path(__file__).resolve(),
        out=args.out,
        alloc_conf=args.alloc_conf,
        force=args.force,
        logger_name=LOGGER_NAME,
        columns=_COLUMNS,
        head=_head,
    )

    print()
    _summarize(lines)
    fresh_points.print_counts(lines, args.out)


if __name__ == "__main__":
    main()

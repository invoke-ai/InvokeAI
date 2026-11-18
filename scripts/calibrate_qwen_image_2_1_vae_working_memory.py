"""Measure the working memory of Qwen-Image-2.1 VAE encodes and decodes, as shipped-data points for `vae.qwen_image_2_1`.

What a point measures: one fresh window (`working_memory/measurement.py`) with the real weights (`--vae`, the model's
`vae` directory) on the device, around one `vae.decode` or `vae.encode` of random input under
`patch_qwen_image_vae_tiling`, as the nodes run it. A window the measurement voids (an allocator retry, a spill, out of
memory) is recorded as invalid, never as a point. Its bytes are the reserved growth of that one cold window, a
`reserved_peak`, not the sibling scripts' larger of a cold and a warm demand: the shipped points were read that way, and
on HIP a demand reads a few hundred MiB above them. Each point runs in a fresh process
(`scripts/working_memory/fresh_points.py`) that first applies the server's runtime, the `MIOPEN_FIND_MODE` default and
the ROCm SDPA guard among them, with `--alloc-conf` or the platform's default allocator. A point whose process raises is
recorded as failed and the run goes on; one predicted not to fit free device memory is skipped unless `--force`, since
Windows drivers page such an allocation instead of failing it.

Runs on CUDA and ROCm, where `--out` collects the point file `scripts/working_memory/build_shipped.py --points` reads:

    python scripts/calibrate_qwen_image_2_1_vae_working_memory.py --vae /path/to/vae --out points.jsonl
"""

import argparse
import dataclasses
import datetime
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# None of these imports torch: a measuring process has to configure the allocator before torch is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.working_memory import fresh_points  # noqa: E402

if TYPE_CHECKING:
    from invokeai.backend.working_memory.measurement import FreshWindow

LOGGER_NAME = "calibrate_qwen_image_2_1_vae_working_memory"
# As `invokeai.backend.qwen_image_2_1.vae`, whose import starts CUDA in the parent, which must stay free of torch.
SPATIAL_SCALE = 16
_COLUMNS = f"{'op':7}{'dtype':9}{'size':>12}{'tile':>6}"


@dataclass(frozen=True)
class Point:
    """One encode or decode to measure, in a process of its own."""

    vae: str
    operation: str
    dtype: str
    height: int
    width: int
    tile: int | None
    fit: bool = True


def plan(
    vae: str,
    operations: Sequence[str],
    dtypes: Sequence[str],
    shapes: Sequence[tuple[int, int]],
    frames: Sequence[int],
    tiles: Sequence[int],
) -> list[Point]:
    """Per operation and dtype, the untiled `shapes`, then each of the square tiled `frames` at each of the `tiles`."""
    runs = [(*shape, None) for shape in shapes] + [(frame, frame, tile) for frame in frames for tile in tiles]
    return [
        Point(vae, operation, dtype, height, width, tile)
        for operation in operations
        for dtype in dtypes
        for height, width, tile in runs
    ]


def point_line(
    point: Point, common: Mapping[str, Any], window: "FreshWindow", facts: Mapping[str, Any], date: str
) -> dict[str, Any]:
    """The line of a measured window: a `reserved_peak` point, or invalid for the window's reasons."""
    line = fresh_points.point_line(
        point,
        common=common,
        provenance=(
            f"scripts/calibrate_qwen_image_2_1_vae_working_memory.py, {date}: {fresh_points.runtime_provenance(facts)}, "
            "real weights, one fresh process per point, max_memory_reserved minus the baseline after the weights loaded."
        ),
        windows=[window],
        raw={"runtime": facts},
        reasons=[] if window.reserved is not None else ["reserved_unreadable"],
    )
    if line["status"] != "point":
        return line
    return line | {"measured_bytes": window.reserved, "method": "reserved_peak"}


def _measure(point: Point, force: bool) -> dict[str, Any]:
    import torch
    from diffusers import AutoencoderKLQwenImage21

    from invokeai.backend.qwen_image_2_1.vae import SPEC, estimate_working_memory
    from invokeai.backend.util.qwen_image_vae import patch_qwen_image_vae_tiling
    from invokeai.backend.working_memory import platform as working_memory_platform
    from invokeai.backend.working_memory.measurement import measure

    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[point.dtype]
    estimate = estimate_working_memory(point.operation, point.height, point.width, point.tile, dtype.itemsize, device)
    facts = fresh_points.runtime_facts(
        device, working_memory_platform.platform_for(device).os, working_memory_platform.__file__
    )
    common = {
        "estimator": SPEC.id,
        "version": SPEC.version,
        "platform": dataclasses.asdict(working_memory_platform.platform_for(device)),
        "features": dict(estimate.features),
        "device": facts["device"],
    }

    vae = (
        AutoencoderKLQwenImage21.from_pretrained(point.vae, torch_dtype=dtype, local_files_only=True).to(device).eval()
    )
    free = torch.cuda.mem_get_info(device)[0]
    if estimate.reservation_bytes > free and not force:
        raw = {"predicted_bytes": estimate.reservation_bytes, "free_bytes": free, "runtime": facts}
        return fresh_points.skipped_line(point, common, raw)

    def operation() -> object:
        height, width = point.height, point.width
        if point.operation == "decode":
            latents = (1, vae.config.z_dim, 1, height // SPATIAL_SCALE, width // SPATIAL_SCALE)
            return vae.decode(torch.randn(*latents, device=device, dtype=dtype), return_dict=False)
        pixels = torch.rand(1, 4, 1, height, width, device=device, dtype=dtype) * 2 - 1
        return vae.encode(pixels).latent_dist.mode()

    with torch.inference_mode(), patch_qwen_image_vae_tiling(vae, point.tile, SPATIAL_SCALE):
        window = measure(device, operation)
    return point_line(point, common, window, facts, datetime.date.today().isoformat())


def _head(line: Mapping[str, Any]) -> str:
    point = line["point"]
    size = f"{point['height']}x{point['width']}"
    return f"{point['operation']:7}{point['dtype']:9}{size:>12}{point['tile'] or '-':>6}"


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def shapes(text: str) -> list[tuple[int, int]]:
    """`512,896x1152` as (height, width) pairs: a bare side is a square."""
    sides = [part.split("x") for part in text.split(",") if part]
    # Unpacking raises the ValueError argparse reports for anything but a side or a pair of them.
    return [(int(height), int(width)) for height, width in (side * 2 if len(side) == 1 else side for side in sides)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", help="The model's `vae` directory, in diffusers layout. Required to measure.")
    parser.add_argument("--out", type=Path, help="The JSON-lines file the points are appended to. Required to measure.")
    parser.add_argument("--operations", default="decode,encode")
    parser.add_argument("--dtypes", default="bf16,fp32")
    parser.add_argument(
        "--sizes", type=shapes, default="512,768,1024,1088,1216,1264,1280,1536", help="Untiled sides or HxW shapes, px."
    )
    parser.add_argument("--frames", type=_ints, default="2048,3072", help="Tiled frames' sides, px.")
    parser.add_argument("--tiles", type=_ints, default="512,768,1024", help="Tile sides, px.")
    parser.add_argument("--alloc-conf", default="", help="As the server's pytorch_cuda_alloc_conf. Empty: the default.")
    parser.add_argument("--force", action="store_true", help="Run points predicted not to fit free device memory.")
    parser.add_argument("--single", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single is not None:
        fresh_points.measure_in_this_process(args.single, point_type=Point, measure=_measure, logger_name=LOGGER_NAME)
    if args.vae is None or args.out is None:
        parser.error("--vae and --out are required to measure")
    fresh_points.check_alloc_conf(parser, args.alloc_conf)
    sides = {side for shape in args.sizes for side in shape} | {*args.frames, *args.tiles}
    # The decode floors to whole latents, so an off-grid side would run smaller than the point it records.
    off_grid = sorted(side for side in sides if side % SPATIAL_SCALE)
    if off_grid:
        parser.error(f"--sizes, --frames and --tiles must be multiples of {SPATIAL_SCALE}; these are not: {off_grid}")
    points = plan(args.vae, args.operations.split(","), args.dtypes.split(","), args.sizes, args.frames, args.tiles)
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
    fresh_points.print_counts(lines, args.out)


if __name__ == "__main__":
    main()

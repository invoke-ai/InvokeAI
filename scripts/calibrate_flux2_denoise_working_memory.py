"""Measure the working memory of one FLUX.2 transformer forward, as shipped-data points for `denoise.flux2`.

What a point measures
---------------------
The demand of one transformer forward, measured as a fresh window (`invokeai/backend/working_memory/measurement.py`,
whose definitions the in-server probe's windows share). With the model and its inputs resident: synchronize, release the
allocator's cached blocks, read the baseline and reset the peaks; run the forward; the demand is then the largest growth
of three counters over the baseline: the driver's figure for this process, torch's reserved peak and its allocated peak.
A forward whose allocations failed, or only succeeded after the allocator released its cache and retried (a squeezed
peak), or after which Windows keeps more than `measurement.SPILL_TOLERANCE_BYTES` of the process in shared memory, is
invalid: it is recorded as such, not as a point.

Every point runs in a fresh process (`scripts/working_memory/fresh_points.py`, which the calibration scripts share),
which first applies the server's runtime (`apply_server_runtime` in `invokeai/app/run_app.py`): the `MIOPEN_FIND_MODE`
default, the integrated-GPU check, the allocator, the AOTriton decision and the monkeypatches, the ROCm SDPA guard among
them, in the server's order. Its settings are those of a server without a config file, except the allocator
configuration: `--alloc-conf`, or without one the platform default the server uses (expandable segments for ROCm on
Windows, torch's native allocator elsewhere). This script's own process runs the integrated-GPU check once before the
first point: the measuring processes inherit the `HIP_VISIBLE_DEVICES` it sets, which their own check keeps, as the
server keeps an exported one.

The forward then runs twice, cold and warm, and the point takes the larger demand (`measure_cold_and_warm` and
`measured_bytes_and_method` in `measurement.py`): a `demand_peak` where both forwards read the driver's figure, else a
`reserved_peak` of torch's counters alone.

The transformer is `Flux2Transformer2DModel` at the geometry the model loader builds for a variant's width (heads of
128, the variant's caption width, guidance embeddings) with random bf16 weights, because memory depends on shapes, not
on weight values. It has 2 double-stream and 2 single-stream blocks: a no-grad forward frees each block's
intermediates, which `--extra-blocks 4` re-checks at 1024px per width. The inputs are what the node hands it: the
image's and the reference images' tokens on one stream, a caption, the timestep and the guidance, a sample of each per
batch entry. A point carries the features of `estimate_flux2_denoise_working_memory`, taken in the measuring process
after the runtime setup, so its `score_matrix_bytes` reflects the attention kernel this build really runs.

Each point is priced before it runs: the estimate for it plus its weights. It is skipped when that exceeds free device
memory, unless `--force`. An oversized allocation does not always fail: where the driver backs it with system RAM
(NVIDIA's sysmem fallback, AMD on Windows) it pages, which can stall the whole machine. `--dry-run` prints that price
per point as a CPU prices it: a CPU is an unmeasured platform, so that is cuDNN's native-allocator entry floored at the
legacy constants (`unmeasured_floor`), above what the point's measured platform reserves wherever the fit lies below
them.

Modes
-----
- `fit` (default): the fit grid, per `--widths` (3072, 4096 and 6144: Klein 4B, Klein 9B and [dev]). The `--sizes`
  (512, 768, 1024, 1440, 2048 px) run with a `--text-tokens` caption (512). Those among `--short-text-sizes` (1024,
  1440) also run with a `--short-text-tokens` caption (77), for the text slope. At 1024px, `--references` (1 and 3) runs
  with that many `--reference-px` (1024) reference images, and `--batches` (2) with that many samples. `--extra-blocks
  N` adds one forward with N blocks per stream at 1024px per width.
- `large`: long sequences, per `--widths` (default 3072 and 4096; [dev] fits 24 GB only below 33025 tokens): 2048px
  with one 2024px reference, 33025 tokens with a 512-token caption; 1536px with one 2024px reference, 25857 tokens; and
  1328px with three 1328px references, which the node resizes to 1024px each, 19689 tokens. The node caps neither the
  image's side nor the number of references; longer sequences price on the line extrapolated.
- `math`: SDPA forced onto the math kernel at 1024px, next to the default kernel, per width: what a materialized score
  matrix costs per element, against what the estimate charges for it.

A reference image's tokens are what `Flux2RefImageExtension` concatenates for it: resized as it resizes them (one
reference to at most 2024x2024 pixels, each of several to at most 1024x1024), encoded at 1/8 of the pixels, padded to
whole 2x2 patches. A 1024px reference is 4096 tokens, a 2024px one 16129.

The fit grid's and the large grid's points are fit points (`"fit": true`): the plain forwards, reference images and
batches included, which the estimator's line admits (`denoise_working_memory._line`). The checks' points, the
extra-blocks forward included, test analytic terms or assumptions rather than the line: they are marked `"fit": false`,
and their `in_fit` is false. A fit point is `in_fit` only where the line admits its features: on a build whose SDPA
kernel materializes the score matrix, the fit grid's forwards carry a score term on top of the line, so they measure
that term rather than the line, as a check does.

Output
------
Every measured point is appended to `--out` as one JSON object per line, the point file
`scripts/working_memory/build_shipped.py --points` reads. It carries the fields of a `MeasuredPoint`
(`invokeai/backend/working_memory/coefficients.py`):

- `estimator`, `version` and `features`, the estimate's own;
- `platform`, as `platform_for` gives it, OS included; the builder names the OS only where the data needs it;
- `measured_bytes` (the demand), `method`, `device` and `in_fit`;
- `provenance`, which names the GPU, the torch build, the OS, the allocator configuration, on ROCm `MIOPEN_FIND_MODE`,
  the AOTriton switch and the SDPA guard, the block count and the date.

The other keys are the script's, which the builder ignores: `fit`, `point` (the grid entry) and `raw` (each forward's
counter deltas, the prediction and the runtime facts). An invalid, skipped or failed forward is no point: it shows in
the table on stdout, with its reasons, and stays out of the file.

Usage
-----
RTX 4090 (CUDA): the native allocator, then `cudaMallocAsync`; the math check once.

    python scripts/calibrate_flux2_denoise_working_memory.py --out 4090-native.jsonl --extra-blocks 4
    python scripts/calibrate_flux2_denoise_working_memory.py --out 4090-async.jsonl --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_flux2_denoise_working_memory.py --out 4090-checks.jsonl --mode math
    python scripts/calibrate_flux2_denoise_working_memory.py --out 4090-native-large.jsonl --mode large
    python scripts/calibrate_flux2_denoise_working_memory.py --out 4090-async-large.jsonl --mode large \
        --alloc-conf backend:cudaMallocAsync

RX 9060 XT (ROCm on Windows): the server's default expandable segments, then the native allocator; the math check once.

    python scripts/calibrate_flux2_denoise_working_memory.py --out 9060xt-expandable.jsonl --extra-blocks 4
    python scripts/calibrate_flux2_denoise_working_memory.py --out 9060xt-native.jsonl --alloc-conf expandable_segments:False
    python scripts/calibrate_flux2_denoise_working_memory.py --out 9060xt-checks.jsonl --mode math

`--dry-run` prints the planned points without touching a device, each priced as a CPU prices it, e.g.

    python scripts/calibrate_flux2_denoise_working_memory.py --dry-run --mode large
"""

import argparse
import dataclasses
import datetime
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# None of these imports torch: a measuring process has to configure the allocator before torch is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR  # noqa: E402
from scripts.working_memory import fresh_points  # noqa: E402

if TYPE_CHECKING:
    import torch

    from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate

GIB = 2**30
MIB = 2**20

PATCH_SIZE = 2
"""FLUX.2 packs 2x2 latent patches into one token."""
LATENT_CHANNELS = 128
"""A packed token's channels: 32 latent channels times the 2x2 patch."""
HEAD_DIM = 128
CONTEXT_DIMS = {3072: 7680, 4096: 12288, 6144: 15360}
"""The caption width per transformer width: 3x the text encoder's hidden size (Qwen3-4B, Qwen3-8B, Mistral Small 3.1),
as `model_manager/configs/flux2_variant.py` declares them."""
CHECK_PX = 1024
"""The size the reference, batch and extra-blocks points and the checks run at."""
REFERENCE_PX = 1024
"""The fit grid's reference images' side, `--reference-px`'s default."""
LARGE_GRID = ((2048, 1, 2024), (1536, 1, 2024), (1328, 3, 1328))
"""The large grid's forwards: image side, reference count and reference side."""
LARGE_WIDTHS = (3072, 4096)
DTYPE = "bfloat16"
"""The node runs the transformer in bf16 only."""
TIMESTEP = 0.5
GUIDANCE = 4.0

LOGGER_NAME = "calibrate_flux2_denoise_working_memory"

Kind = Literal["fit", "blocks", "baseline", "math"]
Mode = Literal["fit", "large", "math"]


@dataclass(frozen=True)
class Point:
    """One transformer forward to measure, in a process of its own."""

    kind: Kind
    """`fit` for a point of the fit grid; otherwise the check it serves: another block count (`blocks`), the
    default-kernel forward a check compares with (`baseline`) or the forced math kernel (`math`)."""
    hidden_size: int
    px: int
    text_tokens: int
    references: int = 0
    """How many reference images of `reference_px` square the forward attends to."""
    reference_px: int = REFERENCE_PX
    batch: int = 1
    blocks: int = 2
    """Double-stream and single-stream blocks, each."""
    force_math: bool = False

    @property
    def fit(self) -> bool:
        return self.kind == "fit"


def image_tokens(px: int) -> int:
    """A square image's tokens: its latent at 1/8 of the pixels, in 2x2 patches."""
    side = px // LATENT_SCALE_FACTOR // PATCH_SIZE
    return side * side


def reference_tokens(point: Point) -> int:
    """What `Flux2RefImageExtension` concatenates onto the image stream for the point's references: each resized as it
    resizes them, its latent at 1/8 of the pixels padded to whole 2x2 patches."""
    if not point.references:
        return 0
    from PIL import Image

    from invokeai.backend.flux2.ref_image_extension import (
        MAX_PIXELS_MULTI_REF,
        MAX_PIXELS_SINGLE_REF,
        resize_image_to_max_pixels,
    )

    cap = MAX_PIXELS_SINGLE_REF if point.references == 1 else MAX_PIXELS_MULTI_REF
    width, height = resize_image_to_max_pixels(Image.new("L", (point.reference_px, point.reference_px)), cap).size
    rows, columns = (-(-(side // LATENT_SCALE_FACTOR) // PATCH_SIZE) for side in (height, width))
    return point.references * rows * columns


def plan(
    mode: Mode,
    *,
    widths: Sequence[int],
    sizes: Sequence[int],
    text_tokens: int,
    short_text_tokens: int,
    short_text_sizes: Sequence[int],
    references: Sequence[int],
    batches: Sequence[int],
    blocks: int,
    extra_blocks: int,
    reference_px: int = REFERENCE_PX,
) -> list[Point]:
    """The forwards a run measures, in order: per width, the fit grid, the large grid or the math check's pair of
    forwards."""
    points: list[Point] = []
    for width in widths:
        if mode == "large":
            points += [
                Point("fit", width, px, text_tokens, references=n, reference_px=ref_px, blocks=blocks)
                for px, n, ref_px in LARGE_GRID
            ]
            continue
        if mode == "math":
            points.append(Point("baseline", width, CHECK_PX, text_tokens, blocks=blocks))
            points.append(Point("math", width, CHECK_PX, text_tokens, blocks=blocks, force_math=True))
            continue
        points += [Point("fit", width, px, text_tokens, blocks=blocks) for px in sizes]
        points += [Point("fit", width, px, short_text_tokens, blocks=blocks) for px in sizes if px in short_text_sizes]
        points += [
            Point("fit", width, CHECK_PX, text_tokens, references=n, reference_px=reference_px, blocks=blocks)
            for n in references
            if n
        ]
        points += [Point("fit", width, CHECK_PX, text_tokens, batch=n, blocks=blocks) for n in batches if n != 1]
        if extra_blocks:
            points.append(Point("blocks", width, CHECK_PX, text_tokens, blocks=extra_blocks))
    return points


def provenance(point: Point, facts: Mapping[str, Any], date: str) -> str:
    """A measured point's `provenance`: where and how it was measured."""
    regime = ", SDPA forced onto the math kernel" if point.force_math else ""
    blocks = f"{point.blocks} double-stream and {point.blocks} single-stream block{'' if point.blocks == 1 else 's'}"
    return (
        f"scripts/calibrate_flux2_denoise_working_memory.py ({date}): {fresh_points.runtime_provenance(facts)}. "
        "Random-init FLUX.2 "
        f"transformer of width {point.hidden_size} with the model loader's geometry, {blocks}{regime}; memory does not "
        "depend on weight values. A fresh process with the server's runtime, the larger demand of a cold and a warm "
        "forward; no failed allocation, allocator retry or spill."
    )


# Everything below imports torch. In a measuring process it runs only after `fresh_points.apply_server_runtime`.


def _estimate(point: Point, device: "torch.device") -> "WorkingMemoryEstimate":
    """The estimate for the point's forward on `device`, priced in the kernel regime the caller holds."""
    import torch

    from invokeai.backend.flux2.denoise_working_memory import estimate_flux2_denoise_working_memory

    return estimate_flux2_denoise_working_memory(
        # The point runs a randomly initialized dense model, which dequantizes and casts nothing per forward.
        torch.nn.Module(),
        hidden_size=point.hidden_size,
        image_seq_len=image_tokens(point.px),
        ref_image_seq_len=reference_tokens(point),
        text_seq_len=point.text_tokens,
        batch_size=point.batch,
        lora_count=0,
        # Measured whole on the device: nothing is streamed per forward.
        weights_resident=True,
        device=device,
        dtype=getattr(torch, DTYPE),
    )


def _transformer(hidden_size: int, blocks: int) -> "torch.nn.Module":
    """The FLUX.2 transformer of a width, as the model loader builds it, with `blocks` blocks per stream, on the meta
    device."""
    import torch
    from diffusers import Flux2Transformer2DModel

    with torch.device("meta"):
        return Flux2Transformer2DModel(
            in_channels=LATENT_CHANNELS,
            out_channels=LATENT_CHANNELS,
            num_layers=blocks,
            num_single_layers=blocks,
            attention_head_dim=HEAD_DIM,
            num_attention_heads=hidden_size // HEAD_DIM,
            joint_attention_dim=CONTEXT_DIMS[hidden_size],
            patch_size=1,
        )


def _weight_bytes(transformer: "torch.nn.Module") -> int:
    """A meta-device transformer's weights in bf16."""
    return sum(param.numel() for param in transformer.parameters()) * 2


def _inputs(point: Point, device: "torch.device") -> dict[str, Any]:
    """What the node hands the transformer per forward: the image's and the references' tokens on one stream, their
    position ids, a caption, the timestep and the guidance, `point.batch` samples of each."""
    import torch

    from invokeai.backend.flux2.sampling_utils import generate_img_ids_flux2

    dtype = getattr(torch, DTYPE)
    latent = point.px // LATENT_SCALE_FACTOR
    references = reference_tokens(point)
    img_ids = generate_img_ids_flux2(h=latent, w=latent, batch_size=point.batch, device=device)
    if references:
        ref_ids = torch.zeros(point.batch, references, 4, device=device, dtype=torch.long)
        img_ids = torch.cat([img_ids, ref_ids], dim=1)
    txt_ids = torch.zeros(point.batch, point.text_tokens, 4, device=device, dtype=torch.long)
    txt_ids[..., 3] = torch.arange(point.text_tokens, device=device, dtype=torch.long)
    tokens = image_tokens(point.px) + references
    return {
        "hidden_states": torch.randn(point.batch, tokens, LATENT_CHANNELS, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(
            point.batch, point.text_tokens, CONTEXT_DIMS[point.hidden_size], device=device, dtype=dtype
        ),
        "timestep": torch.full((point.batch,), TIMESTEP, device=device, dtype=dtype),
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "guidance": torch.full((point.batch,), GUIDANCE, device=device, dtype=dtype),
        "return_dict": False,
    }


def _measure(point: Point, force: bool) -> dict[str, Any]:
    """The output line of `point`, measured in this process after `fresh_points.apply_server_runtime`."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from invokeai.backend.flux2 import denoise_working_memory
    from invokeai.backend.util.devices import TorchDevice
    from invokeai.backend.working_memory.measurement import measure_cold_and_warm
    from invokeai.backend.working_memory.platform import platform_for

    # The device the node itself takes.
    device = TorchDevice.choose_torch_device()
    if device.type != "cuda":
        raise SystemExit(f"Needs a CUDA or ROCm device; this process would run on {device}.")

    def kernel():
        return sdpa_kernel(SDPBackend.MATH) if point.force_math else nullcontext()

    with torch.inference_mode():
        # In the forward's kernel regime, so the score matrix is priced for the kernel that runs.
        with kernel():
            estimate = _estimate(point, device)
        platform = platform_for(device)
        facts = fresh_points.runtime_facts(device, platform.os, denoise_working_memory.__file__)
        common = {
            "estimator": estimate.estimator,
            "version": estimate.version,
            "platform": dataclasses.asdict(platform),
            "features": dict(estimate.features),
            "device": facts["device"],
        }

        transformer = _transformer(point.hidden_size, point.blocks)
        weights = _weight_bytes(transformer)
        free, _ = TorchDevice.cuda_mem_get_info(device)
        raw = {
            "reservation_bytes": estimate.reservation_bytes,
            "weight_bytes": weights,
            "predicted_bytes": estimate.reservation_bytes + weights,
            "free_bytes": free,
            "runtime": facts,
        }
        if raw["predicted_bytes"] > free and not force:
            return fresh_points.skipped_line(point, common, raw)

        # Allocated directly in bf16 on the device and randomly initialized: [dev]'s blocks would otherwise pass
        # through the host in float32 first.
        transformer = transformer.to(getattr(torch, DTYPE)).to_empty(device=device)
        for param in transformer.parameters():
            torch.nn.init.normal_(param, std=0.02)
        transformer.eval()
        inputs = _inputs(point, device)

        def forward() -> object:
            with kernel():
                return transformer(**inputs)

        windows = measure_cold_and_warm(device, forward)
    date = datetime.date.today().isoformat()
    return fresh_points.point_line(
        point, common=common, provenance=provenance(point, facts, date), windows=windows, raw=raw
    )


_COLUMNS = (
    f"{'kind':>8} {'width':>5} {'blocks':>6} {'px':>5} {'text':>5} {'refs':>4} {'ref px':>6} {'batch':>5} {'tokens':>7}"
)


def _grid(point: Mapping[str, Any]) -> str:
    return (
        f"{point['kind']:>8} {point['hidden_size']:>5} {point['blocks']:>6} {point['px']:>5} {point['text_tokens']:>5} "
        f"{point['references']:>4} {point['reference_px'] if point['references'] else '':>6} {point['batch']:>5}"
    )


def _attended(line: Mapping[str, Any]) -> int:
    """The tokens of every sample of a measured line: its size on the line."""
    return line["features"]["tokens"] * line["features"]["batch_size"]


def _head(line: Mapping[str, Any]) -> str:
    """A line's grid columns in the table."""
    return f"{_grid(line['point'])} {_attended(line) if 'features' in line else '':>7}"


def _px(line: Mapping[str, Any]) -> int:
    return line["point"]["px"]


def _summarize_fit(lines: Sequence[Mapping[str, Any]], text_tokens: int) -> None:
    measured = [line for line in lines if line["status"] == "point"]
    grid = [line for line in measured if line["fit"]]
    # The line's slopes from the points that fit it; the extra-blocks forward compares with any grid point.
    fit = [line for line in grid if line["in_fit"]]
    for width in dict.fromkeys(line["point"]["hidden_size"] for line in fit):
        of_width = [line for line in fit if line["point"]["hidden_size"] == width]
        plain = [line for line in of_width if line["point"]["references"] == 0 and line["point"]["batch"] == 1]
        main = sorted((line for line in plain if line["point"]["text_tokens"] == text_tokens), key=_attended)
        for line in main:
            print(f"width {width}, {_px(line)}px: {line['measured_bytes'] / _attended(line) / MIB:.4f} MiB per token")
        if len(main) < 2:
            continue
        lo, hi = main[0], main[-1]
        slope = (hi["measured_bytes"] - lo["measured_bytes"]) / (_attended(hi) - _attended(lo))
        base = lo["measured_bytes"] - slope * _attended(lo)
        print(f"width {width}, {_px(lo)}-{_px(hi)}px: {slope / MIB:.4f} MiB per token, intercept {base / GIB:.3f} GiB")
        # Reference images and batches sit on the same line, which `build_shipped.py` fits on all of them.
        for line in of_width:
            if line in main:
                continue
            on_line = base + slope * _attended(line)
            point = line["point"]
            print(
                f"width {width}, {_px(line)}px, {point['text_tokens']} text, {point['references']} references, batch "
                f"{point['batch']}: {line['measured_bytes'] / GIB:.3f} GiB at {_attended(line)} tokens, "
                f"{line['measured_bytes'] / on_line:.3f}x the {_px(lo)}-{_px(hi)}px line"
            )
    for line in (line for line in measured if line["point"]["kind"] == "blocks"):
        same = ("hidden_size", "px", "text_tokens", "references", "batch")
        base_line = next((ref for ref in grid if all(ref["point"][key] == line["point"][key] for key in same)), None)
        if base_line is not None:
            extra = line["measured_bytes"] - base_line["measured_bytes"]
            print(
                f"{line['point']['blocks']} against {base_line['point']['blocks']} blocks per stream, width "
                f"{line['point']['hidden_size']}, {_px(line)}px: {extra / MIB:+.0f} MiB "
                f"({line['measured_bytes'] / base_line['measured_bytes']:.3f}x)"
            )


def _summarize_large(lines: Sequence[Mapping[str, Any]]) -> None:
    for line in (line for line in lines if line["status"] == "point"):
        point, reserved = line["point"], line["raw"]["reservation_bytes"]
        print(
            f"width {point['hidden_size']}, {_px(line)}px + {point['references']}x{point['reference_px']}px: "
            f"{line['measured_bytes'] / GIB:.3f} GiB at {_attended(line)} tokens, "
            f"{line['measured_bytes'] / _attended(line) / MIB:.4f} MiB per token; the shipped estimate reserves "
            f"{reserved / GIB:.3f} GiB ({reserved / line['measured_bytes']:.3f}x)"
        )


def _summarize_checks(lines: Sequence[Mapping[str, Any]]) -> None:
    from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    measured = [line for line in lines if line["status"] == "point"]
    for line in (line for line in measured if line["point"]["kind"] == "math"):
        point = line["point"]
        baseline = next(
            (
                ref
                for ref in measured
                if ref["point"]["kind"] == "baseline" and ref["point"]["hidden_size"] == point["hidden_size"]
            ),
            None,
        )
        if baseline is None:
            print(f"math kernel, width {point['hidden_size']}: no default-kernel forward to compare with")
            continue
        extra = line["measured_bytes"] - baseline["measured_bytes"]
        charged = line["raw"]["reservation_bytes"] - baseline["raw"]["reservation_bytes"]
        tokens = line["features"]["tokens"]
        elements = point["hidden_size"] // HEAD_DIM * line["features"]["batch_size"] * tokens * tokens
        print(
            f"math kernel, width {point['hidden_size']}, {point['px']}px: {extra / GIB:.3f} GiB over the default "
            f"kernel, {extra / elements:.2f} bytes per score element; the estimate adds {charged / GIB:.3f} GiB "
            f"({SDPA_MATH_BYTES_PER_SCORE_ELEMENT} bytes per element, one chunk where the ROCm guard chunks)"
        )


def _print_plan(points: Sequence[Point]) -> None:
    """The planned points, each priced as a CPU prices it, plus the weights: the estimate with cuDNN's native-allocator
    values floored at the legacy constants, as every unmeasured platform's, the forced math kernel's score matrix
    included."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print(f"{_COLUMNS} {'estimate GiB':>12} {'weights GiB':>11} {'total GiB':>9}")
    cpu = torch.device("cpu")
    weights_of: dict[tuple[int, int], int] = {}
    for point in points:
        with sdpa_kernel(SDPBackend.MATH) if point.force_math else nullcontext():
            estimate = _estimate(point, cpu)
        geometry = (point.hidden_size, point.blocks)
        if geometry not in weights_of:
            weights_of[geometry] = _weight_bytes(_transformer(*geometry))
        weights = weights_of[geometry]
        tokens = estimate.features["tokens"] * point.batch
        print(
            f"{_grid(dataclasses.asdict(point))} {tokens:>7} {estimate.reservation_bytes / GIB:>12.2f} "
            f"{weights / GIB:>11.2f} {(estimate.reservation_bytes + weights) / GIB:>9.2f}"
            f"{'  forced math' if point.force_math else ''}"
        )
    print(f"{len(points)} forwards, one process each")


def _ints(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of integers: {text!r}")


def _sizes(text: str) -> list[int]:
    sizes = _ints(text)
    multiple = LATENT_SCALE_FACTOR * PATCH_SIZE
    if any(px <= 0 or px % multiple for px in sizes):
        raise argparse.ArgumentTypeError(f"sizes must be positive multiples of {multiple} px, got {text!r}")
    return sizes


def _widths(text: str) -> list[int]:
    widths = _ints(text)
    if not widths or not set(widths) <= set(CONTEXT_DIMS):
        raise argparse.ArgumentTypeError(f"widths must be among {', '.join(map(str, CONTEXT_DIMS))}, got {text!r}")
    return widths


def _side(text: str) -> int:
    try:
        side = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a side length in px: {text!r}")
    if side < 2 * LATENT_SCALE_FACTOR:
        raise argparse.ArgumentTypeError(f"a reference must be at least {2 * LATENT_SCALE_FACTOR} px, got {side}")
    return side


def _counts(text: str) -> list[int]:
    counts = _ints(text)
    if any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError(f"counts must not be negative, got {text!r}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fit", "large", "math"), default="fit", help="Default: fit.")
    parser.add_argument("--out", type=Path, help="The JSON-lines file the lines are appended to. Required to measure.")
    parser.add_argument(
        "--widths", type=_widths, help="Transformer widths. Default: all, and 3072 and 4096 in large mode."
    )
    parser.add_argument("--sizes", type=_sizes, default="512,768,1024,1440,2048", help="The fit grid's sizes, px.")
    parser.add_argument("--text-tokens", type=int, default=512, help="The caption at every size. Default 512.")
    parser.add_argument("--short-text-tokens", type=int, default=77, help="The text-slope caption. Default 77.")
    parser.add_argument(
        "--short-text-sizes", type=_sizes, default="1024,1440", help="Fit sizes that also run the text-slope caption."
    )
    parser.add_argument(
        "--references", type=_counts, default="1,3", help="Reference images at 1024px, each run. Default 1,3."
    )
    parser.add_argument(
        "--reference-px", type=_side, default=REFERENCE_PX, help="The fit grid's reference side, px. Default 1024."
    )
    parser.add_argument("--batches", type=_counts, default="2", help="Batch sizes at 1024px beyond 1. Default 2.")
    parser.add_argument("--blocks", type=int, default=2, help="Blocks per stream. Default 2.")
    parser.add_argument("--extra-blocks", type=int, default=0, help="Fit mode: this block count at 1024px as well.")
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
    widths = args.widths or list(LARGE_WIDTHS if args.mode == "large" else CONTEXT_DIMS)
    points = plan(
        args.mode,
        widths=widths,
        sizes=args.sizes,
        text_tokens=args.text_tokens,
        short_text_tokens=args.short_text_tokens,
        short_text_sizes=args.short_text_sizes,
        references=args.references,
        batches=args.batches,
        blocks=args.blocks,
        extra_blocks=args.extra_blocks,
        reference_px=args.reference_px,
    )
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
    if args.mode == "fit":
        _summarize_fit(lines, args.text_tokens)
    elif args.mode == "large":
        _summarize_large(lines)
    else:
        _summarize_checks(lines)
    fresh_points.print_counts(lines, args.out)


if __name__ == "__main__":
    main()

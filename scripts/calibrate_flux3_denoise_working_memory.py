"""Measure the working memory of one FLUX.3 transformer forward, as shipped-data points for `denoise.flux3`.

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
server keeps an exported one. A measuring process is `sys.executable` running this file, so it imports `invokeai` from
the interpreter that started the run.

The forward then runs twice, cold and warm, and the point takes the larger demand (`measure_cold_and_warm` and
`measured_bytes_and_method` in `measurement.py`): a `demand_peak` where both forwards read the driver's figure, else a
`reserved_peak` of torch's counters alone.

The transformer is diffusers' `Flux3Transformer2DModel` at the geometry the model loader reads off the released
checkpoints (`flux3/conversion.py`: width 3072 in 24 heads of 128, a 20480-wide caption, feedforward ratio 3), with
random weights in the measured dtype and the QK norms in float32, as the loader keeps them, because memory depends on
shapes, not on weight values. It has 2 per-mode blocks per stream (`Flux3ModeTransformerBlock`, one stack each for the
text, the image and the references) and 2 joint blocks (`Flux3SingleTransformerBlock`): a no-grad forward frees each
block's intermediates, which `--extra-blocks 4` re-checks at 1024px. It is Klein, the flow map with its target-timestep
embedder, unless a point says Klein Base. The inputs are what the node hands it: the image's tokens, the references'
tokens on a stream of their own with their position ids, a caption, the timestep and, for Klein, the timestep the step
jumps to, a sample of each per batch entry. A point carries the features of `estimate_flux3_denoise_working_memory`,
taken in the measuring process after the runtime setup from the very transformer it measures, so its
`score_matrix_bytes` reflects the attention kernel this build really runs.

Each point is priced before it runs: the estimate for it plus its weights. It is skipped when that exceeds free device
memory, unless `--force`. An oversized allocation does not always fail: where the driver backs it with system RAM
(NVIDIA's sysmem fallback, AMD on Windows) it pages, which can stall the whole machine. `--dry-run` prints that price
per point as a CPU prices it: a CPU is an unmeasured platform, so that is cuDNN's native-allocator entry floored at the
legacy constants (`unmeasured_floor`).

Modes
-----
- `fit` (default): the fit grid, in each of `--dtypes` (bfloat16, float16 and float32, the dtypes
  `TorchDevice.choose_bfloat16_safe_dtype` runs the transformer in). The `--sizes` (512, 768, 1024, 1440, 2048 px) run
  with a `--text-tokens` caption (560). Those among `--short-text-sizes` (1024, 1440) also run with a
  `--short-text-tokens` caption (80), for the text slope. At 1024px, `--references` (1, 3 and 10; at most 3 in float32,
  `FLOAT32_MAX_REFERENCES`) runs with that many `--reference-px` (1024) reference images, `--batches` (2) with that many
  samples, and Klein Base beside Klein (unless `--no-base`). `--extra-blocks N` adds one forward with N blocks per
  stream at 1024px, in the first dtype.
- `large`: long sequences, in each of `--dtypes` (default bfloat16): 2048px with one 2048px reference, 33328 tokens with
  a 560-token caption; 1536px with one 2048px reference, 26160; 2048px with three 1024px references, 29232; and 2048px
  with ten, 57904. The node caps neither the image's side nor the references' (one of up to 4 MP, or ten of up to 1 MP
  each); longer sequences price on the line extrapolated.
- `math`: SDPA forced onto the math kernel at 1024px, next to the default kernel, in each of `--dtypes` (default
  bfloat16): what a materialized score matrix costs per element, against what the estimate charges for it.
- `real`: the released model itself, a `--checkpoint` bf16 safetensors single file in BFL's layout (Klein or Klein
  Base, read off its keys), loaded through the model loader's conversion (`flux3_transformer_from_native`) straight onto
  the device and fully resident: 1024 and 2048 px with a `--text-tokens` caption, and 1024 px with one and with three
  `--reference-px` references. Each forward is set against the shipped 2-block point of the same features on this
  platform and card, and against the shipped estimate, both as the calibration script prices it (fully resident) and as
  the node prices it (the streamed largest layer included). Its weights take ~10.3 GiB beside the forward. It refuses
  the options it would ignore, and any other checkpoint (GGUF, another dtype).

  Its points are checks (`"fit": false`), and shipped they stay out of the fit: `build_shipped.py --points` takes them
  as out-of-fit points, which bound their estimate from below only. The data test holds every shipped estimate to at
  least each such point, and the builder refuses (exit 1) any refit that would price one below what it demanded. Such a
  point never raises a line: only the fit points decide it.

Captions are multiples of 80 tokens, as `flux3/text_encoding.py` pads them, and the transformer attends to the padding.
80 is a short prompt's window; 560 the smallest multiple of 80 at or above the 512-token caption the FLUX.2 lines are
measured with, a long prompt's. A reference image's tokens are what `flux3/ref_images.py` makes of it: resized as
`fit_reference_image` resizes it (one reference to at most 4 MP, each of several to at most 1 MP), center-cropped to
whole 16 px cells, one token per cell. A 1024px reference is 4096 tokens, a 2048px single one 16384.

The fit grid's and the large grid's points are fit points (`"fit": true`): the plain forwards, reference images, batches
and Klein Base included, which the estimator's line admits (`denoise_working_memory._line`). The checks' points, the
extra-blocks and the real model's forwards included, test analytic terms or assumptions rather than the line: they are
marked `"fit": false`, and their `in_fit` is false. A fit point is `in_fit` only where the line admits its features: on
a build whose SDPA kernel materializes the score matrix, the fit grid's forwards carry a score term on top of the line,
so they measure that term rather than the line, as a check does.

Output
------
Every measured point is appended to `--out` as one JSON object per line, the point file
`scripts/working_memory/build_shipped.py --points` reads. It carries the fields of a `MeasuredPoint`
(`invokeai/backend/working_memory/coefficients.py`):

- `estimator`, `version` and `features`, the estimate's own;
- `platform`, as `platform_for` gives it, OS included; the builder names the OS only where the data needs it;
- `measured_bytes` (the demand), `method`, `device` and `in_fit`;
- `provenance`, which names the GPU, the torch build, the OS, the allocator configuration, on ROCm `MIOPEN_FIND_MODE`,
  the AOTriton switch and the SDPA guard, the variant, the block count and the date.

The other keys are the script's, which the builder ignores: `fit`, `point` (the grid entry) and `raw` (each forward's
counter deltas, the prediction and the runtime facts). An invalid, skipped or failed forward is no point: it shows in
the table on stdout, with its reasons, and stays out of the file.

Usage
-----
RTX 4090 (CUDA): the native allocator, then `cudaMallocAsync`; the math check once.

    python scripts/calibrate_flux3_denoise_working_memory.py --out 4090-native.jsonl --extra-blocks 4
    python scripts/calibrate_flux3_denoise_working_memory.py --out 4090-native-large.jsonl --mode large
    python scripts/calibrate_flux3_denoise_working_memory.py --out 4090-async.jsonl --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_flux3_denoise_working_memory.py --out 4090-async-large.jsonl --mode large \
        --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_flux3_denoise_working_memory.py --out 4090-checks.jsonl --mode math
    python scripts/calibrate_flux3_denoise_working_memory.py --out 4090-real.jsonl --mode real \
        --checkpoint E:/ModelStuffSSD/flux-3-image-klein-comfy/diffusion_models/flux3_image_klein_sd_bf16.safetensors

RX 9060 XT (ROCm on Windows): the server's default expandable segments, then the native allocator; the math check once.

    python scripts/calibrate_flux3_denoise_working_memory.py --out 9060xt-expandable.jsonl --extra-blocks 4
    python scripts/calibrate_flux3_denoise_working_memory.py --out 9060xt-expandable-large.jsonl --mode large
    python scripts/calibrate_flux3_denoise_working_memory.py --out 9060xt-native.jsonl \
        --alloc-conf expandable_segments:False
    python scripts/calibrate_flux3_denoise_working_memory.py --out 9060xt-native-large.jsonl --mode large \
        --alloc-conf expandable_segments:False
    python scripts/calibrate_flux3_denoise_working_memory.py --out 9060xt-checks.jsonl --mode math
    python scripts/calibrate_flux3_denoise_working_memory.py --out 9060xt-real.jsonl --mode real \
        --checkpoint E:/ModelStuffSSD/flux-3-image-klein-comfy/diffusion_models/flux3_image_klein_sd_bf16.safetensors

Radeon Pro W7900 (ROCm on Linux): the server's default native allocator; the math check once.

    python scripts/calibrate_flux3_denoise_working_memory.py --out w7900-native.jsonl --extra-blocks 4
    python scripts/calibrate_flux3_denoise_working_memory.py --out w7900-native-large.jsonl --mode large
    python scripts/calibrate_flux3_denoise_working_memory.py --out w7900-checks.jsonl --mode math
    python scripts/calibrate_flux3_denoise_working_memory.py --out w7900-real.jsonl --mode real \
        --checkpoint flux3_image_klein_sd_bf16.safetensors

`--dry-run` prints the planned points without touching a device, each priced as a CPU prices it, e.g.

    python scripts/calibrate_flux3_denoise_working_memory.py --dry-run --mode large
"""

import argparse
import dataclasses
import datetime
import json
import struct
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# None of these imports torch: a measuring process has to configure the allocator before torch is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.working_memory import fresh_points  # noqa: E402

if TYPE_CHECKING:
    import torch

    from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate

GIB = 2**30
MIB = 2**20

CELL_PX = 16
"""One token per 16x16 pixel cell: the VAE's 8x downsampling times the 2x2 packing."""
TEXT_PAD_MULTIPLE = 80
"""`flux3/text_encoding.py` pads every caption to a multiple of this; kept literal so no torch import is needed."""
MAX_TEXT_TOKENS = 8192
MAX_REFERENCES = 10
GEOMETRY: Mapping[str, Any] = {
    "in_channels": 128,
    "cond_in_channels": 128,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 20480,
    "timestep_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
}
"""What `flux3/conversion.py`'s `flux3_transformer_config` reads off the released Klein and Klein Base checkpoints,
less the block counts and the flow map, which a point sets."""
CHECK_PX = 1024
"""The size the reference, batch, Klein Base and extra-blocks points and the checks run at."""
REFERENCE_PX = 1024
"""The fit grid's reference images' side, `--reference-px`'s default."""
LARGE_GRID = ((2048, 1, 2048), (1536, 1, 2048), (2048, 3, 1024), (2048, 10, 1024))
"""The large grid's forwards: image side, reference count and reference side."""
DTYPES = ("bfloat16", "float16", "float32")
"""The dtypes `TorchDevice.choose_bfloat16_safe_dtype` runs the transformer in."""
FLOAT32_MAX_REFERENCES = 3
"""The most references a float32 fit-grid forward attends to. The RTX 4090's float32 lines price ten (45616 tokens) at
19.4 GiB under `cudaMallocAsync` and 23.4 under the native allocator, which with 4.7 GiB of float32 weights at 2 blocks
is more than its 24 GB hold, so a run there skips them. At most three (16944 tokens) keep the float32 grid the same on
every card."""
REAL_GRID = ((1024, 0), (2048, 0), (1024, 1), (1024, 3))
"""The real model's forwards: image side and reference count."""
REAL_DTYPE = "bfloat16"
"""The dtype the real model is measured in: the released single file's, and the node's on every measured card."""
FLOW_MAP_KEY = "ref_time_in.in_layer.weight"
"""`flux3/conversion.py`'s `FLUX3_NATIVE_FLOW_MAP_KEY`, kept literal so the driving process imports no torch: only the
step-distilled checkpoint has it."""
TIMESTEP = 0.5
TARGET_TIMESTEP = 0.4

LOGGER_NAME = "calibrate_flux3_denoise_working_memory"

Kind = Literal["fit", "blocks", "baseline", "math", "real"]
Mode = Literal["fit", "large", "math", "real"]


@dataclass(frozen=True)
class Point:
    """One transformer forward to measure, in a process of its own."""

    kind: Kind
    """`fit` for a point of the fit grid; otherwise the check it serves: another block count (`blocks`), the
    default-kernel forward a check compares with (`baseline`), the forced math kernel (`math`) or the released model
    (`real`)."""
    dtype: str
    """A torch dtype name, one of `DTYPES`."""
    px: int
    text_tokens: int
    references: int = 0
    """How many reference images of `reference_px` square the forward attends to."""
    reference_px: int = REFERENCE_PX
    batch: int = 1
    blocks: int = 2
    """Per-mode blocks per stream, and joint blocks; a real point's checkpoint sets its own."""
    flow_map: bool = True
    """Klein, the step-distilled flow map; False for Klein Base."""
    force_math: bool = False
    checkpoint: str = ""
    """A `real` point's single file; empty for a random-init transformer of `blocks` blocks."""

    @property
    def fit(self) -> bool:
        return self.kind == "fit"


def image_tokens(px: int) -> int:
    """A square image's tokens: one per 16x16 cell."""
    side = px // CELL_PX
    return side * side


def reference_grids(point: Point) -> list[tuple[int, int]]:
    """Each reference's token grid, rows by columns, as `flux3/ref_images.py` sizes it: resized as
    `fit_reference_image` resizes one of `point.references`, center-cropped to whole 16 px cells."""
    if not point.references:
        return []
    from PIL import Image

    from invokeai.backend.flux3.ref_images import fit_reference_image

    fitted = fit_reference_image(Image.new("RGB", (point.reference_px, point.reference_px)), point.references)
    return [(fitted.height // CELL_PX, fitted.width // CELL_PX)] * point.references


def reference_tokens(point: Point) -> int:
    return sum(rows * columns for rows, columns in reference_grids(point))


def checkpoint_header(path: str) -> dict[str, Any]:
    """A safetensors file's header: per tensor its dtype, shape and offsets, read without torch."""
    with open(path, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(length))
    header.pop("__metadata__", None)
    return header


def checkpoint_is_flow_map(path: str) -> bool:
    """Whether a single file in BFL's layout is Klein, the flow map, rather than Klein Base, under any key prefix."""
    return any(key.endswith(FLOW_MAP_KEY) for key in checkpoint_header(path))


def checkpoint_problem(path: str) -> str | None:
    """Why real mode cannot measure `path`, or None: it measures a bf16 safetensors single file in BFL's layout."""
    if Path(path).suffix != ".safetensors":
        return f"{path} is no safetensors single file (a GGUF build is not measured in real mode)"
    dtypes = sorted({entry["dtype"] for entry in checkpoint_header(path).values()} - {"BF16"})
    return f"{path} holds {', '.join(dtypes)} tensors; real mode measures a bf16 single file" if dtypes else None


REAL_MODE_IGNORES = (
    "dtypes",
    "sizes",
    "short_text_tokens",
    "short_text_sizes",
    "references",
    "batches",
    "no_base",
    "blocks",
    "extra_blocks",
)
"""The options real mode has no use for: it measures `REAL_GRID` in `REAL_DTYPE` on the checkpoint's own blocks."""


def plan(
    mode: Mode,
    *,
    dtypes: Sequence[str],
    sizes: Sequence[int],
    text_tokens: int,
    short_text_tokens: int,
    short_text_sizes: Sequence[int],
    references: Sequence[int],
    batches: Sequence[int],
    blocks: int,
    extra_blocks: int,
    base: bool = True,
    reference_px: int = REFERENCE_PX,
    checkpoint: str = "",
) -> list[Point]:
    """The forwards a run measures, in order: per dtype, the fit grid, the large grid or the math check's pair of
    forwards; in fit mode then the extra-blocks forward, in the first dtype; in real mode the real model's forwards of
    `checkpoint`, in `REAL_DTYPE`."""
    if mode == "real":
        flow_map = checkpoint_is_flow_map(checkpoint)
        return [
            Point(
                "real",
                REAL_DTYPE,
                px,
                text_tokens,
                references=n,
                reference_px=reference_px,
                flow_map=flow_map,
                checkpoint=checkpoint,
            )
            for px, n in REAL_GRID
        ]
    points: list[Point] = []
    for dtype in dtypes:
        if mode == "large":
            points += [
                Point("fit", dtype, px, text_tokens, references=n, reference_px=ref_px, blocks=blocks)
                for px, n, ref_px in LARGE_GRID
            ]
            continue
        if mode == "math":
            points.append(Point("baseline", dtype, CHECK_PX, text_tokens, blocks=blocks))
            points.append(Point("math", dtype, CHECK_PX, text_tokens, blocks=blocks, force_math=True))
            continue
        points += [Point("fit", dtype, px, text_tokens, blocks=blocks) for px in sizes]
        points += [Point("fit", dtype, px, short_text_tokens, blocks=blocks) for px in sizes if px in short_text_sizes]
        points += [
            Point("fit", dtype, CHECK_PX, text_tokens, references=n, reference_px=reference_px, blocks=blocks)
            for n in references
            if n and (dtype != "float32" or n <= FLOAT32_MAX_REFERENCES)
        ]
        points += [Point("fit", dtype, CHECK_PX, text_tokens, batch=n, blocks=blocks) for n in batches if n != 1]
        if base:
            points.append(Point("fit", dtype, CHECK_PX, text_tokens, blocks=blocks, flow_map=False))
    if mode == "fit" and extra_blocks:
        points.append(Point("blocks", dtypes[0], CHECK_PX, text_tokens, blocks=extra_blocks))
    return points


def provenance(point: Point, facts: Mapping[str, Any], date: str) -> str:
    """A measured point's `provenance`: where and how it was measured."""
    regime = ", SDPA forced onto the math kernel" if point.force_math else ""
    variant = "Klein (flow map)" if point.flow_map else "Klein Base"
    blocks = f"{point.blocks} per-mode block{'' if point.blocks == 1 else 's'} per stream and {point.blocks} joint"
    model = (
        f"The real FLUX.3 {variant} single file {Path(point.checkpoint).name}, loaded in {point.dtype} through the "
        "model loader's conversion and fully resident"
        if point.checkpoint
        else f"Random-init FLUX.3 {variant} transformer in {point.dtype} with the model loader's geometry, {blocks}"
        f"{regime}; memory does not depend on weight values"
    )
    return (
        f"scripts/calibrate_flux3_denoise_working_memory.py ({date}): {fresh_points.runtime_provenance(facts)}. "
        f"{model}. A fresh process with the server's runtime, the larger demand of a cold and a warm forward; no "
        "failed allocation, allocator retry or spill."
    )


# Everything below imports torch. In a measuring process it runs only after `fresh_points.apply_server_runtime`.


def _transformer(point: Point) -> "torch.nn.Module":
    """The FLUX.3 transformer of the point's variant, block count and dtype, on the meta device: as the model loader
    builds it, the QK norms in float32. A real point's has the geometry its checkpoint's shapes give, as the loader
    reads them (`_build_transformer`)."""
    import torch
    from diffusers import Flux3Transformer2DModel

    from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
    from invokeai.backend.model_manager.load.model_loaders.flux3 import _build_transformer, _keep_qk_norms_in_fp32

    if point.checkpoint:
        shapes = {key: _Shape(tuple(entry["shape"])) for key, entry in checkpoint_header(point.checkpoint).items()}
        model = _build_transformer(CheckpointPrefix.detect(shapes).strip(shapes))
        if bool(model.config.flow_map) != point.flow_map:
            raise SystemExit(f"{point.checkpoint} is not the variant the point names (flow_map={point.flow_map}).")
    else:
        with torch.device("meta"):
            model = Flux3Transformer2DModel(
                **GEOMETRY, num_layers=point.blocks, num_single_layers=point.blocks, flow_map=point.flow_map
            )
    # torch's own cast: diffusers' `ModelMixin.to` warns about the QK norms, which the next line keeps in float32.
    model = torch.nn.Module.to(model, getattr(torch, point.dtype))
    _keep_qk_norms_in_fp32(model)
    return model


@dataclass(frozen=True)
class _Shape:
    """A header entry's shape, which is all `flux3_transformer_config` reads of a tensor."""

    shape: tuple[int, ...]


def _load_real(point: Point, device: "torch.device") -> "torch.nn.Module":
    """The point's checkpoint, built as `Flux3CheckpointModel` builds it (`flux3_transformer_from_native`), except
    straight onto `device` and outside the RAM cache."""
    import torch
    from safetensors.torch import load_file

    from invokeai.backend.model_manager.load.model_loaders.flux3 import flux3_transformer_from_native

    model = flux3_transformer_from_native(load_file(point.checkpoint, device=str(device)), getattr(torch, point.dtype))
    return model.eval()


def _weight_bytes(transformer: "torch.nn.Module") -> int:
    """A meta-device transformer's weights as they will be allocated."""
    return sum(param.numel() * param.element_size() for param in transformer.parameters())


def _estimate(
    point: Point, device: "torch.device", transformer: "torch.nn.Module", *, resident: bool = True
) -> "WorkingMemoryEstimate":
    """The estimate for the point's forward on `device`, priced in the kernel regime the caller holds: fully resident,
    as measured, unless `resident` is False, as the node prices it."""
    import torch

    from invokeai.backend.flux3.denoise_working_memory import estimate_flux3_denoise_working_memory

    return estimate_flux3_denoise_working_memory(
        transformer,
        image_seq_len=image_tokens(point.px),
        ref_image_seq_len=reference_tokens(point),
        text_seq_len=point.text_tokens,
        batch_size=point.batch,
        # Measured whole on the device: nothing is streamed per forward.
        weights_resident=resident,
        device=device,
        dtype=getattr(torch, point.dtype),
    )


def _inputs(point: Point, device: "torch.device") -> dict[str, Any]:
    """What the node hands the transformer per forward (`flux3/denoise.py`): the image's tokens, the references' on a
    stream of their own with their position ids, a caption, the timestep and, for Klein, the timestep it jumps to,
    `point.batch` samples of each."""
    import torch

    from invokeai.backend.flux3.sampling import image_position_ids, text_position_ids

    dtype = getattr(torch, point.dtype)
    side = point.px // CELL_PX
    inputs: dict[str, Any] = {
        "hidden_states": torch.randn(point.batch, side * side, GEOMETRY["in_channels"], device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(
            point.batch, point.text_tokens, GEOMETRY["joint_attention_dim"], device=device, dtype=dtype
        ),
        "timestep": torch.full((point.batch,), TIMESTEP, device=device, dtype=dtype),
        "img_ids": image_position_ids(side, side, l_coord=0, device=device),
        "txt_ids": text_position_ids(point.text_tokens, device=device),
        "target_timestep": torch.full((point.batch,), TARGET_TIMESTEP, device=device, dtype=dtype)
        if point.flow_map
        else None,
        "return_dict": False,
    }
    grids = reference_grids(point)
    if grids:
        inputs["image_cond_hidden_states"] = torch.randn(
            point.batch, reference_tokens(point), GEOMETRY["cond_in_channels"], device=device, dtype=dtype
        )
        inputs["image_cond_ids"] = torch.cat(
            [
                image_position_ids(rows, columns, l_coord=-(1 + index), device=device)
                for index, (rows, columns) in enumerate(grids)
            ]
        )
    return inputs


def _measure(point: Point, force: bool) -> dict[str, Any]:
    """The output line of `point`, measured in this process after `fresh_points.apply_server_runtime`."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from invokeai.backend.flux3 import denoise_working_memory
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
        transformer = _transformer(point)
        # In the forward's kernel regime, so the score matrix is priced for the kernel that runs.
        with kernel():
            estimate = _estimate(point, device, transformer)
        platform = platform_for(device)
        facts = fresh_points.runtime_facts(device, platform.os, denoise_working_memory.__file__)
        common = {
            "estimator": estimate.estimator,
            "version": estimate.version,
            "platform": dataclasses.asdict(platform),
            "features": dict(estimate.features),
            "device": facts["device"],
        }

        weights = _weight_bytes(transformer)
        free, _ = TorchDevice.cuda_mem_get_info(device)
        raw = {
            "reservation_bytes": estimate.reservation_bytes,
            "weight_bytes": weights,
            "predicted_bytes": estimate.reservation_bytes + weights,
            "free_bytes": free,
            "runtime": facts,
        }
        if point.checkpoint:
            raw["node_reservation_bytes"] = _estimate(point, device, transformer, resident=False).reservation_bytes
        if raw["predicted_bytes"] > free and not force:
            return fresh_points.skipped_line(point, common, raw)

        if point.checkpoint:
            transformer = _load_real(point, device)
        else:
            # Allocated directly on the device in the dtypes the meta model holds and randomly initialized, so no
            # block passes through the host in float32 first.
            transformer = transformer.to_empty(device=device)
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
    f"{'kind':>8} {'dtype':>8} {'base':>4} {'blocks':>6} {'px':>5} {'text':>5} {'refs':>4} {'ref px':>6} "
    f"{'batch':>5} {'tokens':>7}"
)


def _grid(point: Mapping[str, Any]) -> str:
    return (
        f"{point['kind']:>8} {point['dtype']:>8} {'' if point['flow_map'] else 'yes':>4} "
        f"{'ckpt' if point['checkpoint'] else point['blocks']:>6} "
        f"{point['px']:>5} {point['text_tokens']:>5} {point['references']:>4} "
        f"{point['reference_px'] if point['references'] else '':>6} {point['batch']:>5}"
    )


def _attended(line: Mapping[str, Any]) -> int:
    """The tokens of every sample of a measured line: its size on the line."""
    return line["features"]["tokens"] * line["features"]["batch_size"]


def _head(line: Mapping[str, Any]) -> str:
    """A line's grid columns in the table."""
    return f"{_grid(line['point'])} {_attended(line) if 'features' in line else '':>7}"


def _px(line: Mapping[str, Any]) -> int:
    return line["point"]["px"]


_SHAPE = ("dtype", "px", "text_tokens", "references", "reference_px", "batch")
"""What makes two grid points the same forward but for the variant or the block count."""


def _same_shape(line: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    return all(line["point"][key] == other["point"][key] for key in _SHAPE)


def _summarize_fit(lines: Sequence[Mapping[str, Any]], text_tokens: int) -> None:
    measured = [line for line in lines if line["status"] == "point"]
    grid = [line for line in measured if line["fit"]]
    # The line's slopes from the points that fit it; the extra-blocks forward compares with any grid point.
    fit = [line for line in grid if line["in_fit"]]
    for dtype in dict.fromkeys(line["point"]["dtype"] for line in fit):
        of_dtype = [line for line in fit if line["point"]["dtype"] == dtype]
        plain = [
            line
            for line in of_dtype
            if line["point"]["references"] == 0 and line["point"]["batch"] == 1 and line["point"]["flow_map"]
        ]
        main = sorted((line for line in plain if line["point"]["text_tokens"] == text_tokens), key=_attended)
        for line in main:
            print(f"{dtype}, {_px(line)}px: {line['measured_bytes'] / _attended(line) / MIB:.4f} MiB per token")
        if len(main) < 2:
            continue
        lo, hi = main[0], main[-1]
        slope = (hi["measured_bytes"] - lo["measured_bytes"]) / (_attended(hi) - _attended(lo))
        base = lo["measured_bytes"] - slope * _attended(lo)
        print(f"{dtype}, {_px(lo)}-{_px(hi)}px: {slope / MIB:.4f} MiB per token, intercept {base / GIB:.3f} GiB")
        # Reference images, batches and Klein Base sit on the same line, which `build_shipped.py` fits on all of them.
        for line in of_dtype:
            if line in main:
                continue
            on_line = base + slope * _attended(line)
            point = line["point"]
            print(
                f"{dtype}{'' if point['flow_map'] else ' Klein Base'}, {_px(line)}px, {point['text_tokens']} text, "
                f"{point['references']} references, batch {point['batch']}: {line['measured_bytes'] / GIB:.3f} GiB "
                f"at {_attended(line)} tokens, {line['measured_bytes'] / on_line:.3f}x the {_px(lo)}-{_px(hi)}px line"
            )
    for line in (line for line in grid if not line["point"]["flow_map"]):
        klein = next((ref for ref in grid if ref["point"]["flow_map"] and _same_shape(ref, line)), None)
        if klein is not None:
            extra = line["measured_bytes"] - klein["measured_bytes"]
            print(
                f"Klein Base against Klein, {line['point']['dtype']}, {_px(line)}px: {extra / MIB:+.0f} MiB "
                f"({line['measured_bytes'] / klein['measured_bytes']:.3f}x)"
            )
    for line in (line for line in measured if line["point"]["kind"] == "blocks"):
        base_line = next((ref for ref in grid if ref["point"]["flow_map"] and _same_shape(ref, line)), None)
        if base_line is not None:
            extra = line["measured_bytes"] - base_line["measured_bytes"]
            print(
                f"{line['point']['blocks']} against {base_line['point']['blocks']} blocks per stream, "
                f"{line['point']['dtype']}, {_px(line)}px: {extra / MIB:+.0f} MiB "
                f"({line['measured_bytes'] / base_line['measured_bytes']:.3f}x)"
            )


def _summarize_large(lines: Sequence[Mapping[str, Any]]) -> None:
    for line in (line for line in lines if line["status"] == "point"):
        point, reserved = line["point"], line["raw"]["reservation_bytes"]
        print(
            f"{point['dtype']}, {_px(line)}px + {point['references']}x{point['reference_px']}px: "
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
            (ref for ref in measured if ref["point"]["kind"] == "baseline" and ref["point"]["dtype"] == point["dtype"]),
            None,
        )
        if baseline is None:
            print(f"math kernel, {point['dtype']}: no default-kernel forward to compare with")
            continue
        extra = line["measured_bytes"] - baseline["measured_bytes"]
        charged = line["raw"]["reservation_bytes"] - baseline["raw"]["reservation_bytes"]
        tokens = line["features"]["tokens"]
        elements = GEOMETRY["num_attention_heads"] * line["features"]["batch_size"] * tokens * tokens
        print(
            f"math kernel, {point['dtype']}, {point['px']}px: {extra / GIB:.3f} GiB over the default kernel, "
            f"{extra / elements:.2f} bytes per score element; the estimate adds {charged / GIB:.3f} GiB "
            f"({SDPA_MATH_BYTES_PER_SCORE_ELEMENT} bytes per element, one chunk where the ROCm guard chunks)"
        )


def _summarize_real(lines: Sequence[Mapping[str, Any]]) -> None:
    """Each real forward against the shipped 2-block point of the same features on this platform and card, and against
    the shipped estimate, fully resident and as the node prices it."""
    from invokeai.backend.working_memory.coefficients import shipped_data

    shipped = [point for data in shipped_data().values() for point in data.points]
    for line in (line for line in lines if line["status"] == "point"):
        point, raw, demand = line["point"], line["raw"], line["measured_bytes"]
        shape = f"{_px(line)}px + {point['references']}x{point['reference_px']}px, {point['text_tokens']} text"
        two_block = next(
            (
                ref
                for ref in shipped
                if ref.in_fit
                and ref.estimator == line["estimator"]
                # Shipped points name the card short (`RTX 4090`), the measuring process as the driver does.
                and ref.device in line["device"]
                and ref.features == line["features"]
                and ref.platform.items() <= line["platform"].items()
            ),
            None,
        )
        against = (
            f"{demand / two_block.measured_bytes:.3f}x the shipped 2-block point ({two_block.measured_bytes / GIB:.3f} "
            "GiB)"
            if two_block is not None
            else "no shipped 2-block point of these features on this platform and card"
        )
        print(
            f"real {point['dtype']}, {shape}: {demand / GIB:.3f} GiB at {_attended(line)} tokens, {against}; the "
            f"shipped estimate reserves {raw['reservation_bytes'] / demand:.3f}x it fully resident, "
            f"{raw['node_reservation_bytes'] / demand:.3f}x with the streamed layer the node adds"
        )


def _print_plan(points: Sequence[Point]) -> None:
    """The planned points, each priced as a CPU prices it, plus the weights: the estimate with cuDNN's native-allocator
    values floored at the legacy constants, as every unmeasured platform's, the forced math kernel's score matrix
    included."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print(f"{_COLUMNS} {'estimate GiB':>12} {'weights GiB':>11} {'total GiB':>9}")
    cpu = torch.device("cpu")
    models: dict[tuple[str, int, bool, str], torch.nn.Module] = {}
    for point in points:
        geometry = (point.dtype, point.blocks, point.flow_map, point.checkpoint)
        if geometry not in models:
            models[geometry] = _transformer(point)
        transformer = models[geometry]
        with sdpa_kernel(SDPBackend.MATH) if point.force_math else nullcontext():
            estimate = _estimate(point, cpu, transformer)
        weights = _weight_bytes(transformer)
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
    if any(px <= 0 or px % CELL_PX for px in sizes):
        raise argparse.ArgumentTypeError(f"sizes must be positive multiples of {CELL_PX} px, got {text!r}")
    return sizes


def _caption(text: str) -> int:
    try:
        tokens = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a token count: {text!r}")
    if tokens <= 0 or tokens % TEXT_PAD_MULTIPLE or tokens > MAX_TEXT_TOKENS:
        raise argparse.ArgumentTypeError(
            f"a caption is a positive multiple of {TEXT_PAD_MULTIPLE} tokens up to {MAX_TEXT_TOKENS}, as the node "
            f"pads it, got {tokens}"
        )
    return tokens


def _dtypes(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names or not set(names) <= set(DTYPES):
        raise argparse.ArgumentTypeError(f"dtypes must be among {', '.join(DTYPES)}, got {text!r}")
    return names


def _side(text: str) -> int:
    try:
        side = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a side length in px: {text!r}")
    if side < CELL_PX:
        raise argparse.ArgumentTypeError(f"a reference must be at least {CELL_PX} px, got {side}")
    return side


def _references(text: str) -> list[int]:
    counts = _ints(text)
    if any(count < 0 or count > MAX_REFERENCES for count in counts):
        raise argparse.ArgumentTypeError(f"reference counts must lie between 0 and {MAX_REFERENCES}, got {text!r}")
    return counts


def _counts(text: str) -> list[int]:
    counts = _ints(text)
    if any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError(f"counts must not be negative, got {text!r}")
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fit", "large", "math", "real"), default="fit", help="Default: fit.")
    parser.add_argument("--checkpoint", default="", help="Real mode: the FLUX.3 single file to measure.")
    parser.add_argument("--out", type=Path, help="The JSON-lines file the lines are appended to. Required to measure.")
    parser.add_argument(
        "--dtypes", type=_dtypes, help="Transformer dtypes. Default: all three in fit mode, bfloat16 otherwise."
    )
    parser.add_argument("--sizes", type=_sizes, default="512,768,1024,1440,2048", help="The fit grid's sizes, px.")
    parser.add_argument("--text-tokens", type=_caption, default=560, help="The caption at every size. Default 560.")
    parser.add_argument("--short-text-tokens", type=_caption, default=80, help="The text-slope caption. Default 80.")
    parser.add_argument(
        "--short-text-sizes", type=_sizes, default="1024,1440", help="Fit sizes that also run the text-slope caption."
    )
    parser.add_argument(
        "--references", type=_references, default="1,3,10", help="Reference images at 1024px, each run. Default 1,3,10."
    )
    parser.add_argument(
        "--reference-px", type=_side, default=REFERENCE_PX, help="The fit grid's reference side, px. Default 1024."
    )
    parser.add_argument("--batches", type=_counts, default="2", help="Batch sizes at 1024px beyond 1. Default 2.")
    parser.add_argument("--no-base", action="store_true", help="Fit mode: leave out the Klein Base forward at 1024px.")
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
    args = parser.parse_args(argv)

    if args.single is not None:
        fresh_points.measure_in_this_process(args.single, point_type=Point, measure=_measure, logger_name=LOGGER_NAME)
    if (args.mode == "real") != bool(args.checkpoint):
        parser.error("--checkpoint names the single file real mode measures, and only real mode takes one")
    if args.checkpoint and not Path(args.checkpoint).is_file():
        parser.error(f"--checkpoint {args.checkpoint} is not a file")
    if args.mode == "real":
        defaults = vars(parser.parse_args([]))
        ignored = [f"--{name.replace('_', '-')}" for name in REAL_MODE_IGNORES if getattr(args, name) != defaults[name]]
        if ignored:
            parser.error(
                f"real mode measures its own grid on the checkpoint's blocks and takes no {', '.join(ignored)}"
            )
        problem = checkpoint_problem(args.checkpoint)
        if problem is not None:
            parser.error(problem)
    points = plan(
        args.mode,
        dtypes=args.dtypes or (list(DTYPES) if args.mode == "fit" else ["bfloat16"]),
        sizes=args.sizes,
        text_tokens=args.text_tokens,
        short_text_tokens=args.short_text_tokens,
        short_text_sizes=args.short_text_sizes,
        references=args.references,
        batches=args.batches,
        blocks=args.blocks,
        extra_blocks=args.extra_blocks,
        base=not args.no_base,
        reference_px=args.reference_px,
        checkpoint=args.checkpoint,
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
    elif args.mode == "real":
        _summarize_real(lines)
    else:
        _summarize_checks(lines)
    fresh_points.print_counts(lines, args.out)


if __name__ == "__main__":
    main()

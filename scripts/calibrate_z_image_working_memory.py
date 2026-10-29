"""Measure the working memory of one Z-Image transformer forward, as shipped-data points for `denoise.z_image`.

What a point measures
---------------------
The demand of one transformer forward, measured as a fresh window (`invokeai/backend/working_memory/measurement.py`,
whose definitions the in-server probe's windows share). With the model and its inputs resident: synchronize, release the
allocator's cached blocks, read the baseline and reset the peaks; run the forward; the demand is then the largest growth
of three counters over the baseline: the driver's figure for this process, torch's reserved peak and its allocated peak.
A forward whose allocations failed, or only succeeded after the allocator released its cache and retried (a squeezed
peak), or after which Windows keeps more than `measurement.SPILL_TOLERANCE_BYTES` of the process in shared memory, is
invalid: it is recorded as such, not as a point.

Every point runs in a fresh process, which first applies the server's runtime (`apply_server_runtime` in
`invokeai/app/run_app.py`): the `MIOPEN_FIND_MODE` default, the integrated-GPU check, the allocator, the AOTriton
decision and the monkeypatches, the ROCm SDPA guard among them, in the server's order. Its settings are those of a server
without a config file, except the allocator configuration: `--alloc-conf`, or without one the platform default the
server uses (expandable segments for ROCm on Windows, torch's native allocator elsewhere). This script's own process runs
the integrated-GPU check once before the first point: the measuring processes inherit the `HIP_VISIBLE_DEVICES` it sets,
which their own check keeps, as the server keeps an exported one.

The forward then runs twice. The first forward after a load is a real peak in a server (kernel modules, library
workspaces), and so is a later one, so the point takes the larger demand. Where both forwards read the driver's figure
the point is a `demand_peak`. Where one could not, the point falls back to torch's counters alone, as a `reserved_peak`.

The transformer has the real geometry (dim 3840, 30 heads of 128, caption features 2560 wide) and random weights in the
measured dtype, because memory depends on shapes, not on weight values. It has 2 main blocks and one refiner block each:
a no-grad forward frees each block's intermediates, which `--extra-blocks 4` re-checks at 1024px. A point carries the
features of `estimate_z_image_denoise_working_memory`, taken in the measuring process after the runtime setup, so its
`score_matrix_bytes` reflects the attention kernel this build really runs.

Each point is priced before it runs: the estimate for it plus its weights in its dtype. It is skipped when that exceeds
free device memory, unless `--force`. An oversized allocation does not always fail: where the driver backs it with
system RAM (NVIDIA's sysmem fallback, AMD on Windows) it pages, which can stall the whole machine.

Modes
-----
- `fit` (default): the fit grid, in each of `--dtypes` (bfloat16, float16, float32). The `--sizes` (512, 768, 1024,
  1440, 2048 px) run with a `--text-tokens` caption (512). Those among `--short-text-sizes` (1024, 1440) also run with
  a `--short-text-tokens` caption (77), for the text slope. `--extra-blocks N` adds one forward with N main blocks at
  1024px, in the first dtype.
- `math`: SDPA forced onto the math kernel at 1024px, next to the default kernel, per dtype: what a materialized score
  matrix costs per element, against what the estimate charges for it.
- `control`: a `--control-blocks` ControlNet adapter (15) at 1024px, next to no control, per dtype: how many
  hint-sized tensors it keeps, against the n + 4 the estimate charges.

Only the fit grid's points are fit points (`"fit": true`). The checks' points, the extra-blocks forward included, test
analytic terms or assumptions rather than the line: they are marked `"fit": false`, and their `in_fit` is false. A fit
point is `in_fit` only where the estimator's line admits its features (`Line.fits`): on a build whose SDPA kernel
materializes the score matrix, the fit grid's forwards carry a score term on top of the line, so they measure that term
rather than the line, as a check does.

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
RTX 4090 (CUDA): the native allocator, then `cudaMallocAsync`; the checks once.

    python scripts/calibrate_z_image_working_memory.py --out 4090-native.jsonl --extra-blocks 4
    python scripts/calibrate_z_image_working_memory.py --out 4090-async.jsonl --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_z_image_working_memory.py --out 4090-checks.jsonl --mode math
    python scripts/calibrate_z_image_working_memory.py --out 4090-checks.jsonl --mode control

RX 9060 XT (ROCm on Windows): the server's default expandable segments, then the native allocator; the checks once.

    python scripts/calibrate_z_image_working_memory.py --out 9060xt-expandable.jsonl --extra-blocks 4
    python scripts/calibrate_z_image_working_memory.py --out 9060xt-native.jsonl --alloc-conf expandable_segments:False
    python scripts/calibrate_z_image_working_memory.py --out 9060xt-checks.jsonl --mode math
    python scripts/calibrate_z_image_working_memory.py --out 9060xt-checks.jsonl --mode control

`--dry-run` prints the planned points without touching a device.
"""

import argparse
import dataclasses
import datetime
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# None of these imports torch: a measuring process has to configure the allocator before torch is imported.
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.util.rocm_aotriton import AOTRITON_EXPERIMENTAL_ENV
from invokeai.app.util.torch_cuda_allocator import ALLOCATOR_CONF_ENV_VARS

if TYPE_CHECKING:
    import torch

    from invokeai.backend.working_memory.measurement import FreshWindow

GIB = 2**30
MIB = 2**20

CAP_FEAT_DIM = 2560
PATCH_SIZE = 2
CONTROL_IN_DIM = 33  # V2.0 adapters; the channel count only sizes the patch embedder
CHECK_PX = 1024
"""The size the checks and the extra-blocks forward run at."""

DTYPES = ("bfloat16", "float16", "float32")
LOGGER_NAME = "calibrate_z_image_working_memory"
RESULT_PREFIX = "RESULT "
POINT_TIMEOUT_S = 1800
"""How long one measuring process may take. Generous: it only keeps a driver that hangs from stalling a whole run."""

Kind = Literal["fit", "blocks", "reference", "math", "control"]
Mode = Literal["fit", "math", "control"]
PointMethod = Literal["demand_peak", "reserved_peak"]


@dataclass(frozen=True)
class Point:
    """One transformer forward to measure, in a process of its own."""

    kind: Kind
    """`fit` for a point of the fit grid; otherwise the check it serves: another block count (`blocks`), the
    default-kernel forward a check compares with (`reference`), the forced math kernel (`math`) or a ControlNet
    (`control`)."""
    px: int
    text_tokens: int
    """The caption's tokens, unpadded."""
    dtype: str
    """A torch dtype name, one of `DTYPES`."""
    blocks: int
    force_math: bool = False
    control_blocks: int = 0

    @property
    def fit(self) -> bool:
        return self.kind == "fit"


def plan(
    mode: Mode,
    *,
    sizes: Sequence[int],
    dtypes: Sequence[str],
    text_tokens: int,
    short_text_tokens: int,
    short_text_sizes: Sequence[int],
    blocks: int,
    extra_blocks: int,
    control_blocks: int,
) -> list[Point]:
    """The forwards a run measures, in order: per dtype, the fit grid or a check's pair of forwards."""
    points: list[Point] = []
    for dtype in dtypes:
        if mode == "fit":
            points += [Point("fit", px, text_tokens, dtype, blocks) for px in sizes]
            points += [Point("fit", px, short_text_tokens, dtype, blocks) for px in sizes if px in short_text_sizes]
            continue
        points.append(Point("reference", CHECK_PX, text_tokens, dtype, blocks))
        points.append(
            Point(
                mode,
                CHECK_PX,
                text_tokens,
                dtype,
                blocks,
                force_math=mode == "math",
                control_blocks=control_blocks if mode == "control" else 0,
            )
        )
    if mode == "fit" and extra_blocks:
        points.append(Point("blocks", CHECK_PX, text_tokens, dtypes[0], extra_blocks))
    return points


def image_tokens(px: int) -> int:
    side = px // LATENT_SCALE_FACTOR // PATCH_SIZE
    return side * side


def measured_bytes_and_method(windows: Sequence["FreshWindow"]) -> tuple[int, PointMethod]:
    """A valid point's bytes and `MeasuredPoint.method`, from its forwards' windows.

    `demand_peak` where every forward read the driver's figure: the largest demand. Otherwise `reserved_peak`, from
    torch's counters alone, so that the method names what the bytes were read from.
    """
    if all(window.driver is not None for window in windows):
        return max(window.demand for window in windows if window.demand is not None), "demand_peak"
    torch_counters = [delta for window in windows for delta in (window.reserved, window.allocated) if delta is not None]
    return max(torch_counters), "reserved_peak"


def point_line(
    point: Point,
    *,
    common: Mapping[str, Any],
    provenance: str,
    windows: Sequence["FreshWindow"],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """The output line of a measured point: a `point`, or `invalid` with the reasons of its forwards.

    `common` holds the `MeasuredPoint` fields the measurement does not decide: estimator, version, platform, features
    and device. `windows` are the forwards' fresh windows, cold first; `raw` the point's other facts.
    """
    reasons = list(dict.fromkeys(reason for window in windows for reason in window.reasons))
    line = {"status": "invalid" if reasons else "point", "fit": point.fit, "point": dataclasses.asdict(point), **common}
    raw = {**raw, **dict(zip(("cold", "warm"), map(dataclasses.asdict, windows), strict=False))}
    if reasons:
        return line | {"reasons": reasons, "raw": raw}
    # Imports torch: a point line is taken in the measuring process, after `_apply_server_runtime`.
    from invokeai.backend.z_image.denoise_working_memory import Z_IMAGE_DENOISE_SPEC

    measured, method = measured_bytes_and_method(windows)
    on_line = Z_IMAGE_DENOISE_SPEC.line(common["features"])
    return line | {
        "measured_bytes": measured,
        "method": method,
        "provenance": provenance,
        "in_fit": point.fit and on_line is not None and on_line.fits,
        "raw": raw,
    }


def failed_line(point: Point, error: str) -> dict[str, Any]:
    return {"status": "failed", "fit": point.fit, "point": dataclasses.asdict(point), "error": error}


def file_record(line: Mapping[str, Any]) -> dict[str, Any] | None:
    """What `--out` receives of a line: a measured point, without the status only the table reads. None for a forward
    that is no point: the data builder refuses a point file record that is no `MeasuredPoint`."""
    if line["status"] != "point":
        return None
    return {key: value for key, value in line.items() if key != "status"}


def _alloc_conf(facts: Mapping[str, Any]) -> str:
    return ", ".join(f"{var}={value}" for var, value in facts["alloc_conf"].items()) or "no allocator configuration"


def provenance(point: Point, facts: Mapping[str, Any], date: str) -> str:
    """A measured point's `provenance`: where and how it was measured."""
    build = f"torch {facts['torch']}" + (f" (HIP {facts['hip']})" if facts["hip"] else "")
    parts = [
        facts["device"],
        build,
        facts["os"].capitalize(),
        f"{_alloc_conf(facts)} (torch reports the {facts['allocator_backend']} allocator)",
    ]
    if facts["hip"]:
        parts += [
            f"MIOPEN_FIND_MODE={facts['miopen_find_mode'] or 'unset'}",
            f"{AOTRITON_EXPERIMENTAL_ENV}={facts['aotriton_experimental'] or 'unset'}",
            f"ROCm SDPA guard {'installed' if facts['rocm_sdpa_guard'] else 'not installed'}",
        ]
    regime = {"math": ", SDPA forced onto the math kernel", "control": f", a {point.control_blocks}-block ControlNet"}
    blocks = f"{point.blocks} main block{'' if point.blocks == 1 else 's'}"
    return (
        f"scripts/calibrate_z_image_working_memory.py ({date}): {', '.join(parts)}. Random-init Z-Image transformer "
        f"with the real geometry, {blocks} and one refiner block each{regime.get(point.kind, '')}; "
        "memory does not depend on weight values. A fresh process with the server's runtime, the larger demand of a "
        "cold and a warm forward; no failed allocation, allocator retry or spill."
    )


# The measuring process. Everything below imports torch, so it runs only after `_apply_server_runtime`.


def _apply_server_runtime(alloc_conf: str) -> None:
    """Apply the server's runtime (`run_app.apply_server_runtime`) before torch is imported, with the settings of a
    server without a config file and the allocator configuration this run measures: `alloc_conf`, or none for the
    platform's default."""
    from invokeai.app.run_app import apply_server_runtime
    from invokeai.app.services.config.config_default import get_config
    from invokeai.backend.util.logging import InvokeAILogger

    config = get_config().model_copy(update={"pytorch_cuda_alloc_conf": alloc_conf or None})
    apply_server_runtime(config, InvokeAILogger.get_logger(LOGGER_NAME, config=config))


def _runtime_facts(device: "torch.device", os_name: str) -> dict[str, Any]:
    """The runtime a point is measured in, as its provenance and `raw` record it."""
    import torch

    from invokeai.backend.util.attention import rocm_sdpa_chunks_math
    from invokeai.backend.working_memory.counters import backend_for, method_verified
    from invokeai.backend.z_image import denoise_working_memory

    name = torch.cuda.get_device_name(device)
    if torch.version.hip is not None:
        name = f"{name} ({torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]})"
    backend = backend_for(device)
    return {
        "device": name,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "os": os_name,
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "alloc_conf": {var: os.environ[var] for var in ALLOCATOR_CONF_ENV_VARS if os.environ.get(var)},
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        "aotriton_experimental": os.environ.get(AOTRITON_EXPERIMENTAL_ENV),
        "rocm_sdpa_guard": rocm_sdpa_chunks_math(device.type),
        "driver_method_verified": backend is not None and method_verified(backend),
        # The checkout the point was priced from. A module's file, not the package's: `invokeai` can be imported as a
        # namespace package, whose `__file__` is None.
        "estimator_module": str(Path(denoise_working_memory.__file__).resolve()),
    }


def _transformer(blocks: int) -> "torch.nn.Module":
    """The Z-Image transformer's geometry with `blocks` main blocks and one refiner block each, on the meta device."""
    import torch
    from diffusers import ZImageTransformer2DModel

    from invokeai.backend.z_image.denoise_working_memory import Z_IMAGE_ATTENTION_HEADS, Z_IMAGE_HIDDEN_SIZE

    with torch.device("meta"):
        return ZImageTransformer2DModel(
            all_patch_size=(PATCH_SIZE,),
            all_f_patch_size=(1,),
            in_channels=16,
            dim=Z_IMAGE_HIDDEN_SIZE,
            n_layers=blocks,
            n_refiner_layers=1,
            n_heads=Z_IMAGE_ATTENTION_HEADS,
            n_kv_heads=Z_IMAGE_ATTENTION_HEADS,
            norm_eps=1e-05,
            qk_norm=True,
            cap_feat_dim=CAP_FEAT_DIM,
            rope_theta=256.0,
            t_scale=1000.0,
            axes_dims=[32, 48, 48],
            axes_lens=[1024, 512, 512],
        )


def _control_adapter(blocks: int) -> "torch.nn.Module":
    """A ControlNet adapter with `blocks` control blocks at the transformer's width, on the meta device."""
    import torch

    from invokeai.backend.z_image.denoise_working_memory import Z_IMAGE_ATTENTION_HEADS, Z_IMAGE_HIDDEN_SIZE
    from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter

    with torch.device("meta"):
        return ZImageControlAdapter(
            num_control_blocks=blocks,
            control_in_dim=CONTROL_IN_DIM,
            dim=Z_IMAGE_HIDDEN_SIZE,
            n_heads=Z_IMAGE_ATTENTION_HEADS,
            n_kv_heads=Z_IMAGE_ATTENTION_HEADS,
        )


def _materialize(module: "torch.nn.Module", device: "torch.device", dtype: "torch.dtype") -> "torch.nn.Module":
    """A meta-device module allocated directly in `dtype` on `device`, randomly initialized: a 15-block adapter is ~3B
    parameters, which would otherwise pass through the host in float32 first."""
    import torch

    module = module.to(dtype).to_empty(device=device)
    for param in module.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return module.eval()


def _measure(point: Point, force: bool) -> dict[str, Any]:
    """The output line of `point`, measured in this process after `_apply_server_runtime`."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from invokeai.backend.util.devices import TorchDevice
    from invokeai.backend.working_memory.measurement import measure
    from invokeai.backend.working_memory.platform import platform_for
    from invokeai.backend.z_image.denoise_working_memory import estimate_z_image_denoise_working_memory
    from invokeai.backend.z_image.z_image_controlnet_extension import (
        ZImageControlNetExtension,
        z_image_forward_with_control,
    )

    # The device the node itself takes.
    device = TorchDevice.choose_torch_device()
    if device.type != "cuda":
        raise SystemExit(f"Needs a CUDA or ROCm device; this process would run on {device}.")
    dtype = getattr(torch, point.dtype)

    def kernel():
        return sdpa_kernel(SDPBackend.MATH) if point.force_math else nullcontext()

    with torch.inference_mode():
        # In the forward's kernel regime, so the score matrix is priced for the kernel that runs.
        with kernel():
            estimate = estimate_z_image_denoise_working_memory(
                # The point runs a randomly initialized dense model, which dequantizes nothing per forward.
                torch.nn.Module(),
                image_seq_len=image_tokens(point.px),
                text_seq_len=point.text_tokens,
                lora_count=0,
                control_blocks=point.control_blocks,
                # The ControlNet forward always builds a padding mask; the plain one passes none at batch size 1.
                has_attention_mask=point.control_blocks > 0,
                device=device,
                dtype=dtype,
            )
        platform = platform_for(device)
        facts = _runtime_facts(device, platform.os)
        common = {
            "estimator": estimate.estimator,
            "version": estimate.version,
            "platform": dataclasses.asdict(platform),
            "features": dict(estimate.features),
            "device": facts["device"],
        }

        transformer = _transformer(point.blocks)
        adapter = _control_adapter(point.control_blocks) if point.control_blocks else None
        on_meta = [module for module in (transformer, adapter) if module is not None]
        weight_bytes = sum(param.numel() for module in on_meta for param in module.parameters()) * dtype.itemsize
        free, _ = TorchDevice.cuda_mem_get_info(device)
        raw = {
            "reservation_bytes": estimate.reservation_bytes,
            "weight_bytes": weight_bytes,
            "predicted_bytes": estimate.reservation_bytes + weight_bytes,
            "free_bytes": free,
            "runtime": facts,
        }
        if raw["predicted_bytes"] > free and not force:
            return {"status": "skipped", "fit": point.fit, "point": dataclasses.asdict(point), **common, "raw": raw}

        transformer = _materialize(transformer, device, dtype)
        latent = point.px // LATENT_SCALE_FACTOR
        x = [torch.randn(16, 1, latent, latent, device=device, dtype=dtype)]
        t = torch.full((1,), 0.5, device=device, dtype=dtype)
        cap_feats = [torch.randn(point.text_tokens, CAP_FEAT_DIM, device=device, dtype=dtype)]
        extension = None
        if adapter is not None:
            extension = ZImageControlNetExtension(
                control_adapter=_materialize(adapter, device, dtype),
                control_cond=torch.randn(CONTROL_IN_DIM, 1, latent, latent, device=device, dtype=dtype),
            )

        def forward() -> object:
            with kernel():
                if extension is None:
                    return transformer(x=x, t=t, cap_feats=cap_feats)
                return z_image_forward_with_control(
                    transformer=transformer, x=x, t=t, cap_feats=cap_feats, control_extension=extension
                )

        windows = [measure(device, forward)]
        if "oom" not in windows[0].reasons:
            windows.append(measure(device, forward))
    date = datetime.date.today().isoformat()
    return point_line(point, common=common, provenance=provenance(point, facts, date), windows=windows, raw=raw)


def _measure_in_this_process(request: str) -> None:
    """Measure one point in this fresh process and print its line.

    Leaves without Python's shutdown, whose HIP teardown can hang on Windows while its DLLs unload (see
    `invokeai/app/util/rocm_integrated_gpu.py`)."""
    spec = json.loads(request)
    try:
        _apply_server_runtime(spec["alloc_conf"])
        line = _measure(Point(**spec["point"]), spec["force"])
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    print(RESULT_PREFIX + json.dumps(line), flush=True)
    sys.stderr.flush()
    os._exit(0)


# The driving process: plans the points, measures each in a fresh process, appends the lines, prints the table.


def _run(point: Point, alloc_conf: str, force: bool) -> dict[str, Any]:
    request = json.dumps({"point": dataclasses.asdict(point), "alloc_conf": alloc_conf, "force": force})
    command = [sys.executable, str(Path(__file__).resolve()), "--single", request]
    try:
        done = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=POINT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return failed_line(point, f"no result within {POINT_TIMEOUT_S} s")
    for output in done.stdout.splitlines():
        if output.startswith(RESULT_PREFIX):
            return json.loads(output[len(RESULT_PREFIX) :])
    tail = done.stderr.strip().splitlines()[-1:] or ["no output"]
    return failed_line(point, f"exit code {done.returncode}: {tail[0]}")


_COLUMNS = f"{'kind':>9} {'dtype':>8} {'blocks':>6} {'control':>7} {'px':>5} {'text':>5} {'tokens':>7}"


def _grid(point: Mapping[str, Any]) -> str:
    return (
        f"{point['kind']:>9} {point['dtype']:>8} {point['blocks']:>6} {point['control_blocks']:>7} "
        f"{point['px']:>5} {point['text_tokens']:>5}"
    )


def _gib(value: int | None) -> str:
    return "" if value is None else f"{value / GIB:.3f}"


def _row(line: Mapping[str, Any]) -> str:
    head = f"{_grid(line['point'])} {line.get('features', {}).get('tokens', ''):>7}"
    status = line["status"]
    raw = line.get("raw", {})
    cold, warm = (_gib(raw[run]["demand"]) if run in raw else "" for run in ("cold", "warm"))
    if status == "point":
        return f"{head} {_gib(line['measured_bytes']):>10} {cold:>8} {warm:>8}  {line['method']}"
    if status == "invalid":
        return f"{head} {'invalid':>10} {cold:>8} {warm:>8}  {', '.join(line['reasons'])}"
    if status == "skipped":
        raw = line["raw"]
        return (
            f"{head} {'skipped':>10}  needs ~{raw['predicted_bytes'] / GIB:.1f} GiB, {raw['free_bytes'] / GIB:.1f} GiB "
            "free (--force runs it)"
        )
    return f"{head} {'FAILED':>10}  {line['error']}"


def _runtime_header(facts: Mapping[str, Any], platform: Mapping[str, str]) -> str:
    rocm = ""
    if facts["hip"]:
        guard = "installed" if facts["rocm_sdpa_guard"] else "NOT installed"
        rocm = f" | {AOTRITON_EXPERIMENTAL_ENV}={facts['aotriton_experimental'] or 'unset'} | SDPA guard {guard}"
    return (
        f"{facts['device']} | torch {facts['torch']} | {'/'.join(platform.values())} | {_alloc_conf(facts)}{rocm} | "
        f"estimator from {facts['estimator_module']}"
    )


def _tokens(line: Mapping[str, Any]) -> int:
    return line["features"]["tokens"]


def _px(line: Mapping[str, Any]) -> int:
    return line["point"]["px"]


def _summarize_fit(lines: Sequence[Mapping[str, Any]], text_tokens: int) -> None:
    measured = [line for line in lines if line["status"] == "point"]
    grid = [line for line in measured if line["fit"]]
    # The line's slopes from the points that fit it; the extra-blocks forward compares with any grid point.
    fit = [line for line in grid if line["in_fit"]]
    for dtype in dict.fromkeys(line["point"]["dtype"] for line in fit):
        of_dtype = [line for line in fit if line["point"]["dtype"] == dtype]
        main = sorted((line for line in of_dtype if line["point"]["text_tokens"] == text_tokens), key=_tokens)
        for line in main:
            print(f"{dtype}, {_px(line)}px: {line['measured_bytes'] / _tokens(line) / MIB:.4f} MiB per attended token")
        if len(main) >= 2:
            lo, hi = main[0], main[-1]
            slope = (hi["measured_bytes"] - lo["measured_bytes"]) / (_tokens(hi) - _tokens(lo))
            base = lo["measured_bytes"] - slope * _tokens(lo)
            print(f"{dtype}, {_px(lo)}-{_px(hi)}px: {slope / MIB:.4f} MiB per token, intercept {base / GIB:.3f} GiB")
        for short in (line for line in of_dtype if line["point"]["text_tokens"] != text_tokens):
            long = next((line for line in main if _px(line) == _px(short)), None)
            if long is not None and _tokens(long) != _tokens(short):
                slope = (long["measured_bytes"] - short["measured_bytes"]) / (_tokens(long) - _tokens(short))
                print(
                    f"{dtype}, {_px(short)}px, caption of {short['point']['text_tokens']} -> {text_tokens} tokens: "
                    f"{slope / MIB:.4f} MiB per caption token"
                )
    for line in (line for line in measured if line["point"]["kind"] == "blocks"):
        same = ("px", "text_tokens", "dtype")
        base = next((ref for ref in grid if all(ref["point"][key] == line["point"][key] for key in same)), None)
        if base is not None:
            extra = line["measured_bytes"] - base["measured_bytes"]
            print(
                f"{line['point']['blocks']} against {base['point']['blocks']} main blocks, {line['point']['dtype']}, "
                f"{_px(line)}px: {extra / MIB:+.0f} MiB ({line['measured_bytes'] / base['measured_bytes']:.3f}x)"
            )


def _summarize_checks(lines: Sequence[Mapping[str, Any]]) -> None:
    from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT
    from invokeai.backend.z_image.denoise_working_memory import Z_IMAGE_ATTENTION_HEADS, Z_IMAGE_HIDDEN_SIZE

    measured = [line for line in lines if line["status"] == "point"]
    for line in (line for line in measured if line["point"]["kind"] in ("math", "control")):
        point = line["point"]
        reference = next(
            (
                ref
                for ref in measured
                if ref["point"]["kind"] == "reference" and ref["point"]["dtype"] == point["dtype"]
            ),
            None,
        )
        if reference is None:
            print(f"{point['kind']}, {point['dtype']}: no default-kernel forward to compare with")
            continue
        extra = line["measured_bytes"] - reference["measured_bytes"]
        charged = line["raw"]["reservation_bytes"] - reference["raw"]["reservation_bytes"]
        tokens = _tokens(line)
        if point["kind"] == "math":
            elements = Z_IMAGE_ATTENTION_HEADS * tokens * tokens
            print(
                f"math kernel, {point['dtype']}, {point['px']}px: {extra / GIB:.3f} GiB over the default kernel, "
                f"{extra / elements:.2f} bytes per score element; the estimate adds {charged / GIB:.3f} GiB "
                f"({SDPA_MATH_BYTES_PER_SCORE_ELEMENT} bytes per element, one chunk where the ROCm guard chunks)"
            )
        else:
            hint_bytes = tokens * Z_IMAGE_HIDDEN_SIZE * (4 if point["dtype"] == "float32" else 2)
            blocks = point["control_blocks"]
            print(
                f"control, {blocks} blocks, {point['dtype']}, {point['px']}px: {extra / GIB:.3f} GiB over no control, "
                f"{extra / hint_bytes:.2f} hint-sized tensors; the estimate adds {charged / GIB:.3f} GiB "
                f"({blocks} + 4 hints, and any score matrix the mask brings)"
            )


def _print_plan(points: Sequence[Point]) -> None:
    from invokeai.backend.z_image.denoise_working_memory import padded_tokens

    print(_COLUMNS)
    for point in points:
        tokens = padded_tokens(image_tokens(point.px)) + padded_tokens(point.text_tokens)
        print(f"{_grid(dataclasses.asdict(point))} {tokens:>7}{'  forced math' if point.force_math else ''}")
    print(f"{len(points)} forwards, one process each")


def _sizes(text: str) -> list[int]:
    try:
        sizes = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of sizes: {text!r}")
    multiple = LATENT_SCALE_FACTOR * PATCH_SIZE
    if any(px <= 0 or px % multiple for px in sizes):
        raise argparse.ArgumentTypeError(f"sizes must be positive multiples of {multiple} px, got {text!r}")
    return sizes


def _dtypes(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names or not set(names) <= set(DTYPES):
        raise argparse.ArgumentTypeError(f"dtypes must be among {', '.join(DTYPES)}, got {text!r}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fit", "math", "control"), default="fit", help="Default: fit.")
    parser.add_argument("--out", type=Path, help="The JSON-lines file the lines are appended to. Required to measure.")
    parser.add_argument("--sizes", type=_sizes, default="512,768,1024,1440,2048", help="The fit grid's sizes, px.")
    parser.add_argument("--dtypes", type=_dtypes, default=",".join(DTYPES), help="Default: all three.")
    parser.add_argument("--text-tokens", type=int, default=512, help="The caption at every size. Default 512.")
    parser.add_argument("--short-text-tokens", type=int, default=77, help="The text-slope caption. Default 77.")
    parser.add_argument(
        "--short-text-sizes", type=_sizes, default="1024,1440", help="Fit sizes that also run the text-slope caption."
    )
    parser.add_argument("--blocks", type=int, default=2, help="Main transformer blocks. Default 2.")
    parser.add_argument("--extra-blocks", type=int, default=0, help="Fit mode: this block count at 1024px as well.")
    parser.add_argument("--control-blocks", type=int, default=15, help="The control check's adapter. Default 15.")
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
        _measure_in_this_process(args.single)
        return
    points = plan(
        args.mode,
        sizes=args.sizes,
        dtypes=args.dtypes,
        text_tokens=args.text_tokens,
        short_text_tokens=args.short_text_tokens,
        short_text_sizes=args.short_text_sizes,
        blocks=args.blocks,
        extra_blocks=args.extra_blocks,
        control_blocks=args.control_blocks,
    )
    if args.dry_run:
        _print_plan(points)
        return
    if args.out is None:
        parser.error("--out is required to measure")
    exported = [var for var in ALLOCATOR_CONF_ENV_VARS if os.environ.get(var)]
    if args.alloc_conf and exported:
        # The server keeps an exported value and only warns; a measurement must not run another allocator than asked.
        parser.error(f"--alloc-conf cannot apply while {', '.join(exported)} is set in the environment")

    from invokeai.app.util.rocm_integrated_gpu import hide_integrated_gpus_on_rocm_windows
    from invokeai.backend.util.logging import InvokeAILogger

    # The runtime's integrated-GPU check, once for the whole run, before torch is imported anywhere: the measuring
    # processes inherit the HIP_VISIBLE_DEVICES it sets instead of each asking HIP in a child process of its own.
    hide_integrated_gpus_on_rocm_windows(InvokeAILogger.get_logger(LOGGER_NAME))

    print(f"{len(points)} forwards, one fresh process each")
    lines: list[dict[str, Any]] = []
    with args.out.open("a", encoding="utf-8") as out:
        for point in points:
            line = _run(point, args.alloc_conf, args.force)
            if "raw" in line and not any("raw" in earlier for earlier in lines):
                print(_runtime_header(line["raw"]["runtime"], line["platform"]))
            if not lines:
                print(f"{_COLUMNS} {'demand GiB':>10} {'cold':>8} {'warm':>8}")
            record = file_record(line)
            if record is not None:
                out.write(json.dumps(record) + "\n")
                out.flush()
            print(_row(line), flush=True)
            lines.append(line)

    print()
    if args.mode == "fit":
        _summarize_fit(lines, args.text_tokens)
    else:
        _summarize_checks(lines)
    statuses = [line["status"] for line in lines]
    counts = ", ".join(f"{statuses.count(status)} {status}" for status in ("point", "invalid", "skipped", "failed"))
    print(f"{counts}; the points appended to {args.out}")


if __name__ == "__main__":
    main()

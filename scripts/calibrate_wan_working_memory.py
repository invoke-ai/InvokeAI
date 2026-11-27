"""Measure the working memory of one Wan 2.2 transformer forward, as shipped-data points for `wan.denoise`.

What a point measures
---------------------
The demand of one transformer forward, measured as a fresh window (`invokeai/backend/working_memory/measurement.py`)
in a fresh process with the server's runtime (`scripts/working_memory/fresh_points.py`), the larger of a cold and a warm
forward, as `scripts/calibrate_z_image_working_memory.py` describes in full. The transformer has the real geometry of
the A14B experts or of TI2V-5B, random weights in the measured dtype (memory depends on shapes, not on weight values) and
2 blocks: a no-grad forward frees each block's intermediates, which `--extra-blocks 4` re-checks. The caption is 512
tokens of 4096 features, as the UMT5 encoder pads it. A point carries the features of
`estimate_wan_denoise_working_memory`, taken in the measuring process after the runtime setup, so its `score_matrix_bytes`
reflects the attention kernel this build really runs.

Each point is priced before it runs, the estimate plus its weights, and skipped when that exceeds free device memory,
unless `--force`.

Modes
-----
- `fit` (default): per dtype, the A14B, TI2V-5B and TI2V-5B per-token-timestep lines over latent grids from one frame to
  the largest video the app sets. `--extra-blocks N` adds one A14B forward with N blocks, in the first dtype.
- `sage`: at the largest grid of each geometry in the first dtype, the plain forward next to one whose every attention
  call first allocates and holds what SageAttention's kernels hold beside SDPA's own work: an int8 copy of Q and of K,
  and a float16 and an 8-bit copy of V. Whether its quantized copies raise the forward's peak, which needs no NVIDIA GPU
  to answer. Its points are checks (`"fit": false`): keep them in a file of their own, out of the shipped data, whose
  features do not tell them from the plain forward's.

Output
------
Every measured point is appended to `--out`, the point file `scripts/working_memory/build_shipped.py --points` reads.

Usage
-----
    python scripts/calibrate_wan_working_memory.py --out w7900-native.jsonl --extra-blocks 4
    python scripts/calibrate_wan_working_memory.py --out w7900-sage-check.jsonl --mode sage --dtypes bfloat16

`--dry-run` prints the planned points without touching a device.
"""

import argparse
import dataclasses
import datetime
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# None of these imports torch: a measuring process has to configure the allocator before torch is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.working_memory import fresh_points  # noqa: E402

if TYPE_CHECKING:
    import torch

GIB = 2**30
MIB = 2**20
TEXT_TOKENS = 512
TEXT_DIM = 4096
DTYPES = ("bfloat16", "float16", "float32")
LOGGER_NAME = "calibrate_wan_working_memory"

Geometry = Literal["a14b", "ti2v", "ti2v_token_timesteps"]
GRIDS: Mapping[str, tuple[tuple[int, int, int], ...]] = {
    # (latent frames, latent height, latent width): 1280x720 one frame, 832x480x81, 1280x720x81 and x121.
    "a14b": ((1, 90, 160), (21, 60, 104), (21, 90, 160), (31, 90, 160)),
    # 1280x704 one frame, x61 and x121, 1792x1024x121.
    "ti2v": ((1, 44, 80), (16, 44, 80), (31, 44, 80), (31, 64, 112)),
    "ti2v_token_timesteps": ((1, 44, 80), (16, 44, 80), (31, 44, 80), (31, 64, 112)),
}
_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "a14b": {"num_attention_heads": 40, "in_channels": 16, "out_channels": 16, "ffn_dim": 13824},
    "ti2v": {"num_attention_heads": 24, "in_channels": 48, "out_channels": 48, "ffn_dim": 14336},
}

Kind = Literal["fit", "blocks", "reference", "sage"]
Mode = Literal["fit", "sage"]


@dataclass(frozen=True)
class Point:
    """One transformer forward to measure, in a process of its own."""

    kind: Kind
    geometry: Geometry
    frames: int
    latent_height: int
    latent_width: int
    dtype: str
    blocks: int

    @property
    def fit(self) -> bool:
        return self.kind == "fit"

    @property
    def tokens(self) -> int:
        return self.frames * (self.latent_height // 2) * (self.latent_width // 2)


def plan(mode: Mode, *, dtypes: Sequence[str], blocks: int, extra_blocks: int) -> list[Point]:
    points: list[Point] = []
    if mode == "fit":
        for dtype in dtypes:
            for geometry, grid in GRIDS.items():
                points += [Point("fit", geometry, *shape, dtype, blocks) for shape in grid]  # type: ignore[arg-type]
        if extra_blocks:
            points.append(Point("blocks", "a14b", *GRIDS["a14b"][2], dtypes[0], extra_blocks))
        return points
    for geometry in ("a14b", "ti2v"):
        shape = GRIDS[geometry][-1]
        points.append(Point("reference", geometry, *shape, dtypes[0], blocks))  # type: ignore[arg-type]
        points.append(Point("sage", geometry, *shape, dtypes[0], blocks))  # type: ignore[arg-type]
    return points


def provenance(point: Point, facts: Mapping[str, Any], date: str) -> str:
    timesteps = "one timestep per token" if point.geometry == "ti2v_token_timesteps" else "one timestep per sample"
    model = "A14B expert" if point.geometry == "a14b" else "TI2V-5B"
    sage = (
        ", every attention call holding SageAttention-sized quantized copies of Q, K and V"
        if point.kind == "sage"
        else ""
    )
    return (
        f"scripts/calibrate_wan_working_memory.py ({date}): {fresh_points.runtime_provenance(facts)}. Random-init Wan "
        f"2.2 {model} transformer with the real geometry and {point.blocks} blocks, {timesteps}, a {TEXT_TOKENS}-token "
        f"caption{sage}; memory does not depend on weight values. A fresh process with the server's runtime, the larger "
        "demand of a cold and a warm forward; no failed allocation, allocator retry or spill."
    )


# The measuring process. Everything below imports torch, so it runs only after `fresh_points.apply_server_runtime`.


def _transformer(geometry: str, blocks: int) -> "torch.nn.Module":
    import torch
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    with torch.device("meta"):
        return WanTransformer3DModel(
            patch_size=(1, 2, 2),
            attention_head_dim=128,
            text_dim=TEXT_DIM,
            freq_dim=256,
            num_layers=blocks,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=1024,
            **_CONFIGS["a14b" if geometry == "a14b" else "ti2v"],
        )


def _materialize(module: "torch.nn.Module", device: "torch.device", dtype: "torch.dtype") -> "torch.nn.Module":
    import torch

    module = module.to(dtype)
    # As `from_pretrained` loads them: the norms, the timestep embedder and the modulation tables stay in float32.
    for name, child in module.named_modules():
        if any(part in name.split(".") for part in module._keep_in_fp32_modules):
            child.to(torch.float32)
    module = module.to_empty(device=device)
    for param in module.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return module.eval()


def _holding_sage_copies():
    """`F.scaled_dot_product_attention` holding, through each call, what SageAttention's kernels allocate beside it."""
    import contextlib

    import torch
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention

    def sdpa(query, key, value, *args, **kwargs):
        held = (
            torch.empty_like(query, dtype=torch.int8),
            torch.empty_like(key, dtype=torch.int8),
            torch.empty_like(value, dtype=torch.float16),
            torch.empty_like(value, dtype=torch.int8),
        )
        out = original(query, key, value, *args, **kwargs)
        del held
        return out

    @contextlib.contextmanager
    def scope():
        F.scaled_dot_product_attention = sdpa
        try:
            yield
        finally:
            F.scaled_dot_product_attention = original

    return scope()


def _measure(point: Point, force: bool) -> dict[str, Any]:
    import contextlib

    import torch

    from invokeai.backend.model_manager.taxonomy import ModelFormat, WanVariantType
    from invokeai.backend.util.devices import TorchDevice
    from invokeai.backend.wan import working_memory
    from invokeai.backend.wan.working_memory import estimate_wan_denoise_working_memory
    from invokeai.backend.working_memory.measurement import measure_cold_and_warm
    from invokeai.backend.working_memory.platform import platform_for

    device = TorchDevice.choose_torch_device()
    if device.type != "cuda":
        raise SystemExit(f"Needs a CUDA or ROCm device; this process would run on {device}.")
    dtype = getattr(torch, point.dtype)
    channels = _CONFIGS["a14b" if point.geometry == "a14b" else "ti2v"]["in_channels"]
    shape = torch.Size((1, channels, point.frames, point.latent_height, point.latent_width))

    with torch.inference_mode():
        estimate = estimate_wan_denoise_working_memory(
            # The point runs a randomly initialized dense model, which dequantizes nothing per forward.
            torch.nn.Module(),
            variant=WanVariantType.T2V_A14B if point.geometry == "a14b" else WanVariantType.TI2V_5B,
            latent_shape=shape,
            token_timesteps=point.geometry == "ti2v_token_timesteps",
            model_format=ModelFormat.Diffusers,
            device=device,
            dtype=dtype,
        )
        platform = platform_for(device)
        facts = fresh_points.runtime_facts(device, platform.os, working_memory.__file__)
        common = {
            "estimator": estimate.estimator,
            "version": estimate.version,
            # The spec is keyed by no platform trait: its points name none (the card and runtime are in the provenance).
            "platform": {},
            "features": dict(estimate.features),
            "device": facts["device"],
        }

        transformer = _transformer(point.geometry, point.blocks)
        weight_bytes = sum(param.numel() for param in transformer.parameters()) * dtype.itemsize
        free, _ = TorchDevice.cuda_mem_get_info(device)
        raw = {
            "reservation_bytes": estimate.reservation_bytes,
            "weight_bytes": weight_bytes,
            "predicted_bytes": estimate.reservation_bytes + weight_bytes,
            "free_bytes": free,
            "runtime": facts,
        }
        if raw["predicted_bytes"] > free and not force:
            return fresh_points.skipped_line(point, common, raw)

        transformer = _materialize(transformer, device, dtype)
        hidden_states = torch.randn(shape, device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(1, TEXT_TOKENS, TEXT_DIM, device=device, dtype=dtype)
        if point.geometry == "ti2v_token_timesteps":
            timestep = torch.full((1, point.tokens), 500.0, device=device, dtype=dtype)
        else:
            timestep = torch.full((1,), 500.0, device=device)

        def forward() -> object:
            holding = _holding_sage_copies() if point.kind == "sage" else contextlib.nullcontext()
            with holding:
                return transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

        windows = measure_cold_and_warm(device, forward)
    date = datetime.date.today().isoformat()
    return fresh_points.point_line(
        point, common=common, provenance=provenance(point, facts, date), windows=windows, raw=raw
    )


_COLUMNS = f"{'kind':>9} {'geometry':>20} {'dtype':>8} {'blocks':>6} {'latent':>12} {'tokens':>7}"


def _grid(point: Mapping[str, Any]) -> str:
    latent = f"{point['frames']}x{point['latent_height']}x{point['latent_width']}"
    tokens = point["frames"] * (point["latent_height"] // 2) * (point["latent_width"] // 2)
    return (
        f"{point['kind']:>9} {point['geometry']:>20} {point['dtype']:>8} {point['blocks']:>6} {latent:>12} {tokens:>7}"
    )


def _head(line: Mapping[str, Any]) -> str:
    return _grid(line["point"])


def _summarize(lines: Sequence[Mapping[str, Any]]) -> None:
    measured = [line for line in lines if line["status"] == "point"]
    for line in measured:
        point, tokens = line["point"], line["features"]["tokens"]
        print(
            f"{point['kind']}, {point['geometry']}, {point['dtype']}, {point['blocks']} blocks, {tokens} tokens: "
            f"{line['measured_bytes'] / GIB:.3f} GiB, {line['measured_bytes'] / tokens / MIB:.4f} MiB per token; "
            f"estimate {line['raw']['reservation_bytes'] / GIB:.3f} GiB"
        )
    for line in (line for line in measured if line["point"]["kind"] == "sage"):
        reference = next(
            (
                ref
                for ref in measured
                if ref["point"]["kind"] == "reference" and ref["point"]["geometry"] == line["point"]["geometry"]
            ),
            None,
        )
        if reference is not None:
            print(
                f"sage copies, {line['point']['geometry']}: {line['measured_bytes'] / reference['measured_bytes']:.3f}x "
                "the plain forward's demand"
            )


def _dtypes(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names or not set(names) <= set(DTYPES):
        raise argparse.ArgumentTypeError(f"dtypes must be among {', '.join(DTYPES)}, got {text!r}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fit", "sage"), default="fit", help="Default: fit.")
    parser.add_argument("--out", type=Path, help="The JSON-lines file the lines are appended to. Required to measure.")
    parser.add_argument("--dtypes", type=_dtypes, default=",".join(DTYPES), help="Default: all three.")
    parser.add_argument("--blocks", type=int, default=2, help="Transformer blocks. Default 2.")
    parser.add_argument("--extra-blocks", type=int, default=0, help="Fit mode: this block count at A14B 720p x81 too.")
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
    points = plan(args.mode, dtypes=args.dtypes, blocks=args.blocks, extra_blocks=args.extra_blocks)
    if args.dry_run:
        print(_COLUMNS)
        for point in points:
            print(_grid(dataclasses.asdict(point)))
        print(f"{len(points)} forwards, one process each")
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

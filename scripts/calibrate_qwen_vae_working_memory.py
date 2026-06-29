"""Calibrate the Qwen Image VAE working-memory estimate against measured peak CUDA/HIP memory.

Background
----------
``estimate_vae_working_memory_qwen_image`` models peak working memory as a linear function of
spatial area::

    working_memory = h * w * element_size * scaling_constant

This script measures the *actual* peak reserved memory the VAE consumes during decode/encode across
a grid of resolutions so the ``scaling_constant`` can be fit from several points instead of one, and
so we can check whether the pure-linear model holds or whether a super-linear (attention) term
appears at high resolution.

The estimate is consumed by the model cache via ``free >= estimate`` to decide whether to evict, so
it MUST be an upper bound: we measure peak *reserved* (not just allocated) memory, the conservative
quantity that includes caching-allocator overhead and kernel scratch/workspace.

Portability
-----------
Backend-agnostic: uses only ``torch.cuda.*``, which works on both NVIDIA/CUDA and AMD/ROCm (HIP)
builds of PyTorch. Run the SAME script on each backend and compare the ``implied_constant`` columns
-- the curve *shape* (linear vs. super-linear) is architectural and should match, but the absolute
constant can differ (cuDNN vs. MIOpen conv workspaces, flash-attention availability, allocator
rounding). Ship ``max`` across backends plus headroom.

Each (operation, resolution) point is measured in a FRESH SUBPROCESS so the caching allocator's
fragmentation history from earlier points cannot contaminate the reserved-delta reading. A point
that OOMs is recorded as ``oom`` rather than aborting the run, so the grid can probe up to the
card's ceiling safely.

Usage
-----
    python scripts/calibrate_qwen_vae_working_memory.py [--vae /path/to/vae_dir] [--csv out.csv]

If ``--vae`` is omitted, the script auto-discovers an ``AutoencoderKLQwenImage`` under
``$INVOKEAI_ROOT/models``.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

LATENT_SCALE_FACTOR = 8

# (height, width) pixel-space resolutions. Squares to test linearity in area, plus non-square
# points (incl. the original 1248x832 calibration point) to confirm area = h*w is the right
# predictor rather than max(h, w) or perimeter. Subprocess isolation + OOM capture means we can
# list aggressive resolutions; ones that don't fit are simply recorded as oom.
DEFAULT_RESOLUTIONS = [
    (512, 512),
    (768, 768),
    (832, 1248),  # original single calibration point (as HxW)
    (1024, 1024),
    (1088, 1920),
    (1280, 1280),
    (1536, 1024),
    (1536, 1536),
    (1792, 1792),
    (2048, 2048),
]


def discover_vae() -> Path:
    """Find an AutoencoderKLQwenImage VAE directory under $INVOKEAI_ROOT/models."""
    root = os.environ.get("INVOKEAI_ROOT")
    if not root:
        raise SystemExit("INVOKEAI_ROOT not set; pass --vae explicitly.")
    models = Path(root) / "models"
    for config_path in models.glob("*/vae/config.json"):
        try:
            cfg = json.loads(config_path.read_text())
        except Exception:
            continue
        if cfg.get("_class_name") == "AutoencoderKLQwenImage":
            return config_path.parent
    raise SystemExit(f"No AutoencoderKLQwenImage VAE found under {models}; pass --vae explicitly.")


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _build_input(operation: str, h: int, w: int, z_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Construct the 5D (B, C, num_frames, H, W) input the invocation feeds the VAE.

    decode: latents at latent resolution (H/8, W/8) with z_dim channels.
    encode: image at pixel resolution (H, W) with 3 channels.
    These mirror QwenImageLatentsToImageInvocation / QwenImageImageToLatentsInvocation exactly.
    """
    device = torch.device("cuda")
    if operation == "decode":
        return torch.randn(
            1, z_dim, 1, h // LATENT_SCALE_FACTOR, w // LATENT_SCALE_FACTOR, device=device, dtype=dtype
        )
    return torch.randn(1, 3, 1, h, w, device=device, dtype=dtype)


@torch.inference_mode()
def measure_one(vae_path: str, operation: str, h: int, w: int, dtype: torch.dtype) -> dict:
    """Measure peak reserved-memory growth for a single decode/encode. Runs in a child process."""
    vae = AutoencoderKLQwenImage.from_pretrained(vae_path, local_files_only=True, torch_dtype=dtype)
    vae.to("cuda")
    vae.disable_tiling()  # Qwen invocations never tile; match that.

    param = next(vae.parameters())
    dtype = param.dtype
    element_size = param.element_size()
    z_dim = int(vae.config.z_dim)

    x = _build_input(operation, h, w, z_dim, dtype)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_reserved = torch.cuda.memory_reserved()

    # Measure the COLD first call -- it includes conv-algorithm-search / attention workspace
    # allocation, which is exactly what the real (single-shot) invocation pays.
    try:
        if operation == "decode":
            vae.decode(x, return_dict=False)
        else:
            vae.encode(x).latent_dist.mode()
        torch.cuda.synchronize()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        return {"operation": operation, "h": h, "w": w, "oom": True}

    peak_reserved = torch.cuda.max_memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    reserved_delta = peak_reserved - baseline_reserved

    area = h * w
    return {
        "operation": operation,
        "h": h,
        "w": w,
        "area": area,
        "element_size": element_size,
        "dtype": str(dtype),
        "reserved_delta": reserved_delta,
        "allocated_peak": peak_allocated,
        "reserved_baseline": baseline_reserved,
        # The constant as the estimator parameterizes it: mem = area * element_size * k
        "implied_constant": reserved_delta / (area * element_size),
        "oom": False,
    }


def run_grid(vae_path: str, resolutions: list[tuple[int, int]], dtype_name: str, csv_path: Path | None) -> None:
    rows: list[dict] = []
    print(f"VAE: {vae_path}")
    print(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)} | hip={torch.version.hip} | dtype={dtype_name}\n")
    print(f"{'op':6} {'HxW':>11} {'area':>10} {'reserved(GiB)':>14} {'alloc(GiB)':>11} {'implied_k':>10}")
    print("-" * 70)

    for operation in ("decode", "encode"):
        for h, w in resolutions:
            # Fresh subprocess per point for an uncontaminated reserved-memory reading.
            proc = subprocess.run(
                [sys.executable, __file__, "--single", operation, str(h), str(w),
                 "--vae", vae_path, "--dtype", dtype_name],
                capture_output=True,
                text=True,
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            try:
                row = json.loads(line)
            except Exception:
                print(f"{operation:6} {f'{h}x{w}':>11}  FAILED: {proc.stderr.strip().splitlines()[-1:]}")
                continue
            rows.append(row)
            if row.get("oom"):
                print(f"{operation:6} {f'{h}x{w}':>11} {h * w:>10}  {'OOM':>14}")
                continue
            gib = 1024**3
            print(
                f"{operation:6} {f'{h}x{w}':>11} {row['area']:>10} "
                f"{row['reserved_delta'] / gib:>14.3f} {row['allocated_peak'] / gib:>11.3f} "
                f"{row['implied_constant']:>10.1f}"
            )

    # Summary: the shippable constant is the MAX implied constant over fitting points (upper bound).
    print("\n=== summary (max implied constant = candidate scaling_constant, before headroom) ===")
    for operation in ("decode", "encode"):
        ks = [r["implied_constant"] for r in rows if r["operation"] == operation and not r.get("oom")]
        if ks:
            print(f"{operation:6}: n={len(ks)}  min_k={min(ks):.1f}  max_k={max(ks):.1f}  "
                  f"-> use >= {max(ks):.0f} (+headroom)")

    if csv_path:
        import csv

        fieldnames = [
            "operation", "h", "w", "area", "element_size", "dtype",
            "reserved_delta", "allocated_peak", "reserved_baseline", "implied_constant", "oom",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nWrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", type=str, default=None, help="Path to an AutoencoderKLQwenImage diffusers dir.")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write the raw results as CSV.")
    parser.add_argument(
        "--dtype",
        choices=list(DTYPES),
        default="float16",
        help="Compute dtype. Default float16 to match InvokeAI's default precision on CUDA/ROCm.",
    )
    # Internal: measure a single point in this process and print one JSON line.
    parser.add_argument("--single", nargs=3, metavar=("OP", "H", "W"), default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    vae_path = args.vae or str(discover_vae())
    dtype = DTYPES[args.dtype]

    if args.single:
        op, h, w = args.single[0], int(args.single[1]), int(args.single[2])
        print(json.dumps(measure_one(vae_path, op, h, w, dtype)))
        return

    run_grid(vae_path, DEFAULT_RESOLUTIONS, args.dtype, Path(args.csv) if args.csv else None)


if __name__ == "__main__":
    main()

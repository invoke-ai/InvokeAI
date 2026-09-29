#!/usr/bin/env python
"""Calibrate `estimate_vae_working_memory_flux` against the FLUX.1 autoencoder on this machine.

Why this exists
---------------
The model cache consumes the estimate as ``free >= estimate`` and reserves that much before the
VAE runs, so the estimate **must be an upper bound**: too low and the cache keeps other models
resident and the generation dies in its last step; far too high and it evicts a transformer it did
not need to. Neither failure is visible from the code.

This is the third calibrator of its family and the only one that was never checked in --
``calibrate_qwen_vae_working_memory.py`` covers ``AutoencoderKLQwenImage`` and
``calibrate_wan_vae_working_memory.py`` covers the Wan VAE. Modelled on them deliberately: same
subprocess isolation, same reserved-delta method, same output shape, so the numbers are comparable.

What it measures that its ancestors did not
-------------------------------------------
The **tiled** rows. FLUX.1 gained a tiled encode when its VAE became a diffusers ``AutoencoderKL``,
and the estimator's tiled branch adds a residual term for what tiling does not bound -- the input
image and the assembled moments for an encode, several copies of the assembled image for a decode.
Those residuals are derived from shapes rather than fitted, so the thing worth checking is not a
constant but the estimator's own answer, which is what this compares against.

Runs on CUDA and on ROCm/HIP unchanged (``torch.cuda.*`` covers both). Two cautions on ROCm/Windows,
both measured on an RX 9060 XT: ``max_memory_reserved`` counts host memory the driver silently pages
into, so an over-budget row reports an inflated peak instead of failing; and the fused SDPA backends
are unavailable there, so the VAE's mid-block attention materialises its score matrix and an untiled
3072px encode asks for 81 GiB rather than the 18.8 GiB a 4090 needs.

Usage
-----
    python scripts/calibrate_flux_vae_working_memory.py --vae /path/to/ae.safetensors
    python scripts/calibrate_flux_vae_working_memory.py                     # auto-discover
    python scripts/calibrate_flux_vae_working_memory.py --quick --tiles 512
    python scripts/calibrate_flux_vae_working_memory.py --csv out.csv

Please paste the whole output, including the header and the verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

DEFAULT_RESOLUTIONS = [(512, 512), (1024, 1024), (1536, 1536), (2048, 2048), (3072, 3072)]
QUICK_RESOLUTIONS = [(1024, 1024), (2048, 2048)]
LATENT_SCALE = 8


def discover_vae() -> Path | None:
    """The FLUX VAE under ``INVOKEAI_ROOT``, if there is exactly one obvious candidate."""
    root = os.environ.get("INVOKEAI_ROOT")
    if not root:
        return None
    candidates = sorted(Path(root).rglob("ae.safetensors")) + sorted(
        Path(root).rglob("vae/diffusion_pytorch_model.safetensors")
    )
    return candidates[0] if candidates else None


def _worker(vae_path: str, operation: str, h: int, w: int, dtype_name: str, tile_size: str) -> None:
    import torch

    from invokeai.backend.model_manager.load.model_loaders.flux import FluxVAELoader
    from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling
    from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    tile = None if tile_size == "none" else int(tile_size)

    # Through the product's own loader, so this measures what a generation runs -- including the
    # key conversion, and without reaching for a config over the network.
    from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_FLUX_Config

    loader = FluxVAELoader.__new__(FluxVAELoader)
    loader._torch_dtype = dtype
    loader._torch_device = torch.device("cuda")
    # `__new__` skips `__init__`, so the logger the loader may reach for is not there otherwise.
    loader._logger = logging.getLogger("calibrate_flux_vae")
    # `model_construct` skips validation: the loader reads only `path`, and a calibration run has no
    # model record to take the rest of the fields from.
    vae = loader._load_model(VAE_Checkpoint_FLUX_Config.model_construct(path=vae_path)).to("cuda").eval()

    z = vae.config.latent_channels
    if operation == "decode":
        x = torch.randn(1, z, h // LATENT_SCALE, w // LATENT_SCALE, device="cuda", dtype=dtype)
    else:
        x = torch.randn(1, 3, h, w, device="cuda", dtype=dtype)

    estimate = estimate_vae_working_memory_flux(operation, x, vae, tile_size=tile)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_reserved = torch.cuda.memory_reserved()

    with torch.inference_mode(), scoped_vae_tiling(vae, tile):
        if operation == "decode":
            out = vae.decode(x, return_dict=False)[0]
        else:
            # `.sample(generator)`, not `.mean`: that is what the nodes call, and it allocates a
            # noise tensor and a std tensor the mean path never touches.
            out = vae.encode(x).latent_dist.sample(torch.Generator(device="cuda").manual_seed(0))
    torch.cuda.synchronize()

    delta = max(0, torch.cuda.max_memory_reserved() - before_reserved)
    print(
        "RESULT "
        + json.dumps(
            {
                "operation": operation,
                "h": h,
                "w": w,
                "tile": tile,
                "element_size": next(vae.parameters()).element_size(),
                "reserved_delta": delta,
                "peak_alloc": torch.cuda.max_memory_allocated(),
                "estimate": estimate,
                "headroom": (estimate / delta) if delta else None,
            }
        )
    )
    del out, x, vae


def measure(vae_path: str, operation: str, h: int, w: int, dtype: str, tile: int | None) -> dict:
    """One row, in its own process: peak reserved is process-global and would otherwise carry over."""
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", vae_path, operation, str(h), str(w), dtype, str(tile or "none")],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    err = proc.stderr.lower()
    base = {"operation": operation, "h": h, "w": w, "tile": tile}
    if "out of memory" in err:
        return {**base, "oom": True}
    return {**base, "error": (proc.stderr.strip().splitlines() or ["?"])[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=6, help=argparse.SUPPRESS)
    ap.add_argument("--vae", help="FLUX VAE checkpoint: ae.safetensors or the diffusers-layout file")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tiles", type=int, default=512, help="Tile size for the tiled rows; 0 to skip them")
    ap.add_argument("--csv")
    args = ap.parse_args()

    if args.worker:
        _worker(
            args.worker[0], args.worker[1], int(args.worker[2]), int(args.worker[3]), args.worker[4], args.worker[5]
        )
        return

    import torch

    discovered = discover_vae()
    vae_path = args.vae or (str(discovered) if discovered else None)
    if not vae_path:
        sys.exit("No VAE found. Pass --vae /path/to/ae.safetensors.")
    if not torch.cuda.is_available():
        sys.exit("No GPU visible.")

    prop = torch.cuda.get_device_properties(0)
    kind = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
    print("=" * 96)
    print("FLUX.1 VAE working-memory calibration")
    print("=" * 96)
    print(f"  python   {sys.version.split()[0]}   {platform.platform()}")
    print(f"  torch    {torch.__version__}   cuda={torch.version.cuda}  hip={getattr(torch.version, 'hip', None)}")
    print(f"  device   {prop.name}   {kind}   {prop.total_memory / 2**30:.1f} GiB")
    print(f"  vae      {vae_path}")
    print(f"  dtype    {args.dtype}")
    print()
    print(f"  {'op':7}{'HxW':>12}{'tile':>7}{'measured GiB':>14}{'estimate GiB':>14}{'headroom':>10}   verdict")
    print("  " + "-" * 90)

    tiles: list[int | None] = [None] + ([args.tiles] if args.tiles else [])
    rows: list[dict] = []
    for operation in ("decode", "encode"):
        for h, w in QUICK_RESOLUTIONS if args.quick else DEFAULT_RESOLUTIONS:
            for tile in tiles:
                row = measure(vae_path, operation, h, w, args.dtype, tile)
                rows.append(row)
                label = f"  {operation:7}{f'{h}x{w}':>12}{str(tile or '-'):>7}"
                if row.get("oom"):
                    print(f"{label}{'OOM':>14}")
                    continue
                if row.get("error"):
                    print(f"{label}   failed: {row['error'][:50]}")
                    continue
                headroom = row["headroom"]
                verdict = "OK" if headroom and headroom >= 1.0 else "** UNDER-RESERVED **"
                print(
                    f"{label}{row['reserved_delta'] / 2**30:>14.2f}{row['estimate'] / 2**30:>14.2f}"
                    f"{headroom if headroom else 0:>10.2f}   {verdict}"
                )

    print()
    print("-" * 96)
    print("VERDICT")
    print("-" * 96)
    measured = [r for r in rows if "headroom" in r and r["headroom"]]
    under = [r for r in measured if r["headroom"] < 1.0]
    if under:
        print(f"  {len(under)} row(s) reserve LESS than the VAE actually used:")
        for r in under:
            print(f"    {r['operation']} {r['h']}x{r['w']} tile={r['tile']}  headroom {r['headroom']:.2f}x")
        print()
        print("  The cache under-reserves here, so it keeps other models resident and the VAE can")
        print("  OOM in the last step of a generation. The constants in `vae_working_memory.py` need")
        print("  a backend branch, the way `estimate_vae_working_memory_qwen_image` already has one.")
    else:
        worst = min(measured, key=lambda r: r["headroom"]) if measured else None
        if worst:
            print(
                f"  Every row is an upper bound. Tightest: {worst['operation']} {worst['h']}x{worst['w']} "
                f"tile={worst['tile']} at {worst['headroom']:.2f}x."
            )
        print("  If a decode still OOMs on this machine, the cause is elsewhere -- report these")
        print("  numbers anyway, they are the control case.")

    if args.csv:
        import csv as _csv

        keys = [
            "operation",
            "h",
            "w",
            "tile",
            "element_size",
            "reserved_delta",
            "peak_alloc",
            "estimate",
            "headroom",
            "oom",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  wrote {args.csv}")
    print("\nPlease paste this entire output, header included, into the issue or chat.")


if __name__ == "__main__":
    main()

"""Benchmark the MiniMax H3 video VAE *encode* path: float32 vs the float16-autocast recipe.

Reference videos (Ref2VA) and keyframes (FL2VA) are encoded through the VAE's convolutional
encoder one 17-frame chunk at a time, spatially tiled. ``vae_encode_autocast`` runs those
convolutions under float16 autocast on CUDA-type devices (NVIDIA and ROCm). This script times
the real ``encode_reference_video`` path both ways on a synthetic clip, reports peak memory and
the numeric drift between the two encodes, and projects the cost of a full-length reference.

    python scripts/benchmark_minimax_h3_vae_encode.py --vae /path/to/MiniMax-H3-components/vae
    python scripts/benchmark_minimax_h3_vae_encode.py --root ~/invokeai   # finds the VAE in the models dir

The same ROCm conv3d handling the model loader applies (the conv2d decomposition on every HIP
version) is applied here, and the active path is printed; ``--conv3d decomposed``
or ``--conv3d native`` forces one path so the two can be A/B'd on the same HIP version. The decoder is not
loaded onto the device — it is a ~4.5 GiB ViT that the encode never touches.

Measured on an RTX 5060 Ti (torch 2.13, cuDNN, TF32 convs on) at the default 768x448 x 39
frames: 4.6 s/chunk float32 -> 3.2 s/chunk autocast; with TF32 off, 9.2 -> 3.2 s/chunk.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

# The server sets this at startup (see run_app.py); a standalone script has to do it itself or the
# warm-up encode spends minutes in MIOpen's exhaustive kernel search on ROCm. Inert elsewhere.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Direct script execution puts ``scripts/`` on sys.path, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invokeai.backend.minimax_h3 import keyframe_conditioning, reference_conditioning  # noqa: E402
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import (  # noqa: E402
    AutoencoderKLMiniMaxH3,
    MiniMaxH3VideoCausalConv3d,
)
from invokeai.backend.minimax_h3.reference_conditioning import (  # noqa: E402
    encode_reference_video,
    snap_reference_num_frames,
)
from invokeai.backend.minimax_h3.rocm_causal_conv3d import (  # noqa: E402
    _SENTINEL as ROCM_DECOMPOSITION_SENTINEL,
)
from invokeai.backend.minimax_h3.rocm_causal_conv3d import (  # noqa: E402
    _patch_minimax_h3_causal_conv3d,
    patch_minimax_h3_causal_conv3d_for_rocm,
)


def _find_vae_in_root(root: Path) -> Path:
    """Locate the H3 components install's ``vae/`` folder under an InvokeAI root's models dir."""
    candidates = []
    for config in (root / "models").glob("*/vae/config.json"):
        try:
            if json.loads(config.read_text()).get("_class_name") == "AutoencoderKLMiniMaxH3":
                candidates.append(config.parent)
        except (OSError, ValueError):
            continue
    if not candidates:
        raise SystemExit(f"No AutoencoderKLMiniMaxH3 folder found under {root / 'models'}; pass --vae explicitly.")
    if len(candidates) > 1:
        print(f"Several H3 VAE folders found, using the first: {[str(c) for c in candidates]}")
    return candidates[0]


_STOCK_FORWARD = MiniMaxH3VideoCausalConv3d.forward  # captured before any patch


def _set_conv_path(path: str) -> None:
    """Force the encoder's conv3d implementation: 'native' (stock forward) or 'decomposed'."""
    if path == "decomposed":
        _patch_minimax_h3_causal_conv3d()
        return
    MiniMaxH3VideoCausalConv3d.forward = _STOCK_FORWARD
    if hasattr(MiniMaxH3VideoCausalConv3d, ROCM_DECOMPOSITION_SENTINEL):
        delattr(MiniMaxH3VideoCausalConv3d, ROCM_DECOMPOSITION_SENTINEL)


def _conv_path_description() -> str:
    decomposed = getattr(MiniMaxH3VideoCausalConv3d, ROCM_DECOMPOSITION_SENTINEL, False)
    if torch.version.hip is None:
        return "conv2d decomposition (forced)" if decomposed else "cuDNN conv3d (CUDA build)"
    return f"HIP {torch.version.hip}: {'conv2d decomposition' if decomposed else 'native MIOpen conv3d'}"


def _run(vae: AutoencoderKLMiniMaxH3, frames: np.ndarray, device: torch.device, repeats: int, mode: str) -> dict:
    def once() -> tuple[torch.Tensor, float]:
        start = time.perf_counter()
        rows = encode_reference_video(vae, frames, device)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        return rows, time.perf_counter() - start

    # Warm-up: kernel selection (MIOpen find can take minutes per new shape set), allocator growth.
    _, warm = once()
    print(f"{mode:>8}: warm-up encode {warm:.1f} s", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    rows = None
    for index in range(repeats):
        rows, elapsed = once()
        times.append(elapsed)
        print(f"{mode:>8}: repeat {index + 1}/{repeats} {elapsed:.1f} s", flush=True)
    assert rows is not None
    result = {"rows": rows, "min_s": min(times), "max_s": max(times)}
    if device.type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--vae", type=Path, help="The H3 components install's vae/ folder (diffusers layout).")
    source.add_argument("--root", type=Path, help="An InvokeAI root; the VAE is located under its models dir.")
    parser.add_argument("--width", type=int, default=768, help="Reference canvas width (default 768).")
    parser.add_argument("--height", type=int, default=448, help="Reference canvas height (default 448).")
    parser.add_argument(
        "--frames",
        type=int,
        default=39,
        help="Frames in the synthetic clip; snapped down to 17n+5, minimum 22 (default 39).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions per mode (default 3).")
    parser.add_argument("--device", default="cuda", help="Torch device (default cuda).")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["fp32", "autocast"],
        default=["fp32", "autocast"],
        help="Which encodes to time, in this order (default: fp32 then autocast; drift needs both).",
    )
    parser.add_argument(
        "--no-tf32", action="store_true", help="Disable TF32 convolutions (NVIDIA only; ROCm has no TF32 path)."
    )
    parser.add_argument(
        "--conv3d",
        choices=["auto", "native", "decomposed", "both"],
        default="auto",
        help=(
            "conv3d path: 'auto' does what the model loader does (the conv2d decomposition on ROCm, stock conv3d "
            "elsewhere); 'decomposed' / 'native' force one path on any build; 'both' runs every mode on native "
            "first, then decomposed, and reports the numeric drift between the two paths (a slow native path "
            "makes this long - shrink the clip, e.g. --width 256 --height 256 --frames 22, the minimum)."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False

    vae_path = args.vae if args.vae is not None else _find_vae_in_root(args.root.expanduser())
    if args.conv3d == "auto":
        patch_minimax_h3_causal_conv3d_for_rocm()
    elif args.conv3d == "decomposed":
        _set_conv_path("decomposed")
    vae = AutoencoderKLMiniMaxH3.from_pretrained(vae_path, local_files_only=True).eval()
    vae.decoder = torch.nn.Identity()  # the ViT decoder is never used by an encode
    vae = vae.to(device)

    num_frames = snap_reference_num_frames(args.frames)
    clip_length = int(vae.config.clip_length)
    num_chunks = -(-num_frames // clip_length)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(num_frames, args.height, args.width, 3), dtype=np.uint8)
    tiles_y = len(vae._split_tiles(args.height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)[0])
    tiles_x = len(vae._split_tiles(args.width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)[0])

    print(f"torch {torch.__version__}, device {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(device)})", end="")
    print(f", conv path: {_conv_path_description()}")
    if torch.version.hip is None and device.type == "cuda":
        print(f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")
    print(
        f"clip: {num_frames} frames at {args.width}x{args.height} -> {num_chunks} chunks of {clip_length}, "
        f"{tiles_y}x{tiles_x} tiles per chunk; {args.repeats} timed repeats per mode after one warm-up",
        flush=True,
    )

    real_autocast = keyframe_conditioning.vae_encode_autocast

    def run_modes(label: str) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for mode in args.modes:
            if mode == "fp32":
                keyframe_conditioning.vae_encode_autocast = lambda device: contextlib.nullcontext()
                reference_conditioning.vae_encode_autocast = keyframe_conditioning.vae_encode_autocast
            else:
                keyframe_conditioning.vae_encode_autocast = real_autocast
                reference_conditioning.vae_encode_autocast = real_autocast
            with torch.inference_mode():
                results[mode] = _run(vae, frames, device, args.repeats, f"{label}{mode}")
            r = results[mode]
            line = (
                f"{label}{mode:>8}: {r['min_s']:.2f} s (max {r['max_s']:.2f} s) for {num_chunks} chunks = "
                f"{r['min_s'] / num_chunks:.2f} s/chunk"
            )
            if "peak_allocated_gib" in r:
                line += f"; peak allocated {r['peak_allocated_gib']:.2f} GiB, reserved {r['peak_reserved_gib']:.2f} GiB"
            print(line, flush=True)

        if "fp32" in results and "autocast" in results:
            rows32, rows16 = results["fp32"]["rows"], results["autocast"]["rows"]
            diff = (rows32 - rows16).abs()
            print(
                f"{label}speedup {results['fp32']['min_s'] / results['autocast']['min_s']:.2f}x; drift autocast vs "
                f"fp32: rel-norm {(diff.norm() / rows32.norm()).item():.2e}, max {diff.max().item():.3f} "
                f"(row std {rows32.std().item():.3f}); finite: {bool(torch.isfinite(rows16).all())}"
            )
        # A 10 s reference at 24 fps snaps to 226 frames = 14 chunks.
        for mode, r in results.items():
            print(f"{label}projected 10 s reference ({mode}): {14 * r['min_s'] / num_chunks:.0f} s", flush=True)
        return results

    if args.conv3d != "both":
        run_modes("")
        return

    by_path: dict[str, dict[str, dict]] = {}
    for path in ("native", "decomposed"):
        _set_conv_path(path)
        print(f"--- conv path: {_conv_path_description()}", flush=True)
        by_path[path] = run_modes(f"[{path}] ")
    for mode in args.modes:
        native, decomposed = by_path["native"][mode]["rows"], by_path["decomposed"][mode]["rows"]
        diff = (native - decomposed).abs()
        print(
            f"{mode}: decomposed vs native drift rel-norm {(diff.norm() / native.norm()).item():.2e}, "
            f"max {diff.max().item():.3f} (row std {native.std().item():.3f}); "
            f"finite: {bool(torch.isfinite(decomposed).all())}; "
            f"decomposed speedup {by_path['native'][mode]['min_s'] / by_path['decomposed'][mode]['min_s']:.2f}x"
        )


if __name__ == "__main__":
    main()

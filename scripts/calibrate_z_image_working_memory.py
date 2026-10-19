"""Calibrate the Z-Image denoise working-memory estimate against measured peak CUDA/HIP memory.

Background
----------
The Z-Image transformer attends over one unified sequence -- image tokens and caption tokens, each padded to a
multiple of 32 -- through diffusers' `dispatch_attention_fn`. Three numbers decide how much VRAM the model cache has
to keep free for one transformer forward (``ZImageDenoiseInvocation._estimate_working_memory``):

1. the activation cost per attended token on a fused SDPA kernel, which scales linearly with the sequence;
2. the materialized score matrix, only where SDPA has no fused kernel for these shapes -- ROCm/Windows without
   the experimental AOTriton kernels, MPS -- which `sdpa_score_matrix_bytes` charges at the shipped
   `SDPA_MATH_BYTES_PER_SCORE_ELEMENT`;
3. with a ControlNet attached, its hints: one per control block, live next to the transformer's own activations
   until the last layer the control adds to.

This measures all three on the real geometry (dim 3840, 30 heads of 128, caption features 2560 wide) with randomly
initialized models and a reduced transformer block count: memory depends on shapes, not on weight values, and a
no-grad forward frees each block's intermediates. `--extra-blocks` re-checks that on the build at hand. The ControlNet
point runs the control through the denoise's hooks with its hints on consecutive main blocks: they are all computed as
the first main block starts and dropped after the last one they land on, so where in between they land does not move
the peak, and packing them keeps the transformer at one block per hint. Every point runs in a fresh subprocess, so the
caching allocator's history cannot contaminate the peak-reserved reading, and a point that runs out of memory is
recorded as such rather than aborting the run.

Peak *reserved*, not allocated: the estimate is compared against free memory, so it has to bound what the
allocator actually takes off the card.

Every point is priced before it runs -- the node's own estimate for it, the forced-math score matrix where that is
the point, and the weights -- and skipped when that exceeds free device memory. An oversized allocation does not
always fail: where the driver backs it with system RAM (NVIDIA's sysmem fallback, AMD on Windows) it pages and can
stall the whole machine. On a build whose default kernel materializes the score matrix, that skips every size above
1024px. `--force` runs them anyway.

Usage
-----
    python scripts/calibrate_z_image_working_memory.py
    python scripts/calibrate_z_image_working_memory.py --extra-blocks 4 --control-variant 2.1-lite
    python scripts/calibrate_z_image_working_memory.py --max-px 1024 --control-variant none
"""

import argparse
import dataclasses
import json
import subprocess
import sys
from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.z_image.z_image_denoise import (
    Z_IMAGE_ATTENTION_HEADS,
    Z_IMAGE_HIDDEN_SIZE,
    ZImageDenoiseInvocation,
    _padded,
)
from invokeai.backend.fun_controlnet.z_image import (
    Z_IMAGE_FUN_CONTROL_1_0,
    Z_IMAGE_FUN_CONTROL_2_0,
    Z_IMAGE_FUN_CONTROL_2_1,
    Z_IMAGE_FUN_CONTROL_2_1_LITE,
    ZImageFunControlVariant,
)
from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT

GIB = 1024**3
MIB = 1024**2

CAP_FEAT_DIM = 2560
PATCH_SIZE = 2
REFINER_BLOCKS = 2
# One transformer block at full width: attention 4 dim^2, SwiGLU 8 dim^2, adaLN and norms, rounded up.
BLOCK_PARAMS = 13 * Z_IMAGE_HIDDEN_SIZE**2

SIZES = (1024, 1440, 2048)
TEXT_TOKENS = 512

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
CONTROL_VARIANTS: dict[str, ZImageFunControlVariant | None] = {
    "none": None,
    "1.0": Z_IMAGE_FUN_CONTROL_1_0,
    "2.0": Z_IMAGE_FUN_CONTROL_2_0,
    "2.1": Z_IMAGE_FUN_CONTROL_2_1,
    "2.1-lite": Z_IMAGE_FUN_CONTROL_2_1_LITE,
}


def image_tokens(px: int) -> int:
    side = px // LATENT_SCALE_FACTOR // PATCH_SIZE
    return side * side


def unified_tokens(px: int, text_tokens: int) -> int:
    return _padded(image_tokens(px)) + _padded(text_tokens)


def control_blocks(variant: ZImageFunControlVariant | None) -> int:
    return len(variant.main_layers) if variant is not None else 0


def base_blocks(blocks: int, variant: ZImageFunControlVariant | None) -> int:
    """The main blocks a point builds: one per hint at least, which land on consecutive blocks."""
    return max(blocks, control_blocks(variant))


def packed(variant: ZImageFunControlVariant) -> ZImageFunControlVariant:
    """``variant`` with its hints on main blocks 0, 1, 2, ...: the same peak on the fewest blocks."""
    return dataclasses.replace(variant, main_layers=tuple(range(control_blocks(variant))))


def predicted_bytes(
    px: int, blocks: int, dtype: torch.dtype, force_math: bool, variant: ZImageFunControlVariant | None
) -> int:
    """What a point is expected to take: working memory as the node estimates it, plus the weights."""
    working = ZImageDenoiseInvocation._estimate_working_memory(
        image_seq_len=image_tokens(px),
        text_seq_len=TEXT_TOKENS,
        num_loras=0,
        num_control_blocks=control_blocks(variant),
        device=torch.device("cuda"),
        dtype=dtype,
    )
    if force_math:
        tokens = unified_tokens(px, TEXT_TOKENS)
        working += Z_IMAGE_ATTENTION_HEADS * tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
    # Main blocks plus refiners and embedders; the control's blocks plus its refiners.
    modules = base_blocks(blocks, variant) + 2 * REFINER_BLOCKS
    if variant is not None:
        modules += control_blocks(variant) + (REFINER_BLOCKS if variant.refiner_chain != "main" else 0)
    return working + modules * BLOCK_PARAMS * dtype.itemsize


def _peak_reserved(fn) -> int | None:
    """Run ``fn`` and return the growth in peak reserved bytes, or ``None`` if it ran out of memory."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_reserved()
    try:
        fn()
        torch.cuda.synchronize()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        return None
    return torch.cuda.max_memory_reserved() - baseline


def _control(variant: ZImageFunControlVariant, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """A randomly initialized control, allocated directly in ``dtype`` on the device: 17 blocks are ~3B parameters,
    which would otherwise pass through the host in float32 first."""
    from invokeai.backend.fun_controlnet.z_image import ZImageFunControl

    with torch.device("meta"):
        control = ZImageFunControl(
            variant, hidden_size=Z_IMAGE_HIDDEN_SIZE, num_attention_heads=Z_IMAGE_ATTENTION_HEADS
        )
    control = control.to(dtype).to_empty(device=device)
    for param in control.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return control.eval()


@torch.inference_mode()
def measure(px: int, text_tokens: int, blocks: int, dtype: torch.dtype, force_math: bool, variant_name: str) -> dict:
    """Peak reserved memory for one transformer forward at one resolution."""
    from diffusers import ZImageTransformer2DModel

    from invokeai.backend.fun_controlnet.hooks import install_fun_control

    variant = CONTROL_VARIANTS[variant_name]
    device = torch.device("cuda")
    model = (
        ZImageTransformer2DModel(
            all_patch_size=(PATCH_SIZE,),
            all_f_patch_size=(1,),
            in_channels=16,
            dim=Z_IMAGE_HIDDEN_SIZE,
            n_layers=base_blocks(blocks, variant),
            n_refiner_layers=REFINER_BLOCKS,
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
        .to(device=device, dtype=dtype)
        .eval()
    )
    latent = px // LATENT_SCALE_FACTOR
    x = [torch.randn(16, 1, latent, latent, device=device, dtype=dtype)]
    t = torch.full((1,), 0.5, device=device, dtype=dtype)
    cap_feats = [torch.randn(text_tokens, CAP_FEAT_DIM, device=device, dtype=dtype)]

    control = None
    if variant is not None:
        control = _control(packed(variant), device, dtype)
        context = torch.randn(1, image_tokens(px), control.control_in_dim, device=device, dtype=dtype)

    def forward() -> None:
        with (
            sdpa_kernel(SDPBackend.MATH) if force_math else nullcontext(),
            install_fun_control(model, control, context, scale=0.75) if control is not None else nullcontext(),
        ):
            model(x=x, t=t, cap_feats=cap_feats)

    row = {
        "px": px,
        "tokens": unified_tokens(px, text_tokens),
        "blocks": blocks,
        "force_math": force_math,
        "control_variant": variant_name,
    }
    # Warm up first: the cold call pays one-off workspace costs the per-token slope must not absorb.
    if _peak_reserved(forward) is None:
        return row | {"oom": True}
    peak = _peak_reserved(forward)
    return row | ({"oom": True} if peak is None else {"oom": False, "reserved_delta": peak})


def _run_point(px: int, blocks: int, dtype_name: str, force_math: bool, variant_name: str) -> dict | None:
    args = [str(px), str(TEXT_TOKENS), str(blocks), dtype_name, "1" if force_math else "0", variant_name]
    proc = subprocess.run([sys.executable, __file__, "--single", *args], capture_output=True, text=True)
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except Exception:
        tail = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
        print(f"  FAILED {' '.join(args)}: {tail[0]}")
        return None


def report(dtype_name: str, blocks: int, extra_blocks: int | None, variant_name: str, max_px: int, force: bool) -> None:
    dtype = DTYPES[dtype_name]
    sizes = [px for px in SIZES if px <= max_px]
    free, _ = torch.cuda.mem_get_info()
    print(f"{'kernel':>7} {'blocks':>6} {'control':>7} {'px':>5} {'tokens':>7} {'reserved(GiB)':>14}")
    print("-" * 52)

    def run(kernel: str, count: int, px: int, force_math: bool, control: str) -> int | None:
        need = predicted_bytes(px, count, dtype, force_math, CONTROL_VARIANTS[control])
        if need > free and not force:
            print(
                f"{kernel:>7} {count:>6} {control:>7} {px:>5} {'':>7} {'skipped':>14}"
                f"  needs ~{need / GIB:.1f} GiB, {free / GIB:.1f} GiB free"
            )
            return None
        row = _run_point(px, count, dtype_name, force_math, control)
        if row is None or row.get("oom"):
            print(f"{kernel:>7} {count:>6} {control:>7} {px:>5} {'':>7} {'OOM':>14}")
            return None
        print(f"{kernel:>7} {count:>6} {control:>7} {px:>5} {row['tokens']:>7} {row['reserved_delta'] / GIB:>14.3f}")
        return row["reserved_delta"]

    counts = [blocks] + ([extra_blocks] if extra_blocks else [])
    fused: dict[int, dict[int, int]] = {}
    for count in counts:
        for px in sizes:
            if (peak := run("auto", count, px, False, "none")) is not None:
                fused.setdefault(count, {})[px] = peak
    math_peak = run("math", blocks, SIZES[0], True, "none")
    variant = CONTROL_VARIANTS[variant_name]
    control_peak = run("auto", blocks, SIZES[0], False, variant_name) if variant is not None else None

    print()
    for count, peaks in fused.items():
        for px, peak in peaks.items():
            tokens = unified_tokens(px, TEXT_TOKENS)
            print(f"auto kernel, {count} blocks, {px}px: {peak / tokens / MIB:.4f} MB per attended token")
        if len(peaks) >= 2:
            lo, hi = min(peaks), max(peaks)
            slope = (peaks[hi] - peaks[lo]) / (unified_tokens(hi, TEXT_TOKENS) - unified_tokens(lo, TEXT_TOKENS))
            base = peaks[lo] - slope * unified_tokens(lo, TEXT_TOKENS)
            print(f"auto kernel, {count} blocks: {slope / MIB:.4f} MB/token, intercept {base / GIB:.3f} GiB")

    tokens = unified_tokens(SIZES[0], TEXT_TOKENS)
    reference = fused.get(blocks, {}).get(SIZES[0])
    if math_peak is not None and reference is not None:
        extra = math_peak - reference
        charged = Z_IMAGE_ATTENTION_HEADS * tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        print(
            f"math kernel, {SIZES[0]}px: {extra / GIB:.3f} GiB over the auto kernel, "
            f"{extra / (Z_IMAGE_ATTENTION_HEADS * tokens * tokens):.2f} bytes/score element; shipped constant charges "
            f"{charged / GIB:.3f} GiB ({SDPA_MATH_BYTES_PER_SCORE_ELEMENT})"
        )
    if control_peak is not None and reference is not None:
        extra = control_peak - reference
        hint_bytes = tokens * Z_IMAGE_HIDDEN_SIZE * dtype.itemsize
        count = control_blocks(variant)
        print(
            f"control {variant_name}, {count} blocks, {SIZES[0]}px: {extra / GIB:.3f} GiB over no control, "
            f"{extra / hint_bytes:.2f} hint-sized tensors; the estimate charges n + 4 = {count + 4}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument("--blocks", type=int, default=2, help="Main transformer blocks. Default 2.")
    parser.add_argument("--extra-blocks", type=int, default=None, help="Measure a second block count as well.")
    parser.add_argument(
        "--control-variant", choices=list(CONTROL_VARIANTS), default="2.1", help="ControlNet layout. Default 2.1."
    )
    parser.add_argument("--max-px", type=int, default=SIZES[-1], help="Largest fused-kernel size to measure.")
    parser.add_argument("--force", action="store_true", help="Run points predicted not to fit free device memory.")
    parser.add_argument("--single", nargs="*", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/HIP device available.")
    if args.single:
        px, text_tokens, blocks, dtype_name, force_math, variant_name = args.single
        row = measure(int(px), int(text_tokens), int(blocks), DTYPES[dtype_name], force_math == "1", variant_name)
        print(json.dumps(row))
        return
    print(
        f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)} | hip={torch.version.hip} | {args.dtype}"
    )
    report(args.dtype, args.blocks, args.extra_blocks, args.control_variant, args.max_px, args.force)


if __name__ == "__main__":
    main()

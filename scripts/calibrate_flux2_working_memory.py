"""Calibrate the FLUX.2 working-memory estimates against measured peak CUDA/HIP memory.

Background
----------
Four constants decide how much VRAM the model cache keeps free for a FLUX.2 operation, and every one
of them was fitted on CUDA:

1. ``SDPA_MATH_BYTES_PER_SCORE_ELEMENT`` (``backend/util/attention.py``) -- bytes per element of the
   score matrix, charged only where SDPA has no fused kernel and materializes it.
2. ``estimate_vae_working_memory_flux2``'s 2200 / 1100 bytes per pixel per element byte
   (``backend/util/vae_working_memory.py``).
3. ``Flux2DenoiseInvocation._estimate_working_memory``'s 0.4 MB per token, defined at the Klein 9B
   width (4096) and scaled linearly by the loaded variant's width.
4. The dispatch question underneath all of it: does this build's SDPA fuse these shapes, or does it
   build the score matrix?

The estimate is consumed by the model cache via ``free >= estimate`` to decide what to evict, so it
MUST be an upper bound: this measures peak *reserved* (not merely allocated) memory, the conservative
quantity that includes caching-allocator overhead and kernel scratch.

Why this script exists
----------------------
ROCm answers (1), (2) and (4) differently from CUDA -- its fused kernels cap the head dim at 128, so
the FLUX.2 VAE's 512-wide mid-block head falls back to ``math`` and the score matrix becomes a real,
dominant term. The constants shipped today are known to be short on ROCm for VAE decode in the middle
of the resolution range. Recalibrating needs the hardware, so this puts the whole measurement in one
runnable file: run it on an AMD card and paste the output.

Portability
-----------
Backend-agnostic: only ``torch.cuda.*``, which is the same API on NVIDIA/CUDA and AMD/ROCm (HIP)
builds. Run the SAME script on each backend and compare. The curve *shape* is architectural and
should match; the absolute constants can differ (cuDNN vs MIOpen workspaces, fused-kernel
availability, allocator rounding). Ship the max across backends, plus headroom.

No checkpoints required. Every measurement uses a randomly initialized model at the real geometry --
memory depends on shapes, not on weight values, and a stock-config ``AutoencoderKLFlux2`` reproduces
the real-weight 1024px decode measurement to within 2%. Pass ``--vae`` to measure a specific VAE (the
small-decoder variant has different ``block_out_channels``).

Each point is measured in a FRESH SUBPROCESS so the caching allocator's fragmentation history cannot
contaminate the reserved-delta reading. A point that OOMs is recorded as ``oom`` rather than aborting
the run, so the grid can probe up to the card's ceiling safely.

Usage
-----
    python scripts/calibrate_flux2_working_memory.py
    python scripts/calibrate_flux2_working_memory.py --only vae --csv flux2_rocm.csv
    python scripts/calibrate_flux2_working_memory.py --only denoise --max-px 1024
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from invokeai.app.invocations.flux2_denoise import (
    FLUX2_ATTENTION_HEAD_DIM,
    FLUX2_REFERENCE_HIDDEN_SIZE,
    Flux2DenoiseInvocation,
)
from invokeai.backend.model_manager.configs.flux2_variant import flux2_hidden_size
from invokeai.backend.model_manager.taxonomy import Flux2VariantType
from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT, sdpa_score_matrix_bytes
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux2

GIB = 1024**3
MIB = 1024**2

# The FLUX.2 VAE attends on the 8x-downsampled grid with a single 512-wide head. Mirrors
# `_FLUX2_VAE_*` in vae_working_memory.py.
VAE_SPATIAL_COMPRESSION = 8
VAE_MID_BLOCK_HEAD_DIM = 512
LATENT_SCALE_FACTOR = 8

# (variant, transformer hidden size, context_in_dim). The source of truth is
# `model_manager/configs/flux2_variant.py`; `_check_variant_table` asserts we have not drifted.
VARIANTS = [
    (Flux2VariantType.Klein4B, 3072, 7680),
    (Flux2VariantType.Klein9B, 4096, 12288),
    (Flux2VariantType.Dev, 6144, 15360),
]

DEFAULT_VAE_PX = [512, 768, 1024, 1280, 1536]

# (heads, seq, head_dim) for the score-matrix constant. Spans the shapes the two estimators actually
# produce: the VAE's single 512-wide head, and the transformer's many 128-wide ones.
DEFAULT_SDPA_SHAPES = [
    (1, 4096, 512),
    (1, 8192, 512),
    (1, 16384, 512),
    (4, 4096, 128),
    (48, 4608, 128),
]

# Sequence lengths the denoise slope is taken between. 4608 is a plain 1024x1024 generation
# (4096 image + 512 text); 9216 is that doubled, which is also what a batch of 2 costs.
DENOISE_SEQ_SHORT = 4608
DENOISE_SEQ_LONG = 9216
DENOISE_TEXT_TOKENS = 512

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# `SDPBackend` is a pybind11 enum and is not iterable, so name it by hand.
SDP_BACKEND_NAMES = {
    int(getattr(SDPBackend, name)): name
    for name in ("ERROR", "MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "OVERRIDEABLE")
    if hasattr(SDPBackend, name)
}


def _check_variant_table() -> None:
    """Fail loudly if the widths hard-coded above ever drift from the model manager's table."""
    for variant, hidden, _ in VARIANTS:
        actual = flux2_hidden_size(variant)
        if actual != hidden:
            raise SystemExit(f"Variant table is stale: {variant.value} is {actual}, not {hidden}.")


def _peak_reserved(fn) -> int | None:
    """Run ``fn`` and return the growth in peak reserved bytes, or ``None`` if it OOMed.

    Reserved rather than allocated: that is the quantity the estimate has to bound, because it is
    what the allocator actually takes off the card.
    """
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


# --------------------------------------------------------------------------------------------
# 1. Dispatch: what does this build's SDPA actually do with the shapes the estimators produce?
# --------------------------------------------------------------------------------------------


def measure_dispatch(dtype: torch.dtype) -> list[dict]:
    """Report the kernel torch would pick for each shape. Costs nothing; explains everything else.

    This is the table that decides whether the score-matrix term is charged at all, and it is where
    build-specific assumptions go to die -- an earlier revision of this feature assumed ROCm rejects
    additive masks, which gfx1100 disproves.
    """
    device = torch.device("cuda")
    rows = []
    for head_dim in (128, 512):
        for has_mask in (False, True):
            q = torch.empty((1, 1, 8, head_dim), device=device, dtype=dtype)
            mask = torch.empty((1, 1, 8, 8), device=device, dtype=dtype) if has_mask else None
            try:
                choice = int(torch.ops.aten._fused_sdp_choice(q, q, q, mask, 0.0, False))
                name = SDP_BACKEND_NAMES.get(choice, str(choice))
            except Exception as e:  # noqa: BLE001 - the failure itself is the datum
                name = f"raised ({type(e).__name__})"
            rows.append({"head_dim": head_dim, "mask": has_mask, "choice": name})
    return rows


# --------------------------------------------------------------------------------------------
# 2. SDPA_MATH_BYTES_PER_SCORE_ELEMENT
# --------------------------------------------------------------------------------------------


@torch.inference_mode()
def measure_sdpa(num_heads: int, seq_len: int, head_dim: int, dtype: torch.dtype) -> dict:
    """Peak reserved bytes per element of a materialized score matrix, with MATH forced."""
    device = torch.device("cuda")
    q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def run() -> None:
        with sdpa_kernel([SDPBackend.MATH]):
            F.scaled_dot_product_attention(q, k, v)

    peak = _peak_reserved(run)
    row = {"num_heads": num_heads, "seq_len": seq_len, "head_dim": head_dim, "oom": peak is None}
    if peak is not None:
        elements = num_heads * seq_len * seq_len
        row |= {"reserved_delta": peak, "bytes_per_element": peak / elements}
    return row


# --------------------------------------------------------------------------------------------
# 3. The VAE's linear constants
# --------------------------------------------------------------------------------------------


def _load_vae(vae_path: str | None, dtype: torch.dtype):
    from diffusers import AutoencoderKLFlux2

    if vae_path:
        return AutoencoderKLFlux2.from_pretrained(vae_path, local_files_only=True, torch_dtype=dtype)
    # Weight values do not affect activation memory, only shapes do; the stock config is the real
    # FLUX.2 VAE geometry (32 latent channels, block_out_channels ending at 512).
    return AutoencoderKLFlux2().to(dtype=dtype)


@torch.inference_mode()
def measure_vae(operation: str, px: int, dtype: torch.dtype, force_math: bool, vae_path: str | None) -> dict:
    """Peak reserved memory for one untiled decode/encode, against what the estimator predicts.

    ``force_math`` runs the mid-block attention through SDPA's ``math`` fallback even where a fused
    kernel exists, so a CUDA box can approximate the regime ROCm is in permanently. It is not a
    substitute for measuring on ROCm -- the two are not equivalent, which is itself worth showing.
    """
    device = torch.device("cuda")
    vae = _load_vae(vae_path, dtype).to(device).eval()
    vae.disable_tiling()  # the decode/encode invocations do not tile; match them.

    param = next(vae.parameters())
    element_size = param.element_size()
    if operation == "decode":
        latent_channels = int(vae.config.latent_channels)
        x = torch.randn(1, latent_channels, px // LATENT_SCALE_FACTOR, px // LATENT_SCALE_FACTOR, **_td(device, dtype))
    else:
        x = torch.randn(1, 3, px, px, **_td(device, dtype))

    def run() -> None:
        ctx = sdpa_kernel([SDPBackend.MATH]) if force_math else _null_context()
        with ctx:
            if operation == "decode":
                vae.decode(x, return_dict=False)
            else:
                vae.encode(x, return_dict=False)[0].mode()

    peak = _peak_reserved(run)
    row = {
        "operation": operation,
        "px": px,
        "force_math": force_math,
        "element_size": element_size,
        "oom": peak is None,
    }
    if peak is None:
        return row

    estimate = estimate_vae_working_memory_flux2(operation=operation, image_tensor=x, vae=vae, device=device)
    # The estimate is linear_term + score_matrix. Back the score term out so the linear constant can
    # be fitted on its own -- it is the one the 2200/1100 literals name.
    seq_len = (px // VAE_SPATIAL_COMPRESSION) ** 2
    score_bytes = sdpa_score_matrix_bytes(
        device=device, dtype=param.dtype, num_heads=1, head_dim=VAE_MID_BLOCK_HEAD_DIM, seq_len=seq_len
    )
    if force_math and score_bytes == 0:
        # Fused here, so the estimator charged nothing -- but we forced math, so price it anyway.
        score_bytes = seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        estimate += score_bytes

    return row | {
        "reserved_delta": peak,
        "estimate": estimate,
        "score_term": score_bytes,
        # Only meaningful while the peak actually exceeds the score term. Where it does not, the
        # additive (linear + score) model does not decompose on this build and the number is noise.
        "implied_linear_constant": ((peak - score_bytes) / (px * px * element_size)) if peak > score_bytes else None,
        "covered": peak <= estimate,
    }


# --------------------------------------------------------------------------------------------
# 4. The denoise per-token constant, and its width scaling
# --------------------------------------------------------------------------------------------


@torch.inference_mode()
def measure_denoise(hidden: int, context_dim: int, seq_len: int, blocks: int, dtype: torch.dtype) -> dict:
    """Peak reserved memory for one transformer forward at a given width and sequence length.

    Uses a reduced block count on purpose: the per-token cost is block-count independent, because a
    no-grad forward frees each block's intermediates as it goes. ``--blocks`` measures a second point
    so that assumption can be re-checked on this build rather than inherited.
    """
    from diffusers import Flux2Transformer2DModel

    device = torch.device("cuda")
    model = (
        Flux2Transformer2DModel(
            num_layers=blocks,
            num_single_layers=blocks,
            num_attention_heads=hidden // FLUX2_ATTENTION_HEAD_DIM,
            attention_head_dim=FLUX2_ATTENTION_HEAD_DIM,
            joint_attention_dim=context_dim,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    img_tokens = seq_len - DENOISE_TEXT_TOKENS
    kwargs = {
        "hidden_states": torch.randn(1, img_tokens, 128, **_td(device, dtype)),
        "encoder_hidden_states": torch.randn(1, DENOISE_TEXT_TOKENS, context_dim, **_td(device, dtype)),
        "timestep": torch.full((1,), 0.5, **_td(device, dtype)),
        "img_ids": torch.zeros(img_tokens, 4, **_td(device, dtype)),
        "txt_ids": torch.zeros(DENOISE_TEXT_TOKENS, 4, **_td(device, dtype)),
        "guidance": torch.full((1,), 4.0, **_td(device, dtype)),
        "return_dict": False,
    }

    # Warm up first, then drop the allocator's cache: the cold call pays one-off weight-cast and
    # workspace costs that the per-token slope must not absorb. The slope is a difference between
    # two of these, so any constant overhead cancels either way -- warming just reduces the noise.
    try:
        model(**kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        return {"hidden": hidden, "seq_len": seq_len, "blocks": blocks, "oom": True}

    peak = _peak_reserved(lambda: model(**kwargs))
    row = {"hidden": hidden, "seq_len": seq_len, "blocks": blocks, "oom": peak is None}
    if peak is not None:
        row |= {"reserved_delta": peak}
    return row


# --------------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------------


def _td(device: torch.device, dtype: torch.dtype) -> dict:
    return {"device": device, "dtype": dtype}


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _run_point(args: list[str]) -> dict | None:
    """Run one measurement in a fresh subprocess and return its JSON row."""
    proc = subprocess.run([sys.executable, __file__, "--single", *args], capture_output=True, text=True)
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except Exception:
        tail = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
        print(f"  FAILED {' '.join(args)}: {tail[0]}")
        return None


def report_dispatch(dtype_name: str) -> list[dict]:
    print("\n=== 1. SDPA dispatch: which kernel would this build pick? ===")
    print("The score-matrix term is charged only where the answer is MATH (or the query raises).\n")
    rows = measure_dispatch(DTYPES[dtype_name]) or []
    print(f"{'head_dim':>9} {'mask':>6} {'kernel':>22}")
    print("-" * 40)
    for r in rows:
        print(f"{r['head_dim']:>9} {str(r['mask']):>6} {r['choice']:>22}")
    print("\n  head_dim 512 is the FLUX.2 VAE mid-block; 128 is the transformer, and the masked row")
    print("  is regional prompting. MATH on the 512 row means the VAE estimate needs the score term.")
    return rows


def report_sdpa(shapes: list[tuple[int, int, int]], dtype_name: str) -> list[dict]:
    print("\n=== 2. SDPA_MATH_BYTES_PER_SCORE_ELEMENT ===")
    print(f"Peak reserved per score element with MATH forced. Shipped constant: {SDPA_MATH_BYTES_PER_SCORE_ELEMENT}.\n")
    print(f"{'heads':>6} {'seq':>7} {'head_dim':>9} {'reserved(GiB)':>14} {'bytes/elem':>11}")
    print("-" * 52)
    rows = []
    for heads, seq, head_dim in shapes:
        row = _run_point(["sdpa", str(heads), str(seq), str(head_dim), dtype_name])
        if row is None:
            continue
        rows.append(row)
        if row.get("oom"):
            print(f"{heads:>6} {seq:>7} {head_dim:>9} {'OOM':>14}")
            continue
        print(
            f"{heads:>6} {seq:>7} {head_dim:>9} {row['reserved_delta'] / GIB:>14.3f} {row['bytes_per_element']:>11.2f}"
        )
    fitted = [r["bytes_per_element"] for r in rows if not r.get("oom")]
    if fitted:
        worst = max(fitted)
        verdict = "OK" if SDPA_MATH_BYTES_PER_SCORE_ELEMENT >= worst else "SHORT"
        print(f"\n  max = {worst:.2f}; shipped constant is {SDPA_MATH_BYTES_PER_SCORE_ELEMENT} -> {verdict}")
    return rows


def report_vae(pxs: list[int], dtype_name: str, vae_path: str | None) -> list[dict]:
    print("\n=== 3. VAE linear constants (2200 decode / 1100 encode) ===")
    print("`implied_k` backs the score-matrix term out, so it is comparable to those literals; fit the")
    print("constant on the rows whose `math` column matches what this build really does (section 1).")
    print("`covered` is the question that matters: is the shipped estimate an upper bound here?")
    print("Caveat: forcing math on a build that HAS a fused kernel is not equivalent to a build that")
    print("has none -- on CUDA the forced-math decode measures *below* the fused one, because the")
    print("memory-efficient kernel's workspace is the larger term there. Only a real run on the")
    print("materializing build calibrates it.\n")
    print(
        f"{'op':7} {'px':>5} {'math':>5} {'measured(GiB)':>14} {'estimate(GiB)':>14} {'implied_k':>10} {'covered':>8}"
    )
    print("-" * 70)
    rows = []
    for operation in ("decode", "encode"):
        for px in pxs:
            for force_math in (False, True):
                args = ["vae", operation, str(px), dtype_name, "1" if force_math else "0"]
                if vae_path:
                    args.append(vae_path)
                row = _run_point(args)
                if row is None:
                    continue
                rows.append(row)
                if row.get("oom"):
                    print(f"{operation:7} {px:>5} {str(force_math):>5} {'OOM':>14}")
                    continue
                k = row["implied_linear_constant"]
                print(
                    f"{operation:7} {px:>5} {str(force_math):>5} {row['reserved_delta'] / GIB:>14.3f} "
                    f"{row['estimate'] / GIB:>14.3f} {(f'{k:.0f}' if k else 'n/a'):>10} "
                    f"{('yes' if row['covered'] else 'NO'):>8}"
                )
    print("")
    for operation, shipped in (("decode", 2200), ("encode", 1100)):
        for force_math in (False, True):
            ks = [
                r["implied_linear_constant"]
                for r in rows
                if r["operation"] == operation
                and r["force_math"] is force_math
                and not r.get("oom")
                and r["implied_linear_constant"]
            ]
            if not ks:
                continue
            mode = "math" if force_math else "fused"
            verdict = "OK" if shipped >= max(ks) else "SHORT"
            print(f"  {operation} ({mode}): implied_k max = {max(ks):.0f}, shipped = {shipped} -> {verdict}")
    short = [r for r in rows if not r.get("oom") and not r["covered"]]
    if short:
        print("\n  Points the shipped estimate does NOT cover:")
        for r in short:
            gap = (r["reserved_delta"] - r["estimate"]) / GIB
            print(f"    {r['operation']} {r['px']}px force_math={r['force_math']}: short by {gap:.2f} GiB")
    return rows


def report_denoise(dtype_name: str, blocks: int, extra_blocks: int | None) -> list[dict]:
    print("\n=== 4. Denoise per-token constant and its width scaling ===")
    print(f"Shipped: {0.4:.1f} MB/token at hidden={FLUX2_REFERENCE_HIDDEN_SIZE}, scaled linearly by width.\n")
    print(
        f"{'variant':10} {'hidden':>7} {'blocks':>7} {'short(GiB)':>11} {'long(GiB)':>10} {'MB/token':>9} {'ratio':>7}"
    )
    print("-" * 68)
    rows = []
    slopes: dict[int, float] = {}
    block_counts = [blocks] + ([extra_blocks] if extra_blocks else [])
    for variant, hidden, context_dim in VARIANTS:
        for nb in block_counts:
            pair = []
            for seq in (DENOISE_SEQ_SHORT, DENOISE_SEQ_LONG):
                row = _run_point(["denoise", str(hidden), str(context_dim), str(seq), str(nb), dtype_name])
                if row is None or row.get("oom"):
                    pair = []
                    break
                rows.append(row)
                pair.append(row["reserved_delta"])
            if not pair:
                print(f"{variant.value:10} {hidden:>7} {nb:>7} {'OOM':>11}")
                continue
            slope = (pair[1] - pair[0]) / (DENOISE_SEQ_LONG - DENOISE_SEQ_SHORT)
            if nb == blocks:
                slopes[hidden] = slope
            print(
                f"{variant.value:10} {hidden:>7} {nb:>7} {pair[0] / GIB:>11.3f} {pair[1] / GIB:>10.3f} "
                f"{slope / MIB:>9.4f} {slope / slopes.get(FLUX2_REFERENCE_HIDDEN_SIZE, slope):>7.3f}"
            )

    if extra_blocks:
        print(f"\n  The two block counts should agree per width; if they do not, the '{blocks} blocks stand in")
        print("  for the real model' assumption does not hold on this build and the rest is suspect.")

    reference = slopes.get(FLUX2_REFERENCE_HIDDEN_SIZE)
    if reference:
        shipped = 0.4 * MIB
        verdict = "OK" if shipped >= reference else "SHORT"
        print(
            f"\n  hidden={FLUX2_REFERENCE_HIDDEN_SIZE}: measured {reference / MIB:.4f} MB/token, shipped 0.4 -> {verdict}"
        )
        for hidden, slope in sorted(slopes.items()):
            print(
                f"  hidden={hidden}: slope ratio {slope / reference:.3f} against width ratio "
                f"{hidden / FLUX2_REFERENCE_HIDDEN_SIZE:.3f}"
            )
        print("\n  Those two columns should match: that is the claim that width scales the constant.")
        # What the full estimator would reserve for the case #9500 describes.
        for variant, hidden, _ in VARIANTS:
            est = Flux2DenoiseInvocation._estimate_working_memory(
                None, image_seq_len=4096, ref_image_seq_len=12288, text_seq_len=512, num_loras=0, hidden_size=hidden
            )
            need = slopes.get(hidden, reference) * 16896
            print(
                f"  {variant.value:10} 1024px + 3 refs: estimate {est / GIB:5.2f} GiB, "
                f"measured slope implies {need / GIB:5.2f} GiB"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        choices=["dispatch", "sdpa", "vae", "denoise"],
        action="append",
        help="Run only these sections (repeatable). Default: all four.",
    )
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16", help="Compute dtype. Default bfloat16.")
    parser.add_argument("--max-px", type=int, default=None, help="Skip VAE resolutions above this.")
    parser.add_argument("--vae", type=str, default=None, help="Optional AutoencoderKLFlux2 diffusers dir.")
    parser.add_argument("--blocks", type=int, default=2, help="Transformer blocks per stream. Default 2.")
    parser.add_argument(
        "--extra-blocks",
        type=int,
        default=None,
        help="Measure a second block count too, to re-check block-count independence on this build.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Write the raw rows to CSV.")
    # Internal: measure one point in this process and print one JSON line.
    parser.add_argument("--single", nargs="*", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/HIP device available.")
    _check_variant_table()

    if args.single:
        kind, rest = args.single[0], args.single[1:]
        if kind == "sdpa":
            heads, seq, head_dim, dtype_name = int(rest[0]), int(rest[1]), int(rest[2]), rest[3]
            print(json.dumps(measure_sdpa(heads, seq, head_dim, DTYPES[dtype_name])))
        elif kind == "vae":
            operation, px, dtype_name, force_math = rest[0], int(rest[1]), rest[2], rest[3] == "1"
            vae_path = rest[4] if len(rest) > 4 else None
            print(json.dumps(measure_vae(operation, px, DTYPES[dtype_name], force_math, vae_path)))
        elif kind == "denoise":
            hidden, context_dim, seq, blocks, dtype_name = (
                int(rest[0]),
                int(rest[1]),
                int(rest[2]),
                int(rest[3]),
                rest[4],
            )
            print(json.dumps(measure_denoise(hidden, context_dim, seq, blocks, DTYPES[dtype_name])))
        else:
            raise SystemExit(f"unknown point kind {kind}")
        return

    sections = args.only or ["dispatch", "sdpa", "vae", "denoise"]
    print(
        f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)} | "
        f"hip={torch.version.hip} | dtype={args.dtype}"
    )
    pxs = [p for p in DEFAULT_VAE_PX if args.max_px is None or p <= args.max_px]

    rows: list[dict] = []
    if "dispatch" in sections:
        rows += [{"section": "dispatch", **r} for r in report_dispatch(args.dtype)]
    if "sdpa" in sections:
        rows += [{"section": "sdpa", **r} for r in report_sdpa(DEFAULT_SDPA_SHAPES, args.dtype)]
    if "vae" in sections:
        rows += [{"section": "vae", **r} for r in report_vae(pxs, args.dtype, args.vae)]
    if "denoise" in sections:
        rows += [{"section": "denoise", **r} for r in report_denoise(args.dtype, args.blocks, args.extra_blocks)]

    if args.csv:
        import csv

        fieldnames: list[str] = []
        for r in rows:
            for key in r:
                if key not in fieldnames:
                    fieldnames.append(key)
        with Path(args.csv).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()

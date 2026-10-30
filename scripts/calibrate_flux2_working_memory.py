"""Measure what a materialized SDPA score matrix costs, and whether this build materializes one.

What it measures
----------------
`sdpa_score_matrix_bytes` (``backend/util/attention.py``) charges a working-memory estimate for the score matrix only
where SDPA has no fused kernel for the shapes and builds it, at ``SDPA_MATH_BYTES_PER_SCORE_ELEMENT`` bytes per
element. Two sections check both halves on the build it runs on:

1. ``dispatch``: which kernel this build's SDPA picks for a 128-wide head (the transformers') and a 512-wide one (the
   FLUX.2 VAE mid-block), with and without an additive mask -- the answer the estimators' probe gets.
2. ``sdpa``: peak reserved bytes per score element with the math kernel forced, over the shapes the estimators
   produce. The constant must be at least the largest; the measured table is in ``backend/util/attention.py``.

The estimators' coefficients are measured elsewhere: the denoise lines by fresh-process calibration scripts such as
``scripts/calibrate_flux2_denoise_working_memory.py``, the VAE lines by calibration runs in a running server whose
exports ``scripts/working_memory/build_shipped.py`` turns into shipped points.

Each point is measured in a fresh subprocess, so the caching allocator's history cannot contaminate the reserved-delta
reading; a point that runs out of memory is recorded as ``oom`` rather than aborting the run. Only ``torch.cuda.*`` is
used, the same API on CUDA and ROCm (HIP) builds: run it on each build and compare.

Usage
-----
    python scripts/calibrate_flux2_working_memory.py
    python scripts/calibrate_flux2_working_memory.py --only sdpa --csv sdpa_rocm.csv
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT

GIB = 1024**3


# (heads, seq, head_dim) for the score-matrix constant. Spans the shapes the two estimators actually
# produce: the VAE's single 512-wide head, and the transformer's many 128-wide ones.
DEFAULT_SDPA_SHAPES = [
    (1, 4096, 512),
    (1, 8192, 512),
    (1, 16384, 512),
    (4, 4096, 128),
    (48, 4608, 128),
]

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# Environment that materially changes a peak-*reserved* reading or which kernel runs. Reported in the
# header so a pasted result cannot be ambiguous about it later.
REPORTED_ENV_PREFIXES = (
    "MIOPEN_",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "FLASH_ATTENTION_",
    "TORCH_ROCM_",
    "MIGRAPHX_",
    "HSA_",
    "TORCH_BLAS_",
)

# `SDPBackend` is a pybind11 enum and is not iterable, so name it by hand.
SDP_BACKEND_NAMES = {
    int(getattr(SDPBackend, name)): name
    for name in ("ERROR", "MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "OVERRIDEABLE")
    if hasattr(SDPBackend, name)
}


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
# plumbing
# --------------------------------------------------------------------------------------------


def _report_environment() -> None:
    """Print the env vars that can move these numbers, and warn about the one that invalidates them.

    ``garbage_collection_threshold`` makes the allocator release cached blocks once reserved memory
    passes a fraction of the card -- which is precisely the quantity being measured here. A run with
    it set reports a *lower* peak than the operation actually needs, and the giveaway is a series
    that stops rising with resolution.
    """
    import os

    interesting = {k: v for k, v in sorted(os.environ.items()) if k.startswith(REPORTED_ENV_PREFIXES)}
    if interesting:
        print("env: " + ", ".join(f"{k}={v}" for k, v in interesting.items()))
    else:
        print("env: none of the MIOpen / allocator / flash-attention overrides are set")
    for key, value in interesting.items():
        if "garbage_collection_threshold" in value:
            print(
                f"  WARNING: {key} sets garbage_collection_threshold. That releases cached blocks "
                "once reserved memory crosses the threshold, so peak-reserved readings near the "
                "card's capacity will read LOW. Re-run without it before fitting any constant."
            )


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        choices=["dispatch", "sdpa"],
        action="append",
        help="Run only these sections (repeatable). Default: both.",
    )
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16", help="Compute dtype. Default bfloat16.")
    parser.add_argument("--csv", type=str, default=None, help="Write the raw rows to CSV.")
    # Internal: measure one point in this process and print one JSON line.
    parser.add_argument("--single", nargs="*", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/HIP device available.")

    if args.single:
        kind, rest = args.single[0], args.single[1:]
        if kind == "sdpa":
            heads, seq, head_dim, dtype_name = int(rest[0]), int(rest[1]), int(rest[2]), rest[3]
            print(json.dumps(measure_sdpa(heads, seq, head_dim, DTYPES[dtype_name])))
        else:
            raise SystemExit(f"unknown point kind {kind}")
        return

    sections = args.only or ["dispatch", "sdpa"]
    print(
        f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)} | "
        f"hip={torch.version.hip} | dtype={args.dtype}"
    )
    _report_environment()

    rows: list[dict] = []
    if "dispatch" in sections:
        rows += [{"section": "dispatch", **r} for r in report_dispatch(args.dtype)]
    if "sdpa" in sections:
        rows += [{"section": "sdpa", **r} for r in report_sdpa(DEFAULT_SDPA_SHAPES, args.dtype)]

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

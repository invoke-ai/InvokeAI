"""Triton kernels for W8A8 int8 linears: activation rotation + per-token quantization, and the int8 GEMM.

Imported only by ``int8_matmul`` on first use, never at InvokeAI import time: this module imports Triton at the top,
because ``@triton.jit`` needs it at definition time, and InvokeAI has to import on machines without Triton. Policy --
which calls run here, how rows are chunked, what happens when a kernel fails -- lives in ``int8_matmul``; this module
only computes.

For a convrot ``int8_tensorwise`` weight, stored rotated as ``W_rot = grouped_256(W) @ H`` and quantized to ``W_q``
with per-output-channel scale ``s_w``, ``y = x @ W^T`` is computed as

1. ``x' = grouped_256(x) @ H`` -- ``H`` is symmetric and orthonormal, so ``x @ W^T = x' @ W_rot^T``;
2. ``q = clamp(floor(x' / s_x + 0.5), -127, 127)`` with ``s_x = amax(|x'|) / 127`` per token (row);
3. ``y = (q @ W_q^T) * s_x[m] * s_w[n] + bias``, the GEMM in int8 with int32 accumulation, the rest in float32
   before rounding to the compute dtype once.

Every K handled here is a multiple of 256 (the rotation group), so neither kernel masks along K. Row offsets are
computed in int64: a row index times a row stride passes 2^31 for large token counts (a video transformer's batch
of ~100k tokens times a 16k-wide layer), and an int32 wrap there is an illegal memory access the fallback cannot
recover from.
"""

import threading

import torch
import triton
import triton.language as tl

ROTATION_GROUP_SIZE = 256
# Rows one program of the activation kernel quantizes. Each program reads its rows twice (amax, then quantize), so
# small blocks keep the second read in L2; 4 rows measured 0.9-1.5 TB/s on an RTX 4090.
_QUANT_ROWS = 4
# Flush buffer for autotune timing when the device does not report its L2 size.
_DEFAULT_L2_BYTES = 64 * 1024 * 1024
# Autotune time budget per config, as Triton's own `do_bench` uses (milliseconds).
_BENCH_WARMUP_MS = 25
_BENCH_REPEAT_MS = 100


@triton.jit
def _hadamard_256(x, ROWS: tl.constexpr):
    """``x @ H`` for each row of a ``[ROWS, 256]`` block, ``H`` the normalized regular Hadamard of size 256.

    ``H = H4 (x) H4 (x) H4 (x) H4 / 16`` with the regular seed ``H4 = J - 2 * antidiag``, so ``H4 @ v`` is
    ``sum(v) - 2 * flip(v)``: viewed as ``[ROWS, 4, 4, 4, 4]``, the rotation is that step along each of the four
    axes, in registers, without a matmul or a loaded matrix.
    """
    x = tl.reshape(x, (ROWS, 4, 4, 4, 4))
    x = tl.sum(x, axis=1, keep_dims=True) - 2.0 * tl.flip(x, 1)
    x = tl.sum(x, axis=2, keep_dims=True) - 2.0 * tl.flip(x, 2)
    x = tl.sum(x, axis=3, keep_dims=True) - 2.0 * tl.flip(x, 3)
    x = tl.sum(x, axis=4, keep_dims=True) - 2.0 * tl.flip(x, 4)
    return tl.reshape(x, (ROWS, 256)) * 0.0625


@triton.jit
def _nan_max(a, b):
    # A NaN anywhere in a row makes its scale NaN, so the row's output is NaN as on the dequantized path, instead of
    # the NaN being dropped by the reduction and the row coming out finite.
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _rotated_block(X, row_offsets, row_mask, k, ROWS: tl.constexpr):
    cols = k + tl.arange(0, 256)
    x = tl.load(X + row_offsets[:, None] + cols[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
    return _hadamard_256(x, ROWS)


@triton.jit
def _quantize_activations_kernel(X, Q, S, M, K, stride_xm, stride_qm, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < M
    x_rows = rows.to(tl.int64) * stride_xm
    q_rows = rows.to(tl.int64) * stride_qm
    amax = tl.zeros([ROWS], dtype=tl.float32)
    for k in range(0, K, 256):
        x = _rotated_block(X, x_rows, row_mask, k, ROWS)
        amax = _nan_max(amax, tl.reduce(tl.abs(x), 1, _nan_max))
    amax = _nan_max(amax, 1e-12)
    inverse = 127.0 / amax
    tl.store(S + rows, amax / 127.0, mask=row_mask)
    # Second pass recomputes the rotation rather than keeping K floats per row resident.
    for k in range(0, K, 256):
        x = _rotated_block(X, x_rows, row_mask, k, ROWS)
        q = tl.minimum(tl.maximum(tl.floor(x * inverse[:, None] + 0.5), -127.0), 127.0)
        tl.store(Q + q_rows[:, None] + (k + tl.arange(0, 256))[None, :], q.to(tl.int8), mask=row_mask[:, None])


def gemm_configs() -> list[triton.Config]:
    """Tiles the autotuner picks from. A tile a GPU lacks the shared memory for fails to compile with
    ``OutOfResources``, which the autotuner scores as never fastest; the largest ones sit close to the ~99 KB of
    sm_86 and sm_89."""
    return [
        triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8}, num_stages=stages, num_warps=warps)
        for bm, bn, bk, stages, warps in (
            (128, 256, 64, 3, 8),
            (256, 128, 64, 3, 8),
            (128, 128, 128, 3, 8),
            (128, 128, 64, 4, 4),
            (128, 128, 64, 4, 8),
            (64, 256, 64, 4, 4),
            (128, 64, 128, 4, 4),
            (64, 128, 128, 4, 4),
            (64, 64, 128, 4, 4),
            (32, 64, 128, 4, 4),
            (128, 256, 128, 2, 8),
            (256, 128, 128, 2, 8),
        )
    ]


@triton.jit
def _int8_gemm_kernel(
    A, W, A_SCALE, W_SCALE, BIAS, C, M, N, K, M_BUCKET, stride_am, stride_wn, stride_wk, stride_cm, stride_ws,
    HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
    # M_BUCKET is only an autotune key. Grouped launch order: GROUP_M row blocks share each weight column block while
    # it is in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    # Out-of-range rows and columns read row/column 0 instead of masking every load of the hot loop; the store
    # discards them. K is a multiple of 256 and therefore of BK.
    a_ptrs = A + tl.where(rm < M, rm, 0).to(tl.int64)[:, None] * stride_am + rk[None, :]
    w_ptrs = W + tl.where(rn < N, rn, 0).to(tl.int64)[None, :] * stride_wn + rk[:, None] * stride_wk
    acc = tl.zeros([BM, BN], dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(w_ptrs), acc, out_dtype=tl.int32)
        a_ptrs += BK
        w_ptrs += BK * stride_wk
    a_scale = tl.load(A_SCALE + rm, mask=rm < M, other=0.0)
    w_scale = tl.load(W_SCALE + rn * stride_ws, mask=rn < N, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * a_scale[:, None] * w_scale[None, :]
    if HAS_BIAS:
        out += tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    c_mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :], out.to(C.dtype.element_ty), mask=c_mask)


def _benchmark_with_l2_flush(device: torch.device):
    """Triton's ``do_bench`` timing loop, with an L2 flush buffer the size of this GPU's L2.

    Triton's own flushes with a fixed 256 MB buffer, allocated per config while a layer is being tuned -- several
    times what the working-memory estimate of a small model's largest layer leaves room for. A buffer the size of the
    L2 (72 MB on an RTX 4090) evicts it just the same.
    """
    flush_bytes = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0) or _DEFAULT_L2_BYTES

    def bench(kernel_call, quantiles):
        flush = torch.empty(flush_bytes // 4, dtype=torch.int32, device=device)
        kernel_call()
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            flush.zero_()
            kernel_call()
        end.record()
        torch.cuda.synchronize(device)
        estimate_ms = max(start.elapsed_time(end) / 5, 1e-3)
        for _ in range(max(1, int(_BENCH_WARMUP_MS / estimate_ms))):
            kernel_call()
        repeats = max(1, int(_BENCH_REPEAT_MS / estimate_ms))
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for begin, finish in zip(starts, ends, strict=True):
            flush.zero_()
            begin.record()
            kernel_call()
            finish.record()
        torch.cuda.synchronize(device)
        times = torch.tensor([begin.elapsed_time(finish) for begin, finish in zip(starts, ends, strict=True)])
        return torch.quantile(times, torch.tensor(quantiles, dtype=times.dtype)).tolist()

    return bench


# One autotuner per device. Its in-memory result cache is keyed by shape only, so a machine with two different GPUs
# would otherwise run the second on the first one's tiles; and its ``run`` keeps per-call state on the instance, which
# two sessions tuning at once on different GPUs would trample. On disk, results are keyed by GPU architecture too, so
# identical GPUs share them across restarts.
_gemm_by_device: dict[torch.device, triton.runtime.Autotuner] = {}
_gemm_lock = threading.Lock()


def _gemm_for(device: torch.device) -> triton.runtime.Autotuner:
    gemm = _gemm_by_device.get(device)
    if gemm is None:
        with _gemm_lock:
            gemm = _gemm_by_device.get(device)
            if gemm is None:
                gemm = triton.autotune(
                    configs=gemm_configs(),
                    key=["M_BUCKET", "N", "K"],
                    cache_results=True,
                    do_bench=_benchmark_with_l2_flush(device),
                )(_int8_gemm_kernel)
                _gemm_by_device[device] = gemm
    return gemm


def quantize_activations(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and quantize ``[M, K]`` rows to int8, one float32 scale per row. Rows may be strided along M."""
    m, k = x2d.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x2d.device)
    scale = torch.empty((m,), dtype=torch.float32, device=x2d.device)
    _quantize_activations_kernel[(triton.cdiv(m, _QUANT_ROWS),)](
        x2d, q, scale, m, k, x2d.stride(0), q.stride(0), ROWS=_QUANT_ROWS
    )
    return q, scale


def launch_gemm(
    kernel,
    q: torch.Tensor,
    q_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    m_bucket: int,
    **meta,
) -> None:
    """Write ``(q @ weight^T) * q_scale[:, None] * weight_scale[None, :] + bias`` into ``out``.

    ``kernel`` is the per-device autotuner, or the bare kernel with a tile in ``meta``. ``weight_scale`` holds ``N``
    per-channel values or one per-tensor value; ``out`` may be a row slice of a larger output.
    """
    m, k = q.shape
    n = weight.shape[0]
    weight_scale = weight_scale.reshape(-1)

    def grid(launch: dict) -> tuple[int]:
        return (triton.cdiv(m, launch["BM"]) * triton.cdiv(n, launch["BN"]),)

    kernel[grid](
        q, weight, q_scale, weight_scale, bias if bias is not None else weight_scale, out, m, n, k, m_bucket,
        q.stride(0), weight.stride(0), weight.stride(1), out.stride(0), 1 if weight_scale.numel() > 1 else 0,
        HAS_BIAS=bias is not None, **meta,
    )  # fmt: skip


def w8a8_rows(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    m_bucket: int,
) -> None:
    """``F.linear`` of half-precision rows ``x2d`` with a convrot int8 weight, as W8A8, written into ``out``.

    Every tensor must be on ``x2d``'s CUDA device, and ``x2d`` must have unit stride along K. ``m_bucket`` is the
    autotune key for the row count. The kernels launch on that device whichever device is current: a session worker
    pins its thread to its GPU, a single-device install with ``device: cuda:1`` does not.
    """
    if x2d.shape[-1] % ROTATION_GROUP_SIZE != 0:
        raise ValueError(f"in_features {x2d.shape[-1]} is not a multiple of {ROTATION_GROUP_SIZE}")
    with torch.cuda.device(x2d.device):
        q, q_scale = quantize_activations(x2d)
        launch_gemm(_gemm_for(x2d.device), q, q_scale, weight, weight_scale, bias, out, m_bucket)

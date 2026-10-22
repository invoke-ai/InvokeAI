"""int8 compute: W8A8 matmuls for convrot ``int8_tensorwise`` linears on NVIDIA GPUs, opt-in via ``int8_compute``.

By default an :class:`~invokeai.backend.quantization.int8_convrot.Int8ConvrotLinear` dequantizes its int8 weight to
the compute dtype on every forward and multiplies in bf16/fp16 -- on an RTX 4090 that is 0.75-0.97x the speed of a
dense bf16 layer. With ``int8_compute`` on, a layer quantized with convrot instead rotates and quantizes its
*activation* per token to int8 as well and multiplies on the int8 tensor cores (``int8_matmul_kernels``): 2.8-3.2x a
dense bf16 layer at 4608 tokens, 1.9x at 128.

That changes numerics, which is why it is opt-in: against the dense weight, W8A8 measures ~0.8-0.9 % relative output
error where the dequantized path measures ~0.3-0.4 %, so images differ from that path at the same seed. Only rotated
layers take it: the rotation spreads outlier channels across the group before the per-token scale is taken, and
unrotated layers, where one outlier channel sets the scale for the whole token, were not measured. A layer whose int8
weight partial loading left in RAM streams it to the GPU per call, as the dequantized path does, and gets the same
result as when resident.

Memory: a call whose kernels are tuned and validated never needs more than the dequantized path's per-forward transient,
``Int8ConvrotLinear.dequant_transient_bytes`` (two weight-sized tensors in the compute dtype, 4·N·K bytes). It allocates
the int8 activation copy for at most ``2·N`` rows at a time (``_chunk_rows``), 2·N·K bytes, plus the int8 weight (N·K)
when it is streamed; the output is the one ``F.linear`` would allocate too. The first call per tuning key may briefly
need a little more: tuning allocates an L2-sized flush buffer (see ``int8_matmul_kernels``), and the validation
afterwards runs ``_FIRST_CALL_ROWS`` rows through the dequantized path -- its own transient, after the kernels' streamed
weight is released -- and holds that slice of the reference output.

This module owns the policy: which calls may take the kernels, where the kernels exist, and what happens when they
fail. The kernels live in ``int8_matmul_kernels``, which imports Triton and is therefore imported here lazily, on the
first call that could use it -- InvokeAI imports and runs without Triton.

Failures never fail a generation the dequantized path could finish (see ``triton_kernels``):

* An out-of-memory error propagates like any other: outside a key's first call, the fallback would not fit either.
* A Triton cache ``OSError`` falls back for that call; ``MAX_CACHE_FAILURES`` in a row retire the device.
* Any other kernel error retires the device for int8 compute until restart, with one warning.
* The first call per autotune key compares a slice of its rows with the dequantized path and retires the device if
  they disagree by more than ``_FIRST_CALL_MAX_ERROR``: the kernels are JIT-compiled per GPU architecture and tile, so
  a miscompile would otherwise reach images silently.
"""

import functools
import importlib
import logging
import math
import threading
from collections.abc import Callable
from types import ModuleType

import torch

from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.triton_kernels import (
    TRITON_REQUIREMENT,
    FailureOutcome,
    TritonKernelHealth,
    is_missing_triton,
    relative_error,
)

logger = InvokeAILogger.get_logger(__name__)

_KERNELS_MODULE = "invokeai.backend.quantization.int8_matmul_kernels"
# The rotation the activation kernel implements in registers. Weights rotated in other group sizes fall back.
KERNEL_ROTATION_GROUP_SIZE = 256
_HALF_PRECISION = (torch.float16, torch.bfloat16)
# int8 tensor-core GEMMs through Triton: Ampere and newer. Measured on sm_89 only; older GPUs fall back.
_MIN_CAPABILITY = (8, 0)
# Largest relative L2 difference between the first kernel result of an autotune key and the dequantized path. W8A8
# measures ~0.8-0.9 % against the dense weight. A broken kernel lands near 1 or above: an all-zero output scores
# exactly 1, an unrelated one of the same magnitude ~1.4, a missing scale far more. 0.3 sits well clear of both.
_FIRST_CALL_MAX_ERROR = 0.3
# Rows of a key's first call that are recomputed on the dequantized path for that comparison, from the end so the
# last, partial tile is among them. A kernel that miscompiled garbles every tile, so a slice catches what the whole
# output would, without holding a second output-sized tensor.
_FIRST_CALL_ROWS = 256
# Autotune results are keyed by the next power of two of the row count, capped, rather than by the row count: every
# resolution inside one bucket reuses the result instead of tuning again.
_MAX_M_BUCKET = 16384

# Opt-in for the same reason as fp8 compute: an existing install would start producing different images at the same
# seed. Tests override it process-wide without a configured app.
_int8_compute_override: bool | None = None


def set_int8_compute_enabled(enabled: bool | None) -> None:
    """Override the configured setting process-wide. Pass ``None`` to revert to the config value."""
    global _int8_compute_override
    _int8_compute_override = enabled


def is_int8_compute_enabled() -> bool:
    if _int8_compute_override is not None:
        return _int8_compute_override
    try:
        from invokeai.app.services.config.config_default import get_config

        return bool(get_config().int8_compute)
    except Exception:
        # Backend code may run outside a configured app (scripts, tests). Default to the dequantized path.
        return False


_health = TritonKernelHealth()
_lock = threading.Lock()
# Held around the kernels of every tuning key not yet validated, i.e. while it may still be tuning. Two GPUs tuning the
# same key at once compile the same twelve tiles into the same Triton cache files; on Windows that collides with an
# OSError, each collision falls back and tunes again on the next layer, and three in a row retire the device.
# Serialized, the second GPU finds the first one's results and compiled kernels on disk. Once a key is validated it
# is never taken again.
_tuning_lock = threading.Lock()
# Per device: whether the kernels can run there at all, decided once (and explained once in the log).
_available: dict[torch.device, bool] = {}


@functools.cache
def _load_kernels() -> ModuleType | str:
    """The kernels module, or why it cannot be imported. Cached: an install does not gain Triton while running."""
    try:
        return importlib.import_module(_KERNELS_MODULE)
    except ModuleNotFoundError as e:
        if is_missing_triton(e):
            return f"it needs {TRITON_REQUIREMENT}"
        return f"loading its kernels failed ({e!r})"
    except Exception as e:
        return f"loading Triton failed ({e!r})"


def _unavailable_reason(device: torch.device) -> tuple[str, int] | None:
    """Why the kernels cannot run on ``device``, and how loudly to say so; None where they can."""
    if device.type != "cuda" or torch.version.hip is not None:
        return "it runs on NVIDIA GPUs only", logging.INFO
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < _MIN_CAPABILITY:
        required = ".".join(map(str, _MIN_CAPABILITY))
        return f"it needs compute capability {required} or newer, this GPU has {major}.{minor}", logging.INFO
    kernels = _load_kernels()
    # Missing or broken Triton on a GPU that could run the kernels is the one case the user can fix.
    return (kernels, logging.WARNING) if isinstance(kernels, str) else None


def _kernels_available(device: torch.device) -> bool:
    available = _available.get(device)
    if available is None:
        with _lock:
            available = _available.get(device)
            if available is None:
                unavailable = _unavailable_reason(device)
                available = unavailable is None
                _available[device] = available
                if unavailable is not None:
                    reason, level = unavailable
                    logger.log(
                        level,
                        f"int8_compute is enabled, but not used on {device}: {reason}. int8 layers there dequantize.",
                    )
    return available and not _health.is_retired(device)


def can_use_int8_matmul(x: torch.Tensor, in_features: int) -> bool:
    """Whether a forward of ``x`` through a rotated int8 layer of this width may run on the int8 kernels.

    Every condition falls back to the dequantized path rather than raising: a mid-generation failure is far worse
    than losing the speedup. Checked per forward, so the cheap ones go first. Where the weight lives does not matter:
    a streamed one is copied to ``x``'s device.
    """
    if not is_int8_compute_enabled():
        return False
    if x.dtype not in _HALF_PRECISION or x.numel() == 0:
        return False
    # A width mismatch is F.linear's to report; the kernels would read past the weight.
    if x.shape[-1] != in_features or x.stride(-1) != 1:
        return False
    # No backward, and autocast would have changed the dtype F.linear computes in.
    if torch.is_grad_enabled() and x.requires_grad:
        return False
    if torch.is_autocast_enabled(x.device.type):
        return False
    return _kernels_available(x.device)


def _m_bucket(rows: int) -> int:
    return min(1 << max(rows - 1, 0).bit_length(), _MAX_M_BUCKET)


def _chunk_rows(out_features: int) -> int:
    """Rows quantized and multiplied at once: their int8 copy (2·N·K bytes) plus a streamed int8 weight (N·K) stays
    below the dequantized path's 4·N·K, leaving a weight's worth of headroom for the scales and allocator rounding."""
    return 2 * out_features


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    fallback: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``F.linear`` of ``x`` with a convrot int8 weight on the int8 kernels, or ``fallback(x)`` where they failed.

    Call only where :func:`can_use_int8_matmul` agreed. ``fallback`` computes the same layer on the dequantized path
    for any rows of ``x``.
    """
    kernels = _load_kernels()
    assert not isinstance(kernels, str), "can_use_int8_matmul() was not consulted"
    in_features = x.shape[-1]
    try:
        rows = x.view(-1, in_features)
    except RuntimeError:
        # Leading dimensions that cannot be merged without a copy; that copy is not in the memory bound.
        return fallback(x)
    device = x.device
    out_features = weight.shape[0]
    streamed = (weight.to(device), weight_scale.to(device), bias.to(device) if bias is not None else None)
    out = torch.empty((rows.shape[0], out_features), dtype=x.dtype, device=device)
    chunk = _chunk_rows(out_features)
    failure: tuple[FailureOutcome, str] | None = None
    unvalidated: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
    for start in range(0, rows.shape[0], chunk):
        chunk_rows, chunk_out = rows[start : start + chunk], out[start : start + chunk]
        key = _tuning_key(chunk_rows, chunk_out, *streamed)
        if _health.is_validated(key):
            failure = _launch(kernels, chunk_rows, *streamed, chunk_out)
        else:
            with _tuning_lock:
                failure = _launch(kernels, chunk_rows, *streamed, chunk_out)
            tail = slice(-_FIRST_CALL_ROWS, None)
            unvalidated.setdefault(key, (chunk_rows[tail], chunk_out[tail]))
        if failure is not None:
            break
    # The comparison runs the dequantized path, which streams the weight itself; the kernels' copy goes first.
    del streamed
    for key, (key_rows, key_out) in unvalidated.items():
        if failure is not None:
            break
        failure = _validate(key, key_rows, key_out, fallback)
    del unvalidated
    if failure is None:
        _health.record_success(device)
        if _health.first_use(device):
            logger.info(f"int8 compute: rotated int8_tensorwise layers on {device} run on the int8 tensor cores.")
        return out.view(*x.shape[:-1], out_features)

    # Out of the `except` that caught the failure, and with the partial output released, before the fallback runs.
    del out
    outcome, error = failure
    if outcome is FailureOutcome.RETIRED:
        logger.warning(
            f"int8 compute failed on {device} for a {out_features}x{in_features} layer, input {tuple(x.shape)} "
            f"{x.dtype} ({error}). int8 layers on this device dequantize until restart; set `int8_compute: false` "
            "to stop using int8 compute."
        )
    return fallback(x)


def _tuning_key(
    rows: torch.Tensor, out: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None
) -> tuple:
    """What selects a different tuned and compiled kernel for this chunk, so each one is validated on its own.

    The autotune key (device, row bucket, N, K, dtypes), plus what Triton specializes a compiled kernel on: whether
    pointers start 16-byte aligned and integers are multiples of 16. A chunk that starts mid-output for an odd N, or a
    row count off the 16 grid, runs a different binary than the first chunk did.
    """
    return (
        rows.device,
        rows.dtype,
        None if bias is None else bias.dtype,
        weight_scale.numel() == 1,
        weight.shape[0],
        weight.shape[1],
        _m_bucket(rows.shape[0]),
        rows.data_ptr() % 16 == 0,
        out.data_ptr() % 16 == 0,
        rows.shape[0] % 16 == 0,
        rows.stride(0) % 16 == 0,
    )


def _validate(
    key: tuple, rows: torch.Tensor, out: torch.Tensor, fallback: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[FailureOutcome, str] | None:
    """Compare a key's first kernel rows with the dequantized path; the failure to act on, or None.

    Not inside the kernel's ``try``: a dequantized path that fails is not int8 compute's failure. A reference that is
    not finite (an fp16 overflow, a NaN input) or all zero (zero rows without a bias) says nothing about the kernel:
    the key is left unvalidated for a later call to check, and the kernel's output stands -- it ran without error,
    and recomputing the whole layer on the dequantized path would cost the memory the chunking saves.
    """
    expected = fallback(rows)
    reference = torch.linalg.vector_norm(expected, dtype=torch.float32).item()
    if not math.isfinite(reference) or reference == 0.0:
        return None
    error = relative_error(out, expected)
    if not math.isfinite(error) or error > _FIRST_CALL_MAX_ERROR:
        mismatch = RuntimeError(f"its first result differs from the dequantized path by {error:.2f} (relative L2)")
        return _health.record_failure(rows.device, mismatch), repr(mismatch)
    _health.mark_validated(key)
    return None


def _launch(
    kernels: ModuleType,
    rows: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
) -> tuple[FailureOutcome, str] | None:
    """The failure and its description, or None. Only the description leaves the `except`: the exception's
    traceback would keep the failed kernel's tensors alive through the fallback."""
    try:
        kernels.w8a8_rows(rows, weight, weight_scale, bias, out, _m_bucket(rows.shape[0]))
    except Exception as e:
        if isinstance(e, RuntimeError) and is_oom_error(e):
            raise
        return _health.record_failure(rows.device, e), repr(e)
    return None

"""
Utility routine used for autodetection of optimal slice size
for attention mechanism.
"""

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from typing import Any, Callable

import psutil
import torch
from torch.nn.attention import SDPBackend

from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.sdpa_scope import (
    SDPA_KERNELS,
    cudnn_attention_demoted,
    demote_cudnn_attention,
    describe_sdpa_priority_order,
    sdpa_priority_order,
)


def auto_detect_slice_size(latents: torch.Tensor) -> str:
    bytes_per_element_needed_for_baddbmm_duplication = latents.element_size() + 4
    max_size_required_for_baddbmm = (
        16
        * latents.size(dim=2)
        * latents.size(dim=3)
        * latents.size(dim=2)
        * latents.size(dim=3)
        * bytes_per_element_needed_for_baddbmm_duplication
    )
    if latents.device.type in {"cpu", "mps"}:
        mem_free = psutil.virtual_memory().free
    elif latents.device.type == "cuda":
        mem_free, _ = torch.cuda.mem_get_info(latents.device)
    elif latents.device.type == "xpu":
        mem_free, _ = TorchDevice.xpu_mem_get_info(latents.device)
    else:
        raise ValueError(f"unrecognized device {latents.device}")

    if max_size_required_for_baddbmm > (mem_free * 3.0 / 4.0):
        return "max"
    elif torch.backends.mps.is_available():
        return "max"
    else:
        return "balanced"


# SDPA computes attention one of two ways: a fused kernel (flash / memory-efficient / cuDNN) that
# never materializes the O(S^2) score matrix, or the `math` fallback, which does. Which one runs is
# not a property of FLUX.2 -- it is a property of the build, the device, the dtype, the head dim and
# whether an attention mask was passed, and the rules differ per build in ways that are not worth
# hard-coding. Measured, for the FLUX.2 VAE's 512-wide mid-block head: CUDA takes it on the
# memory-efficient kernel; on ROCm the answer varied by card and torch build (gfx1100 on torch 2.10
# reported `math` from a probe, gfx1201 took it on flash) -- and on torch 2.13 the fused kernels
# that do accept it return wrong output, see `ROCM_FUSED_SDPA_MAX_HEAD_DIM`. So on ROCm the wide
# head is decided by policy (`rocm_sdpa_uses_math_kernel`), and everywhere else, and for every
# narrower head, the helpers below ask torch instead of assuming. Masks are not a discriminator
# anywhere measured.

# Peak *reserved* bytes per element of the materialized score matrix, with `SDPBackend.MATH` forced,
# each point in a fresh process. The same figures hold for bf16, fp16 and fp32 inputs, because the
# fallback's softmax intermediates are fp32 regardless -- so this is an absolute byte count, not a
# multiple of the element size.
#
#   heads  seq    head_dim   CUDA    gfx1100   gfx1201
#       1  4096        512   12.88     13.62     16.38
#       1  8192        512   10.28     10.78     13.47
#       1 16384        512    9.71      9.76     11.99
#       4  4096        128    9.97     10.16     12.84
#      48  4608        128    9.58      9.59     11.69
#
# 17 is an upper bound on every measured point on all three. Note how far apart the two ROCm cards
# are: this is not a "CUDA number and a ROCm number", it is per-build, and the cost of guessing low
# is an OOM the estimate exists to prevent. `scripts/calibrate_flux2_working_memory.py` reproduces
# this table on any build.
SDPA_MATH_BYTES_PER_SCORE_ELEMENT = 17

# `_fused_sdp_choice` reports which kernel `F.scaled_dot_product_attention` would pick. These are
# the answers that mean "a fused kernel"; `MATH` -- and `ERROR`, which torch returns when it cannot
# pick anything at all -- mean the score matrix gets built.
_FUSED_SDP_CHOICES = frozenset(
    int(getattr(SDPBackend, name))
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "OVERRIDEABLE")
    if hasattr(SDPBackend, name)
)

# Widest head the fused SDPA kernels are trusted with on ROCm. torch 2.13.0+rocm7.2 (ROCm 7.2.4,
# gfx1100) returns wrong output from `F.scaled_dot_product_attention` at head_dim 512: the flash
# kernel is pure garbage at 16k tokens, the memory-efficient kernel is garbage or NaN depending on
# the inputs' alignment, and the default dispatch lands on one of them. head_dim 256 and below is
# correct in every backend, and the math backend is correct at any width. 512 is exactly the
# mid-block self-attention of the FLUX.2 VAE, the SD/SDXL AutoencoderKL and the FLUX.1 autoencoder:
# one head over the full channel width -- which is why the failure looked like a broken VAE. Because
# which kernel runs depends on the caching allocator's state, a decode that was fine turns into
# noise or a black image after enough allocation churn (measured: 14 of 30 decodes of the same
# latents in a fresh single-GPU process) and stays that way until the memory layout changes again.
#
# The guard keys on the torch build, not the card: only gfx1100 was measured, and a wrong image is
# the worse failure, so every ROCm device takes the math kernel for wide heads until a build is
# known to be fixed. The cost is the math kernel for those calls: slower, and a materialized score
# matrix (~4.3 GiB for a 1024px VAE decode, ~21 GiB at 1536px), which the guard computes in chunks of
# `SDPA_MATH_CHUNK_BYTES` and the working-memory estimators price accordingly. A user on a build where
# the fused kernels are correct can raise the threshold, or switch the guard off with a large value:
#     INVOKE_ROCM_FUSED_SDPA_MAX_HEAD_DIM=100000
_ROCM_FUSED_SDPA_MAX_HEAD_DIM_DEFAULT = 256
_ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV = "INVOKE_ROCM_FUSED_SDPA_MAX_HEAD_DIM"


def _read_rocm_fused_sdpa_max_head_dim() -> int:
    raw = os.environ.get(_ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV)
    if raw is None:
        return _ROCM_FUSED_SDPA_MAX_HEAD_DIM_DEFAULT
    try:
        return int(raw)
    except ValueError:
        InvokeAILogger.get_logger(__name__).warning(
            f"{_ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV}={raw!r} is not an integer; "
            f"keeping the default of {_ROCM_FUSED_SDPA_MAX_HEAD_DIM_DEFAULT}."
        )
        return _ROCM_FUSED_SDPA_MAX_HEAD_DIM_DEFAULT


ROCM_FUSED_SDPA_MAX_HEAD_DIM = _read_rocm_fused_sdpa_max_head_dim()

_IS_ROCM = torch.version.hip is not None
_ROCM_SDPA_GUARD_SENTINEL = "_invokeai_rocm_sdpa_guard"

# Largest score matrix one math-kernel SDPA call may materialize on ROCm; larger calls run in chunks of at most this
# (`_chunked_sdpa`). Where no fused kernel exists -- ROCm on Windows has none, measured on an RX 9060 XT with torch
# 2.12+rocm7.14 -- an unchunked call costs heads x seq^2 x ~12-17 bytes: 8 GiB for Z-Image at 1024px, 12.9 GiB for
# Krea-2, 40 GiB for Z-Image at 1536px, against a 16 GB card. Measured per call at 1 GiB, against unchunked: Z-Image
# shape +8 % time, FLUX VAE mid-block (one 512-wide head, 1024px) +12 %; both within one bf16 ulp of the unchunked
# result. Read at call time, so tests can lower it.
SDPA_MATH_CHUNK_BYTES = 1 << 30


def _rocm_cuda(device_type: str) -> bool:
    return _IS_ROCM and device_type == "cuda"


def rocm_sdpa_uses_math_kernel(device_type: str, head_dim: int) -> bool:
    """True when the ROCm guard sends an SDPA call of this shape to the math kernel.

    The single rule shared by the guard itself and by the working-memory estimators, so a reserved
    score matrix always corresponds to a kernel that really runs.
    """
    return _rocm_cuda(device_type) and head_dim > ROCM_FUSED_SDPA_MAX_HEAD_DIM


def rocm_sdpa_chunks_math(device_type: str) -> bool:
    """True when math-kernel attention on this device runs in chunks of at most `SDPA_MATH_CHUNK_BYTES`.

    That is a ROCm device with the guard installed on the live binding -- not a guard that failed to install, and not
    a process that never installed it (scripts and tests that skip `apply_monkeypatches`), where the whole score
    matrix is still built and must be budgeted in full.
    """
    return _rocm_cuda(device_type) and getattr(
        torch.nn.functional.scaled_dot_product_attention, _ROCM_SDPA_GUARD_SENTINEL, False
    )


def _math_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """`F.scaled_dot_product_attention` on the math kernel, without touching torch's global backend flags."""
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        # The Python entry point turns a boolean mask (True = attend) into an additive one before
        # dispatching; the aten op expects the additive form.
        attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(~attn_mask, float("-inf"))
    return torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, attn_mask, dropout_p, is_causal, None, scale=scale, enable_gqa=enable_gqa
    )[0]


def _math_score_bytes(query: torch.Tensor, key: torch.Tensor) -> int:
    """Upper bound on the score matrix a math-kernel call over these (4-D) inputs materializes."""
    batch, heads, query_len, _ = query.shape
    return batch * heads * query_len * key.shape[-2] * SDPA_MATH_BYTES_PER_SCORE_ELEMENT


def _chunkable_kv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    """Whether `_chunked_sdpa` slices K/V correctly: 4-D and of the query's batch. Other broadcasts that math SDPA
    accepts (3-D K/V, a wider K/V batch) stay whole."""
    batch = query.shape[0]
    return key.dim() == value.dim() == 4 and key.shape[0] == value.shape[0] == batch


def sdpa_backend_choice(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> SDPBackend | None:
    """The kernel `F.scaled_dot_product_attention` would run this exact call on, under the backend selection in force
    now; None when no kernel would, or it cannot be told.

    The one wrapper around torch's private dispatch query, so it is torch's own answer rather than a reimplementation
    of its eligibility rules: it honours the per-backend enable flags and the priority order alike, whichever
    `sdpa_policy` window is open -- this thread's or another session's. Not cached: that state is process-global and a
    cache key cannot honestly enumerate it (`sdpa_kernel(..., set_priority=True)` reorders with every flag unchanged:
    measured, `EFFICIENT` before and `MATH` inside). It costs 2.5us, measured on CPU for the Krea-2 shape.

    With math enabled `_fused_sdp_choice` is asked directly: math is then always eligible, so torch never finds nothing
    and stays silent. With math disabled -- only inside an exclusive window -- a call no permitted kernel takes makes
    torch re-run every check in debug mode, printing C++ warnings straight to stderr that no Python warning filter
    reaches, and raise; on a per-attention-call path that is thousands of lines a generation. There the per-kernel
    `can_use_*` checks, asked silently in the dispatcher's own priority order, give its answer. They exist for CUDA
    tensors only, so elsewhere a math-disabled call answers None.

    It is the dispatcher's choice, before InvokeAI's SDPA wrappers act on it: on ROCm the wide-head guard
    (`rocm_sdpa_uses_math_kernel`) sends heads wider than `ROCM_FUSED_SDPA_MAX_HEAD_DIM` to math whatever this says, so
    a caller asking about such heads consults that first.
    """
    try:
        if torch.backends.cuda.math_sdp_enabled():
            choice = SDPBackend(
                int(
                    torch.ops.aten._fused_sdp_choice(
                        query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa
                    )
                )
            )
            return None if choice == SDPBackend.ERROR else choice
        if not query.is_cuda:
            return None
        cuda = torch.backends.cuda
        params = cuda.SDPAParams(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
        permitted = {
            SDPBackend.FLASH_ATTENTION: lambda: cuda.flash_sdp_enabled() and cuda.can_use_flash_attention(params),
            SDPBackend.EFFICIENT_ATTENTION: lambda: (
                cuda.mem_efficient_sdp_enabled() and cuda.can_use_efficient_attention(params)
            ),
            SDPBackend.CUDNN_ATTENTION: lambda: cuda.cudnn_sdp_enabled() and cuda.can_use_cudnn_attention(params),
        }
        for backend in sdpa_priority_order() or ():
            if backend in permitted and permitted[backend]():
                return backend
        return None
    except Exception:
        return None


def sdpa_runs_fused(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> bool:
    """Whether `sdpa_backend_choice` names a fused kernel, one that does not build the O(S^2) score matrix.

    "No kernel" and "cannot tell" answer False: the conservative case for every caller.
    """
    return sdpa_kernel_is_fused(sdpa_backend_choice(query, key, value, attn_mask, scale=scale, enable_gqa=enable_gqa))


def sdpa_kernel_is_fused(choice: SDPBackend | None) -> bool:
    """Whether a `sdpa_backend_choice` answer is a fused kernel. None -- no kernel, or cannot tell -- is not."""
    return choice is not None and int(choice) in _FUSED_SDP_CHOICES


def _chunked_sdpa(
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """Run a 4-D `(batch, heads, seq, dim)` attention as calls whose score matrices stay within `SDPA_MATH_CHUNK_BYTES`.

    Softmax normalizes each query row over all keys, so splitting heads or query rows is exact in exact arithmetic,
    and both orders agree to within one bf16 ulp in practice. Head groups come first: every call keeps its full
    `query x key` shape, which is bitwise the unchunked result wherever the kernel's arithmetic does not depend on
    the batch it runs over (it does on CPU; a rocBLAS batched GEMM re-tiles when 30 head-batches become groups of 2,
    measured at 0.0005 on a bf16 Z-Image shape). Query rows are split only when a single head (or GQA group) alone
    exceeds the budget -- the one-head VAE mid-block -- or when K/V are broadcast across heads rather than grouped. A
    mask is sliced along the axis being split wherever it is not broadcast there. The output is allocated from the
    first chunk's result, which carries the dtype autocast chose.
    """
    budget = SDPA_MATH_CHUNK_BYTES
    batch, heads, query_len, _ = query.shape
    key_len, kv_heads = key.shape[-2], key.shape[-3]
    per_head = batch * query_len * key_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    if kv_heads == heads:
        ratio: int | None = 1
    elif enable_gqa and kv_heads > 0 and heads % kv_heads == 0:
        ratio = heads // kv_heads
    else:
        ratio = None
    group = 0 if ratio is None else (budget // per_head) // ratio * ratio

    out: torch.Tensor | None = None
    if group > 0:
        mask_per_head = attn_mask is not None and attn_mask.dim() >= 3 and attn_mask.shape[-3] == heads
        assert ratio is not None
        for start in range(0, heads, group):
            end = min(start + group, heads)
            mask = attn_mask[..., start:end, :, :] if mask_per_head else attn_mask
            kv = slice(start // ratio, end // ratio)
            part = call(query[:, start:end], key[:, kv], value[:, kv], mask)
            if out is None:
                out = part.new_empty((*query.shape[:-1], part.shape[-1]))
            out[:, start:end] = part
    else:
        rows = max(1, budget // (batch * heads * key_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT))
        mask_per_row = attn_mask is not None and attn_mask.dim() >= 2 and attn_mask.shape[-2] == query_len
        for start in range(0, query_len, rows):
            end = min(start + rows, query_len)
            mask = attn_mask[..., start:end, :] if mask_per_row else attn_mask
            part = call(query[:, :, start:end], key, value, mask)
            if out is None:
                out = part.new_empty((*query.shape[:-1], part.shape[-1]))
            out[:, :, start:end] = part
    assert out is not None
    return out


def install_rocm_sdpa_guard() -> None:
    """Rebind ``torch.nn.functional.scaled_dot_product_attention`` on ROCm so that attention runs on a kernel that is
    both correct and bounded in memory.

    Two rules, both only for ROCm devices:

    * A call whose head_dim exceeds ``ROCM_FUSED_SDPA_MAX_HEAD_DIM`` runs the math kernel instead of a fused one, which
      returns wrong output there.
    * A call that runs on the math kernel -- routed by the first rule, or chosen by torch because no fused kernel takes
      it (all of them, on ROCm under Windows) -- and would materialize more than ``SDPA_MATH_CHUNK_BYTES`` of scores
      runs in chunks (`_chunked_sdpa`). Causal and dropout calls, and anything but 4-D inputs, are left whole.

    The math kernel is invoked as the aten op directly rather than through ``sdpa_kernel(MATH)``:
    that context manager writes torch's process-global backend flags, so on a multi-GPU install it
    would also decide the kernel for whatever the other worker is running (see `sdpa_scope`). The
    op call touches no shared state, so two sessions can take different paths at once.

    Callers that resolve the function through the module at call time inherit the guard -- that is
    diffusers' `AttnProcessor2_0` and its attention dispatcher, and every autoencoder in this repo.
    Two differences from the fused path for the wide-head calls only: torch's argument validation is
    skipped (an invalid mask dtype is not rejected up front), and the aten op is not on autocast's
    cast list, so fp32 inputs under ``torch.autocast`` compute in fp32 instead of the autocast
    dtype. No current wide-head caller runs under autocast. Chunks of narrower heads go through the
    original function, keeping its semantics.

    Idempotent; a no-op on non-ROCm builds. If the aten op is missing or rejects the call shape
    this wrapper relies on, the guard is not installed and the log says so, rather than failing
    inside the first generation. Installed once at startup via ``apply_monkeypatches``.
    """
    if not _IS_ROCM:
        return
    functional = torch.nn.functional
    if getattr(functional.scaled_dot_product_attention, _ROCM_SDPA_GUARD_SENTINEL, False):
        return
    logger = InvokeAILogger.get_logger(__name__)
    try:
        probe = torch.zeros((1, 1, 2, 4))
        _math_sdpa(probe, probe, probe, torch.ones((2, 2), dtype=torch.bool), 0.0, False, None, False)
    except Exception as e:
        logger.warning(
            f"ROCm SDPA guard NOT installed: the math attention op rejected the call shape ({e!r}). "
            f"Attention with head_dim > {ROCM_FUSED_SDPA_MAX_HEAD_DIM} may return wrong output on this build, and "
            "math-kernel attention builds its whole score matrix."
        )
        return
    original = functional.scaled_dot_product_attention

    @wraps(original)
    def guarded_scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *args: Any,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        wide_head = rocm_sdpa_uses_math_kernel(query.device.type, query.shape[-1])
        if wide_head and (args or kwargs):
            # A torch that grew new SDPA arguments would need the math call updated to match;
            # refuse rather than silently drop them.
            raise TypeError(
                f"scaled_dot_product_attention received arguments the ROCm SDPA guard does not forward: "
                f"{args!r}, {sorted(kwargs)!r}"
            )
        chunk = (
            _rocm_cuda(query.device.type)
            and query.dim() == 4
            and not query.is_nested
            and not is_causal
            and dropout_p == 0.0
            and not args
            and not kwargs
            and _math_score_bytes(query, key) > SDPA_MATH_CHUNK_BYTES
            and _chunkable_kv(query, key, value)
            and (wide_head or not sdpa_runs_fused(query, key, value, attn_mask, scale=scale, enable_gqa=enable_gqa))
        )
        if wide_head:
            if not chunk:
                return _math_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
            return _chunked_sdpa(
                lambda q, k, v, m: _math_sdpa(q, k, v, m, 0.0, False, scale, enable_gqa),
                query,
                key,
                value,
                attn_mask,
                enable_gqa,
            )
        if chunk:
            return _chunked_sdpa(
                lambda q, k, v, m: original(q, k, v, m, 0.0, False, scale=scale, enable_gqa=enable_gqa),
                query,
                key,
                value,
                attn_mask,
                enable_gqa,
            )
        return original(
            query, key, value, attn_mask, dropout_p, is_causal, *args, scale=scale, enable_gqa=enable_gqa, **kwargs
        )

    setattr(guarded_scaled_dot_product_attention, _ROCM_SDPA_GUARD_SENTINEL, True)
    functional.scaled_dot_product_attention = guarded_scaled_dot_product_attention
    logger.info(
        f"ROCm: attention with head_dim > {ROCM_FUSED_SDPA_MAX_HEAD_DIM} runs on the math SDPA kernel "
        f"(the fused kernels return wrong output there; {_ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV} overrides the threshold), "
        f"and math-kernel attention runs in chunks of at most {SDPA_MATH_CHUNK_BYTES / 2**30:g} GiB of scores."
    )


_CUDNN_FALLBACK_SENTINEL = "_invokeai_cudnn_sdpa_fallback"
# Serialises the decision to demote cuDNN, so of two GPU workers whose cuDNN calls fail at once exactly one demotes it
# and warns, and the other retries.
_CUDNN_FALLBACK_LOCK = threading.Lock()
# Bumped once per demotion and read before every call: a call that failed while another session demoted cuDNN can then
# tell, and retry, although by the time it asks, the dispatcher no longer names cuDNN for it.
_cudnn_demotions = 0


def install_cudnn_sdpa_fallback() -> None:
    """Rebind ``torch.nn.functional.scaled_dot_product_attention`` on CUDA builds so attention that fails inside cuDNN
    is retried on the next kernel instead of failing the generation.

    Under the process-wide order (`sdpa_scope`) cuDNN serves most CUDA attention, and torch checks its eligibility
    statically: a failure at run time -- a plan the cuDNN frontend cannot build, an old cuDNN library swapped in by hand,
    a GPU generation it mishandles -- raises instead of falling back. When a call raises a RuntimeError that is not an
    out-of-memory error (VRAM is the model cache's business), math is enabled, and the dispatcher names cuDNN for that
    exact call -- which, with math eligible, means cuDNN was ranked ahead of math -- cuDNN is ranked behind math for the
    rest of the process (`demote_cudnn_attention`), one warning is logged, and the call is retried once. An error from
    the retry propagates. Math is disabled only inside an exclusive window, such as ``INVOKE_KREA2_SDPA_BACKEND=cudnn``:
    there cuDNN was asked for explicitly, so its error propagates too. A ``torch.AcceleratorError`` is no cuDNN failure
    either: an asynchronous CUDA error from earlier work surfaces at whatever launch comes next and poisons the context,
    so it propagates rather than blaming cuDNN. The enable flags are never touched.

    A grouped-query call (``enable_gqa`` with fewer K/V heads than query heads) is retried with K/V expanded to the
    query heads, which is the same attention: it only reached the kernel unexpanded because cuDNN took it, and with
    cuDNN demoted no fused kernel takes grouped-query attention (memory-efficient refuses it, flash refuses masks and is
    absent on Windows), so the unchanged call would build the full score matrix on math -- gigabytes per call, priced
    nowhere. Expanded, memory-efficient serves it.

    The cost is one Python frame and a ``try`` per call: 0.2-0.35us, measured on CPU against a bare 5.5us call.

    Callers that resolve the function through the module at call time inherit it: InvokeAI's own models, diffusers'
    attention processors and dispatcher, transformers' SDPA attention, and ``torch.nn.functional``'s own callers such
    as ``MultiheadAttention``. Code that bound the function before startup patching bypasses it, as it bypasses the
    ROCm guard: the vendored PiD network (``pid/_src/networks/pixeldit_official.py``) imports the bare name and is
    loaded with ``invokeai.app.invocations.baseinvocation``, before ``apply_monkeypatches`` runs, so a cuDNN failure in
    PiD decode attention still fails the decode.

    Idempotent; a no-op on builds without CUDA, ROCm included, which have no cuDNN attention. Installed once at startup
    via ``apply_monkeypatches``, before SageAttention, so that SageAttention's own fallback calls go through it too.
    """
    if torch.version.cuda is None or torch.version.hip is not None:
        return
    functional = torch.nn.functional
    if getattr(functional.scaled_dot_product_attention, _CUDNN_FALLBACK_SENTINEL, False):
        return
    original = functional.scaled_dot_product_attention

    @wraps(original)
    def scaled_dot_product_attention_with_cudnn_fallback(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *args: Any,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        demotions = _cudnn_demotions
        try:
            return original(
                query, key, value, attn_mask, dropout_p, is_causal, *args, scale=scale, enable_gqa=enable_gqa, **kwargs
            )
        except RuntimeError as error:
            if not _cudnn_failed_and_is_demoted(
                error, demotions, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
            ):
                raise
            if enable_gqa and key.dim() >= 3 and key.shape[-3] != query.shape[-3]:
                # See the docstring: unchanged, the grouped call would now run on math.
                key = key.repeat_interleave(query.shape[-3] // key.shape[-3], dim=-3)
                value = value.repeat_interleave(query.shape[-3] // value.shape[-3], dim=-3)
                enable_gqa = False
            return original(
                query, key, value, attn_mask, dropout_p, is_causal, *args, scale=scale, enable_gqa=enable_gqa, **kwargs
            )

    setattr(scaled_dot_product_attention_with_cudnn_fallback, _CUDNN_FALLBACK_SENTINEL, True)
    functional.scaled_dot_product_attention = scaled_dot_product_attention_with_cudnn_fallback


def _cudnn_failed_and_is_demoted(
    error: RuntimeError,
    demotions: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> bool:
    """Whether a failed SDPA call was cuDNN's and cuDNN now ranks behind math, so a retry runs on another kernel."""
    global _cudnn_demotions

    if is_oom_error(error) or _is_sticky_cuda_error(error) or not torch.backends.cuda.math_sdp_enabled():
        return False
    with _CUDNN_FALLBACK_LOCK:
        if _cudnn_demotions != demotions:
            return True
        choice = sdpa_backend_choice(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa
        )
        if choice != SDPBackend.CUDNN_ATTENTION or not demote_cudnn_attention():
            return False
        _cudnn_demotions += 1
    # Outside the lock: a stalled log handler must not hold up the other worker's attention.
    _warn_cudnn_demoted(error, query.device)
    return True


def _is_sticky_cuda_error(error: RuntimeError) -> bool:
    # A torch without the class raises such errors as plain RuntimeErrors, which this cannot tell apart.
    accelerator_error = getattr(torch, "AcceleratorError", None)
    return accelerator_error is not None and isinstance(error, accelerator_error)


def _warn_cudnn_demoted(error: RuntimeError, device: torch.device) -> None:
    try:
        gpu = torch.cuda.get_device_name(device)
    except Exception:
        gpu = str(device)
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:
        cudnn = None
    InvokeAILogger.get_logger(__name__).warning(
        f"cuDNN attention failed on {gpu} (torch {torch.__version__}, cuDNN {cudnn}): {error}. InvokeAI now ranks "
        "cuDNN behind PyTorch's other attention kernels for the rest of this run, and generation continues on the next "
        "kernel; images may differ slightly from those cuDNN produced. Please report this at "
        "https://github.com/invoke-ai/InvokeAI/issues, including this message."
    )


# The code whose attention calls this thread makes, for reports about them: a node's invocation type.
_attention_caller: ContextVar[str | None] = ContextVar("invokeai_attention_caller", default=None)


@contextmanager
def attention_caller(name: str) -> Iterator[None]:
    """Attribute the attention calls this thread makes inside the block to `name`.

    The session processor enters it around every node's execution with the invocation type (``flux_denoise``,
    ``z_image_text_encoder``), so a report about attention names the node it concerns, for every model family alike. A
    ContextVar, so per thread: sessions on other GPUs keep their own. Calls on threads a node starts itself are not
    attributed.
    """
    token = _attention_caller.set(name)
    try:
        yield
    finally:
        _attention_caller.reset(token)


_PINNED_REPORT_SENTINEL = "_invokeai_pinned_sdpa_report"
_PINNED_NAMES = {kernel: name for name, kernel in SDPA_KERNELS.items()}
_KERNEL_LABELS = {
    SDPBackend.FLASH_ATTENTION: "flash",
    SDPBackend.CUDNN_ATTENTION: "cuDNN",
    SDPBackend.EFFICIENT_ATTENTION: "memory-efficient",
    SDPBackend.MATH: "math",
}
_KERNEL_ENABLED = {
    SDPBackend.FLASH_ATTENTION: "flash_sdp_enabled",
    SDPBackend.CUDNN_ATTENTION: "cudnn_sdp_enabled",
    SDPBackend.EFFICIENT_ATTENTION: "mem_efficient_sdp_enabled",
    SDPBackend.MATH: "math_sdp_enabled",
}
# torch's per-kernel eligibility checks, in `torch.backends.cuda`. Math has none: it takes every call.
_KERNEL_CHECKS = {
    SDPBackend.FLASH_ATTENTION: "can_use_flash_attention",
    SDPBackend.CUDNN_ATTENTION: "can_use_cudnn_attention",
    SDPBackend.EFFICIENT_ATTENTION: "can_use_efficient_attention",
}
# Reports logged so far, each once. Past this many, new ones are dropped, and that is said once.
_MAX_PINNED_REPORTS = 256
_pinned_reports: set[tuple[Any, ...]] = set()
_pinned_reports_dropped = False
_PINNED_REPORTS_LOCK = threading.Lock()
# (device, kernel) -> whether the kernel serves a plain call on that GPU at all, asked once per GPU.
_plain_call_served: dict[tuple[str, SDPBackend], bool] = {}
# The device types the report asks about. torch registers `_fused_sdp_choice` for these and the CPU; on any other --
# MPS -- the query raises, with about 7.7 KB of message and 79us a call (measured on CPU for an equally raising query).
# The CPU is left out because the ranking never applies there: on a CPU-only install, or for CPU attention on a GPU
# install, every node type would be reported with nothing to act on.
_REPORTED_DEVICE_TYPES = frozenset({"cuda", "xpu"})


def install_pinned_sdpa_kernel_report(pinned: SDPBackend) -> None:
    """Rebind ``torch.nn.functional.scaled_dot_product_attention`` so that attention the pinned kernel does not serve is
    logged, once per node type, kernel and reason.

    ``attention_backend`` pins a kernel by ranking it first (`sdpa_scope`). That is never exclusive: a call the kernel
    cannot serve runs on the next one, and torch says nothing. Before each call the wrapper asks the dispatcher which
    kernel will serve it (`sdpa_backend_choice`); when that is not the pinned one, it logs one warning per distinct
    (node type from `attention_caller`, pinned kernel, kernel that serves the call, reason), and the call runs
    unchanged. The reasons are torch's answers, not a reimplementation of its eligibility rules:

    * The tensors are on an XPU, where torch picks the kernel without the ranking.
    * The pinned kernel's enable flag is off, as inside an exclusive window such as ``INVOKE_KREA2_SDPA_BACKEND``'s.
    * The kernel serves no plain call on this GPU at all -- flash in Windows CUDA builds, cuDNN on ROCm. Asked once per
      GPU, with torch's ``can_use_*`` check on an fp16, unmasked, head_dim 64 call.
    * cuDNN is pinned, and was demoted after failing at run time (`install_cudnn_sdpa_fallback`).
    * On ROCm, a head wider than ``ROCM_FUSED_SDPA_MAX_HEAD_DIM``, which the guard runs on math.
    * Otherwise the kernel cannot serve this call. The warning lists the call's properties, and then torch's own check
      runs once in debug mode, which prints torch's exact reason to stderr (as C++ warnings, not Python ones).

    A call on the CPU, where torch picks the kernel without the ranking (flash wherever it can, math otherwise), or on a
    device the dispatcher cannot answer for -- MPS -- is not asked about at all, and one it names no kernel
    for, which inside an exclusive window can happen, is not reported. At most ``_MAX_PINNED_REPORTS`` reports are
    kept.

    Callers that resolve the function through the module at call time are covered: InvokeAI's own models, diffusers'
    attention processors and dispatcher, transformers' SDPA attention. Code that bound the function before startup
    patching bypasses the report, as it bypasses the cuDNN fallback: the vendored PiD network
    (``pid/_src/networks/pixeldit_official.py``) runs under the ranking like everything else, but what its attention
    runs on is not reported.

    The cost is the dispatcher query on every call, plus a small key for a call the pinned kernel does not serve.
    Measured on CPU, one thread, against a bare 8.3us call (fp32, [1, 2, 8, 16]): +5.6us on a call the pinned kernel
    serves, 3.5us of it the dispatcher query, and +7.6us on one already reported.

    Installed by ``apply_monkeypatches`` while a kernel is pinned, after the ROCm guard and before the cuDNN fallback.
    Outside the guard, it sees the wide heads the guard sends to math; inside the fallback, the retry after a cuDNN
    demotion is reported with the kernel that serves it. It stays a frame of its own rather than part of the fallback's:
    that one exists on CUDA builds only, this one on every build. Idempotent. Logs the order in force, and warns when
    the build is known to lack the pinned kernel without asking a GPU (no CUDA context is created): flash in a CUDA
    build where ``torch.backends.cuda.is_flash_attention_available()`` is False, cuDNN on ROCm.
    """
    functional = torch.nn.functional
    if getattr(functional.scaled_dot_product_attention, _PINNED_REPORT_SENTINEL, False):
        return
    original = functional.scaled_dot_product_attention
    guarded = bool(getattr(original, _ROCM_SDPA_GUARD_SENTINEL, False))

    @wraps(original)
    def scaled_dot_product_attention_reporting_pinned_kernel(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *args: Any,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            _report_unpinned_call(
                pinned, guarded, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
            )
        except Exception:
            # A diagnostic must never be what fails attention.
            _warn_pinned_report_failed()
        return original(
            query, key, value, attn_mask, dropout_p, is_causal, *args, scale=scale, enable_gqa=enable_gqa, **kwargs
        )

    setattr(scaled_dot_product_attention_reporting_pinned_kernel, _PINNED_REPORT_SENTINEL, True)
    functional.scaled_dot_product_attention = scaled_dot_product_attention_reporting_pinned_kernel

    logger = InvokeAILogger.get_logger(__name__)
    logger.info(
        f"attention_backend '{_PINNED_NAMES[pinned]}': PyTorch SDPA kernels ranked {describe_sdpa_priority_order()}. "
        f"Attention the {_KERNEL_LABELS[pinned]} kernel cannot serve runs on the next one, logged once per node type "
        "and reason."
    )
    if missing := _build_lacks_kernel(pinned):
        logger.warning(
            f"attention_backend '{_PINNED_NAMES[pinned]}': {missing}, so GPU attention runs on the next kernel in "
            "that order. `attention_backend: auto` is the default."
        )
    if pinned == SDPBackend.MATH:
        logger.warning(
            "attention_backend 'math' is a debugging mode: the math kernel materializes the full attention score "
            "matrix, and the working-memory estimates of most model families do not include it, so generations may "
            "run out of memory. On ROCm the SDPA guard splits large non-causal math-kernel calls into bounded chunks. "
            "`attention_backend: auto` is the default."
        )


def _build_lacks_kernel(kernel: SDPBackend) -> str | None:
    """Why this build has no `kernel` for its GPUs, where the build alone tells; else None. On ROCm flash depends on the
    GPU as well, which the per-call check answers."""
    if torch.version.hip is not None:
        return "ROCm builds of PyTorch have no cuDNN attention kernel" if kernel == SDPBackend.CUDNN_ATTENTION else None
    if torch.version.cuda is None:
        return None
    if kernel == SDPBackend.FLASH_ATTENTION and not torch.backends.cuda.is_flash_attention_available():
        return "this PyTorch build has no flash-attention kernel for GPUs"
    return None


@lru_cache(maxsize=1)
def _warn_pinned_report_failed() -> None:
    """Say once, with the traceback, that the pinned-kernel report broke. Called from the handler, so it has one."""
    InvokeAILogger.get_logger(__name__).warning(
        "Could not tell which SDPA kernel serves an attention call; attention_backend reports may be missing.",
        exc_info=True,
    )


def _report_unpinned_call(
    pinned: SDPBackend,
    guarded: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> None:
    global _pinned_reports_dropped

    device_type = query.device.type
    if device_type not in _REPORTED_DEVICE_TYPES:
        return
    if guarded and rocm_sdpa_uses_math_kernel(device_type, query.shape[-1]):
        if pinned == SDPBackend.MATH:
            return
        served: SDPBackend | None = SDPBackend.MATH
        reason: tuple[Any, ...] = ("rocm_guard",)
    else:
        served = sdpa_backend_choice(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa
        )
        if served is None or served == pinned:
            return
        reason = _pinned_kernel_bypass(pinned, query, key, attn_mask, dropout_p, is_causal, enable_gqa)
    report = (_attention_caller.get(), pinned, served, reason)
    if report in _pinned_reports or _pinned_reports_dropped:
        return
    with _PINNED_REPORTS_LOCK:
        if report in _pinned_reports or _pinned_reports_dropped:
            return
        dropped = len(_pinned_reports) >= _MAX_PINNED_REPORTS
        if dropped:
            _pinned_reports_dropped = True
        else:
            _pinned_reports.add(report)
    # Outside the lock: a stalled log handler must not hold up the other worker's attention.
    logger = InvokeAILogger.get_logger(__name__)
    name = _PINNED_NAMES[pinned]
    if dropped:
        logger.warning(
            f"attention_backend '{name}': {_MAX_PINNED_REPORTS} kinds of attention calls were reported that the kernel "
            "does not serve; further ones are not logged."
        )
        return
    caller = report[0]
    torch_check = reason[0] == "call" and query.is_cuda and pinned in _KERNEL_CHECKS
    logger.warning(
        f"attention_backend '{name}' is not possible for "
        f"{f'attention in {caller}' if caller else 'attention outside a node'}: "
        f"{_describe_bypass(pinned, reason)}; it runs on {served.name}."
        + (" PyTorch's own check follows on stderr." if torch_check else "")
    )
    if torch_check:
        cuda = torch.backends.cuda
        getattr(cuda, _KERNEL_CHECKS[pinned])(
            cuda.SDPAParams(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa), True
        )


def _pinned_kernel_bypass(
    pinned: SDPBackend,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool,
) -> tuple[Any, ...]:
    """Why a call the dispatcher sends past the pinned kernel goes there, as a small hashable key."""
    device = query.device
    if device.type != "cuda":
        return ("device", device.type)
    if not getattr(torch.backends.cuda, _KERNEL_ENABLED[pinned])():
        return ("disabled",)
    if not _serves_plain_calls(pinned, device):
        return ("build", str(device))
    if pinned == SDPBackend.CUDNN_ATTENTION and cudnn_attention_demoted():
        return ("demoted",)
    mask = "no" if attn_mask is None else "bool" if attn_mask.dtype == torch.bool else "float"
    heads = (query.shape[-3], key.shape[-3]) if query.dim() >= 3 and key.dim() >= 3 else None
    return ("call", query.dtype, query.shape[-1], mask, heads, enable_gqa, is_causal, dropout_p, query.dim())


def _serves_plain_calls(kernel: SDPBackend, device: torch.device) -> bool:
    """Whether `kernel` serves a plain call on this CUDA device: fp16, no mask, head_dim 64. False means the build or
    the GPU has no such kernel. torch's own check, asked once per device and kernel."""
    check = _KERNEL_CHECKS.get(kernel)
    if check is None:
        return True
    cache_key = (str(device), kernel)
    served = _plain_call_served.get(cache_key)
    if served is None:
        cuda = torch.backends.cuda
        enabled = getattr(cuda, _KERNEL_ENABLED[kernel])
        enabled_before = enabled()
        probe = torch.empty((1, 1, 128, 64), device=device, dtype=torch.float16)
        served = bool(getattr(cuda, check)(cuda.SDPAParams(probe, probe, probe, None, 0.0, False, False), False))
        # The check also reads the enable flag, so an answer given while a window had the kernel switched off is about
        # the window, not the build. It is kept only when the flag was on before the check and is still on after it.
        if enabled_before and enabled():
            _plain_call_served[cache_key] = served
    return served


def _describe_bypass(pinned: SDPBackend, reason: tuple[Any, ...]) -> str:
    label = _KERNEL_LABELS[pinned]
    kind = reason[0]
    if kind == "device":
        return f"on {reason[1]} PyTorch picks the kernel itself, without the ranking"
    if kind == "disabled":
        return (
            f"the {label} kernel is switched off for this call (by an exclusive window, such as "
            "INVOKE_KREA2_SDPA_BACKEND's)"
        )
    if kind == "build":
        return f"this PyTorch build or GPU ({reason[1]}) has no {label} kernel"
    if kind == "demoted":
        return "cuDNN attention failed earlier in this run and now ranks behind math"
    if kind == "rocm_guard":
        return (
            f"on ROCm, heads wider than {ROCM_FUSED_SDPA_MAX_HEAD_DIM} run on math "
            f"({_ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV} sets the width)"
        )
    _, dtype, head_dim, mask, heads, enable_gqa, is_causal, dropout_p, ndim = reason
    properties = [str(dtype).removeprefix("torch."), f"head_dim {head_dim}", f"{mask} mask"]
    if heads is not None and (heads[0] != heads[1] or enable_gqa):
        gqa = "enable_gqa" if enable_gqa else "without enable_gqa"
        properties.append(f"{heads[0]} query / {heads[1]} key-value heads, {gqa}")
    if is_causal:
        properties.append("causal")
    if dropout_p:
        properties.append(f"dropout {dropout_p:g}")
    if ndim != 4:
        properties.append(f"{ndim}-D inputs")
    return f"the {label} kernel cannot serve this call ({', '.join(properties)})"


_DISPATCH_TORCH = "torch"
_DISPATCH_FUSED = "fused"
_DISPATCH_MATH = "math"


@lru_cache(maxsize=1)
def _warn_unknown_diffusers_dispatch() -> None:
    """Say once per process that estimates are running blind. Rate-limited, not cached for truth."""
    InvokeAILogger.get_logger(__name__).warning(
        "Could not determine the active diffusers attention backend; budgeting working memory as if "
        "attention materializes its score matrix. Estimates will be conservative."
    )


def _diffusers_attention_dispatch() -> str:
    """Report how the diffusers attention dispatcher will route a diffusers model's attention calls.

    Diffusers models do not call `F.scaled_dot_product_attention` directly -- they go through
    `dispatch_attention_fn`, which honours the `DIFFUSERS_ATTN_BACKEND` environment variable and the
    `attention_backend()` context manager. Only the default `native` backend hands the call to
    torch; the others pin a specific kernel, and `_native_math` pins the materializing one. A
    torch-level probe alone would report "fused" for a user who has forced math.

    Read live on every estimate, never cached: the active backend is mutable process state, and a
    cached answer would keep reserving zero after a switch to `_native_math` -- the one case this
    lookup exists to catch. It is a dict lookup against an already-imported module, priced once per
    invocation.

    Reading the process-wide backend also covers per-model overrides, which is why the estimate does
    not need the model in hand (it is priced before the model is loaded). `set_attention_backend()`
    stamps the choice onto the model's attention processors *and* calls
    `_AttentionBackendRegistry.set_active_backend()` -- deliberately, "so that it propagates
    gracefully throughout". `reset_attention_backend()` clears only the processors, leaving the
    registry pinned, which errs towards over-reserving rather than under-reserving.

    Returns ``_DISPATCH_TORCH`` when torch decides, ``_DISPATCH_FUSED`` for a backend that never
    materializes the score matrix, or ``_DISPATCH_MATH`` when one is built -- including when we
    cannot tell, since under-reserving is the failure this whole term exists to prevent.
    """
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        backend, _ = _AttentionBackendRegistry.get_active_backend()
        name = str(getattr(backend, "value", backend))
    except Exception:
        # A private diffusers attribute that moved, or a selected backend whose kernel failed to
        # register. Budget the materializing case, but say so: silently adding several GB to every
        # FLUX.2 estimate is not something that should pass unnoticed.
        _warn_unknown_diffusers_dispatch()
        return _DISPATCH_MATH

    if name == "native":
        return _DISPATCH_TORCH
    if "math" in name:
        return _DISPATCH_MATH
    # Every other backend diffusers offers -- flash, sage, xformers, flex, aiter, the pinned
    # `_native_*` kernels -- exists precisely to avoid materializing the score matrix.
    return _DISPATCH_FUSED


def _torch_sdpa_materializes_score_matrix(
    device_type: str, device_index: int | None, dtype: torch.dtype, head_dim: int, has_attn_mask: bool
) -> bool:
    """Ask torch whether `F.scaled_dot_product_attention` would build the O(S^2) score matrix.

    Eligibility depends on the dtype, the head dim and the presence of a mask, not on the sequence
    length, so a tiny probe answers for the real forward. Not cached, for the reasons
    `sdpa_backend_choice` gives; it costs microseconds against a multi-second forward.

    Anything that goes wrong reports the materializing path, which is both the conservative answer
    and, for the most common cause, the correct one: torch registers `_fused_sdp_choice` for CPU,
    CUDA/ROCm and XPU only, so the call raises on MPS -- and MPS is exactly where
    `scaled_dot_product_attention` finds no fused kernel either and runs
    `_scaled_dot_product_attention_math_for_mps`, an MPSGraph transcription of `Q @ K^T` -> softmax
    -> `@ V` that holds the score tensor as a real intermediate. The remaining causes (an allocation
    failure inside the probe, a torch that predates the op) leave us knowing nothing at all, and
    there the asymmetry decides: a shortfall costs an OOM, an over-estimate costs some residency.
    """
    try:
        device = torch.device(device_type) if device_index is None else torch.device(device_type, device_index)
        q = torch.empty((1, 1, 8, head_dim), device=device, dtype=dtype)
        mask = torch.empty((1, 1, 8, 8), device=device, dtype=dtype) if has_attn_mask else None
    except Exception:
        return True
    return not sdpa_runs_fused(q, q, q, mask)


def sdpa_score_matrix_bytes(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    has_attn_mask: bool = False,
    via_diffusers_dispatch: bool = False,
) -> int:
    """Bytes SDPA spends on a materialized score matrix for one attention call, 0 if fused.

    Add this to a working-memory estimate whose linear term was calibrated on a fused kernel. On
    CUDA it is almost always 0; where the build has no fused kernel for the shapes -- ROCm/gfx1100
    for a 512-wide head, MPS for anything at all -- it is the dominant term: a 1536px FLUX.2 VAE
    decode materializes 36864^2 scores, ~21GB of them.

    That is a large term to add on a probe, so it is logged when it fires: a user who suddenly sees
    their model pushed out of VRAM should be able to find out why from the log rather than by
    reading this file.

    Set ``via_diffusers_dispatch`` for attention that runs inside a diffusers model (the FLUX.2
    transformer does; the FLUX.2 VAE's mid-block attention does not -- it still reaches
    `F.scaled_dot_product_attention` directly through `AttnProcessor2_0`). It consults the
    process-wide default backend, which is the one that applies here: estimates are priced before
    the model is loaded and outside any `attention_backend()` scope.

    Where the ROCm guard computes math attention in chunks (`rocm_sdpa_chunks_math`), one chunk is
    what is alive at a time, so the term is capped at ``SDPA_MATH_CHUNK_BYTES``. That cap assumes the
    call is one the guard will chunk: a plain 4-D, non-causal, dropout-free call with K/V of the
    query's batch, which is what every caller here prices. A causal or dropout call, or one with
    broadcast K/V, runs whole and would need the full figure.
    """
    if seq_len <= 0 or num_heads <= 0:
        return 0

    score_matrix_bytes = num_heads * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
    chunked = rocm_sdpa_chunks_math(device.type)
    if chunked:
        score_matrix_bytes = min(score_matrix_bytes, SDPA_MATH_CHUNK_BYTES)
    suffix = ", in chunks" if chunked else ""

    if via_diffusers_dispatch:
        dispatch = _diffusers_attention_dispatch()
        if dispatch == _DISPATCH_FUSED:
            return 0
        if dispatch == _DISPATCH_MATH:
            return _log_and_return(score_matrix_bytes, device, head_dim, f"the diffusers backend{suffix}")
        # _DISPATCH_TORCH: diffusers forwards to `F.scaled_dot_product_attention`, so torch decides.

    if rocm_sdpa_uses_math_kernel(device.type, head_dim):
        # `install_rocm_sdpa_guard` sends these calls to the math kernel regardless of what
        # torch's dispatch would pick, so the probe would answer for a kernel that never runs.
        return _log_and_return(score_matrix_bytes, device, head_dim, f"the ROCm SDPA guard{suffix}")
    if not _torch_sdpa_materializes_score_matrix(device.type, device.index, dtype, head_dim, has_attn_mask):
        return 0
    return _log_and_return(score_matrix_bytes, device, head_dim, f"this torch build{suffix}")


@lru_cache(maxsize=None)
def _log_score_matrix_reservation(device_type: str, head_dim: int, reason: str, gib: str) -> None:
    """Announce the materializing path once per (device, head dim, reason). It is the difference
    between a model that stays resident and one that does not, and nothing else in the log says so.
    """
    InvokeAILogger.get_logger(__name__).info(
        f"SDPA materializes its attention score matrix on {device_type} for head_dim={head_dim} "
        f"({reason}), so working-memory estimates reserve an extra ~{gib} GiB for it."
    )


def _log_and_return(score_matrix_bytes: int, device: torch.device, head_dim: int, reason: str) -> int:
    _log_score_matrix_reservation(device.type, head_dim, reason, f"{score_matrix_bytes / 1024**3:.1f}")
    return score_matrix_bytes

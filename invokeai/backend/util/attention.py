"""
Utility routine used for autodetection of optimal slice size
for attention mechanism.
"""

import os
import threading
import warnings
from functools import lru_cache, wraps
from typing import Any

import psutil
import torch
from torch.nn.attention import SDPBackend

from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


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

# Guards the process-global warning filter list that `catch_warnings` swaps; see the probe below.
_WARNING_FILTER_LOCK = threading.Lock()

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
# known to be fixed. The cost is a materialized score matrix for those calls (~4.3 GiB for a 1024px
# VAE decode, ~21 GiB at 1536px; the working-memory estimators price it). A user on a build where
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
_ROCM_SDPA_GUARD_SENTINEL = "_invokeai_rocm_head_dim_guard"


def rocm_sdpa_uses_math_kernel(device_type: str, head_dim: int) -> bool:
    """True when the ROCm guard sends an SDPA call of this shape to the math kernel.

    The single rule shared by the guard itself and by the working-memory estimators, so a reserved
    score matrix always corresponds to a kernel that really runs.
    """
    return _IS_ROCM and device_type == "cuda" and head_dim > ROCM_FUSED_SDPA_MAX_HEAD_DIM


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


def install_rocm_sdpa_head_dim_guard() -> None:
    """Rebind ``torch.nn.functional.scaled_dot_product_attention`` so that, on ROCm, a call whose
    head_dim exceeds ``ROCM_FUSED_SDPA_MAX_HEAD_DIM`` runs the math kernel instead of a fused one.

    The math kernel is invoked as the aten op directly rather than through ``sdpa_kernel(MATH)``:
    that context manager writes torch's process-global backend flags, so on a multi-GPU install it
    would also decide the kernel for whatever the other worker is running (see `sdpa_scope`). The
    op call touches no shared state, so two sessions can take different paths at once.

    Callers that resolve the function through the module at call time inherit the guard -- that is
    diffusers' `AttnProcessor2_0` and its attention dispatcher, and every autoencoder in this repo.
    Two differences from the fused path for the routed calls only: torch's argument validation is
    skipped (an invalid mask dtype is not rejected up front), and the aten op is not on autocast's
    cast list, so fp32 inputs under ``torch.autocast`` compute in fp32 instead of the autocast
    dtype. No current wide-head caller runs under autocast.

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
            f"ROCm head-dim SDPA guard NOT installed: the math attention op rejected the call shape ({e!r}). "
            f"Attention with head_dim > {ROCM_FUSED_SDPA_MAX_HEAD_DIM} may return wrong output on this build."
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
        if not rocm_sdpa_uses_math_kernel(query.device.type, query.shape[-1]):
            return original(
                query, key, value, attn_mask, dropout_p, is_causal, *args, scale=scale, enable_gqa=enable_gqa, **kwargs
            )
        if args or kwargs:
            # A torch that grew new SDPA arguments would need the math call updated to match;
            # refuse rather than silently drop them.
            raise TypeError(
                f"scaled_dot_product_attention received arguments the ROCm head-dim guard does not forward: "
                f"{args!r}, {sorted(kwargs)!r}"
            )
        return _math_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    setattr(guarded_scaled_dot_product_attention, _ROCM_SDPA_GUARD_SENTINEL, True)
    functional.scaled_dot_product_attention = guarded_scaled_dot_product_attention
    logger.info(
        f"ROCm: attention with head_dim > {ROCM_FUSED_SDPA_MAX_HEAD_DIM} runs on the math SDPA kernel "
        f"(the fused kernels return wrong output there; {_ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV} overrides the threshold)."
    )


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

    `_fused_sdp_choice` is the same dispatch query torch's own `scaled_dot_product_attention` runs
    to pick a kernel, so this is its real answer rather than a reimplementation of its rules.
    Eligibility depends on the dtype, the head dim and the presence of a mask, not on the sequence
    length, so a tiny probe answers for the real forward.

    Not cached. The answer turns on global torch state a cache key cannot honestly enumerate: the
    per-backend enable flags, but also the *priority order*, which `sdpa_kernel(..., set_priority=
    True)` reorders while leaving every flag untouched -- measured, same flags, `EFFICIENT` before
    and `MATH` inside. Each item added to such a key is one more thing to get wrong later, and the
    probe costs ~6us against a multi-second forward, so it just runs every time.

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
        with _WARNING_FILTER_LOCK, warnings.catch_warnings():
            # When no fused kernel is eligible, torch re-runs every check in debug mode to warn why
            # each one was rejected. That is the case we are deliberately probing for; we do not
            # want a wall of warnings every time an estimate is priced.
            #
            # `catch_warnings` swaps the process-global filter list, and this probe is deliberately
            # uncached, so with concurrent session workers two estimates could interleave their
            # enter/exit and leave a stale filter set behind. The lock makes the swap atomic against
            # other probes; it costs nothing at ~6us of hold time.
            warnings.simplefilter("ignore")
            choice = int(torch.ops.aten._fused_sdp_choice(q, q, q, mask, 0.0, False))
    except Exception:
        return True

    return choice not in _FUSED_SDP_CHOICES


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
    """
    if seq_len <= 0 or num_heads <= 0:
        return 0

    score_matrix_bytes = num_heads * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    if via_diffusers_dispatch:
        dispatch = _diffusers_attention_dispatch()
        if dispatch == _DISPATCH_FUSED:
            return 0
        if dispatch == _DISPATCH_MATH:
            return _log_and_return(score_matrix_bytes, device, head_dim, "the diffusers backend")
        # _DISPATCH_TORCH: diffusers forwards to `F.scaled_dot_product_attention`, so torch decides.

    if rocm_sdpa_uses_math_kernel(device.type, head_dim):
        # `install_rocm_sdpa_head_dim_guard` sends these calls to the math kernel regardless of
        # what torch's dispatch would pick, so the probe would answer for a kernel that never runs.
        return _log_and_return(score_matrix_bytes, device, head_dim, "the ROCm head-dim guard")
    if not _torch_sdpa_materializes_score_matrix(device.type, device.index, dtype, head_dim, has_attn_mask):
        return 0
    return _log_and_return(score_matrix_bytes, device, head_dim, "this torch build")


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

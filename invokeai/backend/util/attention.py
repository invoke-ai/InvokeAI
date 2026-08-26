# Copyright (c) 2023 Lincoln Stein and the InvokeAI Team
"""
Utility routine used for autodetection of optimal slice size
for attention mechanism.
"""

import warnings
from functools import lru_cache

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
# whether an attention mask was passed. CUDA's memory-efficient kernel accepts head dims well past
# 128 and arbitrary additive masks; ROCm's fused kernels reject both; MPS has no fused SDPA kernel
# at all. A working-memory estimate that assumes the fused path is therefore only correct on the
# build it was measured on, which is why the helpers below ask instead of assuming.

# Peak *reserved* bytes per element of the materialized score matrix, measured on CUDA with
# `SDPBackend.MATH` forced, each point in a fresh process: 12.9 bytes/element at 4k tokens, 10.3 at
# 8k, 9.7 at 16k -- and the same figures for bf16, fp16 and fp32 inputs, because the fallback's
# softmax intermediates are fp32 regardless. So this is an absolute byte count, not a multiple of
# the element size. 13 is an upper bound on every measured point from 4k tokens up; below that it
# can fall a couple of MB short of the allocator's rounding, which is noise next to the GB-scale
# linear terms this is added to. It is a CUDA measurement standing in for every materializing
# backend -- no other was available to calibrate against -- but the intermediates it prices (an
# fp32 score matrix and its softmax) have the same shape wherever the fallback runs.
SDPA_MATH_BYTES_PER_SCORE_ELEMENT = 13

# `_fused_sdp_choice` reports which kernel `F.scaled_dot_product_attention` would pick. These are
# the answers that mean "a fused kernel"; `MATH` -- and `ERROR`, which torch returns when it cannot
# pick anything at all -- mean the score matrix gets built.
_FUSED_SDP_CHOICES = frozenset(
    int(getattr(SDPBackend, name))
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "OVERRIDEABLE")
    if hasattr(SDPBackend, name)
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


def _sdp_kernel_toggles() -> tuple[bool, ...]:
    """The global switches that gate each fused SDPA kernel, as `_fused_sdp_choice` sees them.

    `torch.backends.cuda.enable_flash_sdp(False)` and `sdpa_kernel([...])` flip these at runtime and
    the dispatch answer flips with them, so they belong in the probe's cache key rather than being
    baked into a permanent result.
    """
    cuda = torch.backends.cuda
    return tuple(
        bool(getattr(cuda, name)())
        for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled")
        if hasattr(cuda, name)
    )


def _torch_sdpa_materializes_score_matrix(
    device_type: str, device_index: int | None, dtype: torch.dtype, head_dim: int, has_attn_mask: bool
) -> bool:
    """Ask torch whether `F.scaled_dot_product_attention` would build the O(S^2) score matrix.

    `_fused_sdp_choice` is the same dispatch query torch's own `scaled_dot_product_attention` runs
    to pick a kernel, so this is its real answer rather than a reimplementation of its rules.
    Eligibility depends on the dtype, the head dim and the presence of a mask, not on the sequence
    length, so a tiny probe answers for the real forward.

    Anything that goes wrong reports the materializing path, which is both the conservative answer
    and, for the most common cause, the correct one: torch registers `_fused_sdp_choice` for CPU,
    CUDA/ROCm and XPU only, so the call raises on MPS -- and MPS is exactly where
    `scaled_dot_product_attention` finds no fused kernel either and runs
    `_scaled_dot_product_attention_math_for_mps`, an MPSGraph transcription of `Q @ K^T` -> softmax
    -> `@ V` that holds the score tensor as a real intermediate. The remaining causes (an allocation
    failure inside the probe, a torch that predates the op) leave us knowing nothing at all, and
    there the asymmetry decides: a shortfall costs an OOM, an over-estimate costs some residency.
    """
    return _probe_sdpa_dispatch(device_type, device_index, dtype, head_dim, has_attn_mask, _sdp_kernel_toggles())


@lru_cache(maxsize=None)
def _probe_sdpa_dispatch(
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
    head_dim: int,
    has_attn_mask: bool,
    sdp_kernel_toggles: tuple[bool, ...],
) -> bool:
    """Cached body of the probe above. Every input torch's answer depends on is part of the key --
    `sdp_kernel_toggles` is not read here, it is carried so a runtime change invalidates the entry.
    """
    try:
        device = torch.device(device_type) if device_index is None else torch.device(device_type, device_index)
        q = torch.empty((1, 1, 8, head_dim), device=device, dtype=dtype)
        mask = torch.empty((1, 1, 8, 8), device=device, dtype=dtype) if has_attn_mask else None
        with warnings.catch_warnings():
            # When no fused kernel is eligible, torch re-runs every check in debug mode to warn why
            # each one was rejected. That is the case we are deliberately probing for; we do not
            # want a wall of warnings every time an estimate is priced.
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
    CUDA it is almost always 0; where the fused kernels are missing or reject the shapes -- ROCm
    caps the head dim at 128 and does not take arbitrary additive masks, MPS ships no fused SDPA
    kernel at all -- it is the dominant term: a 1536px FLUX.2 VAE decode materializes 36864^2
    scores, ~17GB of them.

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
            return score_matrix_bytes
        # _DISPATCH_TORCH: diffusers forwards to `F.scaled_dot_product_attention`, so torch decides.

    if not _torch_sdpa_materializes_score_matrix(device.type, device.index, dtype, head_dim, has_attn_mask):
        return 0
    return score_matrix_bytes

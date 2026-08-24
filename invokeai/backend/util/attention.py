# Copyright (c) 2023 Lincoln Stein and the InvokeAI Team
"""
Utility routine used for autodetection of optimal slice size
for attention mechanism.
"""

from functools import lru_cache

import psutil
import torch

from invokeai.backend.util.devices import TorchDevice


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
# never materializes the O(S^2) score matrix, or the `math` fallback, which does. torch picks per
# call from the dtype, the head dim and whether an attention mask was passed -- and the answer
# differs between builds. CUDA's memory-efficient kernel accepts head dims well past 128 and
# arbitrary additive masks; ROCm's fused kernels reject both and drop to `math`. A working-memory
# estimate that assumes the fused path is therefore only correct on the build it was measured on,
# which is why the helper below asks torch instead of assuming.

# Peak *reserved* bytes per element of the materialized score matrix, measured on CUDA with
# `SDPBackend.MATH` forced, each point in a fresh process: 12.9 bytes/element at 4k tokens, 10.3 at
# 8k, 9.7 at 16k -- and the same figures for bf16, fp16 and fp32 inputs, because the fallback's
# softmax intermediates are fp32 regardless. So this is an absolute byte count, not a multiple of
# the element size. 13 is an upper bound on every measured point from 4k tokens up; below that it
# can fall a couple of MB short of the allocator's rounding, which is noise next to the GB-scale
# linear terms this is added to.
SDPA_MATH_BYTES_PER_SCORE_ELEMENT = 13


@lru_cache(maxsize=None)
def _sdpa_has_fused_kernel(
    device_type: str, device_index: int | None, dtype: torch.dtype, head_dim: int, has_attn_mask: bool
) -> bool:
    """Ask torch whether any non-materializing SDPA kernel is eligible for these attention shapes.

    Eligibility depends on the dtype, the head dim and the presence of a mask, not on the sequence
    length, so a tiny probe answers for the real forward. Falls back to ``True`` (the status quo
    assumption) whenever torch gives us nothing to go on -- over-reserving many GB on a guess would
    push the model out of VRAM and be worse than the shortfall we are trying to avoid.
    """
    if device_type != "cuda":
        # `can_use_*_attention` is CUDA/ROCm-only. MPS, XPU and CPU all ship fused SDPA kernels, so
        # keep the fused assumption there rather than guessing at their dispatch rules.
        return True

    try:
        from torch.backends.cuda import SDPAParams, can_use_efficient_attention, can_use_flash_attention

        device = torch.device(device_type) if device_index is None else torch.device(device_type, device_index)
        q = torch.empty((1, 1, 8, head_dim), device=device, dtype=dtype)
        mask = torch.empty((1, 1, 8, 8), device=device, dtype=dtype) if has_attn_mask else None
        try:
            params = SDPAParams(q, q, q, mask, 0.0, False, False)
        except TypeError:
            # torch < 2.5: no `enable_gqa` field.
            params = SDPAParams(q, q, q, mask, 0.0, False)

        checks = [can_use_flash_attention, can_use_efficient_attention]
        can_use_cudnn_attention = getattr(torch.backends.cuda, "can_use_cudnn_attention", None)
        if can_use_cudnn_attention is not None:
            checks.append(can_use_cudnn_attention)
        return any(check(params, False) for check in checks)
    except Exception:
        return True


def sdpa_score_matrix_bytes(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    has_attn_mask: bool = False,
) -> int:
    """Bytes SDPA spends on a materialized score matrix for one attention call, 0 if fused.

    Add this to a working-memory estimate whose linear term was calibrated on a fused kernel. On
    CUDA it is almost always 0; on a build whose fused kernels reject the shapes (notably ROCm,
    which caps the head dim at 128 and does not take arbitrary additive masks) it is the dominant
    term -- a 1536px FLUX.2 VAE decode materializes 36864^2 scores, ~17GB of them.
    """
    if seq_len <= 0 or num_heads <= 0:
        return 0
    if _sdpa_has_fused_kernel(device.type, device.index, dtype, head_dim, has_attn_mask):
        return 0
    return num_heads * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT

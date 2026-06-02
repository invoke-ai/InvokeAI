"""Estimate the working (non-weight) VRAM that a denoise forward pass needs, so the model cache can
reserve it. This is the denoise-side counterpart to ``vae_working_memory.py``.

Why this exists
---------------
The model cache reserves headroom for the *activations* an op allocates during its forward pass,
separate from the model weights. VAE encode/decode already estimate this per-operation; the denoise
loop historically passed nothing and inherited a flat reserve regardless of resolution/model. PR
#7509 (which introduced the working-memory system) explicitly noted per-op estimation should be
extended "to other operations" — this module is that extension for denoise.

Architecture-scaled estimate
----------------------------
The peak activation memory of a diffusion forward scales with the size of the activation tensors,
which is roughly ``latent_area * activation_width * element_size * batch``. So::

    working_memory = base + multiplier * latent_area * activation_width * batch * element_size

``activation_width`` is read from the loaded model's OWN config (conv channels for UNets, hidden/inner
dim for transformers), so the estimate scales with model size automatically — there is no per-model
hardcoded constant. ``multiplier`` is the single fitted number (≈ "live activation copies at peak"),
CALIBRATED against our own measured peaks, NOT borrowed from another implementation.

This is the same shape ComfyUI uses (``area * dtype * factor`` with a per-model factor); the
difference is we make the model-size part explicit via ``activation_width`` (read from the model) and
keep only a small per-family ``multiplier`` that we calibrate from our own ``DENOISE_MEM``
measurements (see ``memory-audit/``).

Safety properties
-----------------
The cache honors an op-provided estimate down to ``MIN_DEVICE_WORKING_MEM_GB`` (model_cache.py) but
never below the user-configured reserve, and an absent estimate keeps the full default reserve. A
failure to estimate falls back to ``None`` (= today's behavior). Under-estimates are clamped up to
the minimum; over-estimates only cost streaming speed. See ``resolve_denoise_working_mem_bytes``.
"""

import json
import logging
from logging import Logger
from typing import Any, Optional

import torch

MB = 2**20
GB = 2**30

# Per-family activation multiplier (dimensionless: ~number of live activation copies at peak).
# CALIBRATED against our own measured denoise peaks. UNet is anchored to SDXL: measured ~332/737/1295
# MB at 1024/1536/2048^2 with model_channels=320 => multiplier ~32 (base ~12MB). DiT is PROVISIONAL —
# the FLUX/Anima measurement runs were lost to a cascade OOM, so DiT stays measure-only (not enforced)
# until WE measure it via the DENOISE_MEM records. Do not assume the UNet value transfers to
# transformers; that is exactly what the per-family measurement is for.
# Both values are CALIBRATED from our own DENOISE_MEM measurements on an 8GB card:
#   - unet: SDXL (width 320) measured implied multiplier ~28 at 1024/1536^2; we keep 32 for a small
#     safety margin.
#   - dit: FLUX (width 3072) measured a clean, resolution-stable ~5.3; Anima (width 2048) measured
#     ~3.3 (its measurement also includes the LLM-adapter forward, so it is an upper bound on pure
#     denoise). 6 covers FLUX with margin and safely over-covers Anima. One multiplier for all
#     transformers works because per-model size differences are absorbed by the activation_width term.
ACTIVATION_MULTIPLIER = {
    "unet": 32,
    "dit": 6,
}

# Fallback activation width if it can't be read from the model config (estimate then degrades to a
# flat per-area value, still floored by the cache).
DEFAULT_ACTIVATION_WIDTH = 320

# Fixed scratch overhead independent of resolution.
BASE_WORKING_MEMORY_BYTES = 64 * MB

# Per-family enforcement (see resolve_denoise_working_mem_bytes). A family is enforced once its
# multiplier is calibrated from our own measurements. Both are now calibrated (see ACTIVATION_MULTIPLIER).
ENFORCE_UNET_WORKING_MEMORY = True
ENFORCE_DIT_WORKING_MEMORY = True


def model_activation_width(model: Any) -> int:
    """Read a characteristic activation width from a loaded denoise model.

    Conv UNets -> first block channel count (``block_out_channels[0]``); transformers -> hidden/inner
    dim. Falls back to ``DEFAULT_ACTIVATION_WIDTH`` if nothing usable is found. Never raises.
    """
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            block_out = getattr(cfg, "block_out_channels", None)
            if block_out:
                return int(block_out[0])
            for attr in ("inner_dim", "hidden_size", "joint_attention_dim", "cross_attention_dim", "d_model"):
                v = getattr(cfg, attr, None)
                if isinstance(v, int) and v > 0:
                    return v
            num_heads = getattr(cfg, "num_attention_heads", None)
            head_dim = getattr(cfg, "attention_head_dim", None)
            if isinstance(num_heads, int) and isinstance(head_dim, int) and num_heads > 0 and head_dim > 0:
                return num_heads * head_dim
        # Custom models (e.g. InvokeAI Anima) expose the width as a DIRECT attribute, not on .config.
        for attr in ("hidden_size", "model_channels", "inner_dim", "dim"):
            v = getattr(model, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        # InvokeAI FLUX model exposes its hidden size on model.params.
        params = getattr(model, "params", None)
        if params is not None:
            hidden_size = getattr(params, "hidden_size", None)
            if isinstance(hidden_size, int) and hidden_size > 0:
                return hidden_size
    except Exception:
        pass
    return DEFAULT_ACTIVATION_WIDTH


def estimate_denoise_working_memory(
    latent_area: int,
    activation_width: int,
    batch_size: int,
    element_size: int,
    multiplier: float,
    base_bytes: int = BASE_WORKING_MEMORY_BYTES,
) -> int:
    """Estimate denoise working memory (bytes) from the architecture-scaled activation size.

    :param latent_area: latent spatial area = latent_height * latent_width.
    :param activation_width: the model's activation width (conv channels / hidden dim).
    :param batch_size: the latent batch (number of images); CFG behavior is folded into the
        per-family multiplier, so do NOT double for classifier-free guidance.
    :param element_size: bytes per element of the inference dtype (e.g. 2 for fp16/bf16).
    :param multiplier: the per-family activation multiplier (see ``ACTIVATION_MULTIPLIER``).
    """
    area = max(0, int(latent_area))
    width = max(1, int(activation_width))
    batch = max(1, int(batch_size))
    return int(base_bytes + multiplier * area * width * batch * int(element_size))


def estimate_denoise_working_memory_for_model(
    model: Any,
    latent_height: int,
    latent_width: int,
    batch_size: int,
    element_size: int,
    family: str,
) -> int:
    """Estimate denoise working memory for a loaded model, scaling by its activation width.

    :param model: the loaded denoise model (read its config for the activation width).
    :param family: ``"unet"`` or ``"dit"`` — selects the calibrated multiplier and enforcement.
    """
    multiplier = ACTIVATION_MULTIPLIER.get(family, ACTIVATION_MULTIPLIER["dit"])
    return estimate_denoise_working_memory(
        latent_area=int(latent_height) * int(latent_width),
        activation_width=model_activation_width(model),
        batch_size=batch_size,
        element_size=element_size,
        multiplier=multiplier,
    )


def dtype_element_size(dtype: torch.dtype) -> int:
    """Bytes per element for a torch dtype, robust across torch versions."""
    try:
        return torch.empty((), dtype=dtype).element_size()
    except Exception:
        return 2  # safe default (fp16/bf16)


def resolve_denoise_working_mem_bytes(estimate_bytes: int, family: str) -> Optional[int]:
    """Return the working-memory value to pass to ``model_on_device`` for the given family.

    :param family: ``"unet"`` (SD/SDXL conv UNet) or ``"dit"`` (diffusion transformers).
    Returns the estimate only when that family's enforcement is enabled; otherwise ``None``
    (measure-only: the cache keeps its default reserve while the family's multiplier is calibrated).
    """
    enforce = ENFORCE_UNET_WORKING_MEMORY if family == "unet" else ENFORCE_DIT_WORKING_MEMORY
    return estimate_bytes if enforce else None


def begin_denoise_measure(logger: Logger) -> Optional[int]:
    """Snapshot allocator state immediately before a denoise loop, for calibration diagnostics.

    Only active when DEBUG logging is enabled (the ``DENOISE_MEM`` record is logged at debug level),
    so it adds zero overhead in normal operation. Returns the bytes currently allocated (the
    resident-weights baseline), or ``None`` if disabled / CUDA unavailable. Pass the result to
    :func:`end_denoise_measure`.
    """
    if not logger.isEnabledFor(logging.DEBUG) or not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return alloc_before
    except Exception:
        return None


def end_denoise_measure(
    token: Optional[int],
    logger: Logger,
    *,
    label: str,
    estimate_bytes: int,
    pixel_height: int,
    pixel_width: int,
    batch_size: int,
    element_size: int,
) -> None:
    """Emit a ``DENOISE_MEM`` calibration record: the estimate vs the MEASURED activation peak.

    ``measured_peak_mb`` is the extra VRAM the denoise loop allocated on top of the resident
    weights — the real working-memory need the estimate should match. ``estimate_over_measured``
    > 1 means we are over-estimating (would over-reserve if enforced).
    """
    if token is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        measured = max(0, peak - token)
        ratio = round(estimate_bytes / measured, 2) if measured > 0 else None
        is_unet = label in ("unet", "sdxl", "sdxl-legacy", "sd15", "sd")
        logger.debug(
            "DENOISE_MEM "
            + json.dumps(
                {
                    "label": label,
                    "px_h": int(pixel_height),
                    "px_w": int(pixel_width),
                    "batch": int(batch_size),
                    "elt": int(element_size),
                    "estimate_mb": round(estimate_bytes / MB, 1),
                    "measured_peak_mb": round(measured / MB, 1),
                    "resident_before_mb": round(token / MB, 1),
                    "total_peak_mb": round(peak / MB, 1),
                    "estimate_over_measured": ratio,
                    "enforced": ENFORCE_UNET_WORKING_MEMORY if is_unet else ENFORCE_DIT_WORKING_MEMORY,
                }
            )
        )
    except Exception:
        pass

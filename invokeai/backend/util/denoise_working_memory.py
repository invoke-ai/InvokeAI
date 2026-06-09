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
measurements.

Safety properties
-----------------
The cache honors an op-provided estimate down to ``MIN_DEVICE_WORKING_MEM_GB`` (model_cache.py) but
never below the user-configured reserve, and an absent estimate keeps the full default reserve. A
failure to estimate falls back to ``None`` (= today's behavior). Under-estimates are clamped up to
the minimum; over-estimates only cost streaming speed. See ``resolve_denoise_working_mem_bytes``.
"""

import json
import logging
import time
from typing import Any, Optional, Protocol

import torch


class _LevelLogger(Protocol):
    """Minimal logger surface used for ``DENOISE_MEM`` diagnostics: a level check plus debug emit.

    Both the stdlib :class:`logging.Logger` and InvokeAI's ``LoggerInterface`` wrapper (what an
    invocation's ``context.logger`` is) satisfy this. The wrapper is NOT a real ``Logger``, so these
    functions must not assume the full ``logging.Logger`` API — only ``isEnabledFor`` and ``debug``.
    """

    def isEnabledFor(self, level: int) -> bool: ...

    def debug(self, msg: str) -> None: ...


MB = 2**20
GB = 2**30

# Per-architecture activation multiplier (dimensionless: ~number of live activation copies at peak),
# CALIBRATED from our own DENOISE_MEM measurements on an 8GB card (RTX 4070). It is read per
# ARCHITECTURE, not one value for all transformers: reading activation_width from the model absorbs
# SIZE differences WITHIN a family (FLUX.2 Klein 4B and 9B both back-solve to ~2-3), but block-structure
# differences ACROSS families do not collapse to a single constant (measured implied multipliers:
# sd3 ~2.2, anima ~2.4-3.0, flux2 ~2.1-3.1, z_image ~3.5-3.8, qwen-Edit ~6.8). Each measured value
# below is roughly max-implied x 1.15 for margin.
#   - "unet": SDXL (width 320) measured ~28; kept at 32.
#   - "dit": the DEFAULT for transformers we have NOT measured yet (cogview4, FLUX.1, future archs),
#     kept conservative so an unmeasured arch never under-reserves.
#   - sd3's value folds in its always-on CFG (a doubled-batch single forward). qwen is the Edit variant
#     (its reference image lengthens the sequence ~2x) from a single 1024 point. Both are floor-covered
#     at the resolutions they were run, so their exact value is not yet load-bearing.
ACTIVATION_MULTIPLIER = {
    "unet": 32,  # SD1.5 / SDXL conv UNet
    "dit": 6,  # DEFAULT for unmeasured transformers (cogview4, FLUX.1, future archs)
    "flux2": 2.2,  # FLUX.2 Klein 4B + 9B. Tracks the CONSTRAINED 9B peak (implied 2.1@1536, 2.6@1024).
    # The high 4B implies (~3.1) come from the fully-fitting 4B variant, which has VRAM slack and needs
    # no estimate margin; 3.6 (=3.1x1.15) over-reserved 9B past the 3GB cap and pinned it at ~35%. The
    # partial-load headroom (PARTIAL_LOAD_HEADROOM_MULTIPLIER) now supplies the margin for the partial 9B.
    "z_image": 4.5,  # Z-Image Turbo (2 points)
    "anima": 3.5,  # Anima (2 points, cfg=1)
    "sd3": 3.0,  # SD3.5 Medium (1 point; includes its always-on CFG 2x)
    "qwen": 8.0,  # Qwen-Image Edit (1 point; reference-image sequence inflation)
}

# Families that use the conv-UNet multiplier / enforcement rather than the transformer path. Both an
# estimate `family` and a DENOISE_MEM `label` resolve through family_multiplier()/family_enforced(),
# so a label like "sdxl-legacy" maps to the same multiplier as family "unet".
UNET_FAMILIES = frozenset({"unet", "sdxl", "sdxl-legacy", "sd15", "sd"})

# Fallback activation width if it can't be read from the model config (estimate then degrades to a
# flat per-area value, still floored by the cache).
DEFAULT_ACTIVATION_WIDTH = 320

# Fixed scratch overhead independent of resolution.
BASE_WORKING_MEMORY_BYTES = 64 * MB

# Per-family enforcement (see resolve_denoise_working_mem_bytes). A family is enforced once its
# multiplier is calibrated from our own measurements. Both are now calibrated (see ACTIVATION_MULTIPLIER).
ENFORCE_UNET_WORKING_MEMORY = True
ENFORCE_DIT_WORKING_MEMORY = True


def family_multiplier(family: str) -> float:
    """The calibrated activation multiplier for an architecture / family key.

    UNet families (incl. labels like ``"sdxl-legacy"``) map to the conv-UNet value; an unmeasured
    transformer falls back to the conservative ``"dit"`` default. Accepts either an estimate
    ``family`` or a DENOISE_MEM ``label`` — they resolve identically by construction.
    """
    if family in UNET_FAMILIES:
        return ACTIVATION_MULTIPLIER["unet"]
    return ACTIVATION_MULTIPLIER.get(family, ACTIVATION_MULTIPLIER["dit"])


def family_enforced(family: str) -> bool:
    """Whether the cache should reserve this family's estimate (vs measure-only)."""
    return ENFORCE_UNET_WORKING_MEMORY if family in UNET_FAMILIES else ENFORCE_DIT_WORKING_MEMORY


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
    :param family: an architecture/family key (e.g. ``"unet"``, ``"flux2"``, ``"sd3"``); selects the
        calibrated multiplier, with unknown transformers falling back to the conservative ``"dit"``.
    """
    multiplier = family_multiplier(family)
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

    :param family: an architecture/family key. UNet families use UNet enforcement; every other key
    uses the transformer enforcement flag. Returns the estimate only when that family's enforcement is
    enabled; otherwise ``None`` (measure-only: the cache keeps its default reserve).
    """
    return estimate_bytes if family_enforced(family) else None


def begin_denoise_measure(logger: _LevelLogger) -> Optional[tuple[int, float]]:
    """Snapshot allocator state and a wall-clock start immediately before a denoise loop.

    Only active when DEBUG logging is enabled (the ``DENOISE_MEM`` record is logged at debug level),
    so it adds zero overhead in normal operation. Returns ``(bytes_allocated, start_time)`` — the
    resident-weights baseline and a ``perf_counter`` start taken after a CUDA sync — or ``None`` if
    disabled / CUDA unavailable. Pass the result to :func:`end_denoise_measure`.
    """
    if not logger.isEnabledFor(logging.DEBUG) or not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return alloc_before, time.perf_counter()
    except Exception:
        return None


def end_denoise_measure(
    token: Optional[tuple[int, float]],
    logger: _LevelLogger,
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
    > 1 means we are over-estimating (would over-reserve if enforced). ``elapsed_ms`` is the
    GPU-synced wall time of the denoise loop, so an A/B (e.g. smart_partial_loading on vs off) can
    confirm that loading more of the model speeds up inference rather than overhead slowing it down.
    """
    if token is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        alloc_before, start_time = token
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        peak = torch.cuda.max_memory_allocated()
        measured = max(0, peak - alloc_before)
        ratio = round(estimate_bytes / measured, 2) if measured > 0 else None
        logger.debug(
            "DENOISE_MEM "
            + json.dumps(
                {
                    "label": label,
                    "px_h": int(pixel_height),
                    "px_w": int(pixel_width),
                    "batch": int(batch_size),
                    "elt": int(element_size),
                    "mult": family_multiplier(label),
                    "estimate_mb": round(estimate_bytes / MB, 1),
                    "measured_peak_mb": round(measured / MB, 1),
                    "resident_before_mb": round(alloc_before / MB, 1),
                    "total_peak_mb": round(peak / MB, 1),
                    "elapsed_ms": elapsed_ms,
                    "estimate_over_measured": ratio,
                    "enforced": family_enforced(label),
                }
            )
        )
    except Exception:
        pass

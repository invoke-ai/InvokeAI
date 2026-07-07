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

Usage (every denoise invocation follows this shape)::

    working_memory = estimate_denoise_working_memory_for_model(
        model=transformer_info.model, latent_height=..., latent_width=..., batch_size=...,
        inference_dtype=..., family="...",
    )
    with transformer_info.model_on_device(working_mem_bytes=working_memory.bytes) as (...):
        ...
        mem_probe = working_memory.measure(context.logger, pixel_height=..., pixel_width=...)
        # denoise loop
        mem_probe.end()

Calibration convention: estimates and DENOISE_MEM records use PRE-PACKING latent spatial dims
(``pixel // vae_scale_factor``); any token-packing factor is folded into the family multiplier. The
cleanest calibration data comes from fully-resident runs — under partial loading, streamed-weight
buffers count toward the measured peak.

Safety properties
-----------------
The cache honors an op-provided estimate down to ``MIN_DEVICE_WORKING_MEM_GB`` (model_cache.py) but
never below the user-configured reserve, and an absent estimate keeps the full default reserve.
Under-estimates are clamped up to the minimum; over-estimates only cost streaming speed.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol

import torch


class _LevelLogger(Protocol):
    """Minimal logger surface used for ``DENOISE_MEM`` diagnostics: a level check plus debug emit.

    Both the stdlib :class:`logging.Logger` and InvokeAI's ``LoggerInterface`` wrapper (what an
    invocation's ``context.logger`` is) satisfy this. The wrapper is NOT a real ``Logger``, so this
    module must not assume the full ``logging.Logger`` API — only ``isEnabledFor`` and ``debug``.
    """

    def isEnabledFor(self, level: int) -> bool: ...

    def debug(self, msg: str) -> None: ...


MB = 2**20
GB = 2**30

# The architecture/family keys with a calibrated multiplier. New transformer architectures without
# their own measurements should pass "dit" (the conservative default) until calibrated.
DenoiseFamily = Literal["unet", "dit", "flux2", "z_image", "anima", "sd3", "qwen"]

# Per-architecture activation multiplier (dimensionless: ~number of live activation copies at peak),
# CALIBRATED from our own DENOISE_MEM measurements on an 8GB card (RTX 4070). It is read per
# ARCHITECTURE, not one value for all transformers: reading activation_width from the model absorbs
# SIZE differences WITHIN a family (FLUX.2 Klein 4B and 9B both back-solve to ~2-3), but block-structure
# differences ACROSS families do not collapse to a single constant (measured implied multipliers:
# sd3 ~2.2, anima ~2.4-3.0, flux2 ~2.1-3.1, z_image ~3.5-3.8, qwen-Edit ~6.8). Each measured value
# below is roughly max-implied x 1.15 for margin.
#   - "unet": SDXL (width 320) measured ~28; kept at 32.
#   - "dit": calibrated from FLUX.1 (implied ~5.3); doubles as the conservative DEFAULT for
#     transformers we have NOT measured yet (cogview4, future archs), so an unmeasured arch never
#     under-reserves.
#   - sd3's value folds in its always-on CFG (a doubled-batch single forward). qwen is the Edit variant
#     (its reference image lengthens the sequence ~2x) from a single 1024 point. Both are floor-covered
#     at the resolutions they were run, so their exact value is not yet load-bearing.
ACTIVATION_MULTIPLIER: dict[str, float] = {
    "unet": 32,  # SD1.5 / SDXL conv UNet
    "dit": 6,  # FLUX.1 anchor + DEFAULT for unmeasured transformers (cogview4, future archs)
    "flux2": 2.2,  # FLUX.2 Klein 4B + 9B. Tracks the CONSTRAINED 9B peak (implied 2.1@1536, 2.6@1024).
    # The high 4B implies (~3.1) come from the fully-fitting 4B variant, which has VRAM slack and needs
    # no estimate margin; 3.6 (=3.1x1.15) over-reserved 9B past the 3GB cap and pinned it at ~35%. The
    # partial-load headroom (PARTIAL_LOAD_HEADROOM_MULTIPLIER) now supplies the margin for the partial 9B.
    "z_image": 4.5,  # Z-Image Turbo (2 points)
    "anima": 3.5,  # Anima (2 points, cfg=1)
    "sd3": 3.0,  # SD3.5 Medium (1 point; includes its always-on CFG 2x)
    "qwen": 8.0,  # Qwen-Image Edit (1 point; reference-image sequence inflation)
}

# Fallback activation width if it can't be read from the model config (estimate then degrades to a
# flat per-area value, still floored by the cache).
DEFAULT_ACTIVATION_WIDTH = 320

# Fixed scratch overhead independent of resolution.
BASE_WORKING_MEMORY_BYTES = 64 * MB


def family_multiplier(family: str) -> float:
    """The calibrated activation multiplier for an architecture/family key; an unmeasured or unknown
    family falls back to the conservative ``"dit"`` default."""
    return ACTIVATION_MULTIPLIER.get(family, ACTIVATION_MULTIPLIER["dit"])


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
            # "dim" (probed last, most generic name) covers Z-Image-shaped configs.
            for attr in ("inner_dim", "hidden_size", "joint_attention_dim", "cross_attention_dim", "d_model", "dim"):
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

    :param latent_area: latent spatial area = latent_height * latent_width (pre-packing).
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


def dtype_element_size(dtype: torch.dtype) -> int:
    """Bytes per element for a torch dtype, robust across torch versions."""
    try:
        return torch.empty((), dtype=dtype).element_size()
    except Exception:
        return 2  # safe default (fp16/bf16)


class DenoiseMemProbe:
    """Measures a denoise loop's peak VRAM delta and GPU-synced wall time, emitted as one
    ``DENOISE_MEM`` debug record by :meth:`end`.

    ``measured_peak_mb`` is the extra VRAM the loop allocated on top of the resident weights — the
    real working-memory need the estimate should match (``estimate_over_measured`` > 1 means
    over-reserving). ``elapsed_ms`` lets an A/B (e.g. smart_partial_loading on vs off) confirm that
    loading more of the model speeds up inference.

    Active only when DEBUG logging is enabled and CUDA is available; otherwise construction and
    ``end()`` are no-ops, so it adds zero overhead in normal operation. Never raises.
    """

    def __init__(
        self,
        estimate: "DenoiseWorkingMemory",
        logger: _LevelLogger,
        pixel_height: int,
        pixel_width: int,
        label: str,
    ) -> None:
        self._estimate = estimate
        self._logger = logger
        self._pixel_height = pixel_height
        self._pixel_width = pixel_width
        self._label = label
        self._alloc_before = 0
        self._start_time = 0.0
        self.active = False
        if not logger.isEnabledFor(logging.DEBUG) or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
            self._alloc_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            self._start_time = time.perf_counter()
            self.active = True
        except Exception:
            self.active = False

    def end(self) -> None:
        """Emit the ``DENOISE_MEM`` record (idempotent; no-op when inactive)."""
        if not self.active:
            return
        self.active = False
        try:
            torch.cuda.synchronize()
            elapsed_ms = round((time.perf_counter() - self._start_time) * 1000, 1)
            peak = torch.cuda.max_memory_allocated()
            measured = max(0, peak - self._alloc_before)
            estimate_bytes = self._estimate.bytes
            ratio = round(estimate_bytes / measured, 2) if measured > 0 else None
            self._logger.debug(
                "DENOISE_MEM "
                + json.dumps(
                    {
                        "label": self._label,
                        "px_h": int(self._pixel_height),
                        "px_w": int(self._pixel_width),
                        "batch": int(self._estimate.batch_size),
                        "elt": int(self._estimate.element_size),
                        "mult": self._estimate.multiplier,
                        "estimate_mb": round(estimate_bytes / MB, 1),
                        "measured_peak_mb": round(measured / MB, 1),
                        "resident_before_mb": round(self._alloc_before / MB, 1),
                        "total_peak_mb": round(peak / MB, 1),
                        "elapsed_ms": elapsed_ms,
                        "estimate_over_measured": ratio,
                    }
                )
            )
        except Exception:
            pass


@dataclass(frozen=True)
class DenoiseWorkingMemory:
    """An architecture-scaled working-memory estimate for one denoise operation.

    Pass :attr:`bytes` to ``model_on_device(working_mem_bytes=...)`` so the cache reserves it, and
    wrap the denoise loop with :meth:`measure` / ``probe.end()`` to log the estimate against the
    measured peak for calibration.
    """

    bytes: int
    family: str
    multiplier: float
    element_size: int
    batch_size: int

    def measure(
        self,
        logger: _LevelLogger,
        *,
        pixel_height: int,
        pixel_width: int,
        label: Optional[str] = None,
    ) -> DenoiseMemProbe:
        """Start a ``DENOISE_MEM`` measurement window over the denoise loop; call ``end()`` on the
        returned probe after the loop.

        :param label: record label; defaults to the family. Override where one family covers several
            code paths (e.g. family ``"unet"`` emitting as ``"sdxl"`` vs ``"sdxl-legacy"``).
        """
        return DenoiseMemProbe(
            estimate=self,
            logger=logger,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            label=label or self.family,
        )


def estimate_denoise_working_memory_for_model(
    model: Any,
    latent_height: int,
    latent_width: int,
    batch_size: int,
    inference_dtype: torch.dtype,
    family: DenoiseFamily,
) -> DenoiseWorkingMemory:
    """Estimate denoise working memory for a loaded model, scaling by its activation width.

    :param model: the loaded denoise model (its config supplies the activation width).
    :param latent_height: latent-space height (pre-packing), i.e. ``pixel_height // vae_scale_factor``.
    :param latent_width: latent-space width (pre-packing).
    :param family: the architecture/family key; selects the calibrated multiplier, with unmeasured
        transformers passing ``"dit"`` (the conservative default).
    """
    multiplier = family_multiplier(family)
    element_size = dtype_element_size(inference_dtype)
    working_mem_bytes = estimate_denoise_working_memory(
        latent_area=int(latent_height) * int(latent_width),
        activation_width=model_activation_width(model),
        batch_size=batch_size,
        element_size=element_size,
        multiplier=multiplier,
    )
    return DenoiseWorkingMemory(
        bytes=working_mem_bytes,
        family=family,
        multiplier=multiplier,
        element_size=element_size,
        batch_size=max(1, int(batch_size)),
    )

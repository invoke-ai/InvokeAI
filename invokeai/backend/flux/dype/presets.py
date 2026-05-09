"""DyPE presets and automatic configuration."""

import math
from dataclasses import dataclass
from typing import Literal

from invokeai.backend.flux.dype.base import DyPEConfig

# DyPE preset type - using Literal for proper frontend dropdown support
DyPEPreset = Literal["off", "manual", "auto", "area", "4k"]

# Constants for preset values
DYPE_PRESET_OFF: DyPEPreset = "off"
DYPE_PRESET_MANUAL: DyPEPreset = "manual"
DYPE_PRESET_AUTO: DyPEPreset = "auto"
DYPE_PRESET_AREA: DyPEPreset = "area"
DYPE_PRESET_4K: DyPEPreset = "4k"

# Human-readable labels for the UI
DYPE_PRESET_LABELS: dict[str, str] = {
    "off": "Off",
    "manual": "Manual",
    "auto": "Auto (>1536px)",
    "area": "Area (auto)",
    "4k": "4K Optimized",
}


@dataclass
class DyPEPresetConfig:
    """Preset configuration values."""

    base_resolution: int
    dype_scale: float
    dype_exponent: float
    dype_start_sigma: float


# Predefined preset configurations
DYPE_PRESETS: dict[DyPEPreset, DyPEPresetConfig] = {
    DYPE_PRESET_4K: DyPEPresetConfig(
        base_resolution=1024,
        dype_scale=2.0,
        dype_exponent=2.0,
        dype_start_sigma=1.0,
    ),
}


def get_dype_config_for_resolution(
    width: int,
    height: int,
    base_resolution: int = 1024,
    activation_threshold: int = 1536,
) -> DyPEConfig | None:
    """Automatically determine DyPE config based on target resolution.

    FLUX can handle resolutions up to ~1.5x natively without significant artifacts.
    DyPE is only activated when the resolution exceeds the activation threshold.

    Args:
        width: Target image width in pixels
        height: Target image height in pixels
        base_resolution: Native training resolution of the model (for scale calculation)
        activation_threshold: Resolution threshold above which DyPE is activated

    Returns:
        DyPEConfig if DyPE should be enabled, None otherwise
    """
    max_dim = max(width, height)

    if max_dim <= activation_threshold:
        return None  # FLUX can handle this natively

    # Calculate scaling factor based on base_resolution
    scale = max_dim / base_resolution

    # Dynamic parameters based on scaling
    # Higher resolution = higher dype_scale, capped at 8.0
    dynamic_dype_scale = min(2.0 * scale, 8.0)

    return DyPEConfig(
        enable_dype=True,
        base_resolution=base_resolution,
        dype_scale=dynamic_dype_scale,
        dype_exponent=2.0,
        dype_start_sigma=1.0,
    )


def get_dype_config_for_area(
    width: int,
    height: int,
    base_resolution: int = 1024,
) -> DyPEConfig | None:
    """Automatically determine DyPE config based on target area.

    Uses sqrt(area/base_area) as an effective side-length ratio.
    DyPE is enabled only when target area exceeds base area.

    Returns:
        DyPEConfig if DyPE should be enabled, None otherwise
    """
    area = width * height
    base_area = base_resolution**2

    if area <= base_area:
        return None

    area_ratio = area / base_area
    effective_side_ratio = math.sqrt(area_ratio)
    aspect_ratio = max(width, height) / min(width, height)
    aspect_attenuation = 1.0 if aspect_ratio <= 2.0 else 2.0 / aspect_ratio

    # Retune area mode to be "auto, but area-aware" instead of dramatically
    # stronger than auto. This keeps it closer to the paper-style core DyPE.
    dynamic_dype_scale = 2.4 * effective_side_ratio
    dynamic_dype_scale *= aspect_attenuation
    dynamic_dype_scale = max(0.0, min(dynamic_dype_scale, 8.0))

    # Use a narrower, higher exponent range than the old area heuristic so the
    # paper-style scheduler decays more conservatively and artifacts are reduced.
    exponent_progress = max(0.0, min(effective_side_ratio - 1.0, 1.0))
    dype_exponent = 1.25 + 0.75 * exponent_progress

    return DyPEConfig(
        enable_dype=True,
        base_resolution=base_resolution,
        dype_scale=dynamic_dype_scale,
        dype_exponent=dype_exponent,
        dype_start_sigma=1.0,
    )


def get_dype_config_from_preset(
    preset: DyPEPreset,
    width: int,
    height: int,
    custom_scale: float | None = None,
    custom_exponent: float | None = None,
) -> DyPEConfig | None:
    """Get DyPE configuration from a preset or custom values.

    Args:
        preset: The DyPE preset to use
        width: Target image width
        height: Target image height
        custom_scale: Optional custom dype_scale (only used with 'manual' preset)
        custom_exponent: Optional custom dype_exponent (only used with 'manual' preset)

    Returns:
        DyPEConfig if DyPE should be enabled, None otherwise
    """
    if preset == DYPE_PRESET_OFF:
        return None

    if preset == DYPE_PRESET_MANUAL:
        # Manual mode - custom values can override defaults
        max_dim = max(width, height)
        scale = max_dim / 1024
        dynamic_dype_scale = min(2.0 * scale, 8.0)
        return DyPEConfig(
            enable_dype=True,
            base_resolution=1024,
            dype_scale=custom_scale if custom_scale is not None else dynamic_dype_scale,
            dype_exponent=custom_exponent if custom_exponent is not None else 2.0,
            dype_start_sigma=1.0,
        )

    if preset == DYPE_PRESET_AUTO:
        # Auto preset - custom values are ignored
        return get_dype_config_for_resolution(
            width=width,
            height=height,
            base_resolution=1024,
            activation_threshold=1536,
        )

    if preset == DYPE_PRESET_AREA:
        # Area-based preset - custom values are ignored
        return get_dype_config_for_area(
            width=width,
            height=height,
            base_resolution=1024,
        )

    # Use preset configuration (4K etc.) - custom values are ignored
    preset_config = DYPE_PRESETS.get(preset)
    if preset_config is None:
        return None

    return DyPEConfig(
        enable_dype=True,
        base_resolution=preset_config.base_resolution,
        dype_scale=preset_config.dype_scale,
        dype_exponent=preset_config.dype_exponent,
        dype_start_sigma=preset_config.dype_start_sigma,
    )

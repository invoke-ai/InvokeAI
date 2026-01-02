"""DyPE presets and automatic configuration."""

from dataclasses import dataclass
from enum import Enum

from invokeai.backend.flux.dype.base import DyPEConfig


class DyPEPreset(str, Enum):
    """Predefined DyPE configurations."""

    OFF = "off"  # DyPE disabled
    AUTO = "auto"  # Automatically enable based on resolution
    PRESET_4K = "4k"  # Optimized for 3840x2160 / 4096x2160


@dataclass
class DyPEPresetConfig:
    """Preset configuration values."""

    base_resolution: int
    method: str
    dype_scale: float
    dype_exponent: float
    dype_start_sigma: float


# Predefined preset configurations
DYPE_PRESETS: dict[DyPEPreset, DyPEPresetConfig] = {
    DyPEPreset.PRESET_4K: DyPEPresetConfig(
        base_resolution=1024,
        method="vision_yarn",
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
        method="vision_yarn",
        dype_scale=dynamic_dype_scale,
        dype_exponent=2.0,
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
        custom_scale: Optional custom dype_scale (overrides preset)
        custom_exponent: Optional custom dype_exponent (overrides preset)

    Returns:
        DyPEConfig if DyPE should be enabled, None otherwise
    """
    if preset == DyPEPreset.OFF:
        # Check if custom values are provided even with preset=OFF
        if custom_scale is not None:
            return DyPEConfig(
                enable_dype=True,
                base_resolution=1024,
                method="vision_yarn",
                dype_scale=custom_scale,
                dype_exponent=custom_exponent if custom_exponent is not None else 2.0,
                dype_start_sigma=1.0,
            )
        return None

    if preset == DyPEPreset.AUTO:
        config = get_dype_config_for_resolution(
            width=width,
            height=height,
            base_resolution=1024,
            activation_threshold=1536,
        )
        # Apply custom overrides if provided
        if config is not None:
            if custom_scale is not None:
                config.dype_scale = custom_scale
            if custom_exponent is not None:
                config.dype_exponent = custom_exponent
        return config

    # Use preset configuration
    preset_config = DYPE_PRESETS.get(preset)
    if preset_config is None:
        return None

    return DyPEConfig(
        enable_dype=True,
        base_resolution=preset_config.base_resolution,
        method=preset_config.method,
        dype_scale=custom_scale if custom_scale is not None else preset_config.dype_scale,
        dype_exponent=custom_exponent if custom_exponent is not None else preset_config.dype_exponent,
        dype_start_sigma=preset_config.dype_start_sigma,
    )

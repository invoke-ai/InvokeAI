from dataclasses import dataclass
from enum import Enum


class RasterError(ValueError):
    """Base class for raster type errors."""


class RasterDescriptorError(RasterError):
    """The descriptor, or the array it is paired with, is inconsistent."""


class RasterConversionError(RasterError):
    """A conversion is unsupported or would silently change the pixel meaning."""


class RasterChannels(str, Enum):
    L = "L"
    LA = "LA"
    RGB = "RGB"
    RGBA = "RGBA"
    DATA = "data"


class ColorEncoding(str, Enum):
    SRGB = "srgb"
    LINEAR_SRGB = "linear_srgb"
    NONE = "none"  # non-color data


class ValueRange(str, Enum):
    UNIT = "unit"  # nominal [0, 1]; integers span their full scale
    UNBOUNDED = "unbounded"  # arbitrary float values, no display meaning without an explicit mapping


class AlphaMode(str, Enum):
    NONE = "none"
    STRAIGHT = "straight"
    PREMULTIPLIED = "premultiplied"


FIXED_CHANNEL_COUNTS = {
    RasterChannels.L: 1,
    RasterChannels.LA: 2,
    RasterChannels.RGB: 3,
    RasterChannels.RGBA: 4,
}


@dataclass(frozen=True)
class RasterDescriptor:
    """How to interpret the numbers of a raster. Validated at construction."""

    channels: RasterChannels
    color: ColorEncoding
    value_range: ValueRange
    alpha: AlphaMode
    channel_count: int
    icc_profile: bytes | None = None

    def __post_init__(self) -> None:
        # Valid strings (as stored in files) are coerced to the enums; anything else is rejected.
        for name, enum_type in (
            ("channels", RasterChannels),
            ("color", ColorEncoding),
            ("value_range", ValueRange),
            ("alpha", AlphaMode),
        ):
            try:
                object.__setattr__(self, name, enum_type(getattr(self, name)))
            except (ValueError, TypeError):
                raise RasterDescriptorError(f"invalid {name}: {getattr(self, name)!r}") from None
        if not isinstance(self.channel_count, int) or isinstance(self.channel_count, bool):
            raise RasterDescriptorError(f"channel_count must be an int, got {self.channel_count!r}")
        if self.icc_profile is not None and not isinstance(self.icc_profile, bytes):
            raise RasterDescriptorError("icc_profile must be bytes or None")
        if self.channels is RasterChannels.DATA:
            if self.channel_count < 1:
                raise RasterDescriptorError("DATA rasters need at least one channel")
            if self.alpha is not AlphaMode.NONE:
                raise RasterDescriptorError("DATA rasters cannot carry alpha")
            if self.color is not ColorEncoding.NONE:
                raise RasterDescriptorError("DATA rasters must use ColorEncoding.NONE")
        else:
            expected = FIXED_CHANNEL_COUNTS[self.channels]
            if self.channel_count != expected:
                raise RasterDescriptorError(
                    f"{self.channels.value} needs channel_count {expected}, got {self.channel_count}"
                )
            if self.color is ColorEncoding.NONE:
                raise RasterDescriptorError(f"{self.channels.value} rasters need a color encoding")
            has_alpha_channel = self.channels in (RasterChannels.LA, RasterChannels.RGBA)
            if has_alpha_channel != (self.alpha is not AlphaMode.NONE):
                raise RasterDescriptorError(
                    f"alpha mode {self.alpha.value} does not match channels {self.channels.value}"
                )
        if self.icc_profile is not None and self.color is ColorEncoding.NONE:
            raise RasterDescriptorError("icc_profile requires a color encoding")

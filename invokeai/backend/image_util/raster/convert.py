"""Explicit conversions between RasterImage and PIL, torch and other sample formats.

Every conversion copies. None of them applies a transfer function: the color encoding is carried, never changed.
"""

import numpy as np
import torch
from PIL import Image

from invokeai.backend.image_util.raster.descriptor import (
    FIXED_CHANNEL_COUNTS,
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterConversionError,
    RasterDescriptor,
    ValueRange,
)
from invokeai.backend.image_util.raster.raster import RasterImage

_PIL_MODE_CHANNELS = {
    "L": RasterChannels.L,
    "LA": RasterChannels.LA,
    "RGB": RasterChannels.RGB,
    "RGBA": RasterChannels.RGBA,
}
_PIL_16_BIT_MODES = ("I;16", "I;16L", "I;16B", "I;16N")
_INTEGER_FULL_SCALE = {np.dtype(np.uint8): 255, np.dtype(np.uint16): 65535}
# Conversions run over row blocks of about this many samples, so temporaries stay small next to the image.
_BLOCK_SAMPLES = 1 << 20
_TORCH_DTYPES = {
    torch.uint8: np.dtype(np.uint8),
    torch.uint16: np.dtype(np.uint16),
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
}


def _srgb_descriptor(channels: RasterChannels, icc_profile: bytes | None) -> RasterDescriptor:
    has_alpha = channels in (RasterChannels.LA, RasterChannels.RGBA)
    return RasterDescriptor(
        channels=channels,
        color=ColorEncoding.SRGB,
        value_range=ValueRange.UNIT,
        alpha=AlphaMode.STRAIGHT if has_alpha else AlphaMode.NONE,
        channel_count=FIXED_CHANNEL_COUNTS[channels],
        icc_profile=icc_profile,
    )


def raster_from_pil(image: Image.Image) -> RasterImage:
    """Owned raster from a PIL image: 8-bit L/LA/RGB/RGBA, 16-bit grayscale, `1`, `P` and `PA`; other modes are rejected.

    Palettes and tRNS color keys become alpha when the image has transparency.
    """
    icc_profile = image.info.get("icc_profile")
    if not isinstance(icc_profile, bytes) or not icc_profile:
        icc_profile = None

    mode = image.mode
    if mode in _PIL_16_BIT_MODES:
        pixels = np.asarray(image).astype(np.uint16)[:, :, np.newaxis]
        key = image.info.get("transparency")
        if isinstance(key, int):
            # Pillow has no 16-bit LA: the gray color key becomes a straight uint16 alpha channel.
            alpha = np.where(pixels == key, 0, 65535).astype(np.uint16)
            pixels = np.concatenate([pixels, alpha], axis=2)
            return RasterImage._adopt(pixels, _srgb_descriptor(RasterChannels.LA, icc_profile))
        return RasterImage._adopt(pixels, _srgb_descriptor(RasterChannels.L, icc_profile))
    if mode == "1":
        image = image.convert("LA" if image.has_transparency_data else "L")
    elif mode in ("P", "PA"):
        # has_transparency_data also covers palettes with an alpha column, which carry no info["transparency"].
        image = image.convert("RGBA" if image.has_transparency_data else "RGB")
    elif mode in ("L", "RGB") and image.has_transparency_data:
        # A tRNS color key becomes a straight alpha channel (Pillow maps the keyed color to alpha 0).
        image = image.convert("LA" if mode == "L" else "RGBA")
    elif mode not in _PIL_MODE_CHANNELS:
        raise RasterConversionError(f"unsupported PIL mode {mode}")

    pixels = np.array(image, dtype=np.uint8)
    if pixels.ndim == 2:
        pixels = pixels[:, :, np.newaxis]
    return RasterImage._adopt(pixels, _srgb_descriptor(_PIL_MODE_CHANNELS[image.mode], icc_profile))


def pil_refusal(raster: RasterImage) -> str | None:
    """Why `raster_to_pil` refuses this raster, or None when a gallery PNG represents it exactly."""
    descriptor = raster.descriptor
    if (
        descriptor.channels is RasterChannels.DATA
        or descriptor.color is not ColorEncoding.SRGB
        or descriptor.alpha is AlphaMode.PREMULTIPLIED
    ):
        return (
            f"cannot export {descriptor.channels.value}/{descriptor.color.value}/{descriptor.alpha.value} "
            "to PIL without an explicit conversion"
        )
    if raster.dtype == np.uint8 or (raster.dtype == np.uint16 and descriptor.channels is RasterChannels.L):
        return None
    return f"cannot export {raster.dtype} {descriptor.channels.value} to PIL"


def raster_to_pil(raster: RasterImage, *, share_pixels: bool = False) -> Image.Image:
    """PIL image from uint8 L/LA/RGB/RGBA or uint16 L sRGB straight-alpha rasters; anything else must be converted first.

    The image is independent of the raster. `share_pixels=True` skips that copy for a caller that only reads the image
    and drops the raster with it (a read-only temporary): the image then wraps the raster's read-only pixels.
    """
    descriptor = raster.descriptor
    refusal = pil_refusal(raster)
    if refusal is not None:
        raise RasterConversionError(refusal)
    if raster.dtype == np.uint8:
        pixels = raster.view()
        mode = descriptor.channels.value
    else:
        pixels = raster.view().astype("<u2", copy=False)  # "I;16" is little-endian
        mode = None  # Pillow infers "I;16" from a 2-D uint16 array

    # fromarray shares the (read-only) buffer; the copy makes the image independent of the raster.
    image = Image.fromarray(pixels[:, :, 0] if pixels.shape[2] == 1 else pixels, mode)
    if not share_pixels:
        image = image.copy()
    if descriptor.icc_profile is not None:
        image.info["icc_profile"] = descriptor.icc_profile
    return image


def raster_from_torch(tensor_chw: torch.Tensor, descriptor: RasterDescriptor) -> RasterImage:
    """Owned raster copied from an unbatched CHW tensor (any device). bfloat16 must be widened by the caller."""
    if tensor_chw.ndim != 3:
        raise RasterConversionError(f"expected an unbatched CHW tensor, got shape {tuple(tensor_chw.shape)}")
    if tensor_chw.dtype not in _TORCH_DTYPES:
        raise RasterConversionError(f"unsupported tensor dtype {tensor_chw.dtype}")
    hwc = tensor_chw.detach().to("cpu").permute(1, 2, 0).numpy()
    return RasterImage.from_array(hwc, descriptor)


def decoded_unit_raster(unit_chw: torch.Tensor) -> RasterImage | None:
    """Owned sRGB unit-range RGB raster from the first sample of a decoder output in unit space (CHW or NCHW).

    The values are not clamped: overshoot a gallery PNG would clip is kept. float16 and float32 keep their precision,
    any other dtype (bfloat16) is widened to float32. Returns None when the sample has a non-finite value, which the
    check finds on the tensor's own device before the host copy.
    """
    sample = unit_chw[0] if unit_chw.ndim == 4 else unit_chw
    if not torch.isfinite(sample).all().item():
        return None
    if sample.dtype not in (torch.float16, torch.float32):
        sample = sample.float()
    return raster_from_torch(sample, _srgb_descriptor(RasterChannels.RGB, None))


def raster_to_torch(raster: RasterImage, device: torch.device | str | None = None) -> torch.Tensor:
    """CHW tensor holding a copy of the pixels, optionally moved to `device`."""
    chw = np.array(raster.view().transpose(2, 0, 1), order="C", copy=True)
    tensor = torch.from_numpy(chw)
    return tensor if device is None else tensor.to(device)


def _map_row_blocks(source: np.ndarray, dtype: np.dtype, convert) -> np.ndarray:
    """Apply `convert` to row blocks of `source`, writing a new C-contiguous array of `dtype`."""
    out = np.empty(source.shape, dtype)
    rows = max(1, _BLOCK_SAMPLES // (source.shape[1] * source.shape[2]))
    for start in range(0, source.shape[0], rows):
        np.copyto(out[start : start + rows], convert(source[start : start + rows]), casting="unsafe")
    return out


def quantize_unit(raster: RasterImage, dtype: type[np.uint8] | type[np.uint16]) -> RasterImage:
    """Display-quantize a UNIT float raster to uint8/uint16 (clip, scale, round half to even), or rescale integers.

    uint8 -> uint16 multiplies by 257; uint16 -> uint8 rounds v / 257. Float input must be finite.
    """
    target = np.dtype(dtype)
    if target not in _INTEGER_FULL_SCALE:
        raise RasterConversionError(f"cannot quantize to {target}")
    descriptor = raster.descriptor
    source = raster.view()

    if source.dtype == target:
        return RasterImage.from_array(source, descriptor)
    if source.dtype.kind == "u":
        if target == np.uint16:

            def convert(block):
                wide = block.astype(np.uint16)
                wide *= 257
                return wide
        else:

            def convert(block):
                # 257 is odd, so v / 257 never lands exactly on a tie: round half up and half even coincide.
                wide = block.astype(np.uint32)
                wide += 128
                wide //= 257
                return wide

        return RasterImage._adopt(_map_row_blocks(source, target, convert), descriptor)

    if descriptor.value_range is not ValueRange.UNIT:
        raise RasterConversionError("only ValueRange.UNIT rasters can be quantized")
    full_scale = _INTEGER_FULL_SCALE[target]

    def convert(block):
        if not np.isfinite(block).all():
            raise RasterConversionError("cannot quantize non-finite values")
        # float64 keeps value * 255 / 65535 exact for float16/float32 inputs, so rint sees true ties.
        scaled = block.astype(np.float64)
        np.clip(scaled, 0.0, 1.0, out=scaled)
        np.multiply(scaled, full_scale, out=scaled)
        return np.rint(scaled, out=scaled)

    return RasterImage._adopt(_map_row_blocks(source, target, convert), descriptor)


def to_unit_float(raster: RasterImage, dtype: type[np.float16] | type[np.float32]) -> RasterImage:
    """UNIT float raster from an integer UNIT raster (divide by full scale) or a float cast.

    float32 targets divide in float32: the integer and the full scale are exactly representable, so the correctly
    rounded float32 quotient equals the float64 quotient cast to float32. float16 targets divide in float64.
    """
    target = np.dtype(dtype)
    if target not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise RasterConversionError(f"cannot convert to {target}")
    source = raster.view()

    if source.dtype.kind == "u":
        full_scale = _INTEGER_FULL_SCALE[source.dtype]
        work_dtype = np.float32 if target == np.float32 else np.float64

        def convert(block):
            values = block.astype(work_dtype)
            values /= work_dtype(full_scale)
            return values

    elif source.dtype == np.float32 and target == np.float16:

        def convert(block):
            with np.errstate(over="ignore"):
                narrowed = block.astype(np.float16)
            if np.isinf(narrowed).sum() != np.isinf(block).sum():
                raise RasterConversionError(f"values exceed the {target} range")
            return narrowed

    else:

        def convert(block):
            return block

    return RasterImage._adopt(_map_row_blocks(source, target, convert), raster.descriptor)

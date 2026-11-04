# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
import torch
from PIL import Image

from invokeai.backend.image_util.color_conversion import (
    gamut_clip_tensor,
    linear_srgb_from_oklab,
    oklab_from_srgb,
    srgb_from_linear_srgb,
)
from invokeai.backend.image_util.raster.resample import MIN_VISIBLE_ALPHA

_SUPPORTED_MODES = {"RGB", "RGBA", "L", "LA"}
_TILE_ROWS = 256
# Linear sRGB overshoot treated as conversion noise and clamped rather than gamut clipped. Converting stored
# float32 Oklab back reaches about 2e-7 past a 0 or 1 channel, and gamut_clip_tensor visibly moves colors that
# sit on the gamut boundary, so a flat in-gamut color must not reach it. 1e-5 is under 0.03 of an 8-bit level.
_GAMUT_TOLERANCE = 1e-5


def _check_image(image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow Image")
    if image.mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported image mode: {image.mode}")


def _check_pixels(pixels: np.ndarray) -> None:
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.float32 or pixels.ndim != 3:
        raise ValueError("pixels must be an HxWxC float32 NumPy array")
    if pixels.shape[2] not in (1, 2, 3, 4):
        raise ValueError("pixels must have 1 to 4 channels")


def _with_hidden_filled(rgb: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """`rgb` (HxWx3, uint8 or float32) with the pixels where `hidden` is set replaced by their nearest visible pixel.

    Hidden pixels carry arbitrary color; filling them keeps it out of the filters' neighbourhoods. Returns `rgb`
    itself when nothing is hidden, else a copy. All hidden pixels become 0 when none is visible.
    """
    if not hidden.any():
        return rgb
    rgb = rgb.copy()
    rgb[hidden] = 0
    if not hidden.all():
        _, labels = cv2.distanceTransformWithLabels(
            np.ascontiguousarray(hidden, dtype=np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        visible = ~hidden
        label_to_rgb = np.zeros((int(labels.max()) + 1, 3), dtype=rgb.dtype)
        label_to_rgb[labels[visible]] = rgb[visible]
        rgb[hidden] = label_to_rgb[labels[hidden]]
    return rgb


def _oklab_from_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Oklab (3xHxW float32) of HxWx3 sRGB: uint8 levels, or finite float32 unit values (beyond 0-1 allowed).

    Converted in float64 row tiles and stored as float32, so the stored values do not depend on how torch
    vectorizes float32 arithmetic and the canvas preview's port reproduces them.
    """
    height, width = rgb.shape[:2]
    oklab = torch.empty((3, height, width), dtype=torch.float32)
    for row_start in range(0, height, _TILE_ROWS):
        row_end = min(row_start + _TILE_ROWS, height)
        # np.array: a writable copy of the tile (the source may be read-only), which torch.from_numpy requires.
        tile = torch.from_numpy(np.array(rgb[row_start:row_end])).permute(2, 0, 1).to(dtype=torch.float64)
        if rgb.dtype == np.uint8:
            tile.div_(255.0)
        elif not torch.isfinite(tile).all():
            raise ValueError("pixels must contain only finite values")
        oklab[:, row_start:row_end] = oklab_from_srgb(tile)
    return oklab


def _restored_srgb_tiles(color_tensor: torch.Tensor, lightness: np.ndarray) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Row tiles (start, end, 3xhxw float64 sRGB clamped to 0-1) of the Oklab image with its lightness replaced.

    The lightness is clamped to [0, 1] and out-of-gamut pixels are gamut-clipped, so every restored pixel lands in 0-1. Callers quantize or store the tiles.
    """
    height = color_tensor.shape[1]
    for row_start in range(0, height, _TILE_ROWS):
        row_end = min(row_start + _TILE_ROWS, height)
        tile_color = color_tensor[:, row_start:row_end]
        if not torch.isfinite(tile_color).all():
            raise ValueError("color_tensor must contain only finite values")
        tile_lightness = lightness[row_start:row_end]
        if not np.isfinite(tile_lightness).all():
            raise ValueError("lightness must contain only finite values")

        # float64 from here on: in float32 the gamut clip's cusp approximation can switch branches on rounding
        # noise at hues such as pure blue, making the result depend on the tile.
        adjusted = tile_color.to(dtype=torch.float64)
        adjusted[0] = torch.from_numpy(np.ascontiguousarray(tile_lightness)).clamp(0.0, 1.0)
        linear_rgb = linear_srgb_from_oklab(adjusted)
        if not torch.isfinite(linear_rgb).all():
            raise ValueError("lightness produced non-finite RGB values")
        out_of_gamut = (linear_rgb.min(0).values < -_GAMUT_TOLERANCE) | (
            linear_rgb.max(0).values > 1.0 + _GAMUT_TOLERANCE
        )
        if torch.any(out_of_gamut):
            # The clip is per pixel, so only the pixels that need it pay for it.
            linear_rgb[:, out_of_gamut] = gamut_clip_tensor(linear_rgb[:, out_of_gamut].unsqueeze(-1)).squeeze(-1)
        yield row_start, row_end, srgb_from_linear_srgb(linear_rgb).clamp(0.0, 1.0)


def prepare_lightness(image: Image.Image) -> tuple[torch.Tensor, np.ndarray, np.ndarray | None]:
    """Convert an 8-bit Pillow image to Oklab, lightness, and optional alpha."""
    try:
        _check_image(image)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        alpha: np.ndarray | None = None
        if image.mode in {"RGBA", "LA"}:
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint8).copy()
            rgb = _with_hidden_filled(rgb, alpha == 0)
        oklab = _oklab_from_rgb(rgb)
        lightness = np.ascontiguousarray(oklab[0].numpy(), dtype=np.float32)
        return oklab, lightness, alpha
    except MemoryError as exc:
        raise RuntimeError("Not enough memory to prepare image color data") from exc


def restore_lightness(
    image: Image.Image,
    color_tensor: torch.Tensor,
    lightness: np.ndarray,
    alpha: np.ndarray | None,
) -> Image.Image:
    """Replace Oklab lightness and restore the result in the image's original mode."""
    try:
        _check_image(image)
        width, height = image.size
        if not isinstance(color_tensor, torch.Tensor) or color_tensor.shape != (3, height, width):
            raise ValueError("color_tensor must have shape (3, H, W) matching the image")
        if color_tensor.device.type != "cpu" or color_tensor.dtype != torch.float32:
            raise ValueError("color_tensor must be a CPU float32 tensor")
        if not isinstance(lightness, np.ndarray) or lightness.shape != (height, width) or lightness.dtype != np.float32:
            raise ValueError("lightness must be an HxW float32 NumPy array matching the image")

        source_alpha: np.ndarray | None = None
        if image.mode in {"RGBA", "LA"}:
            if not isinstance(alpha, np.ndarray) or alpha.shape != (height, width) or alpha.dtype != np.uint8:
                raise ValueError("alpha must be an HxW uint8 array matching the image")
            source_alpha = np.ascontiguousarray(alpha)
        elif alpha is not None:
            raise ValueError("alpha must be None for images without an alpha channel")

        original = np.asarray(image, dtype=np.uint8) if source_alpha is not None else None
        grayscale = image.mode in {"L", "LA"}
        restored = np.empty((height, width) if grayscale else (height, width, 3), dtype=np.uint8)
        for row_start, row_end, srgb in _restored_srgb_tiles(color_tensor, lightness):
            if grayscale:
                gray = srgb[0] * 0.299 + srgb[1] * 0.587 + srgb[2] * 0.114
                restored[row_start:row_end] = gray.mul(255.0).round().to(torch.uint8).numpy()
                if source_alpha is not None and original is not None:
                    hidden = source_alpha[row_start:row_end] == 0
                    restored[row_start:row_end][hidden] = original[row_start:row_end][hidden, 0]
            else:
                restored[row_start:row_end] = (
                    srgb.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
                )
                if source_alpha is not None and original is not None:
                    hidden = source_alpha[row_start:row_end] == 0
                    restored[row_start:row_end][hidden] = original[row_start:row_end][hidden, :3]

        if grayscale:
            if source_alpha is not None:
                return Image.fromarray(np.dstack((restored, source_alpha)), mode="LA")
            return Image.fromarray(restored, mode="L")
        if source_alpha is not None:
            return Image.fromarray(np.dstack((restored, source_alpha)), mode="RGBA")
        return Image.fromarray(restored, mode="RGB")
    except MemoryError as exc:
        raise RuntimeError("Not enough memory to restore image color data") from exc


def prepare_lightness_unit(pixels: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Oklab (3xHxW float32) and its lightness (HxW float32) of float32 unit-range L, LA, RGB or RGBA pixels.

    The float counterpart of `prepare_lightness`, for precise rasters: nothing is quantized to 8 bits. Straight alpha;
    values beyond 0-1 enter the conversion unclamped (the sRGB transfer is extended linearly below 0 and as a power
    above 1). Transparent pixels (alpha below `MIN_VISIBLE_ALPHA`, half an 8-bit step, which is what 8-bit alpha 0 means; precise resizes leave such alphas with color 0) are filled from their nearest visible neighbour, as in `prepare_lightness`.
    """
    _check_pixels(pixels)
    height, width, count = pixels.shape
    rgb = pixels[:, :, :3] if count >= 3 else np.broadcast_to(pixels[:, :, :1], (height, width, 3))
    if count in (2, 4):
        rgb = _with_hidden_filled(rgb, pixels[:, :, -1] < MIN_VISIBLE_ALPHA)
    oklab = _oklab_from_rgb(rgb)
    return oklab, np.ascontiguousarray(oklab[0].numpy(), dtype=np.float32)


def restore_lightness_unit(pixels: np.ndarray, color_tensor: torch.Tensor, lightness: np.ndarray) -> np.ndarray:
    """New float32 pixels in the layout of `pixels`: Oklab with the given lightness, back to gamut-mapped sRGB.

    The float counterpart of `restore_lightness`. Alpha is copied unchanged. Transparent pixels (alpha below
    `MIN_VISIBLE_ALPHA`, half an 8-bit step) keep their original values exactly; every other pixel is gamut-mapped
    into 0-1 (see `_restored_srgb_tiles`).
    """
    _check_pixels(pixels)
    height, width, count = pixels.shape
    if color_tensor.shape != (3, height, width) or color_tensor.device.type != "cpu":
        raise ValueError("color_tensor must be a CPU tensor of shape (3, H, W) matching the pixels")
    if color_tensor.dtype != torch.float32:
        raise ValueError("color_tensor must be a float32 tensor")
    if lightness.shape != (height, width) or lightness.dtype != np.float32:
        raise ValueError("lightness must be an HxW float32 NumPy array matching the pixels")

    has_alpha = count in (2, 4)
    out = np.empty_like(pixels)
    for row_start, row_end, srgb in _restored_srgb_tiles(color_tensor, lightness):
        if count <= 2:
            out[row_start:row_end, :, 0] = (srgb[0] * 0.299 + srgb[1] * 0.587 + srgb[2] * 0.114).numpy()
        else:
            out[row_start:row_end, :, :3] = srgb.permute(1, 2, 0).numpy()
        if has_alpha:
            tile_pixels = pixels[row_start:row_end]
            out[row_start:row_end, :, -1] = tile_pixels[:, :, -1]
            hidden = tile_pixels[:, :, -1] < MIN_VISIBLE_ALPHA
            out[row_start:row_end][hidden] = tile_pixels[hidden]
    return out

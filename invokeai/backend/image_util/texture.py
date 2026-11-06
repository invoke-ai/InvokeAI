# SPDX-License-Identifier: Apache-2.0
"""Edge-aware fine texture adjustment."""

from __future__ import annotations

from math import isfinite
from numbers import Real

import cv2
import numpy as np
from PIL import Image

from invokeai.backend.image_util.color import (
    prepare_lightness,
    prepare_lightness_unit,
    restore_lightness,
    restore_lightness_unit,
)
from invokeai.backend.image_util.multiscale import edge_aware_band

_MIN_AMOUNT = -200
_MAX_AMOUNT = 200
_MIN_SIZE = 1
_MAX_SIZE = 100
_GUIDED_FILTER_EPSILON = 0.0008


def _checked(amount: float, size: float) -> tuple[float, float]:
    if isinstance(amount, bool) or not isinstance(amount, Real) or not isfinite(amount):
        raise ValueError("amount must be a finite number between -200 and 200")
    amount_value = float(amount)
    if not _MIN_AMOUNT <= amount_value <= _MAX_AMOUNT:
        raise ValueError("amount must be between -200 and 200")

    if isinstance(size, bool) or not isinstance(size, Real) or not isfinite(size):
        raise ValueError("size must be a finite number between 1 and 100")
    size_value = float(size)
    if not _MIN_SIZE <= size_value <= _MAX_SIZE:
        raise ValueError("size must be between 1 and 100")
    return amount_value, size_value


def _adjusted_lightness(lightness: np.ndarray, amount: float, size: float) -> np.ndarray:
    """The numeric core shared by the 8-bit and the float path: lightness in, adjusted lightness out."""
    inner_radius = max(1, round(size / 4))
    outer_radius = max(inner_radius + 1, round(size))
    detail = edge_aware_band(lightness, inner_radius, outer_radius, _GUIDED_FILTER_EPSILON)

    if amount > 0:
        rms_radius = max(2, round(size / 3))
        kernel_size = 2 * rms_radius + 1
        mean_square = cv2.boxFilter(
            detail * detail,
            ddepth=cv2.CV_32F,
            ksize=(kernel_size, kernel_size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )
        rms = np.sqrt(np.maximum(mean_square, 0.0))
        weight = np.clip((rms - 0.002) / (0.010 - 0.002), 0.0, 1.0)
        gate = weight * weight * (3.0 - 2.0 * weight)
        detail = detail * gate

    return np.asarray(lightness + (amount / 100.0) * detail, dtype=np.float32)


def texture(source: Image.Image, amount: float = 0, size: float = 8) -> Image.Image:
    """Adjust fine-scale detail while preserving the source image's color and alpha."""
    amount_value, size_value = _checked(amount, size)
    if amount_value == 0:
        return source.copy()

    color_tensor, lightness, alpha = prepare_lightness(source)
    adjusted_lightness = _adjusted_lightness(lightness, amount_value, size_value)
    return restore_lightness(source, color_tensor, adjusted_lightness, alpha)


def texture_unit(pixels: np.ndarray, amount: float = 0, size: float = 8) -> np.ndarray:
    """`texture` on HxWxC float32 unit-range pixels (L, LA, RGB or RGBA), without any 8-bit quantization.

    Works in Oklab lightness (see `color.py`). With an amount other than 0, every pixel that is
    not transparent (alpha at least half an 8-bit step) is gamut-mapped into 0-1; values beyond 0-1 survive only at an
    amount of 0 and in transparent pixels. Alpha is copied unchanged.
    """
    amount_value, size_value = _checked(amount, size)
    if amount_value == 0:
        return pixels.copy()

    color_tensor, lightness = prepare_lightness_unit(pixels)
    return restore_lightness_unit(pixels, color_tensor, _adjusted_lightness(lightness, amount_value, size_value))

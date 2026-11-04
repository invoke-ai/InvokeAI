# SPDX-License-Identifier: Apache-2.0
"""Broad-scale contrast and chroma adjustment inspired by dehazed imagery."""

from __future__ import annotations

from math import isfinite
from numbers import Real

import numpy as np
import torch
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
_GUIDED_FILTER_EPSILON = 0.002


def _checked(amount: float) -> float:
    if isinstance(amount, bool) or not isinstance(amount, Real) or not isfinite(amount):
        raise ValueError("amount must be a finite number between -200 and 200")
    amount_value = float(amount)
    if not _MIN_AMOUNT <= amount_value <= _MAX_AMOUNT:
        raise ValueError("amount must be between -200 and 200")
    return amount_value


def _adjusted_lightness(color_tensor: torch.Tensor, lightness: np.ndarray, amount: float) -> np.ndarray:
    """The numeric core shared by the 8-bit and the float path.

    Returns the adjusted lightness and scales the chroma channels of `color_tensor` in place.
    """
    height, width = lightness.shape
    radius = max(2, min(1000, round(min(width, height) / 8)))
    broad_scale = edge_aware_band(lightness, 0, radius, _GUIDED_FILTER_EPSILON)

    strength = amount / 100.0
    contrast = 1.0 + 0.2 * strength
    adjusted_lightness = np.asarray(0.5 + contrast * (lightness - 0.5) + strength * broad_scale, dtype=np.float32)
    color_tensor[1:] *= 1.0 + 0.2 * strength
    return adjusted_lightness


def contrast_dehaze(source: Image.Image, amount: float = 0) -> Image.Image:
    """Adjust broad-scale contrast and Oklab chroma as an aesthetic approximation.

    Positive amounts increase broad luminance separation and chroma; negative
    amounts reduce both. This is a global image adjustment, not physical haze
    estimation or scene recovery.
    """
    amount_value = _checked(amount)
    if amount_value == 0:
        return source.copy()

    try:
        color_tensor, lightness, alpha = prepare_lightness(source)
        adjusted_lightness = _adjusted_lightness(color_tensor, lightness, amount_value)
        return restore_lightness(source, color_tensor, adjusted_lightness, alpha)
    except MemoryError as exc:
        raise RuntimeError("Not enough memory to adjust image contrast") from exc


def contrast_dehaze_unit(pixels: np.ndarray, amount: float = 0) -> np.ndarray:
    """`contrast_dehaze` on HxWxC float32 unit-range pixels (L, LA, RGB or RGBA), without 8-bit quantization.

    Works in Oklab (see `color.py`). With an amount other than 0, every pixel that is
    not transparent (alpha at least half an 8-bit step) is gamut-mapped into 0-1; values beyond 0-1 survive only at an
    amount of 0 and in transparent pixels. Alpha is copied unchanged.
    """
    amount_value = _checked(amount)
    if amount_value == 0:
        return pixels.copy()

    try:
        color_tensor, lightness = prepare_lightness_unit(pixels)
        adjusted_lightness = _adjusted_lightness(color_tensor, lightness, amount_value)
        return restore_lightness_unit(pixels, color_tensor, adjusted_lightness)
    except MemoryError as exc:
        raise RuntimeError("Not enough memory to adjust image contrast") from exc

# SPDX-License-Identifier: Apache-2.0
"""Broad-scale contrast and chroma adjustment inspired by dehazed imagery."""

from __future__ import annotations

from math import isfinite
from numbers import Real

import numpy as np
from PIL import Image

from invokeai.backend.image_util.color import prepare_lightness, restore_lightness
from invokeai.backend.image_util.multiscale import edge_aware_band

_MIN_AMOUNT = -200
_MAX_AMOUNT = 200
_GUIDED_FILTER_EPSILON = 0.002


def contrast_dehaze(source: Image.Image, amount: float = 0) -> Image.Image:
    """Adjust broad-scale contrast and Oklab chroma as an aesthetic approximation.

    Positive amounts increase broad luminance separation and chroma; negative
    amounts reduce both. This is a global image adjustment, not physical haze
    estimation or scene recovery.
    """
    if isinstance(amount, bool) or not isinstance(amount, Real) or not isfinite(amount):
        raise ValueError("amount must be a finite number between -200 and 200")
    amount_value = float(amount)
    if not _MIN_AMOUNT <= amount_value <= _MAX_AMOUNT:
        raise ValueError("amount must be between -200 and 200")

    if amount_value == 0:
        return source.copy()

    try:
        color_tensor, lightness, alpha = prepare_lightness(source)
        width, height = source.size
        radius = max(2, min(1000, round(min(width, height) / 8)))
        broad_scale = edge_aware_band(lightness, 0, radius, _GUIDED_FILTER_EPSILON)

        strength = amount_value / 100.0
        contrast = 1.0 + 0.2 * strength
        adjusted_lightness = np.asarray(0.5 + contrast * (lightness - 0.5) + strength * broad_scale, dtype=np.float32)
        color_tensor[1:] *= 1.0 + 0.2 * strength

        return restore_lightness(source, color_tensor, adjusted_lightness, alpha)
    except MemoryError as exc:
        raise RuntimeError("Not enough memory to adjust image contrast") from exc

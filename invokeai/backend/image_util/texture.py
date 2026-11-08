# SPDX-License-Identifier: Apache-2.0
"""Edge-aware fine texture adjustment."""

from __future__ import annotations

from math import isfinite
from numbers import Real

import cv2
import numpy as np
from PIL import Image

from invokeai.backend.image_util.color import prepare_lightness, restore_lightness
from invokeai.backend.image_util.multiscale import edge_aware_band

_MIN_AMOUNT = -200
_MAX_AMOUNT = 200
_MIN_SIZE = 1
_MAX_SIZE = 100
_GUIDED_FILTER_EPSILON = 0.0008


def texture(source: Image.Image, amount: float = 0, size: float = 8) -> Image.Image:
    """Adjust fine-scale detail while preserving the source image's color and alpha."""
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

    if amount_value == 0:
        return source.copy()

    color_tensor, lightness, alpha = prepare_lightness(source)
    inner_radius = max(1, round(size_value / 4))
    outer_radius = max(inner_radius + 1, round(size_value))
    detail = edge_aware_band(lightness, inner_radius, outer_radius, _GUIDED_FILTER_EPSILON)

    if amount_value > 0:
        rms_radius = max(2, round(size_value / 3))
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

    adjusted_lightness = np.asarray(lightness + (amount_value / 100.0) * detail, dtype=np.float32)
    return restore_lightness(source, color_tensor, adjusted_lightness, alpha)

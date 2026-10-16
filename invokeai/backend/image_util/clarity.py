# SPDX-License-Identifier: Apache-2.0
"""Edge-aware midtone detail adjustment."""

from __future__ import annotations

from math import isfinite
from numbers import Real

import numpy as np
from PIL import Image

from invokeai.backend.image_util.color import prepare_lightness, restore_lightness
from invokeai.backend.image_util.multiscale import edge_aware_band

_MIN_AMOUNT = -100
_MAX_AMOUNT = 100
_MIN_SIZE = 4
_MAX_SIZE = 1000
_GUIDED_FILTER_EPSILON = 0.002


def clarity(source: Image.Image, amount: float = 0, size: float = 40) -> Image.Image:
    """Adjust local midtone detail while preserving the source image's color and alpha."""
    if isinstance(amount, bool) or not isinstance(amount, Real) or not isfinite(amount):
        raise ValueError("amount must be a finite number between -100 and 100")
    amount_value = float(amount)
    if not _MIN_AMOUNT <= amount_value <= _MAX_AMOUNT:
        raise ValueError("amount must be between -100 and 100")

    if isinstance(size, bool) or not isinstance(size, Real) or not isfinite(size):
        raise ValueError("size must be a finite number between 4 and 1000")
    size_value = float(size)
    if not _MIN_SIZE <= size_value <= _MAX_SIZE:
        raise ValueError("size must be between 4 and 1000")

    if amount_value == 0:
        return source.copy()

    color_tensor, lightness, alpha = prepare_lightness(source)
    inner_radius = max(1, round(size_value / 4))
    outer_radius = max(inner_radius + 1, round(size_value))
    detail = edge_aware_band(lightness, inner_radius, outer_radius, _GUIDED_FILTER_EPSILON)
    clipped_lightness = np.clip(lightness, 0.0, 1.0)
    midtone_gate = 4.0 * clipped_lightness * (1.0 - clipped_lightness)
    adjusted_lightness = np.asarray(lightness + (amount_value / 100.0) * midtone_gate * detail, dtype=np.float32)

    return restore_lightness(source, color_tensor, adjusted_lightness, alpha)

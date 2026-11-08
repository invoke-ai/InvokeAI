# SPDX-License-Identifier: Apache-2.0
"""Edge-aware difference-of-smooths image bands."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import numpy.typing as npt

from invokeai.backend.image_util.guided_filter import guided_filter


def edge_aware_band(
    lightness: npt.ArrayLike,
    inner_radius: int,
    outer_radius: int,
    epsilon: float,
) -> npt.NDArray[np.float32]:
    """Return detail between two self-guided smooths of a 2D lightness image.

    The band is the inner-radius smooth minus the outer-radius smooth. The
    inner radius may be zero, in which case the fine smooth is the input.
    """
    if (
        isinstance(inner_radius, bool)
        or not isinstance(inner_radius, Integral)
        or isinstance(outer_radius, bool)
        or not isinstance(outer_radius, Integral)
        or inner_radius < 0
        or outer_radius <= inner_radius
    ):
        raise ValueError("radii must satisfy 0 <= inner_radius < outer_radius")

    image = np.asarray(lightness, dtype=np.float32)
    fine = guided_filter(image, image, int(inner_radius), epsilon)
    coarse = guided_filter(image, image, int(outer_radius), epsilon)
    return np.asarray(fine - coarse, dtype=np.float32)

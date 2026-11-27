# SPDX-License-Identifier: Apache-2.0
"""Scalar guided image filtering using normalized OpenCV box filters."""

from __future__ import annotations

from numbers import Integral

import cv2
import numpy as np
import numpy.typing as npt


def guided_filter(
    guide: npt.ArrayLike,
    signal: npt.ArrayLike,
    radius: int,
    epsilon: float,
) -> npt.NDArray[np.float32]:
    """Filter a 2D signal while preserving edges present in a 2D guide.

    Local means use OpenCV box filters with ``BORDER_REFLECT_101``. This
    reflects samples around the edge without repeating the edge pixel itself.
    The implementation uses a fixed number of array operations and box filters,
    so its work is O(width × height), independent of the filter radius.
    """
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if not np.isscalar(epsilon):
        raise ValueError("epsilon must be finite and positive")
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("epsilon must be finite and positive") from exc
    if not np.isfinite(epsilon_value) or epsilon_value <= 0:
        raise ValueError("epsilon must be finite and positive")

    guide_array = np.asarray(guide, dtype=np.float32)
    signal_array = np.asarray(signal, dtype=np.float32)
    if guide_array.ndim != 2 or signal_array.ndim != 2:
        raise ValueError("guide and signal must be 2D arrays")
    if guide_array.shape != signal_array.shape:
        raise ValueError("guide and signal must have equal shapes")

    if radius == 0:
        return signal_array.copy()

    kernel_size = 2 * int(radius) + 1

    def box_mean(array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return cv2.boxFilter(
            array,
            ddepth=cv2.CV_32F,
            ksize=(kernel_size, kernel_size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )

    mean_guide = box_mean(guide_array)
    mean_signal = box_mean(signal_array)
    mean_guide_squared = box_mean(guide_array * guide_array)
    mean_guide_signal = box_mean(guide_array * signal_array)

    variance = mean_guide_squared - mean_guide * mean_guide
    np.maximum(variance, 0, out=variance)
    covariance = mean_guide_signal - mean_guide * mean_signal
    coefficient_a = covariance / (variance + epsilon_value)
    coefficient_b = mean_signal - coefficient_a * mean_guide

    mean_a = box_mean(coefficient_a)
    mean_b = box_mean(coefficient_b)
    return np.asarray(mean_a * guide_array + mean_b, dtype=np.float32)

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from invokeai.backend.image_util.color_conversion import (
    gamut_clip_tensor,
    linear_srgb_from_oklab,
    oklab_from_srgb,
    srgb_from_linear_srgb,
)

_SUPPORTED_MODES = {"RGB", "RGBA", "L", "LA"}
_RESTORE_TILE_ROWS = 256


def _check_image(image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow Image")
    if image.mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported image mode: {image.mode}")


def prepare_lightness(image: Image.Image) -> tuple[torch.Tensor, np.ndarray, np.ndarray | None]:
    """Convert an 8-bit Pillow image to Oklab, lightness, and optional alpha."""
    try:
        _check_image(image)
        width, height = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        alpha: np.ndarray | None = None
        if image.mode in {"RGBA", "LA"}:
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint8).copy()
            hidden = alpha == 0
            visible = ~hidden
            if hidden.any():
                rgb[hidden] = 0
                if visible.any():
                    import cv2

                    _, labels = cv2.distanceTransformWithLabels(
                        np.ascontiguousarray(hidden, dtype=np.uint8),
                        cv2.DIST_L2,
                        5,
                        labelType=cv2.DIST_LABEL_PIXEL,
                    )
                    visible_labels = labels[visible]
                    label_to_rgb = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
                    label_to_rgb[visible_labels] = rgb[visible]
                    rgb[hidden] = label_to_rgb[labels[hidden]]

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
        oklab = oklab_from_srgb(rgb_tensor).contiguous()
        lightness = np.ascontiguousarray(oklab[0].numpy(), dtype=np.float32)
        if oklab.shape != (3, height, width):
            raise RuntimeError("Color conversion returned an unexpected image shape")
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
        for row_start in range(0, height, _RESTORE_TILE_ROWS):
            row_end = min(row_start + _RESTORE_TILE_ROWS, height)
            tile_color = color_tensor[:, row_start:row_end]
            if not torch.isfinite(tile_color).all():
                raise ValueError("color_tensor must contain only finite values")
            tile_lightness = lightness[row_start:row_end]
            if not np.isfinite(tile_lightness).all():
                raise ValueError("lightness must contain only finite values")

            adjusted = tile_color.clone()
            adjusted[0] = torch.from_numpy(np.ascontiguousarray(tile_lightness)).clamp(0.0, 1.0)
            linear_rgb = linear_srgb_from_oklab(adjusted)
            if not torch.isfinite(linear_rgb).all():
                raise ValueError("lightness produced non-finite RGB values")
            if torch.any((linear_rgb < 0.0) | (linear_rgb > 1.0)):
                linear_rgb = gamut_clip_tensor(linear_rgb)
            srgb = srgb_from_linear_srgb(linear_rgb).clamp(0.0, 1.0)
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

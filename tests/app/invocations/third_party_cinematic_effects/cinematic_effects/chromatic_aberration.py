# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

import numpy as np
from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)
from PIL import Image
from scipy.ndimage import map_coordinates


@invocation(
    "chromatic_aberration",
    title="Chromatic Aberration",
    tags=["image", "postprocessing", "chromatic", "aberration"],
    version="1.1.0",
)
class ChromaticAberrationInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Simulate realistic chromatic aberration in an image with controllable strength and center point."""

    image: ImageField = InputField(description="The image to apply chromatic aberration to")
    strength: float = InputField(default=1.0, ge=0.0, le=10.0, description="Strength of the chromatic shift (0–10)")
    center_x: float = InputField(default=0.5, ge=0.0, le=1.0, description="X coordinate of center (0–1)")
    center_y: float = InputField(default=0.5, ge=0.0, le=1.0, description="Y coordinate of center (0–1)")

    def srgb_to_linear(self, srgb: np.ndarray) -> np.ndarray:
        """Convert sRGB gamma-encoded to linear RGB."""
        a = 0.055
        return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)

    def linear_to_srgb(self, linear: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB gamma-encoded."""
        a = 0.055
        return np.where(linear <= 0.0031308, 12.92 * linear, (1 + a) * np.power(np.maximum(linear, 0.0), 1 / 2.4) - a)

    def radial_offset(self, channel, dx, dy, strength) -> np.ndarray:
        """Offset a single color channel radially outward or inward."""
        height, width = channel.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        norm = np.sqrt(dx**2 + dy**2)
        shift_x = x + dx / (norm + 1e-6) * strength
        shift_y = y + dy / (norm + 1e-6) * strength

        coords = [np.clip(shift_y, 0, height - 1), np.clip(shift_x, 0, width - 1)]
        return map_coordinates(channel, coords, order=1, mode="reflect")

    def apply_chromatic_aberration(self, image: Image.Image) -> Image.Image:
        """Apply chromatic aberration in linear RGB space."""
        img = np.array(image).astype(np.float32) / 255.0
        if img.shape[2] == 4:
            rgb, alpha = img[..., :3], img[..., 3:]
        else:
            rgb, alpha = img, None

        # Convert to linear RGB
        rgb_linear = self.srgb_to_linear(rgb)

        height, width = rgb.shape[:2]
        cx = self.center_x * width
        cy = self.center_y * height

        dx = np.tile(np.arange(width)[None, :] - cx, (height, 1))
        dy = np.tile(np.arange(height)[:, None] - cy, (1, width))

        strength = self.strength

        red = self.radial_offset(rgb_linear[..., 0], dx, dy, strength)
        green = self.radial_offset(rgb_linear[..., 1], dx, dy, strength * 0.5)
        blue = self.radial_offset(rgb_linear[..., 2], dx, dy, -strength)

        out_rgb_linear = np.stack([red, green, blue], axis=-1)
        out_rgb = np.clip(self.linear_to_srgb(out_rgb_linear), 0.0, 1.0)

        if alpha is not None:
            out = np.concatenate([out_rgb, alpha], axis=2)
        else:
            out = out_rgb

        result = (out * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pil_image = context.images.get_pil(self.image.image_name)
        result_image = self.apply_chromatic_aberration(pil_image)
        image_dto = context.images.save(image=result_image)
        return ImageOutput.build(image_dto)

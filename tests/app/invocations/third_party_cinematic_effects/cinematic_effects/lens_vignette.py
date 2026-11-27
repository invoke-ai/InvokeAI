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

from .util import oklch_to_rgb, rgb_to_oklch  # noqa: TID252


@invocation(
    "lens_vignette",
    title="Lens Vignette",
    tags=["image", "postprocessing", "vignette"],
    category="image",
    version="1.2.0",
)
class LensVignetteInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Apply realistic lens vignetting to an image with configurable parameters."""

    image: ImageField = InputField(description="The image to apply vignetting to")
    intensity: float = InputField(default=0.5, ge=0.0, le=4.0, description="Intensity of the vignette effect (0-4)")
    aperture_factor: float = InputField(
        default=0.5, ge=0.2, le=1.5, description="Aperture falloff factor (>1.2 = strong, <0.5 = weak)"
    )
    center_x: float = InputField(default=0.5, ge=0.0, le=1.0, description="X coordinate of vignette center (0-1)")
    center_y: float = InputField(default=0.5, ge=0.0, le=1.0, description="Y coordinate of vignette center (0-1)")
    preserve_highlights: bool = InputField(default=True, description="Preserve highlights")

    def do_highlight_preservation(
        self, img_oklch: np.ndarray, img_vignetted: np.ndarray, highlights_vignetted: np.ndarray
    ) -> np.ndarray:
        L = img_oklch[..., 0]
        highlight = np.clip((L - 0.7) / 0.4, 0.0, 1.0) ** 1.5

        result_L = highlight * highlights_vignetted[..., 0] + (1.0 - highlight) * img_vignetted[..., 0]

        # Replace only the L channel
        result = img_vignetted.copy()
        result[..., 0] = np.clip(result_L, 0.0, 1.0)
        return result

    def apply_vignette(self, image: Image.Image) -> Image.Image:
        """Apply a cosine-fourth power vignette effect with optional off-center falloff."""

        # Load and normalize image
        img = np.array(image).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = img[..., None]

        ch = img.shape[2]
        if ch == 1:
            rgb, alpha = img, None
        elif ch == 2:
            rgb, alpha = img[..., :1], img[..., 1:2]
        elif ch == 3:
            rgb, alpha = img, None
        elif ch == 4:
            rgb, alpha = img[..., :3], img[..., 3:4]
        else:
            rgb, alpha = img[..., :3], img[..., 3:] if ch > 3 else None

        img_oklch = rgb_to_oklch(rgb)
        img_oklch[..., 2] = np.where(img_oklch[..., 1] > 1e-8, img_oklch[..., 2], 0.0)

        height, width = img.shape[:2]
        y, x = np.mgrid[0:height, 0:width]
        cx = self.center_x * width
        cy = self.center_y * height

        dx = x - cx
        dy = y - cy
        distance = np.sqrt(dx**2 + dy**2)
        max_dist = np.sqrt(
            max(
                cx**2 + cy**2,
                (width - cx) ** 2 + cy**2,
                cx**2 + (height - cy) ** 2,
                (width - cx) ** 2 + (height - cy) ** 2,
            )
        )
        r = distance / max_dist

        # Cosine-fourth law
        vignette = np.cos(np.clip(r * self.aperture_factor, 0.0, 1.0) * (np.pi / 2)) ** 4
        vignette_mask = 1.0 - self.intensity * (1.0 - vignette)
        vignette_mask = vignette_mask[..., None]  # expand for broadcast
        vignette_mask = np.clip(vignette_mask, 0.0, 1.0)

        img_vignetted = img_oklch.copy()
        img_vignetted[..., 0] *= vignette_mask[..., 0]  # apply only to L channel

        if self.preserve_highlights:
            highlight_mask = 1.0 - (self.intensity * 0.5) * (1.0 - vignette)
            highlight_mask = highlight_mask[..., None]
            highlight_mask = np.clip(highlight_mask, 0.0, 1.0)

            highlights_vignetted = img_oklch.copy()
            highlights_vignetted[..., 0] *= highlight_mask[..., 0]

            img_oklch = self.do_highlight_preservation(img_oklch, img_vignetted, highlights_vignetted)
        else:
            img_oklch = img_vignetted

        rgb_srgb = oklch_to_rgb(img_oklch)

        if alpha is not None:
            out = np.concatenate([rgb_srgb, alpha], axis=2)
        else:
            out = rgb_srgb

        result = (out * 255.0).clip(0, 255).astype(np.uint8)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result[:, :, 0]
        return Image.fromarray(result)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pil_image = context.images.get_pil(self.image.image_name)
        result_image = self.apply_vignette(pil_image)

        image_dto = context.images.save(image=result_image)

        return ImageOutput.build(image_dto)

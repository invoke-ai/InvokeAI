# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

from typing import Optional, Tuple

import cv2
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

from .util import MyDepthMapPipeline  # noqa: TID252


@invocation("lens_blur", title="Lens Blur", tags=["image", "lens", "blur"], version="1.1.0")
class LensBlurInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Adds lens blur to the input image, first converting the input image to RGB"""

    image: ImageField = InputField(description="The image to streak")
    depth_map: ImageField = InputField(description="The depth map to use")
    aperture_image: Optional[ImageField] = InputField(description="The aperture image to use for convolution")
    focal_distance: float = InputField(
        description="The distance at which focus is sharp: 0.0 (near) - 1.0 (far)",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    distance_range: float = InputField(
        description="The depth of field range around the focal distance",
        default=0.2,
        ge=0.0,
        le=1.0,
    )
    max_blur_radius: float = InputField(description="Maximum blur radius", default=5.0, gt=0.0)
    max_blur_steps: int = InputField(description="Number of blur maps to use internally", default=32, gt=1)
    anamorphic_factor: float = InputField(description="Anamorphic squeeze factor", default=1.33, ge=1.0)
    highlight_threshold: float = InputField(
        description="The luminance threshold at which highlight enhancement begins",
        default=0.75,
        ge=0.0,
        le=1.0,
    )
    highlight_factor: float = InputField(
        description="Minimum multiplier for highlights at the threshold",
        default=1.0,
        ge=1.0,
    )
    highlight_factor_high: float = InputField(
        description="Maximum multiplier for highlights at full brightness",
        default=2.0,
        ge=1.0,
    )
    reduce_halos: bool = InputField(
        description="Reduce halos around areas of sharp depth distance (SLOW)", default=True
    )

    def compute_blur_indices(self, depth_normalized: np.ndarray, num_blurred_images: int) -> np.ndarray:
        # Use an asymmetric depth of field, using a 1/3 front, 2/3 back rule around the focal plane.
        dof_front = 1 * self.distance_range / 3.0
        dof_back = 2 * self.distance_range / 3.0

        below = depth_normalized < self.focal_distance
        above = ~below

        distance_from_focus = np.zeros_like(depth_normalized)

        # Front of focal plane
        distance_from_focus[below] = np.clip((self.focal_distance - depth_normalized[below] - dof_front), 0.0, 1.0)

        # Behind focal plane
        distance_from_focus[above] = np.clip((depth_normalized[above] - self.focal_distance - dof_back), 0.0, 1.0)

        # Normalize distance from focus
        distance_from_focus = cv2.normalize(distance_from_focus, None, 0.0, 1.0, cv2.NORM_MINMAX)

        # Scale to index range
        indices = distance_from_focus * (num_blurred_images - 1)

        return indices

    def split_indices(self, indices: np.ndarray, num_blurred_images: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower_indices = np.floor(indices).astype(np.int32)
        upper_indices = np.clip(lower_indices + 1, 0, num_blurred_images - 1)
        alpha = (indices - lower_indices)[..., np.newaxis]  # Shape: (H, W, 1)

        return lower_indices, upper_indices, alpha

    def apply_anamorphic_gaussian_blur(self, image: Image, sigma_y: float, anamorphic_squeeze: float) -> np.ndarray:
        sigma_x = sigma_y / anamorphic_squeeze

        image_numpy = image.astype(np.float32)

        # Let OpenCV auto-calculate optimal kernel size from sigma
        ksize = (0, 0)

        # Apply the blur
        blurred = cv2.GaussianBlur(image_numpy, ksize, sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT)

        return blurred

    def fractional_resize(self, aperture_image_in: Image, kernel_width: float, kernel_height: float) -> Image:
        desired_w, desired_h = int(round(kernel_width) // 2 * 2 + 1), int(round(kernel_height) // 2 * 2 + 1)
        (aperture_w, aperture_h) = aperture_image_in.size

        if desired_w == aperture_w and desired_h == aperture_h:
            return aperture_image_in.copy()

        resize_pct_w, resize_pct_h = desired_w / aperture_w, desired_h / aperture_h
        temp_w, temp_h = int(aperture_w * 1.25), int(aperture_h * 1.25)
        temp_scaled_w, temp_scaled_h = int(temp_w * resize_pct_w), int(temp_h * resize_pct_h)

        aperture_image = Image.new("L", (temp_w, temp_h), 0)
        offset = ((temp_w - aperture_w) // 2, (temp_h - aperture_h) // 2)
        aperture_image.paste(aperture_image_in, offset)
        aperture_image = aperture_image.resize((temp_scaled_w, temp_scaled_h), resample=Image.LANCZOS)

        left = (temp_scaled_w - desired_w) // 2
        top = (temp_scaled_h - desired_h) // 2
        right = left + desired_w
        bottom = top + desired_h
        aperture_image = aperture_image.crop((left, top, right, bottom))
        return aperture_image

    def apply_aperture_image_blur(
        self, image: Image, aperture_image: Image, kernel_height: float, anamorphic_squeeze: float
    ) -> np.ndarray:
        # A convenience function to help get the kernel sizing just right
        def size_from_sigma(sigma: float) -> Tuple[int, float]:
            rounded_size = 2 * int(np.ceil(3 * sigma)) + 1
            actual_size = 6 * sigma + 1
            return rounded_size, actual_size

        image_numpy = image.astype(np.float32)

        # Resize the original proportionally
        kernel_width = kernel_height / anamorphic_squeeze
        resized_kernel_img = self.fractional_resize(aperture_image, kernel_width, kernel_height)

        # Extract, flip, and normalize the convolution kernel
        resized_kernel_img = resized_kernel_img.transpose(Image.FLIP_TOP_BOTTOM)
        resized_kernel_img = resized_kernel_img.transpose(Image.FLIP_LEFT_RIGHT)
        kernel = np.array(resized_kernel_img, dtype=np.float32)
        kernel /= kernel.sum()

        # Apply convolution channel-wise
        blurred = np.stack(
            [
                cv2.filter2D(image_numpy[..., c], -1, kernel, borderType=cv2.BORDER_REFLECT)
                for c in range(image_numpy.shape[2])
            ],
            axis=2,
        )

        return blurred

    def apply_blur_based_on_depth(
        self, blurred_images: list[np.ndarray], depth_map: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        num_blurred_images = len(blurred_images)
        lower_indices, upper_indices, alpha = self.split_indices(indices, num_blurred_images)

        output = np.zeros_like(blurred_images[0], dtype=np.float32)

        for i in range(num_blurred_images):
            mask_lower = lower_indices == i
            mask_upper = upper_indices == i

            if blurred_images[0].ndim == 3:
                mask_lower = mask_lower[..., np.newaxis]
                mask_upper = mask_upper[..., np.newaxis]

            output += blurred_images[i] * mask_lower * (1 - alpha) + blurred_images[i] * mask_upper * alpha

        return output

    def enhance_highlights(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0

        # Compute per-pixel luminance (ITU-R BT.709)
        luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]

        # Calculate per-pixel enhancement factor
        mask = luminance >= self.highlight_threshold
        t = np.clip((luminance - self.highlight_threshold) / (1.0 - self.highlight_threshold), 0.0, 1.0)
        factor = self.highlight_factor * (1.0 - t) + self.highlight_factor_high * t

        # Expand factor to 3 channels
        factor_expanded = np.where(mask[..., None], factor[..., None], 1.0)

        # Apply enhancement
        enhanced = image * factor_expanded
        enhanced = enhanced * 255.0

        return enhanced

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        image_mode = image.mode
        image = image.convert("RGB")

        if self.depth_map:
            depth_map = context.images.get_pil(self.depth_map.image_name)

            # In case we got an RGB depth map
            depth_map = depth_map.convert("L")

            depth_map_numpy = np.array(depth_map)
        else:
            # Do our own depth estimation
            model_url = "depth-anything/Depth-Anything-V2-Small-hf"
            loaded_model = context.models.load_remote_model(model_url, MyDepthMapPipeline.load_model)

            with loaded_model as depth_anything_detector:
                depth_map = depth_anything_detector.generate_depth(image)

            """
            # Interpolate to original size
            post_processed_output = self._image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(image.height, image.width)],
            )

            # Visualize the prediction
            predicted_depth = post_processed_output[0]["predicted_depth"]
            depth_map = predicted_depth * 255 / predicted_depth.max()
            depth_map_numpy = depth_map.detach().cpu().numpy()
            """

        # Slightly blur
        depth_map = cv2.GaussianBlur(depth_map_numpy, (0, 0), 3)

        # Normalize and invert depth map
        depth_normalized = 1 - cv2.normalize(depth_map_numpy.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

        # Get blur indices so we can use for blurring and depth compositing
        indices = self.compute_blur_indices(depth_normalized, self.max_blur_steps)

        img_blurred = [None] * self.max_blur_steps

        image_numpy = np.array(image)
        image_numpy = self.enhance_highlights(image_numpy)
        img_blurred[0] = image_numpy

        if self.reduce_halos:
            # Get a mask to remove closer things and use cv2 infill
            lower_indices, _, _ = self.split_indices(indices, self.max_blur_steps)

        aperture_image = context.images.get_pil(self.aperture_image.image_name)
        (aperture_w, aperture_h) = aperture_image.size
        if aperture_w != aperture_h or aperture_w < 512:
            raise ValueError("Provided aperture image is not a square and must be at least 512x512")
        aperture_image = aperture_image.convert("L")

        for blur in range(1, self.max_blur_steps):
            # Copy the image to preserve the original
            image_to_blur = np.array(image)

            if self.reduce_halos:
                # Mask where indices are < blur
                mask = (indices < (blur - self.max_blur_steps // 10)).astype(np.uint8)
                mask = mask * 255

                # Inpaint masked regions
                image_to_blur = cv2.inpaint(image_to_blur, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Add highlights
            image_to_blur = self.enhance_highlights(image_to_blur)

            def bokeh_size(d, MAX):
                return (d / MAX) ** 0.10

            if self.aperture_image is None:
                # Use Gaussian blur
                img_blurred[blur] = self.apply_anamorphic_gaussian_blur(
                    image_to_blur,
                    self.max_blur_radius * bokeh_size(blur, (self.max_blur_steps - 1)),
                    self.anamorphic_factor,
                )
            else:
                # Use aperture blur
                img_blurred[blur] = self.apply_aperture_image_blur(
                    image_to_blur,
                    aperture_image,
                    self.max_blur_radius * bokeh_size(blur, (self.max_blur_steps - 1)),
                    self.anamorphic_factor,
                )

        img_blurred = self.apply_blur_based_on_depth(img_blurred, depth_map_numpy, indices)
        img_blurred = np.clip(img_blurred, 0, 255).astype(np.uint8)
        lens_blurred_image = Image.fromarray(img_blurred).convert(image_mode)

        image_dto = context.images.save(image=lens_blurred_image)

        return ImageOutput.build(image_dto)

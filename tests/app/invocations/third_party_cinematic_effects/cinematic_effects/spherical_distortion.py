# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

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


@invocation("spherical_distortion", title="Spherical Distortion", tags=["image", "anamorphic"], version="1.1.0")
class SphericalDistortionInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Applies spherical distortion to an image and fills the frame with the resulting image"""

    image: ImageField = InputField(description="The image to distort")
    k1: float = InputField(description="k1", default=0.3)
    k2: float = InputField(description="k2", default=0.1)
    p1: float = InputField(description="p1", default=0.0)
    p2: float = InputField(description="p2", default=0.0)

    def calculate_optimal_zoom(self, width, height, k1, k2, p1=0.0, p2=0.0):
        """
        Calculate the optimal zoom factor needed to fill the frame after distortion.

        Args:
            k1, k2: Radial distortion coefficients
            width, height: Dimensions of the image
            p1, p2: Tangential distortion coefficients

        Returns:
            Optimal zoom factor
        """
        # Normalize corner coordinates to [-1, 1] range
        x_corners = [1.0, -1.0, 1.0, -1.0]
        y_corners = [1.0, 1.0, -1.0, -1.0]

        # Calculate maximum distortion at corners
        max_distortion = 1.0
        for x, y in zip(x_corners, y_corners):
            r_squared = (x * width / 2) ** 2 + (y * height / 2) ** 2
            r_quad = r_squared**2
            distortion_factor = 1.0 + k1 * r_squared + k2 * r_quad
            tangential_offset = 2 * p1 * x * y + p2 * (r_squared + 2 * x**2)

            # Combined effect
            distorted_x = x * distortion_factor + tangential_offset
            distorted_y = y * distortion_factor + tangential_offset
            max_distortion = max(max_distortion, abs(distorted_x), abs(distorted_y))

        # Zoom to compensate for distortion
        optimal_zoom = 1.05 * max_distortion if max_distortion > 1.0 else 0.95 / max_distortion
        return optimal_zoom

    def apply_advanced_spherical_distortion_cv2(
        self, pil_image, k1=0.3, k2=0.1, p1=0.0, p2=0.0, auto_zoom=True, zoom=1.0
    ):
        """
        Apply advanced spherical distortion with multiple distortion coefficients using OpenCV.
        Includes both radial (k1, k2) and tangential (p1, p2) distortion parameters.

        Args:
            pil_image: PIL Image object
            k1: Primary radial distortion coefficient
            k2: Secondary radial distortion coefficient
            p1, p2: Tangential distortion coefficients (default 0)
            auto_zoom: Whether to automatically calculate zoom to fill frame (overrides zoom parameter)
            zoom: Manual zoom factor (only used if auto_zoom is False)

        Returns:
            PIL Image with advanced lens distortion applied

        Typical Panavision Anamorphic Lens Parameter Values:
            Classic Panavision (C Series Anamorphic):
                k1 = 0.25 to 0.35, k2 = 0.15 to 0.25, p1 = 0.01, p2 = 0.01

            Panavision Primo Anamorphic:
                k1 = 0.15 to 0.25, k2 = 0.05 to 0.15, p1 = 0.005, p2 = 0.005

            Panavision E Series Anamorphic:
                k1 = 0.30 to 0.40, k2 = 0.20 to 0.30, p1 = 0.01, p2 = 0.01

            Panavision G Series Anamorphic (vintage look):
                k1 = 0.40 to 0.50, k2 = 0.25 to 0.35, p1 = 0.02, p2 = 0.02

            Panavision T Series Anamorphic:
                k1 = 0.20 to 0.30, k2 = 0.10 to 0.20, p1 = 0.005, p2 = 0.005
        """
        # Get the width and height
        width, height = pil_image.size

        # Convert PIL image to OpenCV format
        cv_image = np.array(pil_image)

        # Handle RGB vs RGBA
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)

        # Calculate optimal zoom if requested
        if auto_zoom:
            zoom = self.calculate_optimal_zoom(width, height, k1, k2, p1, p2)

        # Create camera matrix (focal length and principal point)
        focal_length = width
        camera_matrix = np.array(
            [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]], dtype=np.float64
        )

        # Define distortion coefficients [k1, k2, p1, p2, k3]
        # k3 is set to 0 for simplicity
        dist_coeffs = np.array([k1, k2, p1, p2, 0], dtype=np.float64)

        # Generate new camera matrix with adjusted zoom
        alpha = 1.0 / zoom  # Inverse of zoom for OpenCV's convention
        new_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), alpha)[0]

        # Apply distortion using undistort (OpenCV uses inverted distortion model)
        distorted_image = cv2.undistort(cv_image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Convert back to PIL
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            distorted_image = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(distorted_image)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            distorted_image = cv2.cvtColor(distorted_image, cv2.COLOR_BGRA2RGBA)
            result_pil = Image.fromarray(distorted_image)
        else:
            result_pil = Image.fromarray(distorted_image)

        return result_pil

    def crop_to_remove_black_borders(self, pil_image, threshold=0):
        """
        Automatically detect and crop black/empty borders from an image.
        Useful as a post-processing step after distortion.

        Args:
            pil_image: PIL image
            threshold: Value to consider as "black" (default 0)

        Returns:
            Cropped PIL image
        """
        # Convert to numpy array for processing
        np_image = np.array(pil_image)

        # Handle different image modes
        if len(np_image.shape) == 3:
            # For RGB/RGBA images, take sum across color channels
            # If any channel has value, it's not black
            mask = np_image.sum(axis=2) > threshold
        else:
            # For grayscale
            mask = np_image > threshold

        # Find the bounding box of non-black pixels
        coords = np.argwhere(mask)

        if len(coords) == 0:
            # No non-black pixels found
            return pil_image

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop the image to those coordinates
        cropped = pil_image.crop((x_min, y_min, x_max + 1, y_max + 1))

        return cropped

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        distorted_image = self.apply_advanced_spherical_distortion_cv2(
            image,
            self.k1,
            self.k2,
            self.p1,
            self.p2,
        )

        image_dto = context.images.save(image=distorted_image)

        return ImageOutput.build(image_dto)

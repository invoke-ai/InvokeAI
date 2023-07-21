from typing import Literal, Optional

import numpy as np
import mediapipe as mp
from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageDraw
from pydantic import BaseModel, Field
from typing import Union
import cv2

from ..models.image import ImageCategory, ImageField, ResourceOrigin
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)


class PILInvocationConfig(BaseModel):
    """Helper class to provide all PIL invocations with additional config"""

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["PIL", "image"],
            },
        }


class ImageMaskOutputFaceMask(BaseInvocationOutput):
    """Base class for invocations that output an image and a mask"""

    # fmt: off
    type: Literal["image_mask_output"] = "image_mask_output"
    image:      ImageField = Field(default=None, description="The output image")
    width:             int = Field(description="The width of the image in pixels")
    height:            int = Field(description="The height of the image in pixels")
    mask:       ImageField = Field(default=None, description="The output mask")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "image", "width", "height", "mask"]}


class FaceMaskInvocation(BaseInvocation, PILInvocationConfig):
    """MediaPipe face detection to create transparencies in an image"""

    # fmt: off
    type: Literal["img_detect_mask"] = "img_detect_mask"

    # Inputs
    image: Optional[ImageField]  = Field(default=None, description="Image to apply transparency to")
    x_offset: float = Field(default=0.0, description="Offset for the X-axis of the oval mask")
    y_offset: float = Field(default=0.0, description="Offset for the Y-axis of the oval mask")
    invert_mask: bool = Field(default=False, description="Toggle to invert the mask")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Face Mask",
                "tags": ["image", "face", "mask"]
            },
        }

    def generate_face_mask(self, pil_image):
        # Convert the PIL image to a NumPy array.
        np_image = np.array(pil_image, dtype=np.uint8)

        # Check if the input image has four channels (RGBA).
        if np_image.shape[2] == 4:
            # Convert RGBA to RGB by removing the alpha channel.
            np_image = np_image[:, :, :3]

        # Create a FaceMesh object for face landmark detection and mesh generation.
        face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Detect the face landmarks and mesh in the input image.
        results = face_mesh.process(np_image)

        # Generate a binary face mask using the face mesh.
        mask_image = np.zeros_like(np_image[:, :, 0])
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_landmark_points = np.array(
                    [[landmark.x * np_image.shape[1], landmark.y * np_image.shape[0]] for landmark in face_landmarks.landmark]
                )

                # Apply the scaling offsets to the face landmark points.
                x_center = np.mean(face_landmark_points[:, 0])
                y_center = np.mean(face_landmark_points[:, 1])
                x_scaled = face_landmark_points[:, 0] + self.x_offset * (face_landmark_points[:, 0] - x_center)
                y_scaled = face_landmark_points[:, 1] + self.y_offset * (face_landmark_points[:, 1] - y_center)

                convex_hull = cv2.convexHull(np.column_stack((x_scaled, y_scaled)).astype(np.int32))
                cv2.fillConvexPoly(mask_image, convex_hull, 255)

        # Convert the binary mask image to a PIL Image.
        mask_pil = Image.fromarray(mask_image, mode='L')

        return mask_pil

    def invoke(self, context: InvocationContext) -> ImageMaskOutputFaceMask:
        image = context.services.images.get_pil_image(self.image.image_name)

        # Generate the face mesh mask.
        mask_pil = self.generate_face_mask(image)

        # Create an RGBA image with transparency
        rgba_image = image.convert("RGBA")

        if self.invert_mask:
            # Apply the mask to make the face transparent.
            composite_image = Image.composite(rgba_image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask_pil)

        else:
            # Invert the mask to make everything outside the face transparent.
            inverted_mask = ImageOps.invert(mask_pil)
            composite_image = Image.composite(rgba_image, Image.new("RGBA", image.size, (0, 0, 0, 0)), inverted_mask)

        # Create white mask with dimensions as transparency image for use with outpainting
        white_mask = Image.new("L", image.size, color=255)

        image_dto = context.services.images.create(
            image=composite_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )
        white_mask_dto = context.services.images.create(
            image=white_mask,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageMaskOutputFaceMask(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=white_mask_dto.image_name),
        )

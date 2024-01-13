import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypedDict

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh  # type: ignore[import]
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from PIL.Image import Image as ImageType
from pydantic import field_validator

import invokeai.assets.fonts as font_assets
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    WithMetadata,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import ImageField, InputField, OutputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory

if TYPE_CHECKING:
    from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("face_mask_output")
class FaceMaskOutput(ImageOutput):
    """Base class for FaceMask output"""

    mask: ImageField = OutputField(description="The output mask")


@invocation_output("face_off_output")
class FaceOffOutput(ImageOutput):
    """Base class for FaceOff Output"""

    mask: ImageField = OutputField(description="The output mask")
    x: int = OutputField(description="The x coordinate of the bounding box's left side")
    y: int = OutputField(description="The y coordinate of the bounding box's top side")


class FaceResultData(TypedDict):
    image: ImageType
    mask: ImageType
    x_center: float
    y_center: float
    mesh_width: int
    mesh_height: int
    chunk_x_offset: int
    chunk_y_offset: int


class FaceResultDataWithId(FaceResultData):
    face_id: int


class ExtractFaceData(TypedDict):
    bounded_image: ImageType
    bounded_mask: ImageType
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class FaceMaskResult(TypedDict):
    image: ImageType
    mask: ImageType


def create_white_image(w: int, h: int) -> ImageType:
    return Image.new("L", (w, h), color=255)


def create_black_image(w: int, h: int) -> ImageType:
    return Image.new("L", (w, h), color=0)


FONT_SIZE = 32
FONT_STROKE_WIDTH = 4


def coalesce_faces(face1: FaceResultData, face2: FaceResultData) -> FaceResultData:
    face1_x_offset = face1["chunk_x_offset"] - min(face1["chunk_x_offset"], face2["chunk_x_offset"])
    face2_x_offset = face2["chunk_x_offset"] - min(face1["chunk_x_offset"], face2["chunk_x_offset"])
    face1_y_offset = face1["chunk_y_offset"] - min(face1["chunk_y_offset"], face2["chunk_y_offset"])
    face2_y_offset = face2["chunk_y_offset"] - min(face1["chunk_y_offset"], face2["chunk_y_offset"])

    new_im_width = (
        max(face1["image"].width, face2["image"].width)
        + max(face1["chunk_x_offset"], face2["chunk_x_offset"])
        - min(face1["chunk_x_offset"], face2["chunk_x_offset"])
    )
    new_im_height = (
        max(face1["image"].height, face2["image"].height)
        + max(face1["chunk_y_offset"], face2["chunk_y_offset"])
        - min(face1["chunk_y_offset"], face2["chunk_y_offset"])
    )
    pil_image = Image.new(mode=face1["image"].mode, size=(new_im_width, new_im_height))
    pil_image.paste(face1["image"], (face1_x_offset, face1_y_offset))
    pil_image.paste(face2["image"], (face2_x_offset, face2_y_offset))

    # Mask images are always from the origin
    new_mask_im_width = max(face1["mask"].width, face2["mask"].width)
    new_mask_im_height = max(face1["mask"].height, face2["mask"].height)
    mask_pil = create_white_image(new_mask_im_width, new_mask_im_height)
    black_image = create_black_image(face1["mask"].width, face1["mask"].height)
    mask_pil.paste(black_image, (0, 0), ImageOps.invert(face1["mask"]))
    black_image = create_black_image(face2["mask"].width, face2["mask"].height)
    mask_pil.paste(black_image, (0, 0), ImageOps.invert(face2["mask"]))

    new_face = FaceResultData(
        image=pil_image,
        mask=mask_pil,
        x_center=max(face1["x_center"], face2["x_center"]),
        y_center=max(face1["y_center"], face2["y_center"]),
        mesh_width=max(face1["mesh_width"], face2["mesh_width"]),
        mesh_height=max(face1["mesh_height"], face2["mesh_height"]),
        chunk_x_offset=max(face1["chunk_x_offset"], face2["chunk_x_offset"]),
        chunk_y_offset=max(face2["chunk_y_offset"], face2["chunk_y_offset"]),
    )
    return new_face


def prepare_faces_list(
    face_result_list: list[FaceResultData],
) -> list[FaceResultDataWithId]:
    """Deduplicates a list of faces, adding IDs to them."""
    deduped_faces: list[FaceResultData] = []

    if len(face_result_list) == 0:
        return []

    for candidate in face_result_list:
        should_add = True
        candidate_x_center = candidate["x_center"]
        candidate_y_center = candidate["y_center"]
        for idx, face in enumerate(deduped_faces):
            face_center_x = face["x_center"]
            face_center_y = face["y_center"]
            face_radius_w = face["mesh_width"] / 2
            face_radius_h = face["mesh_height"] / 2
            # Determine if the center of the candidate_face is inside the ellipse of the added face
            # p < 1 -> Inside
            # p = 1 -> Exactly on the ellipse
            # p > 1 -> Outside
            p = (math.pow((candidate_x_center - face_center_x), 2) / math.pow(face_radius_w, 2)) + (
                math.pow((candidate_y_center - face_center_y), 2) / math.pow(face_radius_h, 2)
            )

            if p < 1:  # Inside of the already-added face's radius
                deduped_faces[idx] = coalesce_faces(face, candidate)
                should_add = False
                break

        if should_add is True:
            deduped_faces.append(candidate)

    sorted_faces = sorted(deduped_faces, key=lambda x: x["y_center"])
    sorted_faces = sorted(sorted_faces, key=lambda x: x["x_center"])

    # add face_id for reference
    sorted_faces_with_ids: list[FaceResultDataWithId] = []
    face_id_counter = 0
    for face in sorted_faces:
        sorted_faces_with_ids.append(
            FaceResultDataWithId(
                **face,
                face_id=face_id_counter,
            )
        )
        face_id_counter += 1

    return sorted_faces_with_ids


def generate_face_box_mask(
    context: "InvocationContext",
    minimum_confidence: float,
    x_offset: float,
    y_offset: float,
    pil_image: ImageType,
    chunk_x_offset: int = 0,
    chunk_y_offset: int = 0,
    draw_mesh: bool = True,
) -> list[FaceResultData]:
    result = []
    mask_pil = None

    # Convert the PIL image to a NumPy array.
    np_image = np.array(pil_image, dtype=np.uint8)

    # Check if the input image has four channels (RGBA).
    if np_image.shape[2] == 4:
        # Convert RGBA to RGB by removing the alpha channel.
        np_image = np_image[:, :, :3]

    # Create a FaceMesh object for face landmark detection and mesh generation.
    face_mesh = FaceMesh(
        max_num_faces=999,
        min_detection_confidence=minimum_confidence,
        min_tracking_confidence=minimum_confidence,
    )

    # Detect the face landmarks and mesh in the input image.
    results = face_mesh.process(np_image)

    # Check if any face is detected.
    if results.multi_face_landmarks:  # type: ignore # this are via protobuf and not typed
        # Search for the face_id in the detected faces.
        for _face_id, face_landmarks in enumerate(results.multi_face_landmarks):  # type: ignore #this are via protobuf and not typed
            # Get the bounding box of the face mesh.
            x_coordinates = [landmark.x for landmark in face_landmarks.landmark]
            y_coordinates = [landmark.y for landmark in face_landmarks.landmark]
            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)

            # Calculate the width and height of the face mesh.
            mesh_width = int((x_max - x_min) * np_image.shape[1])
            mesh_height = int((y_max - y_min) * np_image.shape[0])

            # Get the center of the face.
            x_center = np.mean([landmark.x * np_image.shape[1] for landmark in face_landmarks.landmark])
            y_center = np.mean([landmark.y * np_image.shape[0] for landmark in face_landmarks.landmark])

            face_landmark_points = np.array(
                [
                    [landmark.x * np_image.shape[1], landmark.y * np_image.shape[0]]
                    for landmark in face_landmarks.landmark
                ]
            )

            # Apply the scaling offsets to the face landmark points with a multiplier.
            scale_multiplier = 0.2
            x_center = np.mean(face_landmark_points[:, 0])
            y_center = np.mean(face_landmark_points[:, 1])

            if draw_mesh:
                x_scaled = face_landmark_points[:, 0] + scale_multiplier * x_offset * (
                    face_landmark_points[:, 0] - x_center
                )
                y_scaled = face_landmark_points[:, 1] + scale_multiplier * y_offset * (
                    face_landmark_points[:, 1] - y_center
                )

                convex_hull = cv2.convexHull(np.column_stack((x_scaled, y_scaled)).astype(np.int32))

                # Generate a binary face mask using the face mesh.
                mask_image = np.ones(np_image.shape[:2], dtype=np.uint8) * 255
                cv2.fillConvexPoly(mask_image, convex_hull, 0)

                # Convert the binary mask image to a PIL Image.
                init_mask_pil = Image.fromarray(mask_image, mode="L")
                w, h = init_mask_pil.size
                mask_pil = create_white_image(w + chunk_x_offset, h + chunk_y_offset)
                mask_pil.paste(init_mask_pil, (chunk_x_offset, chunk_y_offset))

            x_center = float(x_center)
            y_center = float(y_center)
            face = FaceResultData(
                image=pil_image,
                mask=mask_pil or create_white_image(*pil_image.size),
                x_center=x_center + chunk_x_offset,
                y_center=y_center + chunk_y_offset,
                mesh_width=mesh_width,
                mesh_height=mesh_height,
                chunk_x_offset=chunk_x_offset,
                chunk_y_offset=chunk_y_offset,
            )

            result.append(face)

    return result


def extract_face(
    context: "InvocationContext",
    image: ImageType,
    face: FaceResultData,
    padding: int,
) -> ExtractFaceData:
    mask = face["mask"]
    center_x = face["x_center"]
    center_y = face["y_center"]
    mesh_width = face["mesh_width"]
    mesh_height = face["mesh_height"]

    # Determine the minimum size of the square crop
    min_size = min(mask.width, mask.height)

    # Calculate the crop boundaries for the output image and mask.
    mesh_width += 128 + padding  # add pixels to account for mask variance
    mesh_height += 128 + padding  # add pixels to account for mask variance
    crop_size = min(
        max(mesh_width, mesh_height, 128), min_size
    )  # Choose the smaller of the two (given value or face mask size)
    if crop_size > 128:
        crop_size = (crop_size + 7) // 8 * 8  # Ensure crop side is multiple of 8

    # Calculate the actual crop boundaries within the bounds of the original image.
    x_min = int(center_x - crop_size / 2)
    y_min = int(center_y - crop_size / 2)
    x_max = int(center_x + crop_size / 2)
    y_max = int(center_y + crop_size / 2)

    # Adjust the crop boundaries to stay within the original image's dimensions
    if x_min < 0:
        context.logger.warning("FaceTools --> -X-axis padding reached image edge.")
        x_max -= x_min
        x_min = 0
    elif x_max > mask.width:
        context.logger.warning("FaceTools --> +X-axis padding reached image edge.")
        x_min -= x_max - mask.width
        x_max = mask.width

    if y_min < 0:
        context.logger.warning("FaceTools --> +Y-axis padding reached image edge.")
        y_max -= y_min
        y_min = 0
    elif y_max > mask.height:
        context.logger.warning("FaceTools --> -Y-axis padding reached image edge.")
        y_min -= y_max - mask.height
        y_max = mask.height

    # Ensure the crop is square and adjust the boundaries if needed
    if x_max - x_min != crop_size:
        context.logger.warning("FaceTools --> Limiting x-axis padding to constrain bounding box to a square.")
        diff = crop_size - (x_max - x_min)
        x_min -= diff // 2
        x_max += diff - diff // 2

    if y_max - y_min != crop_size:
        context.logger.warning("FaceTools --> Limiting y-axis padding to constrain bounding box to a square.")
        diff = crop_size - (y_max - y_min)
        y_min -= diff // 2
        y_max += diff - diff // 2

    context.logger.info(f"FaceTools --> Calculated bounding box (8 multiple): {crop_size}")

    # Crop the output image to the specified size with the center of the face mesh as the center.
    mask = mask.crop((x_min, y_min, x_max, y_max))
    bounded_image = image.crop((x_min, y_min, x_max, y_max))

    # blur mask edge by small radius
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))

    return ExtractFaceData(
        bounded_image=bounded_image,
        bounded_mask=mask,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def get_faces_list(
    context: "InvocationContext",
    image: ImageType,
    should_chunk: bool,
    minimum_confidence: float,
    x_offset: float,
    y_offset: float,
    draw_mesh: bool = True,
) -> list[FaceResultDataWithId]:
    result = []

    # Generate the face box mask and get the center of the face.
    if not should_chunk:
        context.logger.info("FaceTools --> Attempting full image face detection.")
        result = generate_face_box_mask(
            context=context,
            minimum_confidence=minimum_confidence,
            x_offset=x_offset,
            y_offset=y_offset,
            pil_image=image,
            chunk_x_offset=0,
            chunk_y_offset=0,
            draw_mesh=draw_mesh,
        )
    if should_chunk or len(result) == 0:
        context.logger.info("FaceTools --> Chunking image (chunk toggled on, or no face found in full image).")
        width, height = image.size
        image_chunks = []
        x_offsets = []
        y_offsets = []
        result = []

        # If width == height, there's nothing more we can do... otherwise...
        if width > height:
            # Landscape - slice the image horizontally
            fx = 0.0
            steps = int(width * 2 / height) + 1
            increment = (width - height) / (steps - 1)
            while fx <= (width - height):
                x = int(fx)
                image_chunks.append(image.crop((x, 0, x + height, height)))
                x_offsets.append(x)
                y_offsets.append(0)
                fx += increment
                context.logger.info(f"FaceTools --> Chunk starting at x = {x}")
        elif height > width:
            # Portrait - slice the image vertically
            fy = 0.0
            steps = int(height * 2 / width) + 1
            increment = (height - width) / (steps - 1)
            while fy <= (height - width):
                y = int(fy)
                image_chunks.append(image.crop((0, y, width, y + width)))
                x_offsets.append(0)
                y_offsets.append(y)
                fy += increment
                context.logger.info(f"FaceTools --> Chunk starting at y = {y}")

        for idx in range(len(image_chunks)):
            context.logger.info(f"FaceTools --> Evaluating faces in chunk {idx}")
            result = result + generate_face_box_mask(
                context=context,
                minimum_confidence=minimum_confidence,
                x_offset=x_offset,
                y_offset=y_offset,
                pil_image=image_chunks[idx],
                chunk_x_offset=x_offsets[idx],
                chunk_y_offset=y_offsets[idx],
                draw_mesh=draw_mesh,
            )

        if len(result) == 0:
            # Give up
            context.logger.warning(
                "FaceTools --> No face detected in chunked input image. Passing through original image."
            )

    all_faces = prepare_faces_list(result)

    return all_faces


@invocation("face_off", title="FaceOff", tags=["image", "faceoff", "face", "mask"], category="image", version="1.2.1")
class FaceOffInvocation(BaseInvocation, WithMetadata):
    """Bound, extract, and mask a face from an image using MediaPipe detection"""

    image: ImageField = InputField(description="Image for face detection")
    face_id: int = InputField(
        default=0,
        ge=0,
        description="The face ID to process, numbered from 0. Multiple faces not supported. Find a face's ID with FaceIdentifier node.",
    )
    minimum_confidence: float = InputField(
        default=0.5, description="Minimum confidence for face detection (lower if detection is failing)"
    )
    x_offset: float = InputField(default=0.0, description="X-axis offset of the mask")
    y_offset: float = InputField(default=0.0, description="Y-axis offset of the mask")
    padding: int = InputField(default=0, description="All-axis padding around the mask in pixels")
    chunk: bool = InputField(
        default=False,
        description="Whether to bypass full image face detection and default to image chunking. Chunking will occur if no faces are found in the full image.",
    )

    def faceoff(self, context: "InvocationContext", image: ImageType) -> Optional[ExtractFaceData]:
        all_faces = get_faces_list(
            context=context,
            image=image,
            should_chunk=self.chunk,
            minimum_confidence=self.minimum_confidence,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            draw_mesh=True,
        )

        if len(all_faces) == 0:
            context.logger.warning("FaceOff --> No faces detected. Passing through original image.")
            return None

        if self.face_id > len(all_faces) - 1:
            context.logger.warning(
                f"FaceOff --> Face ID {self.face_id} is outside of the number of faces detected ({len(all_faces)}). Passing through original image."
            )
            return None

        face_data = extract_face(context=context, image=image, face=all_faces[self.face_id], padding=self.padding)
        # Convert the input image to RGBA mode to ensure it has an alpha channel.
        face_data["bounded_image"] = face_data["bounded_image"].convert("RGBA")

        return face_data

    def invoke(self, context) -> FaceOffOutput:
        image = context.images.get_pil(self.image.image_name)
        result = self.faceoff(context=context, image=image)

        if result is None:
            result_image = image
            result_mask = create_white_image(*image.size)
            x = 0
            y = 0
        else:
            result_image = result["bounded_image"]
            result_mask = result["bounded_mask"]
            x = result["x_min"]
            y = result["y_min"]

        image_dto = context.images.save(image=result_image)

        mask_dto = context.images.save(image=result_mask, image_category=ImageCategory.MASK)

        output = FaceOffOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=mask_dto.image_name),
            x=x,
            y=y,
        )

        return output


@invocation("face_mask_detection", title="FaceMask", tags=["image", "face", "mask"], category="image", version="1.2.1")
class FaceMaskInvocation(BaseInvocation, WithMetadata):
    """Face mask creation using mediapipe face detection"""

    image: ImageField = InputField(description="Image to face detect")
    face_ids: str = InputField(
        default="",
        description="Comma-separated list of face ids to mask eg '0,2,7'. Numbered from 0. Leave empty to mask all. Find face IDs with FaceIdentifier node.",
    )
    minimum_confidence: float = InputField(
        default=0.5, description="Minimum confidence for face detection (lower if detection is failing)"
    )
    x_offset: float = InputField(default=0.0, description="Offset for the X-axis of the face mask")
    y_offset: float = InputField(default=0.0, description="Offset for the Y-axis of the face mask")
    chunk: bool = InputField(
        default=False,
        description="Whether to bypass full image face detection and default to image chunking. Chunking will occur if no faces are found in the full image.",
    )
    invert_mask: bool = InputField(default=False, description="Toggle to invert the mask")

    @field_validator("face_ids")
    def validate_comma_separated_ints(cls, v) -> str:
        comma_separated_ints_regex = re.compile(r"^\d*(,\d+)*$")
        if comma_separated_ints_regex.match(v) is None:
            raise ValueError('Face IDs must be a comma-separated list of integers (e.g. "1,2,3")')
        return v

    def facemask(self, context: "InvocationContext", image: ImageType) -> FaceMaskResult:
        all_faces = get_faces_list(
            context=context,
            image=image,
            should_chunk=self.chunk,
            minimum_confidence=self.minimum_confidence,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            draw_mesh=True,
        )

        mask_pil = create_white_image(*image.size)

        id_range = list(range(0, len(all_faces)))
        ids_to_extract = id_range
        if self.face_ids != "":
            parsed_face_ids = [int(id) for id in self.face_ids.split(",")]
            # get requested face_ids that are in range
            intersected_face_ids = set(parsed_face_ids) & set(id_range)

            if len(intersected_face_ids) == 0:
                id_range_str = ",".join([str(id) for id in id_range])
                context.logger.warning(
                    f"Face IDs must be in range of detected faces - requested {self.face_ids}, detected {id_range_str}. Passing through original image."
                )
                return FaceMaskResult(
                    image=image,  # original image
                    mask=mask_pil,  # white mask
                )

            ids_to_extract = list(intersected_face_ids)

        for face_id in ids_to_extract:
            face_data = extract_face(context=context, image=image, face=all_faces[face_id], padding=0)
            face_mask_pil = face_data["bounded_mask"]
            x_min = face_data["x_min"]
            y_min = face_data["y_min"]
            x_max = face_data["x_max"]
            y_max = face_data["y_max"]

            mask_pil.paste(
                create_black_image(x_max - x_min, y_max - y_min),
                box=(x_min, y_min),
                mask=ImageOps.invert(face_mask_pil),
            )

        if self.invert_mask:
            mask_pil = ImageOps.invert(mask_pil)

        # Create an RGBA image with transparency
        image = image.convert("RGBA")

        return FaceMaskResult(
            image=image,
            mask=mask_pil,
        )

    def invoke(self, context) -> FaceMaskOutput:
        image = context.images.get_pil(self.image.image_name)
        result = self.facemask(context=context, image=image)

        image_dto = context.images.save(image=result["image"])

        mask_dto = context.images.save(image=result["mask"], image_category=ImageCategory.MASK)

        output = FaceMaskOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=mask_dto.image_name),
        )

        return output


@invocation(
    "face_identifier", title="FaceIdentifier", tags=["image", "face", "identifier"], category="image", version="1.2.1"
)
class FaceIdentifierInvocation(BaseInvocation, WithMetadata):
    """Outputs an image with detected face IDs printed on each face. For use with other FaceTools."""

    image: ImageField = InputField(description="Image to face detect")
    minimum_confidence: float = InputField(
        default=0.5, description="Minimum confidence for face detection (lower if detection is failing)"
    )
    chunk: bool = InputField(
        default=False,
        description="Whether to bypass full image face detection and default to image chunking. Chunking will occur if no faces are found in the full image.",
    )

    def faceidentifier(self, context: "InvocationContext", image: ImageType) -> ImageType:
        image = image.copy()

        all_faces = get_faces_list(
            context=context,
            image=image,
            should_chunk=self.chunk,
            minimum_confidence=self.minimum_confidence,
            x_offset=0,
            y_offset=0,
            draw_mesh=False,
        )

        # Note - font may be found either in the repo if running an editable install, or in the venv if running a package install
        font_path = [x for x in [Path(y, "inter/Inter-Regular.ttf") for y in font_assets.__path__] if x.exists()]
        font = ImageFont.truetype(font_path[0].as_posix(), FONT_SIZE)

        # Paste face IDs on the output image
        draw = ImageDraw.Draw(image)
        for face in all_faces:
            x_coord = face["x_center"]
            y_coord = face["y_center"]
            text = str(face["face_id"])
            # get bbox of the text so we can center the id on the face
            _, _, bbox_w, bbox_h = draw.textbbox(xy=(0, 0), text=text, font=font, stroke_width=FONT_STROKE_WIDTH)
            x = x_coord - bbox_w / 2
            y = y_coord - bbox_h / 2
            draw.text(
                xy=(x, y),
                text=str(text),
                fill=(255, 255, 255, 255),
                font=font,
                stroke_width=FONT_STROKE_WIDTH,
                stroke_fill=(0, 0, 0, 255),
            )

        # Create an RGBA image with transparency
        image = image.convert("RGBA")

        return image

    def invoke(self, context) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        result_image = self.faceidentifier(context=context, image=image)

        image_dto = context.images.save(image=result_image)

        return ImageOutput.build(image_dto)

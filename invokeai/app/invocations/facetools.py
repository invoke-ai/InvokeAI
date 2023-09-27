import math
from typing import Optional, TypedDict

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from PIL.Image import Image as ImageType

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.models.image import ImageCategory, ResourceOrigin


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


class FaceResultsData(TypedDict):
    pil_image: Optional[ImageType]
    mask_pil: Optional[ImageType]
    x_center: float
    y_center: float
    mesh_width: int
    mesh_height: int
    face_id: Optional[int]


class ExtractFaceData(TypedDict):
    bounded_image: ImageType
    mask_pil: Optional[ImageType]
    x_min: int
    y_min: int
    x_max: int
    y_max: int


def cleanup_faces_list(
    orig: list[FaceResultsData],
) -> list[FaceResultsData]:
    newlist = []

    if len(orig) == 0:
        return orig

    for i in orig:
        should_add = True
        i_x_center = i["x_center"]
        i_y_center = i["y_center"]
        for j in newlist:
            face_center_x = j["x_center"]
            face_center_y = j["y_center"]
            face_radius_w = j["mesh_width"] / 2
            face_radius_h = j["mesh_height"] / 2
            # Determine if the center of the candidate i is inside the ellipse of the added face
            # p < 1 -> Inside
            # p = 1 -> Exactly on the ellipse
            # p > 1 -> Outside
            p = (math.pow((i_x_center - face_center_x), 2) / math.pow(face_radius_w, 2)) + (
                math.pow((i_y_center - face_center_y), 2) / math.pow(face_radius_h, 2)
            )

            if p < 1:  # Inside of the already-added face's radius
                should_add = False
                break

        if should_add is True:
            newlist.append(i)

    newlist = sorted(newlist, key=lambda x: x["y_center"])
    newlist = sorted(newlist, key=lambda x: x["x_center"])

    # add a face_id for reference
    face_id_counter = 1
    for face in newlist:
        face["face_id"] = face_id_counter
        face_id_counter += 1

    return newlist


def generate_face_box_mask(
    context: InvocationContext,
    minimum_confidence: float,
    x_offset: float,
    y_offset: float,
    pil_image: ImageType,
    chunk_x_offset: int = 0,
    chunk_y_offset: int = 0,
    draw_mesh: bool = True,
) -> list[FaceResultsData]:
    result = []
    mask_pil = None

    # Convert the PIL image to a NumPy array.
    np_image = np.array(pil_image, dtype=np.uint8)

    # Check if the input image has four channels (RGBA).
    if np_image.shape[2] == 4:
        # Convert RGBA to RGB by removing the alpha channel.
        np_image = np_image[:, :, :3]

    # Create a FaceMesh object for face landmark detection and mesh generation.
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=999,
        min_detection_confidence=minimum_confidence,
        min_tracking_confidence=minimum_confidence,
    )

    # Detect the face landmarks and mesh in the input image.
    results = face_mesh.process(np_image)

    # Check if any face is detected.
    if results.multi_face_landmarks:
        # Search for the face_id in the detected faces.
        for face_landmarks in results.multi_face_landmarks:
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
                mask_pil = Image.new(mode="L", size=(w + chunk_x_offset, h + chunk_y_offset), color=255)
                mask_pil.paste(init_mask_pil, (chunk_x_offset, chunk_y_offset))

            left_side = x_center - mesh_width
            right_side = x_center + mesh_width
            top_side = y_center - mesh_height
            bottom_side = y_center + mesh_height
            im_width, im_height = pil_image.size
            over_w = im_width * 0.1
            over_h = im_height * 0.1
            if (
                (left_side >= -over_w)
                and (right_side < im_width + over_w)
                and (top_side >= -over_h)
                and (bottom_side < im_height + over_h)
            ):
                x_center = float(x_center)
                y_center = float(y_center)
                face = FaceResultsData(
                    pil_image=pil_image if draw_mesh else None,
                    mask_pil=mask_pil,
                    x_center=x_center + chunk_x_offset,
                    y_center=y_center + chunk_y_offset,
                    mesh_width=mesh_width,
                    mesh_height=mesh_height,
                    face_id=None,
                )

                result.append(face)
            else:
                context.services.logger.info("FaceTools --> Face out of bounds, ignoring.")

    return result


def extract_face(
    context: InvocationContext,
    image: ImageType,
    all_faces: list[FaceResultsData],
    face_id: int,
    padding: int,
) -> ExtractFaceData:
    mask_pil = all_faces[face_id]["mask_pil"]
    center_x = all_faces[face_id]["x_center"]
    center_y = all_faces[face_id]["y_center"]
    mesh_width = all_faces[face_id]["mesh_width"]
    mesh_height = all_faces[face_id]["mesh_height"]

    # Determine the minimum size of the square crop
    min_size = min(mask_pil.width, mask_pil.height)

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
        context.services.logger.warning("FaceTools --> -X-axis padding reached image edge.")
        x_max -= x_min
        x_min = 0
    elif x_max > mask_pil.width:
        context.services.logger.warning("FaceTools --> +X-axis padding reached image edge.")
        x_min -= x_max - mask_pil.width
        x_max = mask_pil.width

    if y_min < 0:
        context.services.logger.warning("FaceTools --> +Y-axis padding reached image edge.")
        y_max -= y_min
        y_min = 0
    elif y_max > mask_pil.height:
        context.services.logger.warning("FaceTools --> -Y-axis padding reached image edge.")
        y_min -= y_max - mask_pil.height
        y_max = mask_pil.height

    # Ensure the crop is square and adjust the boundaries if needed
    if x_max - x_min != crop_size:
        context.services.logger.warning("FaceTools --> Limiting x-axis padding to constrain bounding box to a square.")
        diff = crop_size - (x_max - x_min)
        x_min -= diff // 2
        x_max += diff - diff // 2

    if y_max - y_min != crop_size:
        context.services.logger.warning("FaceTools --> Limiting y-axis padding to constrain bounding box to a square.")
        diff = crop_size - (y_max - y_min)
        y_min -= diff // 2
        y_max += diff - diff // 2

    context.services.logger.info(f"FaceTools --> Calculated bounding box (8 multiple): {crop_size}")

    # Crop the output image to the specified size with the center of the face mesh as the center.
    mask_pil = mask_pil.crop((x_min, y_min, x_max, y_max))
    bounded_image = image.crop((x_min, y_min, x_max, y_max))

    # blur mask edge by small radius
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))

    return ExtractFaceData(
        bounded_image=bounded_image,
        mask_pil=mask_pil,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def get_faces_list(
    context: InvocationContext,
    image: ImageType,
    should_chunk: bool,
    minimum_confidence: float,
    x_offset: float,
    y_offset: float,
    draw_mesh: bool = True,
) -> list[FaceResultsData]:
    result = []

    # Generate the face box mask and get the center of the face.
    if not should_chunk:
        context.services.logger.info("FaceTools --> Attempting full image face detection.")
        result = generate_face_box_mask(
            context,
            minimum_confidence,
            x_offset,
            y_offset,
            image,
            0,
            0,
            draw_mesh,
        )
    if should_chunk or len(result) == 0:
        context.services.logger.info("FaceTools --> Chunking image (chunk toggled on, or no face found in full image).")
        width, height = image.size
        image_chunks = []
        x_offsets = []
        y_offsets = []
        result = []

        # If width == height, there's nothing more we can do... otherwise...
        if width > height:
            # Landscape - slice the image horizontally
            fx = 0.0
            steps = int(width * 2 / height)
            while fx <= (width - height):
                x = int(fx)
                image_chunks.append(image.crop((x, 0, x + height - 1, height - 1)))
                x_offsets.append(x)
                y_offsets.append(0)
                fx += (width - height) / steps
                context.services.logger.info(f"FaceTools --> Chunk starting at x = {x}")
        elif height > width:
            # Portrait - slice the image vertically
            fy = 0.0
            steps = int(height * 2 / width)
            while fy <= (height - width):
                y = int(fy)
                image_chunks.append(image.crop((0, y, width - 1, y + width - 1)))
                x_offsets.append(0)
                y_offsets.append(y)
                fy += (height - width) / steps
                context.services.logger.info(f"FaceTools --> Chunk starting at y = {y}")

        for idx in range(len(image_chunks)):
            context.services.logger.info(f"FaceTools --> Evaluating faces in chunk {idx}")
            result = result + generate_face_box_mask(
                context,
                minimum_confidence,
                x_offset,
                y_offset,
                image_chunks[idx],
                x_offsets[idx],
                y_offsets[idx],
                draw_mesh,
            )

        if len(result) == 0:
            # Give up
            context.services.logger.warning(
                "FaceTools --> No face detected in chunked input image. Passing through original image."
            )

    all_faces = cleanup_faces_list(result)

    return all_faces


@invocation("face_off", title="FaceOff", tags=["image", "faceoff", "face", "mask"], category="image", version="1.0.0")
class FaceOffInvocation(BaseInvocation):
    """Bound, extract, and mask a face from an image using MediaPipe detection"""

    image: ImageField = InputField(description="Image for face detection")
    face_id: int = InputField(
        default=0,
        description="0 for first detected face, single digit for one specific. Multiple faces not supported. Find a face's ID with FaceIdentifier node.",
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

    def faceoff(self, context: InvocationContext, image) -> Optional[FaceOffOutput]:
        all_faces = get_faces_list(
            context,
            image,
            self.chunk,
            self.minimum_confidence,
            self.x_offset,
            self.y_offset,
        )

        face_id = self.face_id - 1 if self.face_id > 0 else self.face_id

        if face_id + 1 > len(all_faces) or face_id < 0:
            context.services.logger.warning(
                f"FaceOff --> Face ID {self.face_id} is outside of the number of faces detected ({len(all_faces)}). Passing through original image."
            )
            return None

        face_data = extract_face(context, image, all_faces, face_id, self.padding)
        bounded_image = face_data["bounded_image"]
        mask_pil = face_data["mask_pil"]
        x_min = face_data["x_min"]
        y_min = face_data["y_min"]

        # Convert the input image to RGBA mode to ensure it has an alpha channel.
        image = bounded_image.convert("RGBA")

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        mask_dto = context.services.images.create(
            image=mask_pil,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return FaceOffOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=mask_dto.image_name),
            x=x_min,
            y=y_min,
        )

    def invoke(self, context: InvocationContext) -> FaceOffOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        result = self.faceoff(context, image)

        if result is None:
            whitemask = Image.new("L", image.size, color=255)
            context.services.logger.info("FaceOff --> No face detected. Passing through original image.")

            image_dto = context.services.images.create(
                image=image,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
                workflow=self.workflow,
            )

            mask_dto = context.services.images.create(
                image=whitemask,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.MASK,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
            )

            result = FaceOffOutput(
                image=ImageField(image_name=image_dto.image_name),
                width=image_dto.width,
                height=image_dto.height,
                mask=ImageField(image_name=mask_dto.image_name),
                x=0,
                y=0,
            )

        return result


@invocation("face_mask_detection", title="FaceMask", tags=["image", "face", "mask"], category="image", version="1.0.0")
class FaceMaskInvocation(BaseInvocation):
    """Face mask creation using mediapipe face detection"""

    image: ImageField = InputField(description="Image to face detect")
    face_ids: str = InputField(
        default="0",
        description="0 for all faces, single digit for one, comma-separated list for multiple specific (1, 2, 4). Find face IDs with FaceIdentifier node.",
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

    def facemask(self, context: InvocationContext, image) -> Optional[FaceMaskOutput]:
        all_faces = get_faces_list(
            context,
            image,
            self.chunk,
            self.minimum_confidence,
            self.x_offset,
            self.y_offset,
        )

        mask_pil = Image.new(mode="L", size=((image.size)), color=255)

        id_range = range(0, len(all_faces))

        # If '0' is entered, mask all faces, else use IDs provided (minus one)
        if self.face_ids.strip() != "0":
            id_range = [(int(id.strip()) - 1) for id in self.face_ids.split(",")]

        if all(face_id + 1 > len(all_faces) for face_id in id_range):
            return None

        for face_id in id_range:
            if face_id >= 0 and face_id < len(all_faces):
                face_data = extract_face(context, image, all_faces, face_id, 0)
                face_mask_pil = face_data["mask_pil"]
                x_min = face_data["x_min"]
                y_min = face_data["y_min"]
                x_max = face_data["x_max"]
                y_max = face_data["y_max"]

                mask_pil.paste(
                    Image.new(mode="L", size=((x_max - x_min, y_max - y_min)), color=0),
                    box=(x_min, y_min),
                    mask=ImageOps.invert(face_mask_pil),
                )

        if self.invert_mask:
            mask_pil = ImageOps.invert(mask_pil)

        # Create an RGBA image with transparency
        image = image.convert("RGBA")

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        mask_dto = context.services.images.create(
            image=mask_pil,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return FaceMaskOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=mask_dto.image_name),
        )

    def invoke(self, context: InvocationContext) -> FaceMaskOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        result = self.facemask(context, image)

        if result is None:
            whitemask = Image.new("L", image.size, color=255)
            context.services.logger.warning("FaceMask --> No face detected. Passing through original image.")

            image_dto = context.services.images.create(
                image=image,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
                workflow=self.workflow,
            )

            mask_dto = context.services.images.create(
                image=whitemask,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.MASK,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
            )

            result = FaceMaskOutput(
                image=ImageField(image_name=image_dto.image_name),
                width=image_dto.width,
                height=image_dto.height,
                mask=ImageField(image_name=mask_dto.image_name),
            )

        return result


@invocation(
    "face_identifier", title="FaceIdentifier", tags=["image", "face", "identifier"], category="image", version="1.0.0"
)
class FaceIdentifierInvocation(BaseInvocation):
    """Outputs an image with detected face IDs printed on each face. For use with other FaceTools."""

    image: ImageField = InputField(description="Image to face detect")
    minimum_confidence: float = InputField(
        default=0.5, description="Minimum confidence for face detection (lower if detection is failing)"
    )
    chunk: bool = InputField(
        default=False,
        description="Whether to bypass full image face detection and default to image chunking. Chunking will occur if no faces are found in the full image.",
    )

    def faceidentifier(self, context: InvocationContext, image) -> Optional[ImageOutput]:
        image = image.copy()

        all_faces = get_faces_list(
            context,
            image,
            self.chunk,
            self.minimum_confidence,
            0,
            0,
            False,
        )

        # Paste face IDs on the output image
        draw = ImageDraw.Draw(image)
        for face in all_faces:
            x_coord = face["x_center"]
            y_coord = face["y_center"]
            face_id = face["face_id"]
            draw.text((x_coord, y_coord), str(face_id), fill=(255, 255, 255, 255))

        # Create an RGBA image with transparency
        image = image.convert("RGBA")

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        result = self.faceidentifier(context, image)

        if result is None:
            image = context.services.images.get_pil_image(self.image.image_name)
            context.services.logger.warning(
                "FaceIdentifier --> No face detected. Passing through original image without drawing FaceIDs."
            )

            image_dto = context.services.images.create(
                image=image,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
                workflow=self.workflow,
            )

            result = ImageOutput(
                image=ImageField(image_name=image_dto.image_name),
                width=image_dto.width,
                height=image_dto.height,
            )

        return result

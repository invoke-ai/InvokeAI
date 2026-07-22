from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import cv2
import mediapipe as mp
import numpy
import numpy.typing as npt
from mediapipe.tasks import python as _mp_python  # type: ignore[import]
from mediapipe.tasks.python import vision as _vision  # type: ignore[import]
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark  # type: ignore[import]

mp_python: Any = _mp_python
vision: Any = _vision


@dataclass(frozen=True)
class DrawingSpec:
    color: tuple[int, int, int]
    thickness: int
    circle_radius: int


FaceLandmarks = Sequence[NormalizedLandmark]
FaceConnection = tuple[int, int]
ImageArray: TypeAlias = npt.NDArray[numpy.uint8]
DrawingSpecMap: TypeAlias = Mapping[int, DrawingSpec] | DrawingSpec

_FACE_LANDMARKER_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

min_face_size_pixels: int = 64
f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec: dict[FaceConnection, DrawingSpec] = {}
for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL:
    edge = (connection.start, connection.end)
    face_connection_spec[edge] = head_draw
for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE:
    edge = (connection.start, connection.end)
    face_connection_spec[edge] = left_eye_draw
for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW:
    edge = (connection.start, connection.end)
    face_connection_spec[edge] = left_eyebrow_draw
# for edge in vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE:
    edge = (connection.start, connection.end)
    face_connection_spec[edge] = right_eye_draw
for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW:
    edge = (connection.start, connection.end)
    face_connection_spec[edge] = right_eyebrow_draw
# for edge in vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
    edge = (connection.start, connection.end)
    face_connection_spec[edge] = mouth_draw
iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}


def _get_face_landmarker_model_path() -> str:
    if not _FACE_LANDMARKER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing vendored MediaPipe model asset at {_FACE_LANDMARKER_MODEL_PATH}. "
            "Reinstall the package or restore face_landmarker.task."
        )
    return str(_FACE_LANDMARKER_MODEL_PATH)


def detect_face_landmarks(img_rgb: ImageArray, max_faces: int, min_confidence: float) -> list[FaceLandmarks]:
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_get_face_landmarker_model_path()),
        num_faces=max_faces,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)

    return results.face_landmarks


def _landmark_xy(landmark: NormalizedLandmark) -> tuple[float, float] | None:
    if landmark.x is None or landmark.y is None:
        return None

    return landmark.x, landmark.y


def _landmark_to_pixel(landmark: NormalizedLandmark, image_rows: int, image_cols: int) -> tuple[int, int] | None:
    coords = _landmark_xy(landmark)
    if coords is None:
        return None

    x_coord, y_coord = coords

    if x_coord < 0 or x_coord > 1 or y_coord < 0 or y_coord > 1:
        return None

    image_x = min(int(x_coord * (image_cols - 1)), image_cols - 1)
    image_y = min(int(y_coord * (image_rows - 1)), image_rows - 1)
    return image_x, image_y


def _draw_connections(image: ImageArray, landmark_list: FaceLandmarks, drawing_spec: Mapping[FaceConnection, DrawingSpec]) -> None:
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")

    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:
        raise ValueError("Input image must contain three channel bgr data.")

    for (start_idx, end_idx), spec in drawing_spec.items():
        if start_idx >= len(landmark_list) or end_idx >= len(landmark_list):
            continue

        start_point = _landmark_to_pixel(landmark_list[start_idx], image_rows, image_cols)
        end_point = _landmark_to_pixel(landmark_list[end_idx], image_rows, image_cols)
        if start_point is None or end_point is None:
            continue

        cv2.line(image, start_point, end_point, spec.color, spec.thickness)


def draw_pupils(image: ImageArray, landmark_list: FaceLandmarks, drawing_spec: DrawingSpecMap, halfwidth: int = 2) -> None:
    """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
    landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError("Input image must contain three channel bgr data.")
    for idx, landmark in enumerate(landmark_list):
        if (landmark.visibility is not None and landmark.visibility < 0.9) or (
            landmark.presence is not None and landmark.presence < 0.5
        ):
            continue
        point = _landmark_to_pixel(landmark, image_rows, image_cols)
        if point is None:
            continue

        image_x, image_y = point
        draw_color = None
        if isinstance(drawing_spec, Mapping):
            if drawing_spec.get(idx) is None:
                continue
            draw_color = drawing_spec[idx].color
        else:
            draw_color = drawing_spec.color

        y_min = max(image_y - halfwidth, 0)
        y_max = min(image_y + halfwidth, image_rows)
        x_min = max(image_x - halfwidth, 0)
        x_max = min(image_x + halfwidth, image_cols)
        image[y_min:y_max, x_min:x_max, :] = draw_color


def reverse_channels(image: ImageArray) -> ImageArray:
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def generate_annotation(img_rgb: ImageArray, max_faces: int, min_confidence: float) -> ImageArray:
    """
    Find up to 'max_faces' inside the provided input image.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    """
    img_height, img_width, img_channels = img_rgb.shape
    assert img_channels == 3

    results = detect_face_landmarks(img_rgb, max_faces=max_faces, min_confidence=min_confidence)

    if len(results) == 0:
        print("No faces detected in controlnet image for Mediapipe face annotator.")
        return numpy.zeros_like(img_rgb)

    # Filter faces that are too small
    filtered_landmarks: list[FaceLandmarks] = []
    for landmarks in results:
        first_coords = _landmark_xy(landmarks[0])
        if first_coords is None:
            continue

        face_rect = [
            first_coords[0],
            first_coords[1],
            first_coords[0],
            first_coords[1],
        ]  # Left, up, right, down.
        for landmark in landmarks:
            coords = _landmark_xy(landmark)
            if coords is None:
                continue

            face_rect[0] = min(face_rect[0], coords[0])
            face_rect[1] = min(face_rect[1], coords[1])
            face_rect[2] = max(face_rect[2], coords[0])
            face_rect[3] = max(face_rect[3], coords[1])
        if min_face_size_pixels > 0:
            face_width = abs(face_rect[2] - face_rect[0])
            face_height = abs(face_rect[3] - face_rect[1])
            face_width_pixels = face_width * img_width
            face_height_pixels = face_height * img_height
            face_size = min(face_width_pixels, face_height_pixels)
            if face_size >= min_face_size_pixels:
                filtered_landmarks.append(landmarks)
        else:
            filtered_landmarks.append(landmarks)

    # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
    empty = numpy.zeros_like(img_rgb)

    # Draw detected faces:
    for face_landmarks in filtered_landmarks:
        _draw_connections(empty, face_landmarks, face_connection_spec)
        draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)

    # Flip BGR back to RGB.
    empty = reverse_channels(empty).copy()

    return empty

# Adapted from https://github.com/huggingface/controlnet_aux

from PIL import Image

from invokeai.backend.image_util.mediapipe_face.mediapipe_face_common import generate_annotation
from invokeai.backend.image_util.util import np_to_pil, pil_to_np


def detect_faces(image: Image.Image, max_faces: int = 1, min_confidence: float = 0.5) -> Image.Image:
    """Detects faces in an image using MediaPipe."""

    np_img = pil_to_np(image)
    detected_map = generate_annotation(np_img, max_faces, min_confidence)
    detected_map_pil = np_to_pil(detected_map)
    return detected_map_pil

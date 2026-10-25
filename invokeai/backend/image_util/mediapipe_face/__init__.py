# Adapted from https://github.com/huggingface/controlnet_aux

from PIL import Image

from invokeai.backend.image_util.util import np_to_pil, pil_to_np
from invokeai.backend.util.host_platform import is_windows_arm64

# mediapipe is imported lazily: the package is absent on Windows ARM64, where its only wheel is the
# Tasks-API build without the legacy `solutions` API these modules use, and every node module is
# imported at startup.
MEDIAPIPE_UNAVAILABLE_MESSAGE = (
    "MediaPipe is not available on this platform (its legacy `solutions` API has no Windows ARM64 build), "
    "so the MediaPipe face detection preprocessor and the Face Tools nodes cannot run here."
)


def mediapipe_import_error(error: ImportError) -> RuntimeError:
    """The error to raise when mediapipe cannot be imported.

    Names the platform only where the platform is the cause; elsewhere an import failure is a broken
    install (a missing native library, say) and the message must carry the original error.
    """
    if is_windows_arm64():
        return RuntimeError(MEDIAPIPE_UNAVAILABLE_MESSAGE)
    return RuntimeError(f"MediaPipe failed to import, so the MediaPipe face nodes cannot run: {error}")


def detect_faces(image: Image.Image, max_faces: int = 1, min_confidence: float = 0.5) -> Image.Image:
    """Detects faces in an image using MediaPipe."""

    try:
        from invokeai.backend.image_util.mediapipe_face.mediapipe_face_common import generate_annotation
    except ImportError as e:
        raise mediapipe_import_error(e) from e

    np_img = pil_to_np(image)
    detected_map = generate_annotation(np_img, max_faces, min_confidence)
    detected_map_pil = np_to_pil(detected_map)
    return detected_map_pil

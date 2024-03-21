import cv2
from PIL import Image

from invokeai.backend.image_util.util import (
    cv2_to_pil,
    fit_image_to_resolution,
    normalize_image_channel_count,
    pil_to_cv2,
)


def get_canny_edges(
    image: Image.Image, low_threshold: int, high_threshold: int, detect_resolution: int, image_resolution: int
) -> Image.Image:
    """Returns the edges of an image using the Canny edge detection algorithm.

    Adapted from https://github.com/huggingface/controlnet_aux (Apache-2.0 license).

    Args:
        image: The input image.
        low_threshold: The lower threshold for the hysteresis procedure.
        high_threshold: The upper threshold for the hysteresis procedure.
        input_resolution: The resolution of the input image. The image will be resized to this resolution before edge detection.
        output_resolution: The resolution of the output image. The edges will be resized to this resolution before returning.

    Returns:
        The Canny edges of the input image.
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    np_image = pil_to_cv2(image)
    np_image = normalize_image_channel_count(np_image)
    np_image = fit_image_to_resolution(np_image, detect_resolution)

    edge_map = cv2.Canny(np_image, low_threshold, high_threshold)
    edge_map = normalize_image_channel_count(edge_map)
    edge_map = fit_image_to_resolution(edge_map, image_resolution)

    return cv2_to_pil(edge_map)

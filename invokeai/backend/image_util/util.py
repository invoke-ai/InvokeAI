from math import ceil, floor, sqrt
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class InitImageResizer:
    """Simple class to create resized copies of an Image while preserving the aspect ratio."""

    def __init__(self, Image):
        self.image = Image

    def resize(self, width=None, height=None) -> Image.Image:
        """
        Return a copy of the image resized to fit within
        a box width x height. The aspect ratio is
        maintained. If neither width nor height are provided,
        then returns a copy of the original image. If one or the other is
        provided, then the other will be calculated from the
        aspect ratio.

        Everything is floored to the nearest multiple of 64 so
        that it can be passed to img2img()
        """
        im = self.image

        ar = im.width / float(im.height)

        # Infer missing values from aspect ratio
        if not (width or height):  # both missing
            width = im.width
            height = im.height
        elif not height:  # height missing
            height = int(width / ar)
        elif not width:  # width missing
            width = int(height * ar)

        w_scale = width / im.width
        h_scale = height / im.height
        scale = min(w_scale, h_scale)
        (rw, rh) = (int(scale * im.width), int(scale * im.height))

        # round everything to multiples of 64
        width, height, rw, rh = (x - x % 64 for x in (width, height, rw, rh))

        # no resize necessary, but return a copy
        if im.width == width and im.height == height:
            return im.copy()

        # otherwise resize the original image so that it fits inside the bounding box
        resized_image = self.image.resize((rw, rh), resample=Image.Resampling.LANCZOS)
        return resized_image


def make_grid(image_list, rows=None, cols=None):
    image_cnt = len(image_list)
    if None in (rows, cols):
        rows = floor(sqrt(image_cnt))  # try to make it square
        cols = ceil(image_cnt / rows)
    width = image_list[0].width
    height = image_list[0].height

    grid_img = Image.new("RGB", (width * cols, height * rows))
    i = 0
    for r in range(0, rows):
        for c in range(0, cols):
            if i >= len(image_list):
                break
            grid_img.paste(image_list[i], (c * width, r * height))
            i = i + 1

    return grid_img


def pil_to_np(image: Image.Image) -> np.ndarray:
    """Converts a PIL image to a numpy array."""
    return np.array(image, dtype=np.uint8)


def np_to_pil(image: np.ndarray) -> Image.Image:
    """Converts a numpy array to a PIL image."""
    return Image.fromarray(image)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Converts a PIL image to a CV2 image."""
    return cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Converts a CV2 image to a PIL image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def normalize_image_channel_count(image: np.ndarray) -> np.ndarray:
    """Normalizes an image to have 3 channels.

    If the image has 1 channel, it will be duplicated 3 times.
    If the image has 1 channel, a third empty channel will be added.
    If the image has 4 channels, the alpha channel will be used to blend the image with a white background.

    Adapted from https://github.com/huggingface/controlnet_aux (Apache-2.0 license).

    Args:
        image: The input image.

    Returns:
        The normalized image.
    """
    assert image.dtype == np.uint8
    if image.ndim == 2:
        image = image[:, :, None]
    assert image.ndim == 3
    _height, _width, channels = image.shape
    assert channels == 1 or channels == 3 or channels == 4
    if channels == 3:
        return image
    if channels == 1:
        return np.concatenate([image, image, image], axis=2)
    if channels == 4:
        color = image[:, :, 0:3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        normalized = color * alpha + 255.0 * (1.0 - alpha)
        normalized = normalized.clip(0, 255).astype(np.uint8)
        return normalized

    raise ValueError("Invalid number of channels.")


def resize_image_to_resolution(input_image: np.ndarray, resolution: int) -> np.ndarray:
    """Resizes an image, fitting it to the given resolution.

    Adapted from https://github.com/huggingface/controlnet_aux (Apache-2.0 license).

    Args:
        input_image: The input image.
        resolution: The resolution to fit the image to.

    Returns:
        The resized image.
    """
    h = float(input_image.shape[0])
    w = float(input_image.shape[1])
    scaling_factor = float(resolution) / min(h, w)
    h *= scaling_factor
    w *= scaling_factor
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64
    if scaling_factor > 1:
        return cv2.resize(input_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(input_image, (w, h), interpolation=cv2.INTER_AREA)


def nms(np_img: np.ndarray, threshold: Optional[int] = None, sigma: Optional[float] = None) -> np.ndarray:
    """
    Apply non-maximum suppression to an image.

    If both threshold and sigma are provided, the image will blurred before the suppression and thresholded afterwards,
    resulting in a binary output image.

    This function is adapted from https://github.com/lllyasviel/ControlNet.

    Args:
        image: The input image.
        threshold: The threshold value for the suppression. Pixels with values greater than this will be set to 255.
        sigma: The standard deviation for the Gaussian blur applied to the image.

    Returns:
        The image after non-maximum suppression.

    Raises:
        ValueError: If only one of threshold and sigma provided.
    """

    # Raise a value error if only one of threshold and sigma is provided
    if (threshold is None) != (sigma is None):
        raise ValueError("Both threshold and sigma must be provided if one is provided.")

    if sigma is not None and threshold is not None:
        # Blurring the image can help to thin out features
        np_img = cv2.GaussianBlur(np_img.astype(np.float32), (0, 0), sigma)

    filter_1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    filter_2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    filter_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    filter_4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    nms_img = np.zeros_like(np_img)

    for f in [filter_1, filter_2, filter_3, filter_4]:
        np.putmask(nms_img, cv2.dilate(np_img, kernel=f) == np_img, np_img)

    if sigma is not None and threshold is not None:
        # We blurred - now threshold to get a binary image
        thresholded = np.zeros_like(nms_img, dtype=np.uint8)
        thresholded[nms_img > threshold] = 255
        return thresholded

    return nms_img


def safe_step(x: np.ndarray, step: int = 2) -> np.ndarray:
    """Apply the safe step operation to an array.

    I don't fully understand the purpose of this function, but it appears to be normalizing/quantizing the array.

    Adapted from https://github.com/huggingface/controlnet_aux (Apache-2.0 license).

    Args:
        x: The input array.
        step: The step value.

    Returns:
        The array after the safe step operation.
    """
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y

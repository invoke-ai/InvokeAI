from enum import Enum

import cv2
import numpy as np
import numpy.typing as npt


class ImageChannel(Enum):
    RGB_R = "RGB_R"
    RGB_G = "RGB_G"
    RGB_B = "RGB_B"

    LAB_L = "LAB_L"
    LAB_A = "LAB_A"
    LAB_B = "LAB_B"

    HSV_H = "HSV_H"
    HSV_S = "HSV_S"
    HSV_V = "HSV_V"


def extract_channel(image: npt.NDArray[np.uint8], channel: ImageChannel) -> npt.NDArray[np.uint8]:
    """Extract a channel from an image.

    Args:
        image (np.ndarray): Shape (H, W, 3) of dtype uint8.
        channel (ImageChannel): The channel to extract.

    Returns:
        np.ndarray: Shape (H, W) of dtype uint8.
    """
    if channel == ImageChannel.RGB_R:
        return image[:, :, 0]
    elif channel == ImageChannel.RGB_G:
        return image[:, :, 1]
    elif channel == ImageChannel.RGB_B:
        return image[:, :, 2]
    elif channel == ImageChannel.LAB_L:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return lab[:, :, 0]
    elif channel == ImageChannel.LAB_A:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return lab[:, :, 1]
    elif channel == ImageChannel.LAB_B:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return lab[:, :, 2]
    elif channel == ImageChannel.HSV_H:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 0]
    elif channel == ImageChannel.HSV_S:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 1]
    elif channel == ImageChannel.HSV_V:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 2]
    else:
        raise ValueError(f"Unknown channel: {channel}")

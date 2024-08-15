# This file contains utilities for Grounded-SAM mask refinement based on:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/a39f33ac1557b02ebfb191ea7753e332b5ca933f/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb


import cv2
import numpy as np
import numpy.typing as npt


def mask_to_polygon(mask: npt.NDArray[np.uint8]) -> list[tuple[int, int]]:
    """Convert a binary mask to a polygon.

    Returns:
        list[list[int]]: List of (x, y) coordinates representing the vertices of the polygon.
    """
    # Find contours in the binary mask.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area.
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour.
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: list[tuple[int, int]], image_shape: tuple[int, int], fill_value: int = 1
) -> npt.NDArray[np.uint8]:
    """Convert a polygon to a segmentation mask.

    Args:
        polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        image_shape (tuple): Shape of the image (height, width) for the mask.
        fill_value (int): Value to fill the polygon with.

    Returns:
        np.ndarray: Segmentation mask with the polygon filled (with value 255).
    """
    # Create an empty mask.
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points.
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255).
    cv2.fillPoly(mask, [pts], color=(fill_value,))

    return mask

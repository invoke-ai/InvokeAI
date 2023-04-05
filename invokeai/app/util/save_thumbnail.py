import os
from PIL import Image


def save_thumbnail(
    image: Image.Image,
    filename: str,
    path: str,
    size: int = 256,
) -> str:
    """
    Saves a thumbnail of an image, returning its path.
    """
    base_filename = os.path.splitext(filename)[0]
    thumbnail_path = os.path.join(path, base_filename + ".webp")

    if os.path.exists(thumbnail_path):
        return thumbnail_path

    image_copy = image.copy()
    image_copy.thumbnail(size=(size, size))

    image_copy.save(thumbnail_path, "WEBP")

    return thumbnail_path

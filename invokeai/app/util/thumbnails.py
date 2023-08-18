import os

from PIL import Image


def get_thumbnail_name(image_name: str) -> str:
    """Formats given an image name, returns the appropriate thumbnail image name"""
    thumbnail_name = os.path.splitext(image_name)[0] + ".webp"
    return thumbnail_name


def make_thumbnail(image: Image.Image, size: int = 256) -> Image.Image:
    """Makes a thumbnail from a PIL Image"""
    thumbnail = image.copy()
    thumbnail.thumbnail(size=(size, size))
    return thumbnail

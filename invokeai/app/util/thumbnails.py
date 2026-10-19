import os

from PIL import Image


def get_thumbnail_name(image_name: str) -> str:
    """Formats given an image name, returns the appropriate thumbnail image name"""
    thumbnail_name = os.path.splitext(image_name)[0] + ".webp"
    return thumbnail_name


def make_thumbnail(image: Image.Image, size: int = 256) -> Image.Image:
    """Makes a thumbnail from a PIL Image"""
    # Pillow cannot resize some source modes (notably large ``I;16`` images). Convert to
    # a mode supported by the WEBP thumbnail writer while preserving transparency.
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    thumbnail = image.convert("RGBA" if has_alpha else "RGB")
    thumbnail.thumbnail(size=(size, size))
    return thumbnail

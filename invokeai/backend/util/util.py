import base64
import io
import os
from pathlib import Path

from PIL import Image

# actual size of a gig
GIG = 1073741824


def directory_size(directory: Path) -> int:
    """
    Return the aggregate size of all files in a directory (bytes).
    """
    sum = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            sum += Path(root, f).stat().st_size
        for d in dirs:
            sum += Path(root, d).stat().st_size
    return sum


def image_to_dataURL(image: Image.Image, image_format: str = "PNG") -> str:
    """
    Converts an image into a base64 image dataURL.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    mime_type = Image.MIME.get(image_format.upper(), "image/" + image_format.lower())
    image_base64 = f"data:{mime_type};base64," + base64.b64encode(buffered.getvalue()).decode("UTF-8")
    return image_base64


class Chdir(object):
    """Context manager to chdir to desired directory and change back after context exits:
    Args:
        path (Path): The path to the cwd
    """

    def __init__(self, path: Path):
        self.path = path
        self.original = Path().absolute()

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.original)

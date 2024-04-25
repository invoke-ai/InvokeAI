import base64
import io
import os
import re
import unicodedata
import warnings
from pathlib import Path

from diffusers import logging as diffusers_logging
from PIL import Image
from transformers import logging as transformers_logging

# actual size of a gig
GIG = 1073741824


def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    Adapted from Django: https://github.com/django/django/blob/main/django/utils/text.py
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[/]", "_", value.lower())
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


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


class SilenceWarnings(object):
    """Context manager to temporarily lower verbosity of diffusers & transformers warning messages."""

    def __enter__(self):
        """Set verbosity to error."""
        self.transformers_verbosity = transformers_logging.get_verbosity()
        self.diffusers_verbosity = diffusers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        diffusers_logging.set_verbosity_error()
        warnings.simplefilter("ignore")

    def __exit__(self, type, value, traceback):
        """Restore logger verbosity to state before context was entered."""
        transformers_logging.set_verbosity(self.transformers_verbosity)
        diffusers_logging.set_verbosity(self.diffusers_verbosity)
        warnings.simplefilter("default")

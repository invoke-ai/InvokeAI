from __future__ import annotations

import base64
import io

from PIL import Image
from PIL.Image import Image as PILImageType


def encode_image_base64(image: PILImageType, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image_base64(encoded: str) -> PILImageType:
    data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(data))
    return image.convert("RGB")

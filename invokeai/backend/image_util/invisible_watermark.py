"""
This module defines a singleton object, "invisible_watermark" that
wraps the invisible watermark model. It respects the global "invisible_watermark"
configuration variable, that allows the watermarking to be supressed.
"""

import cv2
import numpy as np
from PIL import Image

import invokeai.backend.util.logging as logger
from invokeai.backend.image_util.imwatermark.vendor import WatermarkDecoder, WatermarkEncoder


class InvisibleWatermark:
    """
    Wrapper around InvisibleWatermark module.
    """

    @classmethod
    def add_watermark(cls, image: Image.Image, watermark_text: str) -> Image.Image:
        logger.debug(f'Applying invisible watermark "{watermark_text}"')
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        encoder = WatermarkEncoder()
        encoder.set_watermark("bytes", watermark_text.encode("utf-8"))
        bgr_encoded = encoder.encode(bgr, "dwtDct")
        return Image.fromarray(cv2.cvtColor(bgr_encoded, cv2.COLOR_BGR2RGB)).convert("RGBA")

    @classmethod
    def decode_watermark(cls, image: Image.Image, watermark_length: int = 8) -> str:
        """Decode an invisible watermark from an image.

        Args:
            image: The PIL image to decode the watermark from.
            watermark_length: The length of the watermark in bytes (default 8, matching the default "InvokeAI" watermark text).

        Returns:
            The decoded watermark text, or an empty string if decoding fails.
        """
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        decoder = WatermarkDecoder("bytes", watermark_length * 8)
        try:
            raw = decoder.decode(bgr, "dwtDct")
            return raw.rstrip(b"\x00").decode("utf-8", errors="replace")
        except (RuntimeError, ValueError, NameError) as e:
            logger.debug("Failed to decode watermark: %s", e)
            return ""

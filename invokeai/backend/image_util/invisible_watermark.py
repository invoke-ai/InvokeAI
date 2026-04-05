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
    def decode_watermark(cls, image: Image.Image, length: int = 8) -> str:
        """Attempt to decode an invisible watermark from an image.

        Args:
            image: The PIL Image to decode the watermark from.
            length: The expected watermark length in bytes. Must match the length used when encoding.
                The WatermarkDecoder requires the length in bits; this value is multiplied by 8 internally.

        Returns:
            The decoded watermark text, or an empty string if no watermark is detected or decoding fails.
        """
        logger.debug("Attempting to decode invisible watermark")
        try:
            bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            decoder = WatermarkDecoder("bytes", length * 8)
            watermark_bytes = decoder.decode(bgr, "dwtDct")
            return watermark_bytes.decode("utf-8", errors="ignore").rstrip("\x00")
        except Exception:
            logger.debug("Failed to decode invisible watermark")
            return ""

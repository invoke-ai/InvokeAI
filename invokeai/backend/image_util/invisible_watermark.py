"""
This module defines a singleton object, "invisible_watermark" that
wraps the invisible watermark model. It respects the global "invisible_watermark"
configuration variable, that allows the watermarking to be supressed.
"""

import cv2
import numpy as np
from imwatermark import WatermarkEncoder
from PIL import Image

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config

config = get_config()


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

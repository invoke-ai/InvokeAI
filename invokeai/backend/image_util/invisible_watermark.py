"""
This module defines a singleton object, "invisible_watermark" that
wraps the invisible watermark model. It respects the global "invisible_watermark"
configuration variable, that allows the watermarking to be supressed.
"""
import numpy as np
import cv2
from PIL import Image
from imwatermark import WatermarkEncoder
from invokeai.app.services.config import InvokeAIAppConfig
import invokeai.backend.util.logging as logger

config = InvokeAIAppConfig.get_config()


class InvisibleWatermark:
    """
    Wrapper around InvisibleWatermark module.
    """

    @classmethod
    def invisible_watermark_available(self) -> bool:
        return config.invisible_watermark

    @classmethod
    def add_watermark(self, image: Image, watermark_text: str) -> Image:
        if not self.invisible_watermark_available():
            return image
        logger.debug(f'Applying invisible watermark "{watermark_text}"')
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        encoder = WatermarkEncoder()
        encoder.set_watermark("bytes", watermark_text.encode("utf-8"))
        bgr_encoded = encoder.encode(bgr, "dwtDct")
        return Image.fromarray(cv2.cvtColor(bgr_encoded, cv2.COLOR_BGR2RGB)).convert("RGBA")

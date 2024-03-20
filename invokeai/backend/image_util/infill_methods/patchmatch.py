"""
This module defines a singleton object, "patchmatch" that
wraps the actual patchmatch object. It respects the global
"try_patchmatch" attribute, so that patchmatch loading can
be suppressed or deferred
"""

import numpy as np
from PIL import Image

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config


class PatchMatch:
    """
    Thin class wrapper around the patchmatch function.
    """

    patch_match = None
    tried_load: bool = False

    def __init__(self):
        super().__init__()

    @classmethod
    def _load_patch_match(cls):
        if cls.tried_load:
            return
        if get_config().patchmatch:
            from patchmatch import patch_match as pm

            if pm.patchmatch_available:
                logger.info("Patchmatch initialized")
                cls.patch_match = pm
            else:
                logger.info("Patchmatch not loaded (nonfatal)")
        else:
            logger.info("Patchmatch loading disabled")
        cls.tried_load = True

    @classmethod
    def patchmatch_available(cls) -> bool:
        cls._load_patch_match()
        if not cls.patch_match:
            return False
        return cls.patch_match.patchmatch_available

    @classmethod
    def inpaint(cls, image: Image.Image) -> Image.Image:
        if cls.patch_match is None or not cls.patchmatch_available():
            return image

        np_image = np.array(image)
        mask = 255 - np_image[:, :, 3]
        infilled = cls.patch_match.inpaint(np_image[:, :, :3], mask, patch_size=3)
        return Image.fromarray(infilled, mode="RGB")


def infill_patchmatch(image: Image.Image) -> Image.Image:
    IS_PATCHMATCH_AVAILABLE = PatchMatch.patchmatch_available()

    if not IS_PATCHMATCH_AVAILABLE:
        logger.warning("PatchMatch is not available on this system")
        return image

    return PatchMatch.inpaint(image)

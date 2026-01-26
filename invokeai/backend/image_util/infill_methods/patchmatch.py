"""
PatchMatch integration using patchmatch-cython.

Attempt to load the CythonSolver once; if it's missing we report PatchMatch as unavailable.
The CythonSolver ships wheels for CPython 3.10–3.13 on macOS x86_64 (10.9+), macOS arm64 (11+), manylinux/musllinux x86_64, and Windows (win32 & win_amd64).
The PythonSolver is not considered a viable fallback because it is ~20x slower.
"""

import numpy as np
from PIL import Image

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config


class PatchMatch:
    """Thin wrapper with explicit state; keeps legacy imports working."""

    _PATCH_SIZE = 3
    _tried = False
    _inpaint_fn = None

    @classmethod
    def _load_patch_match(cls) -> None:
        if cls._tried:
            return
        cls._tried = True

        if not get_config().patchmatch:
            logger.info("PatchMatch disabled via config")
            return

        try:
            from patchmatch_cython import CythonSolver, inpaint_pyramid

            def _inpaint(image, mask):
                return inpaint_pyramid(image, mask, solver_class=CythonSolver, patch_size=cls._PATCH_SIZE)

            cls._inpaint_fn = _inpaint
            logger.info("PatchMatch loaded")
        except Exception as exc:
            logger.warning("PatchMatch unavailable: %s", exc)

    @classmethod
    def patchmatch_available(cls) -> bool:
        cls._load_patch_match()
        return cls._inpaint_fn is not None

    @classmethod
    def inpaint(cls, image: Image.Image) -> Image.Image:
        cls._load_patch_match()
        if cls._inpaint_fn is None:
            logger.warning("PatchMatch is unavailable; returning original image")
            return image

        np_image = np.array(image)
        if np_image.shape[2] < 4:
            logger.warning("PatchMatch requires an RGBA image; received %s channels", np_image.shape[2])
            return image

        mask = 255 - np_image[:, :, 3]
        infilled = cls._inpaint_fn(np_image[:, :, :3], mask)
        return Image.fromarray(infilled, mode="RGB")


def infill_patchmatch(image: Image.Image) -> Image.Image:
    return PatchMatch.inpaint(image)

"""
This module defines a singleton object, "safety_checker" that
wraps the safety_checker model. It respects the global "nsfw_checker"
configuration variable, that allows the checker to be supressed.
"""

import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from transformers import AutoFeatureExtractor

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.silence_warnings import SilenceWarnings

repo_id = "CompVis/stable-diffusion-safety-checker"
CHECKER_PATH = "core/convert/stable-diffusion-safety-checker"


class NSFWImageException(Exception):
    """Raised when a NSFW image is detected."""

    def __init__(self):
        super().__init__("A potentially NSFW image has been detected.")


class SafetyChecker:
    """Wrapper around SafetyChecker model."""

    feature_extractor = None
    safety_checker = None

    @classmethod
    def _load_safety_checker(cls):
        if cls.safety_checker is not None and cls.feature_extractor is not None:
            return

        try:
            model_path = get_config().models_path / CHECKER_PATH
            if model_path.exists():
                cls.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
                cls.safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path)
            else:
                model_path.mkdir(parents=True, exist_ok=True)
                cls.feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
                cls.feature_extractor.save_pretrained(model_path, safe_serialization=True)
                cls.safety_checker = StableDiffusionSafetyChecker.from_pretrained(repo_id)
                cls.safety_checker.save_pretrained(model_path, safe_serialization=True)
        except Exception as e:
            logger.warning(f"Could not load NSFW checker: {str(e)}")

    @classmethod
    def has_nsfw_concept(cls, image: Image.Image) -> bool:
        cls._load_safety_checker()
        if cls.safety_checker is None or cls.feature_extractor is None:
            return False
        device = TorchDevice.choose_torch_device()
        features = cls.feature_extractor([image], return_tensors="pt")
        features.to(device)
        cls.safety_checker.to(device)

        # Only RGB(A) images are supported, so to prevent an error when a NSFW concept is detect in an image with
        # a different image mode, we _must_ convert it to RGB and then to a normalized, batched np array.
        rgb_image = image.convert("RGB")
        # Convert to normalized (0-1) np array
        x_image = np.array(rgb_image).astype(np.float32) / 255.0
        # Add batch dimension and transpose to NCHW
        x_image = x_image[None].transpose(0, 3, 1, 2)
        # A warning is logged if a NSFW concept is detected - silence those, so we can handle it ourselves.
        with SilenceWarnings():
            # `clip_input` (features) is used to check for NSFW concepts. `images` is required, but it isn't actually
            # checked for NSFW concepts. If a NSFW concept is detected, the the image is replaced with a black image.
            checked_image, has_nsfw_concept = cls.safety_checker(images=x_image, clip_input=features.pixel_values)
        return has_nsfw_concept[0]

    @classmethod
    def raise_if_nsfw(cls, image: Image.Image) -> Image.Image:
        """Raises an exception if the image contains NSFW content."""
        if cls.has_nsfw_concept(image):
            raise NSFWImageException()

        return image

"""
This module defines a singleton object, "safety_checker" that
wraps the safety_checker model. It respects the global "nsfw_checker"
configuration variable, that allows the checker to be supressed.
"""

from pathlib import Path

import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image, ImageFilter
from transformers import AutoFeatureExtractor

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.silence_warnings import SilenceWarnings

repo_id = "CompVis/stable-diffusion-safety-checker"
CHECKER_PATH = "core/convert/stable-diffusion-safety-checker"


class SafetyChecker:
    """
    Wrapper around SafetyChecker model.
    """

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
    def blur_if_nsfw(cls, image: Image.Image) -> Image.Image:
        if cls.has_nsfw_concept(image):
            logger.warning("A potentially NSFW image has been detected. Image will be blurred.")
            blurry_image = image.filter(filter=ImageFilter.GaussianBlur(radius=32))
            caution = cls._get_caution_img()
            # Center the caution image on the blurred image
            x = (blurry_image.width - caution.width) // 2
            y = (blurry_image.height - caution.height) // 2
            blurry_image.paste(caution, (x, y), caution)
            image = blurry_image

        return image

    @classmethod
    def _get_caution_img(cls) -> Image.Image:
        import invokeai.app.assets.images as image_assets

        caution = Image.open(Path(image_assets.__path__[0]) / "caution.png")
        return caution.resize((caution.width // 2, caution.height // 2))

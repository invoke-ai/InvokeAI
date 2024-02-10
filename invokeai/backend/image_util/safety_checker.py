"""
This module defines a singleton object, "safety_checker" that
wraps the safety_checker model. It respects the global "nsfw_checker"
configuration variable, that allows the checker to be supressed.
"""
import numpy as np
from PIL import Image

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.devices import choose_torch_device
from invokeai.backend.util.silence_warnings import SilenceWarnings

config = InvokeAIAppConfig.get_config()

CHECKER_PATH = "core/convert/stable-diffusion-safety-checker"


class SafetyChecker:
    """
    Wrapper around SafetyChecker model.
    """

    safety_checker = None
    feature_extractor = None
    tried_load: bool = False

    @classmethod
    def _load_safety_checker(cls):
        if cls.tried_load:
            return

        if config.nsfw_checker:
            try:
                from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
                from transformers import AutoFeatureExtractor

                cls.safety_checker = StableDiffusionSafetyChecker.from_pretrained(config.models_path / CHECKER_PATH)
                cls.feature_extractor = AutoFeatureExtractor.from_pretrained(config.models_path / CHECKER_PATH)
                logger.info("NSFW checker initialized")
            except Exception as e:
                logger.warning(f"Could not load NSFW checker: {str(e)}")
        else:
            logger.info("NSFW checker loading disabled")
        cls.tried_load = True

    @classmethod
    def safety_checker_available(cls) -> bool:
        cls._load_safety_checker()
        return cls.safety_checker is not None

    @classmethod
    def has_nsfw_concept(cls, image: Image.Image) -> bool:
        if not cls.safety_checker_available():
            return False

        device = choose_torch_device()
        features = cls.feature_extractor([image], return_tensors="pt")
        features.to(device)
        cls.safety_checker.to(device)
        x_image = np.array(image).astype(np.float32) / 255.0
        x_image = x_image[None].transpose(0, 3, 1, 2)
        with SilenceWarnings():
            checked_image, has_nsfw_concept = cls.safety_checker(images=x_image, clip_input=features.pixel_values)
        return has_nsfw_concept[0]

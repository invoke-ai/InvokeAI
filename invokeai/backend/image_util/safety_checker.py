"""
This module defines a singleton object, "safety_checker" that
wraps the safety_checker model. It respects the global "nsfw_checker"
configuration variable, that allows the checker to be supressed.
"""

from pathlib import Path

import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from transformers import AutoFeatureExtractor

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.silence_warnings import SilenceWarnings

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

        try:
            cls.safety_checker = StableDiffusionSafetyChecker.from_pretrained(get_config().models_path / CHECKER_PATH)
            cls.feature_extractor = AutoFeatureExtractor.from_pretrained(get_config().models_path / CHECKER_PATH)
        except Exception as e:
            logger.warning(f"Could not load NSFW checker: {str(e)}")
        cls.tried_load = True

    @classmethod
    def safety_checker_available(cls) -> bool:
        return Path(get_config().models_path, CHECKER_PATH).exists()

    @classmethod
    def has_nsfw_concept(cls, image: Image.Image) -> bool:
        if not cls.safety_checker_available() and cls.tried_load:
            return False
        cls._load_safety_checker()
        if cls.safety_checker is None or cls.feature_extractor is None:
            return False
        device = TorchDevice.choose_torch_device()
        features = cls.feature_extractor([image], return_tensors="pt")
        features.to(device)
        cls.safety_checker.to(device)
        x_image = np.array(image).astype(np.float32) / 255.0
        x_image = x_image[None].transpose(0, 3, 1, 2)
        with SilenceWarnings():
            checked_image, has_nsfw_concept = cls.safety_checker(images=x_image, clip_input=features.pixel_values)
        return has_nsfw_concept[0]

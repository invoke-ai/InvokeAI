'''
SafetyChecker class - checks images against the StabilityAI NSFW filter
and blurs images that contain potential NSFW content.
'''
import diffusers
import numpy as np
import torch
import traceback
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from pathlib import Path
from PIL import Image, ImageFilter
from transformers import AutoFeatureExtractor

import invokeai.assets.web as web_assets
import invokeai.backend.util.logging as logger
from invokeai.app.services.config import get_invokeai_config
from .util import CPU_DEVICE

class SafetyChecker(object):
    CAUTION_IMG = "caution.png"
    
    def __init__(self, device: torch.device):
        path = Path(web_assets.__path__[0]) / self.CAUTION_IMG
        caution = Image.open(path)
        self.caution_img = caution.resize((caution.width // 2, caution.height // 2))
        self.device = device
        config = get_invokeai_config()

        try:
            safety_model_id = "CompVis/stable-diffusion-safety-checker"
            safety_model_path = config.cache_dir
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                safety_model_id,
                local_files_only=True,
                cache_dir=safety_model_path,
            )
            self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
                safety_model_id,
                local_files_only=True,
                cache_dir=safety_model_path,
            )
        except Exception:
            logger.error(
                "An error was encountered while installing the safety checker:"
            )
            print(traceback.format_exc())

    def check(self, image: Image.Image):
        """
        Check provided image against the StabilityAI safety checker and return

        """

        self.safety_checker.to(self.device)
        features = self.safety_feature_extractor([image], return_tensors="pt")
        features.to(self.device)
        
        # unfortunately checker requires the numpy version, so we have to convert back
        x_image = np.array(image).astype(np.float32) / 255.0
        x_image = x_image[None].transpose(0, 3, 1, 2)

        diffusers.logging.set_verbosity_error()
        checked_image, has_nsfw_concept = self.safety_checker(
            images=x_image, clip_input=features.pixel_values
        )
        self.safety_checker.to(CPU_DEVICE) # offload
        if has_nsfw_concept[0]:
            logger.warning(
                "An image with potential non-safe content has been detected. A blurred image will be returned."
            )
            return self.blur(image)
        else:
            return image

    def blur(self, input):
        blurry = input.filter(filter=ImageFilter.GaussianBlur(radius=32))
        try:
            if caution := self.caution_img:
                blurry.paste(caution, (0, 0), caution)
        except FileNotFoundError:
            pass
        return blurry

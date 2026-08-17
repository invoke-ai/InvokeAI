"""
This module defines a singleton object, "safety_checker" that
wraps the safety_checker model. It respects the global "nsfw_checker"
configuration variable, that allows the checker to be supressed.
"""

import os
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import snapshot_download
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.silence_warnings import SilenceWarnings

repo_id = "CompVis/stable-diffusion-safety-checker"
CHECKER_PATH = "core/convert/stable-diffusion-safety-checker"

# Serializes the lazy init below. Lock ordering: this is acquired BEFORE
# MODEL_LOAD_LOCK, never after (see _ModelLoadReadWriteLock), matching the
# service-lock -> MODEL_LOAD_LOCK order used by ImageIndexService.
_INIT_LOCK = threading.Lock()


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

        # Serialize the whole lazy init. Without this, two session workers both fall
        # through the check above and race in three ways: they duplicate the download
        # and the construction, they run save_pretrained() into the same directory
        # concurrently, and — because the first thread's mkdir + feature-extractor save
        # makes model_path exist before the weights land — the second can take the
        # "already downloaded" branch below, fail on the missing weights, and leave
        # safety_checker as None, which has_nsfw_concept() reports as "not NSFW".
        with _INIT_LOCK:
            # Re-check now that we hold the lock: another worker may have finished
            # while we waited. (Same double-checked pattern as ModelLoader._load_and_cache
            # and ModelLoadService.load_remote_model.)
            if cls.safety_checker is not None and cls.feature_extractor is not None:
                return

            try:
                model_path = get_config().models_path / CHECKER_PATH
                if model_path.exists():
                    cls.feature_extractor = AutoImageProcessor.from_pretrained(model_path)
                    # Torch module construction is not thread-safe process-wide (see
                    # _ModelLoadReadWriteLock); serialize with the model-load machinery.
                    with MODEL_LOAD_LOCK.write_lock():
                        cls.safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path)
                else:
                    # Download before constructing: from_pretrained(repo_id) would
                    # otherwise hold the process-wide load lock across a multi-GB
                    # network transfer, stalling every other model load.
                    download_path = snapshot_download(repo_id)
                    feature_extractor = AutoImageProcessor.from_pretrained(download_path)
                    # Torch module construction is not thread-safe process-wide (see
                    # _ModelLoadReadWriteLock); serialize with the model-load machinery.
                    with MODEL_LOAD_LOCK.write_lock():
                        safety_checker = StableDiffusionSafetyChecker.from_pretrained(download_path)
                    # Publish the in-memory objects before persisting them: failing to
                    # cache the checker on disk must not leave this run without one.
                    cls.feature_extractor = feature_extractor
                    cls.safety_checker = safety_checker
                    try:
                        cls._persist_checker(feature_extractor, safety_checker, model_path)
                    except Exception as e:
                        logger.warning(f"Could not cache NSFW checker to {model_path}: {str(e)}")
            except Exception as e:
                logger.warning(f"Could not load NSFW checker: {str(e)}")

    @classmethod
    def _persist_checker(cls, feature_extractor, safety_checker, model_path: Path) -> None:
        """Write the checker into `model_path` atomically.

        `save_pretrained()` creates its own destination, so writing the two objects
        straight into `model_path` would make that directory exist as soon as the FIRST
        save lands. Any later failure — a raising construction, a full disk, a killed
        process — would then leave a weightless directory that the `model_path.exists()`
        branch of `_load_safety_checker` trusts forever: the image processor loads, the
        checker doesn't, the exception is swallowed, and `has_nsfw_concept()` reports a
        missing checker as "not NSFW", so every image ships unblurred and the download is
        never retried. Staging into a sibling directory and renaming makes the install
        either complete or absent, never half-present.
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=model_path.parent, prefix=f".{model_path.name}.tmp-"))
        try:
            feature_extractor.save_pretrained(staging)
            safety_checker.save_pretrained(staging)
            # Atomic on a single filesystem, and staging is a sibling of the target.
            os.replace(staging, model_path)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @classmethod
    def has_nsfw_concept(cls, image: Image.Image) -> bool:
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

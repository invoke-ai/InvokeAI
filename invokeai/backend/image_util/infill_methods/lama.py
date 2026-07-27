import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import invokeai.backend.util.logging as logger
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.model_manager.taxonomy import AnyModel

# Emit the TorchScript-deprecation breadcrumb only once per process, rather than on every infill.
_warned_jit_deprecation = False


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


class LaMA:
    def __init__(self, model: AnyModel):
        self._model = model

    def __call__(self, input_image: Image.Image, *args: Any, **kwds: Any) -> Any:
        image = np.asarray(input_image.convert("RGB"))
        image = norm_img(image)

        mask = input_image.split()[-1]
        mask = np.asarray(mask)
        mask = np.invert(mask)
        mask = norm_img(mask)
        mask = (mask > 0) * 1

        device = get_effective_device(self._model)
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(device)

        with torch.inference_mode():
            infilled_image = self._model(image, mask)

        infilled_image = infilled_image[0].permute(1, 2, 0).detach().cpu().numpy()
        infilled_image = np.clip(infilled_image * 255, 0, 255).astype("uint8")
        infilled_image = Image.fromarray(infilled_image)

        return infilled_image

    @staticmethod
    def load_jit_model(url_or_path: str | Path, device: torch.device | str = "cpu") -> torch.nn.Module:
        model_path = url_or_path
        logger.info(f"Loading model from: {model_path}")
        # LaMa ships as a TorchScript archive, so torch.jit.load is the only way to load it (there is no
        # source model to re-export via torch.export, which is what torch's own warning suggests). It still
        # works on torch 2.12 under Python 3.14+ but emits a DeprecationWarning on every call. Suppress that
        # per-call spam, but surface a single breadcrumb once per process so the deprecation stays on our
        # radar. Note: when torch eventually *removes* jit.load, the call below will raise rather than warn,
        # so infill will fail loudly at that point regardless of this suppression.
        global _warned_jit_deprecation
        if not _warned_jit_deprecation:
            logger.warning(
                "LaMa infill loads a TorchScript model via torch.jit.load, which is deprecated in Python "
                "3.14+ and will eventually be removed from torch. Infill still works for now; this needs a "
                "migration (e.g. a re-exported model) before torch drops jit.load."
            )
            _warned_jit_deprecation = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.jit.load.*", category=DeprecationWarning)
            model: torch.nn.Module = torch.jit.load(model_path, map_location="cpu").to(device)  # type: ignore
        model.eval()
        return model

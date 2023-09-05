import gc
from typing import Any

import numpy as np
import torch
from PIL import Image

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import get_invokeai_config
from invokeai.backend.util.devices import choose_torch_device


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def load_jit_model(url_or_path, device):
    model_path = url_or_path
    logger.info(f"Loading model from: {model_path}")
    model = torch.jit.load(model_path, map_location="cpu").to(device)
    model.eval()
    return model


class LaMA:
    def __call__(self, input_image: Image.Image, *args: Any, **kwds: Any) -> Any:
        device = choose_torch_device()
        model_location = get_invokeai_config().models_path / "core/misc/lama/lama.pt"
        model = load_jit_model(model_location, device)

        image = np.asarray(input_image.convert("RGB"))
        image = norm_img(image)

        mask = input_image.split()[-1]
        mask = np.asarray(mask)
        mask = np.invert(mask)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(device)

        with torch.inference_mode():
            infilled_image = model(image, mask)

        infilled_image = infilled_image[0].permute(1, 2, 0).detach().cpu().numpy()
        infilled_image = np.clip(infilled_image * 255, 0, 255).astype("uint8")
        infilled_image = Image.fromarray(infilled_image)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return infilled_image

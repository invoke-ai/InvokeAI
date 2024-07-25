from pathlib import Path
from typing import Literal

import numpy as np
import torch
from einops import repeat
from PIL import Image

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.image_util.depth_anything.v2.dpt import DepthAnythingV2
from invokeai.backend.util.logging import InvokeAILogger

config = get_config()
logger = InvokeAILogger.get_logger(config=config)

DEPTH_ANYTHING_MODELS = {
    "large": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
    "base": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
    "small": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
}


class DepthAnythingDetector:
    def __init__(self, model: DepthAnythingV2, device: torch.device) -> None:
        self.model = model
        self.device = device

    @staticmethod
    def load_model(
        model_path: Path, device: torch.device, model_size: Literal["large", "base", "small", "giant"] = "small"
    ) -> DepthAnythingV2:
        match model_size:
            case "small":
                model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
            case "base":
                model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
            case "large":
                model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
            case "giant":
                model = DepthAnythingV2(encoder="vitg", features=384, out_channels=[1536, 1536, 1536, 1536])

        model.load_state_dict(torch.load(model_path.as_posix(), map_location="cpu"))
        model.eval()
        model.to(device)
        return model

    def __call__(self, image: Image.Image, resolution: int = 512) -> Image.Image:
        if not self.model:
            logger.warn("DepthAnything model was not loaded. Returning original image")
            return image

        np_image = np.array(image, dtype=np.uint8)
        image_height, image_width = np_image.shape[:2]

        with torch.no_grad():
            depth = self.model.infer_image(np_image)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth_map = repeat(depth, "h w -> h w 3").astype(np.uint8)
        depth_map = Image.fromarray(depth_map)

        new_height = int(image_height * (resolution / image_width))
        depth_map = depth_map.resize((resolution, new_height))

        return depth_map

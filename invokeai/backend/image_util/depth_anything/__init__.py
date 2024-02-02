import pathlib
from typing import Literal, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from PIL import Image
from torchvision.transforms import Compose

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.image_util.depth_anything.model.dpt import DPT_DINOv2
from invokeai.backend.image_util.depth_anything.utilities.util import NormalizeImage, PrepareForNet, Resize
from invokeai.backend.util.devices import choose_torch_device
from invokeai.backend.util.util import download_with_progress_bar

config = InvokeAIAppConfig.get_config()

DEPTH_ANYTHING_MODELS = {
    "large": {
        "url": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth?download=true",
        "local": "any/annotators/depth_anything/depth_anything_vitl14.pth",
    },
    "base": {
        "url": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth?download=true",
        "local": "any/annotators/depth_anything/depth_anything_vitb14.pth",
    },
    "small": {
        "url": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth?download=true",
        "local": "any/annotators/depth_anything/depth_anything_vits14.pth",
    },
}


transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


class DepthAnythingDetector:
    def __init__(self) -> None:
        self.model = None
        self.model_size: Union[Literal["large", "base", "small"], None] = None

    def load_model(self, model_size=Literal["large", "base", "small"]):
        DEPTH_ANYTHING_MODEL_PATH = pathlib.Path(config.models_path / DEPTH_ANYTHING_MODELS[model_size]["local"])
        if not DEPTH_ANYTHING_MODEL_PATH.exists():
            download_with_progress_bar(DEPTH_ANYTHING_MODELS[model_size]["url"], DEPTH_ANYTHING_MODEL_PATH)

        if not self.model or model_size != self.model_size:
            del self.model
            self.model_size = model_size

            match self.model_size:
                case "small":
                    self.model = DPT_DINOv2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
                case "base":
                    self.model = DPT_DINOv2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
                case "large":
                    self.model = DPT_DINOv2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
                case _:
                    raise TypeError("Not a supported model")

            self.model.load_state_dict(torch.load(DEPTH_ANYTHING_MODEL_PATH.as_posix(), map_location="cpu"))
            self.model.eval()

        self.model.to(choose_torch_device())
        return self.model

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, image, resolution=512):
        image = np.array(image, dtype=np.uint8)
        image = image[:, :, ::-1] / 255.0

        image_height, image_width = image.shape[:2]
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(choose_torch_device())

        with torch.no_grad():
            depth = self.model(image)
            depth = F.interpolate(depth[None], (image_height, image_width), mode="bilinear", align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth_map = repeat(depth, "h w -> h w 3").cpu().numpy().astype(np.uint8)
        depth_map = Image.fromarray(depth_map)

        new_height = int(image_height * (resolution / image_width))
        depth_map = depth_map.resize((resolution, new_height))

        return depth_map

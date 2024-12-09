import os
import cv2
import numpy as np
import torch

from einops import rearrange, repeat
from PIL import Image
from safetensors.torch import load_file as load_sft
from torch import nn
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel

class DepthImageEncoder:
    depth_model_name = "LiheYoung/depth-anything-large-hf"
    def __init__(self, device):
        self.device = device
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(self.depth_model_name)
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        hw = img.shape[-2:]
        img = torch.clamp(img, -1.0, 1.0)
        img_byte = ((img + 1.0) * 127.5).byte()
        img = self.processor(img_byte, return_tensors="pt")["pixel_values"]
        depth = self.depth_model(img.to(self.device)).predicted_depth
        depth = repeat(depth, "b h w -> b 3 h w")
        depth = torch.nn.functional.interpolate(depth, hw, mode="bicubic", antialias=True)
        depth = depth / 127.5 - 1.0
        return depth

class CannyImageEncoder:
    def __init__(
        self,
        device,
        min_t: int = 50,
        max_t: int = 200,
    ):
        self.device = device
        self.min_t = min_t
        self.max_t = max_t
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert img.shape[0] == 1, "Only batch size 1 is supported"
        img = rearrange(img[0], "c h w -> h w c")
        img = torch.clamp(img, -1.0, 1.0)
        img_np = ((img + 1.0) * 127.5).numpy().astype(np.uint8)
        # Apply Canny edge detection
        canny = cv2.Canny(img_np, self.min_t, self.max_t)
        # Convert back to torch tensor and reshape
        canny = torch.from_numpy(canny).float() / 127.5 - 1.0
        canny = rearrange(canny, "h w -> 1 1 h w")
        canny = repeat(canny, "b 1 ... -> b 3 ...")
        return canny.to(self.device)

"""Adapted from https://github.com/huggingface/controlnet_aux (Apache-2.0 license)."""

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image

from invokeai.backend.image_util.util import (
    normalize_image_channel_count,
    np_to_pil,
    pil_to_np,
    resize_image_to_resolution,
)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartProcessor:
    """Processor for lineart detection."""

    def __init__(self):
        model_path = hf_hub_download("lllyasviel/Annotators", "sk_model.pth")
        self.model = Generator(3, 1, 3)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

        coarse_model_path = hf_hub_download("lllyasviel/Annotators", "sk_model2.pth")
        self.model_coarse = Generator(3, 1, 3)
        self.model_coarse.load_state_dict(torch.load(coarse_model_path, map_location=torch.device("cpu")))
        self.model_coarse.eval()

    def to(self, device: torch.device):
        self.model.to(device)
        self.model_coarse.to(device)
        return self

    def run(
        self, input_image: Image.Image, coarse: bool = False, detect_resolution: int = 512, image_resolution: int = 512
    ) -> Image.Image:
        """Processes an image to detect lineart.

        Args:
            input_image: The input image.
            coarse: Whether to use the coarse model.
            detect_resolution: The resolution to fit the image to before edge detection.
            image_resolution: The resolution of the output image.

        Returns:
            The detected lineart.
        """
        device = next(iter(self.model.parameters())).device

        np_image = pil_to_np(input_image)
        np_image = normalize_image_channel_count(np_image)
        np_image = resize_image_to_resolution(np_image, detect_resolution)

        model = self.model_coarse if coarse else self.model
        assert np_image.ndim == 3
        image = np_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().to(device)
            image = image / 255.0
            image = rearrange(image, "h w c -> 1 c h w")
            line = model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = line

        detected_map = normalize_image_channel_count(detected_map)

        img = resize_image_to_resolution(np_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = 255 - detected_map

        return np_to_pil(detected_map)

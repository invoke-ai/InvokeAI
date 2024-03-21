"""Adapted from https://github.com/huggingface/controlnet_aux (Apache-2.0 license)."""

import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image

from invokeai.backend.image_util.util import (
    fit_image_to_resolution,
    non_maximum_suppression,
    normalize_image_channel_count,
    np_to_pil,
    pil_to_np,
    safe_step,
)


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(
                in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1
            )
        )
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = torch.nn.Conv2d(
            in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0
        )

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDProcessor:
    """Holistically-Nested Edge Detection.

    On instantiation, loads the HED model from the HuggingFace Hub.
    """

    def __init__(self):
        model_path = hf_hub_download("lllyasviel/Annotators", "ControlNetHED.pth")
        self.network = ControlNetHED_Apache2()
        self.network.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.network.float().eval()

    def to(self, device: torch.device):
        self.network.to(device)
        return self

    def run(
        self,
        input_image: Image.Image,
        detect_resolution: int = 512,
        image_resolution: int = 512,
        safe: bool = False,
        scribble: bool = False,
    ) -> Image.Image:
        """Processes an image and returns the detected edges.

        Args:
            input_image: The input image.
            detect_resolution: The resolution to fit the image to before edge detection.
            image_resolution: The resolution to fit the edges to before returning.
            safe: Whether to apply safe step to the detected edges.
            scribble: Whether to apply non-maximum suppression and Gaussian blur to the detected edges.

        Returns:
            The detected edges.
        """
        device = next(iter(self.network.parameters())).device
        np_image = pil_to_np(input_image)
        np_image = normalize_image_channel_count(np_image)
        np_image = fit_image_to_resolution(np_image, detect_resolution)

        assert np_image.ndim == 3
        height, width, _channels = np_image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(np_image.copy()).float().to(device)
            image_hed = rearrange(image_hed, "h w c -> 1 c h w")
            edges = self.network(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (width, height), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge
        detected_map = normalize_image_channel_count(detected_map)

        img = fit_image_to_resolution(np_image, image_resolution)
        height, width, _channels = img.shape

        detected_map = cv2.resize(detected_map, (width, height), interpolation=cv2.INTER_LINEAR)

        if scribble:
            detected_map = non_maximum_suppression(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        return np_to_pil(detected_map)

"""Adapted from https://github.com/huggingface/controlnet_aux (Apache-2.0 license)."""

import functools
import pathlib
from typing import Optional

import cv2
import huggingface_hub
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
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        ngf: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )  # add the innermost layer
        for _ in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: Optional[int] = None,
        submodule=None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class LineartAnimeProcessor:
    """Processes an image to detect lineart."""

    def __init__(self):
        model_path = hf_hub_download("lllyasviel/Annotators", "netG.pth")
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.model = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
        ckpt = torch.load(model_path)
        for key in list(ckpt.keys()):
            if "module." in key:
                ckpt[key.replace("module.", "")] = ckpt[key]
                del ckpt[key]
        self.model.load_state_dict(ckpt)
        self.model.eval()

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def run(self, input_image: Image.Image, detect_resolution: int = 512, image_resolution: int = 512) -> Image.Image:
        """Processes an image to detect lineart.

        Args:
            input_image: The input image.
            detect_resolution: The resolution to use for detection.
            image_resolution: The resolution to use for the output image.

        Returns:
            The detected lineart.
        """
        device = get_effective_device(self.model)
        np_image = pil_to_np(input_image)

        np_image = normalize_image_channel_count(np_image)
        np_image = resize_image_to_resolution(np_image, detect_resolution)

        H, W, C = np_image.shape
        Hn = 256 * int(np.ceil(float(H) / 256.0))
        Wn = 256 * int(np.ceil(float(W) / 256.0))
        img = cv2.resize(np_image, (Wn, Hn), interpolation=cv2.INTER_CUBIC)
        with torch.no_grad():
            image_feed = torch.from_numpy(img).float().to(device)
            image_feed = image_feed / 127.5 - 1.0
            image_feed = rearrange(image_feed, "h w c -> 1 c h w")

            line = self.model(image_feed)[0, 0] * 127.5 + 127.5
            line = line.cpu().numpy()

            line = cv2.resize(line, (W, H), interpolation=cv2.INTER_CUBIC)
            line = line.clip(0, 255).astype(np.uint8)

        detected_map = line

        detected_map = normalize_image_channel_count(detected_map)

        img = resize_image_to_resolution(np_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = 255 - detected_map

        return np_to_pil(detected_map)


class LineartAnimeEdgeDetector:
    """Simple wrapper around the Lineart Anime model for detecting edges in an image."""

    hf_repo_id = "lllyasviel/Annotators"
    hf_filename = "netG.pth"

    @classmethod
    def get_model_url(cls) -> str:
        """Get the URL to download the model from the Hugging Face Hub."""
        return huggingface_hub.hf_hub_url(cls.hf_repo_id, cls.hf_filename)

    @classmethod
    def load_model(cls, model_path: pathlib.Path) -> UnetGenerator:
        """Load the model from a file."""
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        model = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
        ckpt = torch.load(model_path)
        for key in list(ckpt.keys()):
            if "module." in key:
                ckpt[key.replace("module.", "")] = ckpt[key]
                del ckpt[key]
        model.load_state_dict(ckpt)
        model.eval()
        return model

    def __init__(self, model: UnetGenerator) -> None:
        self.model = model

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def run(self, image: Image.Image) -> Image.Image:
        """Processes an image and returns the detected edges."""
        device = get_effective_device(self.model)

        np_image = pil_to_np(image)

        height, width, _channels = np_image.shape
        new_height = 256 * int(np.ceil(float(height) / 256.0))
        new_width = 256 * int(np.ceil(float(width) / 256.0))

        resized_img = cv2.resize(np_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        with torch.no_grad():
            image_feed = torch.from_numpy(resized_img).float().to(device)
            image_feed = image_feed / 127.5 - 1.0
            image_feed = rearrange(image_feed, "h w c -> 1 c h w")

            line = self.model(image_feed)[0, 0] * 127.5 + 127.5
            line = line.cpu().numpy()

            line = cv2.resize(line, (width, height), interpolation=cv2.INTER_CUBIC)
            line = line.clip(0, 255).astype(np.uint8)

        detected_map = 255 - line

        # The lineart model often outputs a lot of almost-black noise. SD1.5 ControlNets seem to be OK with this, but
        # SDXL ControlNets are not - they need a cleaner map. 12 was experimentally determined to be a good threshold,
        # eliminating all the noise while keeping the actual edges. Other approaches to thresholding may be better,
        # for example stretching the contrast or removing noise.
        detected_map[detected_map < 12] = 0

        output = np_to_pil(detected_map)

        return output

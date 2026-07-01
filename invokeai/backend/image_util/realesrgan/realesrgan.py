import math
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
from cv2.typing import MatLike
from tqdm import tqdm

from invokeai.backend.image_util.basicsr.rrdbnet_arch import RRDBNet
from invokeai.backend.model_manager.taxonomy import AnyModel
from invokeai.backend.util.devices import TorchDevice

"""
Adapted from https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
License is BSD3, copied to `LICENSE` in this directory.

The adaptation here has a few changes:
- Remove print statements, use `tqdm` to show progress
- Remove unused "outscale" logic, which simply scales the final image to a given factor
- Remove `dni_weight` logic, which was only used when multiple models were used
- Remove logic to fetch models from network
- Add types, rename a few things
"""


class ImageMode(str, Enum):
    L = "L"
    RGB = "RGB"
    RGBA = "RGBA"


class RealESRGAN:
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    output: torch.Tensor

    def __init__(
        self,
        scale: int,
        loadnet: AnyModel,
        model: RRDBNet,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: bool = False,
    ) -> None:
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale: Optional[int] = None
        self.half = half
        self.device = TorchDevice.choose_torch_device()

        # prefer to use params_ema
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"

        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)

        if self.half:
            self.model = self.model.half()

    def pre_process(self, img: MatLike) -> None:
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible"""
        img_tensor: torch.Tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img_tensor.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = torch.nn.functional.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale
            self.img = torch.nn.functional.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")

    def process(self) -> None:
        # model inference
        self.output = self.model(self.img)

    def tile_process(self) -> None:
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        total_steps = tiles_y * tiles_x
        for i in tqdm(range(total_steps), desc="Upscaling"):
            y = i // tiles_x
            x = i % tiles_x
            # extract tile from input image
            ofs_x = x * self.tile_size
            ofs_y = y * self.tile_size
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + self.tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + self.tile_size, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - self.tile_pad, 0)
            input_end_x_pad = min(input_end_x + self.tile_pad, width)
            input_start_y_pad = max(input_start_y - self.tile_pad, 0)
            input_end_y_pad = min(input_end_y + self.tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            input_tile = self.img[
                :,
                :,
                input_start_y_pad:input_end_y_pad,
                input_start_x_pad:input_end_x_pad,
            ]

            # upscale tile
            with torch.no_grad():
                output_tile = self.model(input_tile)

            # output tile area on total image
            output_start_x = input_start_x * self.scale
            output_end_x = input_end_x * self.scale
            output_start_y = input_start_y * self.scale
            output_end_y = input_end_y * self.scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
            output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
            output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

            # put tile into output image
            self.output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :,
                :,
                output_start_y_tile:output_end_y_tile,
                output_start_x_tile:output_end_x_tile,
            ]

    def post_process(self) -> torch.Tensor:
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.mod_pad_h * self.scale,
                0 : w - self.mod_pad_w * self.scale,
            ]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.pre_pad * self.scale,
                0 : w - self.pre_pad * self.scale,
            ]
        return self.output

    @torch.no_grad()
    def upscale(self, img: MatLike, esrgan_alpha_upscale: bool = True) -> npt.NDArray[Any]:
        np_img = img.astype(np.float32)
        alpha: Optional[np.ndarray] = None
        if np.max(np_img) > 256:
            # 16-bit image
            max_range = 65535
        else:
            max_range = 255
        np_img = np_img / max_range
        if len(np_img.shape) == 2:
            # grayscale image
            img_mode = ImageMode.L
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
        elif np_img.shape[2] == 4:
            # RGBA image with alpha channel
            img_mode = ImageMode.RGBA
            alpha = np_img[:, :, 3]
            np_img = np_img[:, :, 0:3]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            if esrgan_alpha_upscale:
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = ImageMode.RGB
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(np_img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_tensor = self.post_process()
        output_img: npt.NDArray[Any] = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode is ImageMode.L:
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode is ImageMode.RGBA:
            if esrgan_alpha_upscale:
                assert alpha is not None
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha_tensor = self.post_process()
                output_alpha: npt.NDArray[Any] = output_alpha_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                assert alpha is not None
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(
                    alpha,
                    (w * self.scale, h * self.scale),
                    interpolation=cv2.INTER_LINEAR,
                )

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        return output

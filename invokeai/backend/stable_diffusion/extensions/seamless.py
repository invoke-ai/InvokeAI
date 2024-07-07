from __future__ import annotations

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple
from diffusers import UNet2DConditionModel
from diffusers.models.lora import LoRACompatibleConv
from .base import ExtensionBase


class SeamlessExt(ExtensionBase):
    def __init__(
        self,
        seamless_axes: List[str],
        priority: int,
    ):
        super().__init__(priority=priority)
        self.seamless_axes = seamless_axes

    @contextmanager
    def patch_unet(self, state_dict: dict, unet: UNet2DConditionModel):
        if not self.seamless_axes:
            yield
            return

        # override conv_forward
        # https://github.com/huggingface/diffusers/issues/556#issuecomment-1993287019
        def _conv_forward_asymmetric(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
            self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0)
            self.paddingY = (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
            working = torch.nn.functional.pad(input, self.paddingX, mode=x_mode)
            working = torch.nn.functional.pad(working, self.paddingY, mode=y_mode)
            return torch.nn.functional.conv2d(
                working, weight, bias, self.stride, torch.nn.modules.utils._pair(0), self.dilation, self.groups
            )

        original_layers: List[Tuple[nn.Conv2d, Callable]] = []

        try:
            x_mode = "circular" if "x" in self.seamless_axes else "constant"
            y_mode = "circular" if "y" in self.seamless_axes else "constant"

            conv_layers: List[torch.nn.Conv2d] = []

            for module in unet.modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(module)

            for layer in conv_layers:
                if isinstance(layer, LoRACompatibleConv) and layer.lora_layer is None:
                    layer.lora_layer = lambda *x: 0
                original_layers.append((layer, layer._conv_forward))
                layer._conv_forward = _conv_forward_asymmetric.__get__(layer, torch.nn.Conv2d)

            yield

        finally:
            for layer, orig_conv_forward in original_layers:
                layer._conv_forward = orig_conv_forward

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.lora import LoRACompatibleConv

from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase


class SeamlessExt(ExtensionBase):
    def __init__(
        self,
        seamless_axes: List[str],
    ):
        super().__init__()
        self._seamless_axes = seamless_axes

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, cached_weights: Optional[Dict[str, torch.Tensor]] = None):
        with self.static_patch_model(
            model=unet,
            seamless_axes=self._seamless_axes,
        ):
            yield

    @staticmethod
    @contextmanager
    def static_patch_model(
        model: torch.nn.Module,
        seamless_axes: List[str],
    ):
        if not seamless_axes:
            yield
            return

        x_mode = "circular" if "x" in seamless_axes else "constant"
        y_mode = "circular" if "y" in seamless_axes else "constant"

        # override conv_forward
        # https://github.com/huggingface/diffusers/issues/556#issuecomment-1993287019
        def _conv_forward_asymmetric(
            self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
        ):
            self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0)
            self.paddingY = (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
            working = torch.nn.functional.pad(input, self.paddingX, mode=x_mode)
            working = torch.nn.functional.pad(working, self.paddingY, mode=y_mode)
            return torch.nn.functional.conv2d(
                working, weight, bias, self.stride, torch.nn.modules.utils._pair(0), self.dilation, self.groups
            )

        original_layers: List[Tuple[nn.Conv2d, Callable]] = []
        try:
            for layer in model.modules():
                if not isinstance(layer, torch.nn.Conv2d):
                    continue

                if isinstance(layer, LoRACompatibleConv) and layer.lora_layer is None:
                    layer.lora_layer = lambda *x: 0
                original_layers.append((layer, layer._conv_forward))
                layer._conv_forward = _conv_forward_asymmetric.__get__(layer, torch.nn.Conv2d)

            yield

        finally:
            for layer, orig_conv_forward in original_layers:
                layer._conv_forward = orig_conv_forward

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, List, Union

import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


def _conv_forward_asymmetric(self, input, weight, bias):
    """
    Patch for Conv2d._conv_forward that supports asymmetric padding
    """
    working = nn.functional.pad(input, self.asymmetric_padding["x"], mode=self.asymmetric_padding_mode["x"])
    working = nn.functional.pad(working, self.asymmetric_padding["y"], mode=self.asymmetric_padding_mode["y"])
    return nn.functional.conv2d(
        working,
        weight,
        bias,
        self.stride,
        nn.modules.utils._pair(0),
        self.dilation,
        self.groups,
    )


@contextmanager
def set_seamless(model: Union[UNet2DConditionModel, AutoencoderKL, AutoencoderTiny], seamless_axes: List[str]):
    # Callable: (input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor
    to_restore: list[tuple[nn.Conv2d | nn.ConvTranspose2d, Callable]] = []
    try:
        # Hard coded to skip down block layers, allowing for seamless tiling at the expense of prompt adherence
        skipped_layers = 1
        for m_name, m in model.named_modules():
            if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                continue

            if isinstance(model, UNet2DConditionModel) and m_name.startswith("down_blocks.") and ".resnets." in m_name:
                # down_blocks.1.resnets.1.conv1
                _, block_num, _, resnet_num, submodule_name = m_name.split(".")
                block_num = int(block_num)
                resnet_num = int(resnet_num)

                if block_num >= len(model.down_blocks) - skipped_layers:
                    continue

                # Skip the second resnet (could be configurable)
                if resnet_num > 0:
                    continue

                # Skip Conv2d layers (could be configurable)
                if submodule_name == "conv2":
                    continue

            m.asymmetric_padding_mode = {}
            m.asymmetric_padding = {}
            m.asymmetric_padding_mode["x"] = "circular" if ("x" in seamless_axes) else "constant"
            m.asymmetric_padding["x"] = (
                m._reversed_padding_repeated_twice[0],
                m._reversed_padding_repeated_twice[1],
                0,
                0,
            )
            m.asymmetric_padding_mode["y"] = "circular" if ("y" in seamless_axes) else "constant"
            m.asymmetric_padding["y"] = (
                0,
                0,
                m._reversed_padding_repeated_twice[2],
                m._reversed_padding_repeated_twice[3],
            )

            to_restore.append((m, m._conv_forward))
            m._conv_forward = _conv_forward_asymmetric.__get__(m, nn.Conv2d)

        yield

    finally:
        for module, orig_conv_forward in to_restore:
            module._conv_forward = orig_conv_forward
            if hasattr(module, "asymmetric_padding_mode"):
                del module.asymmetric_padding_mode
            if hasattr(module, "asymmetric_padding"):
                del module.asymmetric_padding

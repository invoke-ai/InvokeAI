from __future__ import annotations

from contextlib import contextmanager
from typing import List, Union

import torch.nn as nn
from diffusers.models import AutoencoderKL, UNet2DConditionModel


def _conv_forward_asymmetric(self, input, weight, bias=None):
    """
    Optimized patch for Conv2d._conv_forward that supports asymmetric padding.
    Combines padding for both axes into a single operation.
    """
    # Calculate the combined padding for both x and y axes
    combined_padding = (
        self.asymmetric_padding["x"][0], self.asymmetric_padding["x"][1],
        self.asymmetric_padding["y"][2], self.asymmetric_padding["y"][3]
    )
    
    # Apply combined padding in a single operation
    working = nn.functional.pad(input, combined_padding, mode=self.asymmetric_padding_mode["x"])
    
    # Perform the convolution with no additional padding (since it's already applied)
    return nn.functional.conv2d(
        working,
        weight,
        bias,
        self.stride,
        (0, 0),  # No additional padding needed as we've already padded
        self.dilation,
        self.groups
    )



@contextmanager
def set_seamless(model: Union[UNet2DConditionModel, AutoencoderKL], seamless_axes: List[str]):
    try:
        to_restore = []

        for m_name, m in model.named_modules():
            if isinstance(model, UNet2DConditionModel):
                if ".attentions." in m_name:
                    continue

                if ".resnets." in m_name:
                    if ".conv2" in m_name:
                        continue
                    if ".conv_shortcut" in m_name:
                        continue

            """
            if isinstance(model, UNet2DConditionModel):
                if False and ".upsamplers." in m_name:
                    continue

                if False and ".downsamplers." in m_name:
                    continue

                if True and ".resnets." in m_name:
                    if True and ".conv1" in m_name:
                        if False and "down_blocks" in m_name:
                            continue
                        if False and "mid_block" in m_name:
                            continue
                        if False and "up_blocks" in m_name:
                            continue

                    if True and ".conv2" in m_name:
                        continue

                    if True and ".conv_shortcut" in m_name:
                        continue

                if True and ".attentions." in m_name:
                    continue

                if False and m_name in ["conv_in", "conv_out"]:
                    continue
            """

            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Directly set padding mode and values without redundant checks
                m.asymmetric_padding_mode = {
                    "x": "circular" if "x" in seamless_axes else "constant",
                    "y": "circular" if "y" in seamless_axes else "constant"
                }
                m.asymmetric_padding = {
                    "x": (m.padding[0], m.padding[1], 0, 0),
                    "y": (0, 0, m.padding[2], m.padding[3])
                }
                # Backup and override the conv forward method
                to_restore.append((m, m._conv_forward))
                m._conv_forward = _conv_forward_asymmetric.__get__(m, nn.Conv2d)

        yield

    finally:
        for module, orig_conv_forward in to_restore:
            module._conv_forward = orig_conv_forward
            del module.asymmetric_padding_mode
            del module.asymmetric_padding

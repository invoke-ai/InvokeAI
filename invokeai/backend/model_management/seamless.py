from __future__ import annotations

from contextlib import contextmanager
from typing import List, Union

import torch.nn as nn
from diffusers.models import AutoencoderKL, UNet2DConditionModel


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
            if hasattr(m, "asymmetric_padding_mode"):
                del m.asymmetric_padding_mode
            if hasattr(m, "asymmetric_padding"):
                del m.asymmetric_padding

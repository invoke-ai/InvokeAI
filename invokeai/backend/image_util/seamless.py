import torch.nn as nn


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


def configure_model_padding(model, seamless, seamless_axes):
    """
    Modifies the 2D convolution layers to use a circular padding mode based on
    the `seamless` and `seamless_axes` options.
    """
    # TODO: get an explicit interface for this in diffusers: https://github.com/huggingface/diffusers/issues/556
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if seamless:
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
                m._conv_forward = _conv_forward_asymmetric.__get__(m, nn.Conv2d)
            else:
                m._conv_forward = nn.Conv2d._conv_forward.__get__(m, nn.Conv2d)
                if hasattr(m, "asymmetric_padding_mode"):
                    del m.asymmetric_padding_mode
                if hasattr(m, "asymmetric_padding"):
                    del m.asymmetric_padding

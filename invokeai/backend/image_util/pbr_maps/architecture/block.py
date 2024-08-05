# Original: https://github.com/joeyballentine/Material-Map-Generator
# Adopted and optimized for Invoke AI

from collections import OrderedDict
from typing import Any, List, Literal, Optional

import torch
import torch.nn as nn

ACTIVATION_LAYER_TYPE = Literal["relu", "leakyrelu", "prelu"]
NORMALIZATION_LAYER_TYPE = Literal["batch", "instance"]
PADDING_LAYER_TYPE = Literal["zero", "reflect", "replicate"]
BLOCK_MODE = Literal["CNA", "NAC", "CNAC"]
UPCONV_BLOCK_MODE = Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"]


def act(act_type: ACTIVATION_LAYER_TYPE, inplace: bool = True, neg_slope: float = 0.2, n_prelu: int = 1):
    """Helper to select Activation Layer"""
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    return layer


def norm(norm_type: NORMALIZATION_LAYER_TYPE, nc: int):
    """Helper to select Normalization Layer"""
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    return layer


def pad(pad_type: PADDING_LAYER_TYPE, padding: int):
    """Helper to select Padding Layer"""
    if padding == 0 or pad_type == "zero":
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    return layer


def get_valid_padding(kernel_size: int, dilation: int):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args: Any):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules: List[nn.Module] = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    pad_type: Optional[PADDING_LAYER_TYPE] = "zero",
    norm_type: Optional[NORMALIZATION_LAYER_TYPE] = None,
    act_type: Optional[ACTIVATION_LAYER_TYPE] = "relu",
    mode: BLOCK_MODE = "CNA",
):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ["CNA", "NAC", "CNAC"], f"Wrong conv mode [{mode}]"
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = act(act_type) if act_type else None
    match mode:
        case "CNA":
            n = norm(norm_type, out_nc) if norm_type else None
            return sequential(p, c, n, a)
        case "NAC":
            if norm_type is None and act_type is not None:
                a = act(act_type, inplace=False)
            n = norm(norm_type, in_nc) if norm_type else None
            return sequential(n, a, p, c)
        case "CNAC":
            n = norm(norm_type, in_nc) if norm_type else None
            return sequential(n, a, p, c)


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule: nn.Module):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = "Identity .. \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule: nn.Module):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlockSPSR(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule: nn.Module):
        super(ShortcutBlockSPSR, self).__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor):
        return x, self.sub

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(
        self,
        in_nc: int,
        mid_nc: int,
        out_nc: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_type: PADDING_LAYER_TYPE = "zero",
        norm_type: Optional[NORMALIZATION_LAYER_TYPE] = None,
        act_type: Optional[ACTIVATION_LAYER_TYPE] = "relu",
        mode: BLOCK_MODE = "CNA",
        res_scale: int = 1,
    ):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(
            in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode
        )
        if mode == "CNA":
            act_type = None
        if mode == "CNAC":  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(
            mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode
        )

        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(
        self,
        nc: int,
        kernel_size: int = 3,
        gc: int = 32,
        stride: int = 1,
        bias: bool = True,
        pad_type: PADDING_LAYER_TYPE = "zero",
        norm_type: Optional[NORMALIZATION_LAYER_TYPE] = None,
        act_type: ACTIVATION_LAYER_TYPE = "leakyrelu",
        mode: BLOCK_MODE = "CNA",
    ):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(
            nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode
        )
        self.conv2 = conv_block(
            nc + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        self.conv3 = conv_block(
            nc + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        self.conv4 = conv_block(
            nc + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        if mode == "CNA":
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(
            nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=last_act, mode=mode
        )

    def forward(self, x: torch.Tensor):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        nc: int,
        kernel_size: int = 3,
        gc: int = 32,
        stride: int = 1,
        bias: bool = True,
        pad_type: PADDING_LAYER_TYPE = "zero",
        norm_type: Optional[NORMALIZATION_LAYER_TYPE] = None,
        act_type: ACTIVATION_LAYER_TYPE = "leakyrelu",
        mode: BLOCK_MODE = "CNA",
    ):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)

    def forward(self, x: torch.Tensor):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


# Upsampler
def pixelshuffle_block(
    in_nc: int,
    out_nc: int,
    upscale_factor: int = 2,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
    pad_type: PADDING_LAYER_TYPE = "zero",
    norm_type: Optional[NORMALIZATION_LAYER_TYPE] = None,
    act_type: ACTIVATION_LAYER_TYPE = "relu",
):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(
        in_nc,
        out_nc * (upscale_factor**2),
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=None,
        act_type=None,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(
    in_nc: int,
    out_nc: int,
    upscale_factor: int = 2,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
    pad_type: PADDING_LAYER_TYPE = "zero",
    norm_type: Optional[NORMALIZATION_LAYER_TYPE] = None,
    act_type: ACTIVATION_LAYER_TYPE = "relu",
    mode: UPCONV_BLOCK_MODE = "nearest",
):
    # Adopted from https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc, out_nc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type
    )
    return sequential(upsample, conv)

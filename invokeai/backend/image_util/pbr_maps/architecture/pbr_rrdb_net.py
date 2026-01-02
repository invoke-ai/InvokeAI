# Original: https://github.com/joeyballentine/Material-Map-Generator
# Adopted and optimized for Invoke AI

import math
from typing import Literal, Optional

import torch
import torch.nn as nn

import invokeai.backend.image_util.pbr_maps.architecture.block as B

UPSCALE_MODE = Literal["upconv", "pixelshuffle"]


class PBR_RRDB_Net(nn.Module):
    def __init__(
        self,
        in_nc: int,
        out_nc: int,
        nf: int,
        nb: int,
        gc: int = 32,
        upscale: int = 4,
        norm_type: Optional[B.NORMALIZATION_LAYER_TYPE] = None,
        act_type: B.ACTIVATION_LAYER_TYPE = "leakyrelu",
        mode: B.BLOCK_MODE = "CNA",
        res_scale: int = 1,
        upsample_mode: UPSCALE_MODE = "upconv",
    ):
        super(PBR_RRDB_Net, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [
            B.RRDB(
                nf,
                kernel_size=3,
                gc=32,
                stride=1,
                bias=True,
                pad_type="zero",
                norm_type=norm_type,
                act_type=act_type,
                mode="CNA",
            )
            for _ in range(nb)
        ]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == "upconv":
            upsample_block = B.upconv_block
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.pixelshuffle_block

        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(
            fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)), *upsampler, HR_conv0, HR_conv1
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

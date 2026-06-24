# TODO: Improve blend modes
# TODO: Add nodes like Hue Adjust for Saturation/Contrast/etc... ?
# TODO: Continue implementing more blend modes/color spaces(?)
# TODO: Custom ICC profiles with PIL.ImageCms?
# TODO: Blend multiple layers all crammed into a tensor(?) or list

# Copyright (c) 2023 Darren Ringer <dwringer@gmail.com>
# Parts based on Oklab: Copyright (c) 2021 Bj�rn Ottosson <https://bottosson.github.io/>
# HSL code based on CPython: Copyright (c) 2001-2023 Python Software Foundation; All Rights Reserved
from math import pi as PI
from pathlib import Path

import torch
from PIL import Image

from invokeai.backend.image_util.color_conversion import (
    gamut_clip_tensor,
)
from invokeai.backend.image_util.color_conversion import (
    srgb_from_linear_srgb as shared_srgb_from_linear_srgb,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor

MAX_FLOAT = torch.finfo(torch.tensor(1.0).dtype).max

# CIE Lab to Uniform Perceptual Lab profile is copyright © 2003 Bruce Justin Lindbloom. All rights reserved. <http://www.brucelindbloom.com>
CIELAB_TO_UPLAB_ICC_PATH = Path(__file__).parent / "assets" / "CIELab_to_UPLab.icc"


def equivalent_achromatic_lightness(lch_tensor: torch.Tensor):
    """Calculate Equivalent Achromatic Lightness accounting for Helmholtz-Kohlrausch effect"""
    # As described by High, Green, and Nussbaum (2023): https://doi.org/10.1002/col.22839

    k = [0.1644, 0.0603, 0.1307, 0.0060]

    h_minus_90 = torch.sub(lch_tensor[2, :, :], PI / 2.0)
    h_minus_90 = torch.sub(torch.remainder(torch.add(h_minus_90, 3 * PI), 2 * PI), PI)

    f_by = torch.add(k[0] * torch.abs(torch.sin(torch.div(h_minus_90, 2.0))), k[1])
    f_r_0 = torch.add(k[2] * torch.abs(torch.cos(lch_tensor[2, :, :])), k[3])

    f_r = torch.zeros(lch_tensor[0, :, :].shape)
    mask_hi = torch.ge(lch_tensor[2, :, :], -1 * (PI / 2.0))
    mask_lo = torch.le(lch_tensor[2, :, :], PI / 2.0)
    mask = torch.logical_and(mask_hi, mask_lo)
    f_r[mask] = f_r_0[mask]

    l_max = torch.ones(lch_tensor[0, :, :].shape)
    l_min = torch.zeros(lch_tensor[0, :, :].shape)
    l_adjustment = torch.tensordot(torch.add(f_by, f_r), lch_tensor[1, :, :], dims=([0, 1], [0, 1]))
    l_max = torch.add(l_max, l_adjustment)
    l_min = torch.add(l_min, l_adjustment)
    l_eal_tensor = torch.add(lch_tensor[0, :, :], l_adjustment)

    l_eal_tensor = torch.add(
        lch_tensor[0, :, :], torch.tensordot(torch.add(f_by, f_r), lch_tensor[1, :, :], dims=([0, 1], [0, 1]))
    )
    l_eal_tensor = torch.div(torch.sub(l_eal_tensor, l_min.min()), l_max.max() - l_min.min())

    return l_eal_tensor


def srgb_from_linear_srgb(linear_srgb_tensor: torch.Tensor, alpha: float = 0.0, steps: int = 1):
    """Get gamma-corrected sRGB from a linear-light sRGB image tensor"""

    if 0.0 < alpha:
        linear_srgb_tensor = gamut_clip_tensor(linear_srgb_tensor, alpha=alpha, steps=steps)
    return shared_srgb_from_linear_srgb(linear_srgb_tensor)


def remove_nans(tensor: torch.Tensor, replace_with: float = MAX_FLOAT):
    return torch.where(torch.isnan(tensor), replace_with, tensor)


def tensor_from_pil_image(img: Image.Image, normalize: bool = False):
    return image_resized_to_grid_as_tensor(img, normalize=normalize, multiple_of=1)


# PSF LICENSE AGREEMENT FOR PYTHON 3.11.5

# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and
#    the Individual or Organization ("Licensee") accessing and otherwise using Python
#    3.11.5 software in source or binary form and its associated documentation.

# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
#    grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
#    analyze, test, perform and/or display publicly, prepare derivative works,
#    distribute, and otherwise use Python 3.11.5 alone or in any derivative
#    version, provided, however, that PSF's License Agreement and PSF's notice of
#    copyright, i.e., "Copyright (c) 2001-2023 Python Software Foundation; All Rights
#    Reserved" are retained in Python 3.11.5 alone or in any derivative version
#    prepared by Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
#    incorporates Python 3.11.5 or any part thereof, and wants to make the
#    derivative work available to others as provided herein, then Licensee hereby
#    agrees to include in any such work a brief summary of the changes made to Python
#    3.11.5.

# 4. PSF is making Python 3.11.5 available to Licensee on an "AS IS" basis.
#    PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.  BY WAY OF
#    EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR
#    WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE
#    USE OF PYTHON 3.11.5 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON 3.11.5
#    FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
#    MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 3.11.5, OR ANY DERIVATIVE
#    THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material breach of
#    its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any relationship
#    of agency, partnership, or joint venture between PSF and Licensee.  This License
#    Agreement does not grant permission to use PSF trademarks or trade name in a
#    trademark sense to endorse or promote products or services of Licensee, or any
#    third party.

# 8. By copying, installing or otherwise using Python 3.11.5, Licensee agrees
#    to be bound by the terms and conditions of this License Agreement.
######################################################################################/

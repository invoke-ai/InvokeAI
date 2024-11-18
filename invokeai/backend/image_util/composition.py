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
from typing import Literal, Optional

import torch
from PIL import Image

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
    linear_srgb_tensor = linear_srgb_tensor.clamp(0.0, 1.0)
    mask = torch.lt(linear_srgb_tensor, 0.0404482362771082 / 12.92)
    rgb_tensor = torch.sub(torch.mul(torch.pow(linear_srgb_tensor, (1 / 2.4)), 1.055), 0.055)
    rgb_tensor[mask] = torch.mul(linear_srgb_tensor[mask], 12.92)

    return rgb_tensor


def linear_srgb_from_srgb(srgb_tensor: torch.Tensor):
    """Get linear-light sRGB from a standard gamma-corrected sRGB image tensor"""

    linear_srgb_tensor = torch.pow(torch.div(torch.add(srgb_tensor, 0.055), 1.055), 2.4)
    linear_srgb_tensor_1 = torch.div(srgb_tensor, 12.92)
    mask = torch.le(srgb_tensor, 0.0404482362771082)
    linear_srgb_tensor[mask] = linear_srgb_tensor_1[mask]

    return linear_srgb_tensor


def max_srgb_saturation_tensor(units_ab_tensor: torch.Tensor, steps: int = 1):
    """Compute maximum sRGB saturation of a tensor of Oklab ab unit vectors"""

    rgb_k_matrix = torch.tensor(
        [
            [1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245],
            [0.73956515, -0.45954494, 0.08285427, 0.12541070, 0.14503204],
            [1.35733652, -0.00915799, -1.15130210, -0.50559606, 0.00692167],
        ]
    )

    rgb_w_matrix = torch.tensor(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    )

    cond_r_tensor = torch.add(
        torch.mul(-1.88170328, units_ab_tensor[0, :, :]), torch.mul(-0.80936493, units_ab_tensor[1, :, :])
    )
    cond_g_tensor = torch.add(
        torch.mul(1.81444104, units_ab_tensor[0, :, :]), torch.mul(-1.19445276, units_ab_tensor[1, :, :])
    )

    terms_tensor = torch.stack(
        [
            torch.ones(units_ab_tensor.shape[1:]),
            units_ab_tensor[0, :, :],
            units_ab_tensor[1, :, :],
            torch.pow(units_ab_tensor[0, :, :], 2.0),
            torch.mul(units_ab_tensor[0, :, :], units_ab_tensor[1, :, :]),
        ]
    )

    s_tensor = torch.empty(units_ab_tensor.shape[1:])
    s_tensor = torch.where(
        torch.gt(cond_r_tensor, 1.0),
        torch.einsum("twh, t -> wh", terms_tensor, rgb_k_matrix[0]),
        torch.where(
            torch.gt(cond_g_tensor, 1.0),
            torch.einsum("twh, t -> wh", terms_tensor, rgb_k_matrix[1]),
            torch.einsum("twh, t -> wh", terms_tensor, rgb_k_matrix[2]),
        ),
    )

    k_lms_matrix = torch.tensor(
        [[0.3963377774, 0.2158037573], [-0.1055613458, -0.0638541728], [-0.0894841775, -1.2914855480]]
    )

    k_lms_tensor = torch.einsum("tc, cwh -> twh", k_lms_matrix, units_ab_tensor)

    for _ in range(steps):
        root_lms_tensor = torch.add(torch.mul(k_lms_tensor, s_tensor), 1.0)
        lms_tensor = torch.pow(root_lms_tensor, 3.0)
        lms_ds_tensor = torch.mul(torch.mul(k_lms_tensor, torch.pow(root_lms_tensor, 2.0)), 3.0)
        lms_ds2_tensor = torch.mul(torch.mul(torch.pow(k_lms_tensor, 2.0), root_lms_tensor), 6.0)
        f_tensor = torch.where(
            torch.gt(cond_r_tensor, 1.0),
            torch.einsum("c, cwh -> wh", rgb_w_matrix[0], lms_tensor),
            torch.where(
                torch.gt(cond_g_tensor, 1.0),
                torch.einsum("c, cwh -> wh", rgb_w_matrix[1], lms_tensor),
                torch.einsum("c, cwh -> wh", rgb_w_matrix[2], lms_tensor),
            ),
        )
        f_tensor_1 = torch.where(
            torch.gt(cond_r_tensor, 1.0),
            torch.einsum("c, cwh -> wh", rgb_w_matrix[0], lms_ds_tensor),
            torch.where(
                torch.gt(cond_g_tensor, 1.0),
                torch.einsum("c, cwh -> wh", rgb_w_matrix[1], lms_ds_tensor),
                torch.einsum("c, cwh -> wh", rgb_w_matrix[2], lms_ds_tensor),
            ),
        )
        f_tensor_2 = torch.where(
            torch.gt(cond_r_tensor, 1.0),
            torch.einsum("c, cwh -> wh", rgb_w_matrix[0], lms_ds2_tensor),
            torch.where(
                torch.gt(cond_g_tensor, 1.0),
                torch.einsum("c, cwh -> wh", rgb_w_matrix[1], lms_ds2_tensor),
                torch.einsum("c, cwh -> wh", rgb_w_matrix[2], lms_ds2_tensor),
            ),
        )
        s_tensor = torch.sub(
            s_tensor,
            torch.div(
                torch.mul(f_tensor, f_tensor_1),
                torch.sub(torch.pow(f_tensor_1, 2.0), torch.mul(torch.mul(f_tensor, f_tensor_2), 0.5)),
            ),
        )

    return s_tensor


def linear_srgb_from_oklab(oklab_tensor: torch.Tensor):
    """Get linear-light sRGB from an Oklab image tensor"""

    # L*a*b* to LMS
    lms_matrix_1 = torch.tensor(
        [[1.0, 0.3963377774, 0.2158037573], [1.0, -0.1055613458, -0.0638541728], [1.0, -0.0894841775, -1.2914855480]]
    )

    lms_tensor_1 = torch.einsum("lwh, kl -> kwh", oklab_tensor, lms_matrix_1)
    lms_tensor = torch.pow(lms_tensor_1, 3.0)

    # LMS to linear RGB
    rgb_matrix = torch.tensor(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    )

    linear_srgb_tensor = torch.einsum("kwh, sk -> swh", lms_tensor, rgb_matrix)

    return linear_srgb_tensor


def oklab_from_linear_srgb(linear_srgb_tensor: torch.Tensor):
    """Get an Oklab image tensor from a tensor of linear-light sRGB"""
    # linear RGB to LMS
    lms_matrix = torch.tensor(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )

    lms_tensor = torch.einsum("cwh, kc -> kwh", linear_srgb_tensor, lms_matrix)

    # LMS to L*a*b*
    lms_tensor_neg_mask = torch.lt(lms_tensor, 0.0)
    lms_tensor[lms_tensor_neg_mask] = torch.mul(lms_tensor[lms_tensor_neg_mask], -1.0)
    lms_tensor_1 = torch.pow(lms_tensor, 1.0 / 3.0)
    lms_tensor[lms_tensor_neg_mask] = torch.mul(lms_tensor[lms_tensor_neg_mask], -1.0)
    lms_tensor_1[lms_tensor_neg_mask] = torch.mul(lms_tensor_1[lms_tensor_neg_mask], -1.0)
    lab_matrix = torch.tensor(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )

    lab_tensor = torch.einsum("kwh, lk -> lwh", lms_tensor_1, lab_matrix)

    return lab_tensor


def find_cusp_tensor(units_ab_tensor: torch.Tensor, steps: int = 1):
    """Compute maximum sRGB lightness and chroma from a tensor of Oklab ab unit vectors"""

    s_cusp_tensor = max_srgb_saturation_tensor(units_ab_tensor, steps=steps)

    oklab_tensor = torch.stack(
        [
            torch.ones(s_cusp_tensor.shape),
            torch.mul(s_cusp_tensor, units_ab_tensor[0, :, :]),
            torch.mul(s_cusp_tensor, units_ab_tensor[1, :, :]),
        ]
    )

    rgb_at_max_tensor = linear_srgb_from_oklab(oklab_tensor)

    l_cusp_tensor = torch.pow(torch.div(1.0, rgb_at_max_tensor.max(0).values), 1.0 / 3.0)
    c_cusp_tensor = torch.mul(l_cusp_tensor, s_cusp_tensor)

    return torch.stack([l_cusp_tensor, c_cusp_tensor])


def find_gamut_intersection_tensor(
    units_ab_tensor: torch.Tensor,
    l_1_tensor: torch.Tensor,
    c_1_tensor: torch.Tensor,
    l_0_tensor: torch.Tensor,
    steps: int = 1,
    steps_outer: int = 1,
    lc_cusps_tensor: Optional[torch.Tensor] = None,
):
    """Find thresholds of lightness intersecting RGB gamut from Oklab component tensors"""

    if lc_cusps_tensor is None:
        lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)

    # if (((l_1 - l_0) * c_cusp -
    #      (l_cusp - l_0) * c_1) <= 0.):
    cond_tensor = torch.sub(
        torch.mul(torch.sub(l_1_tensor, l_0_tensor), lc_cusps_tensor[1, :, :]),
        torch.mul(torch.sub(lc_cusps_tensor[0, :, :], l_0_tensor), c_1_tensor),
    )

    t_tensor = torch.where(
        torch.le(cond_tensor, 0.0),  # cond <= 0
        #  t = (c_cusp * l_0) /
        #      ((c_1 * l_cusp) + (c_cusp * (l_0 - l_1)))
        torch.div(
            torch.mul(lc_cusps_tensor[1, :, :], l_0_tensor),
            torch.add(
                torch.mul(c_1_tensor, lc_cusps_tensor[0, :, :]),
                torch.mul(lc_cusps_tensor[1, :, :], torch.sub(l_0_tensor, l_1_tensor)),
            ),
        ),
        # t = (c_cusp * (l_0-1.)) /
        #     ((c_1 * (l_cusp-1.)) + (c_cusp * (l_0 - l_1)))
        torch.div(
            torch.mul(lc_cusps_tensor[1, :, :], torch.sub(l_0_tensor, 1.0)),
            torch.add(
                torch.mul(c_1_tensor, torch.sub(lc_cusps_tensor[0, :, :], 1.0)),
                torch.mul(lc_cusps_tensor[1, :, :], torch.sub(l_0_tensor, l_1_tensor)),
            ),
        ),
    )

    for _ in range(steps_outer):
        dl_tensor = torch.sub(l_1_tensor, l_0_tensor)
        dc_tensor = c_1_tensor

        k_lms_matrix = torch.tensor(
            [[0.3963377774, 0.2158037573], [-0.1055613458, -0.0638541728], [-0.0894841775, -1.2914855480]]
        )
        k_lms_tensor = torch.einsum("tc, cwh -> twh", k_lms_matrix, units_ab_tensor)

        lms_dt_tensor = torch.add(torch.mul(k_lms_tensor, dc_tensor), dl_tensor)

        for _ in range(steps):
            l_tensor = torch.add(
                torch.mul(l_0_tensor, torch.add(torch.mul(t_tensor, -1.0), 1.0)), torch.mul(t_tensor, l_1_tensor)
            )
            c_tensor = torch.mul(t_tensor, c_1_tensor)

            root_lms_tensor = torch.add(torch.mul(k_lms_tensor, c_tensor), l_tensor)

            lms_tensor = torch.pow(root_lms_tensor, 3.0)
            lms_dt_tensor_1 = torch.mul(torch.mul(torch.pow(root_lms_tensor, 2.0), lms_dt_tensor), 3.0)
            lms_dt2_tensor = torch.mul(torch.mul(torch.pow(lms_dt_tensor, 2.0), root_lms_tensor), 6.0)

            rgb_matrix = torch.tensor(
                [
                    [4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010],
                ]
            )

            rgb_tensor = torch.sub(torch.einsum("qt, twh -> qwh", rgb_matrix, lms_tensor), 1.0)
            rgb_tensor_1 = torch.einsum("qt, twh -> qwh", rgb_matrix, lms_dt_tensor_1)
            rgb_tensor_2 = torch.einsum("qt, twh -> qwh", rgb_matrix, lms_dt2_tensor)

            u_rgb_tensor = torch.div(
                rgb_tensor_1,
                torch.sub(torch.pow(rgb_tensor_1, 2.0), torch.mul(torch.mul(rgb_tensor, rgb_tensor_2), 0.5)),
            )

            t_rgb_tensor = torch.mul(torch.mul(rgb_tensor, -1.0), u_rgb_tensor)

            max_floats = torch.mul(MAX_FLOAT, torch.ones(t_rgb_tensor.shape))

            t_rgb_tensor = torch.where(torch.lt(u_rgb_tensor, 0.0), max_floats, t_rgb_tensor)

            t_tensor = torch.where(
                torch.gt(cond_tensor, 0.0), torch.add(t_tensor, t_rgb_tensor.min(0).values), t_tensor
            )

    return t_tensor


def gamut_clip_tensor(rgb_l_tensor: torch.Tensor, alpha: float = 0.05, steps: int = 1, steps_outer: int = 1):
    """Adaptively compress out-of-gamut linear-light sRGB image tensor colors into gamut"""

    lab_tensor = oklab_from_linear_srgb(rgb_l_tensor)
    epsilon = 0.00001
    chroma_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1, :, :], 2.0), torch.pow(lab_tensor[2, :, :], 2.0)))
    chroma_tensor = torch.where(torch.lt(chroma_tensor, epsilon), epsilon, chroma_tensor)

    units_ab_tensor = torch.div(lab_tensor[1:, :, :], chroma_tensor)

    l_d_tensor = torch.sub(lab_tensor[0], 0.5)
    e_1_tensor = torch.add(torch.add(torch.abs(l_d_tensor), torch.mul(chroma_tensor, alpha)), 0.5)
    l_0_tensor = torch.mul(
        torch.add(
            torch.mul(
                torch.sign(l_d_tensor),
                torch.sub(
                    e_1_tensor, torch.sqrt(torch.sub(torch.pow(e_1_tensor, 2.0), torch.mul(torch.abs(l_d_tensor), 2.0)))
                ),
            ),
            1.0,
        ),
        0.5,
    )

    t_tensor = find_gamut_intersection_tensor(
        units_ab_tensor, lab_tensor[0, :, :], chroma_tensor, l_0_tensor, steps=steps, steps_outer=steps_outer
    )
    l_clipped_tensor = torch.add(
        torch.mul(l_0_tensor, torch.add(torch.mul(t_tensor, -1), 1.0)), torch.mul(t_tensor, lab_tensor[0, :, :])
    )
    c_clipped_tensor = torch.mul(t_tensor, chroma_tensor)

    return torch.where(
        torch.logical_or(torch.gt(rgb_l_tensor.max(0).values, 1.0), torch.lt(rgb_l_tensor.min(0).values, 0.0)),
        linear_srgb_from_oklab(
            torch.stack(
                [
                    l_clipped_tensor,
                    torch.mul(c_clipped_tensor, units_ab_tensor[0, :, :]),
                    torch.mul(c_clipped_tensor, units_ab_tensor[1, :, :]),
                ]
            )
        ),
        rgb_l_tensor,
    )


def st_cusps_from_lc(lc_cusps_tensor: torch.Tensor):
    """Alternative cusp representation with max C as min(S*L, T*(1-L))"""

    return torch.stack(
        [
            torch.div(lc_cusps_tensor[1, :, :], lc_cusps_tensor[0, :, :]),
            torch.div(lc_cusps_tensor[1, :, :], torch.add(torch.mul(lc_cusps_tensor[0, :, :], -1.0), 1)),
        ]
    )


def ok_l_r_from_l_tensor(x_tensor: torch.Tensor):
    """Lightness compensated (Y=1) estimate of lightness in Oklab space"""

    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1.0 + k_1) / (1.0 + k_2)
    #  0.5f * (k_3 * x - k_1 + sqrtf((k_3 * x - k_1) * (k_3 * x - k_1) + 4 * k_2 * k_3 * x));

    return torch.mul(
        torch.add(
            torch.sub(torch.mul(x_tensor, k_3), k_1),
            torch.sqrt(
                torch.add(
                    torch.pow(torch.sub(torch.mul(x_tensor, k_3), k_1), 2.0),
                    torch.mul(torch.mul(torch.mul(x_tensor, k_3), k_2), 4.0),
                )
            ),
        ),
        0.5,
    )


def ok_l_from_lr_tensor(x_tensor: torch.Tensor):
    """Get uncompensated Oklab lightness from the lightness compensated version"""

    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1.0 + k_1) / (1.0 + k_2)

    # (x * x + k_1 * x) / (k_3 * (x + k_2))
    return torch.div(
        torch.add(torch.pow(x_tensor, 2.0), torch.mul(x_tensor, k_1)), torch.mul(torch.add(x_tensor, k_2), k_3)
    )


def srgb_from_okhsv(okhsv_tensor: torch.Tensor, alpha: float = 0.05, steps: int = 1):
    """Get standard gamma-corrected sRGB from an Okhsv image tensor"""

    okhsv_tensor = okhsv_tensor.clamp(0.0, 1.0)

    units_ab_tensor = torch.stack(
        [torch.cos(torch.mul(okhsv_tensor[0, :, :], 2.0 * PI)), torch.sin(torch.mul(okhsv_tensor[0, :, :], 2.0 * PI))]
    )
    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)
    st_max_tensor = st_cusps_from_lc(lc_cusps_tensor)
    s_0_tensor = torch.tensor(0.5).expand(st_max_tensor.shape[1:])
    k_tensor = torch.add(torch.mul(torch.div(s_0_tensor, st_max_tensor[0, :, :]), -1.0), 1)

    # First compute L and V assuming a perfect triangular gamut
    lc_v_base_tensor = torch.add(
        s_0_tensor,
        torch.sub(
            st_max_tensor[1, :, :], torch.mul(st_max_tensor[1, :, :], torch.mul(k_tensor, okhsv_tensor[1, :, :]))
        ),
    )
    lc_v_tensor = torch.stack(
        [
            torch.add(torch.div(torch.mul(torch.mul(okhsv_tensor[1, :, :], s_0_tensor), -1.0), lc_v_base_tensor), 1.0),
            torch.div(
                torch.mul(torch.mul(okhsv_tensor[1, :, :], st_max_tensor[1, :, :]), s_0_tensor), lc_v_base_tensor
            ),
        ]
    )

    lc_tensor = torch.mul(okhsv_tensor[2, :, :], lc_v_tensor)

    l_vt_tensor = ok_l_from_lr_tensor(lc_v_tensor[0, :, :])
    c_vt_tensor = torch.mul(lc_v_tensor[1, :, :], torch.div(l_vt_tensor, lc_v_tensor[0, :, :]))

    l_new_tensor = ok_l_from_lr_tensor(lc_tensor[0, :, :])
    lc_tensor[1, :, :] = torch.mul(lc_tensor[1, :, :], torch.div(l_new_tensor, lc_tensor[0, :, :]))
    lc_tensor[0, :, :] = l_new_tensor

    rgb_scale_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                l_vt_tensor,
                torch.mul(units_ab_tensor[0, :, :], c_vt_tensor),
                torch.mul(units_ab_tensor[1, :, :], c_vt_tensor),
            ]
        )
    )

    scale_l_tensor = torch.pow(
        torch.div(1.0, torch.max(rgb_scale_tensor.max(0).values, torch.zeros(rgb_scale_tensor.shape[1:]))), 1.0 / 3.0
    )
    lc_tensor = torch.mul(lc_tensor, scale_l_tensor.expand(lc_tensor.shape))

    rgb_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                lc_tensor[0, :, :],
                torch.mul(units_ab_tensor[0, :, :], lc_tensor[1, :, :]),
                torch.mul(units_ab_tensor[1, :, :], lc_tensor[1, :, :]),
            ]
        )
    )

    rgb_tensor = srgb_from_linear_srgb(rgb_tensor, alpha=alpha, steps=steps)
    return torch.where(torch.isnan(rgb_tensor), 0.0, rgb_tensor).clamp(0.0, 1.0)


def okhsv_from_srgb(srgb_tensor: torch.Tensor, steps: int = 1):
    """Get Okhsv image tensor from standard gamma-corrected sRGB"""

    lab_tensor = oklab_from_linear_srgb(linear_srgb_from_srgb(srgb_tensor))

    c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1, :, :], 2.0), torch.pow(lab_tensor[2, :, :], 2.0)))
    units_ab_tensor = torch.div(lab_tensor[1:, :, :], c_tensor)

    h_tensor = torch.add(
        torch.div(
            torch.mul(
                torch.atan2(
                    torch.mul(lab_tensor[2, :, :], -1.0),
                    torch.mul(
                        lab_tensor[1, :, :],
                        -1,
                    ),
                ),
                0.5,
            ),
            PI,
        ),
        0.5,
    )

    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)
    st_max_tensor = st_cusps_from_lc(lc_cusps_tensor)
    s_0_tensor = torch.tensor(0.5).expand(st_max_tensor.shape[1:])
    k_tensor = torch.add(torch.mul(torch.div(s_0_tensor, st_max_tensor[0, :, :]), -1.0), 1)

    t_tensor = torch.div(
        st_max_tensor[1, :, :], torch.add(c_tensor, torch.mul(lab_tensor[0, :, :], st_max_tensor[1, :, :]))
    )

    l_v_tensor = torch.mul(t_tensor, lab_tensor[0, :, :])
    c_v_tensor = torch.mul(t_tensor, c_tensor)

    l_vt_tensor = ok_l_from_lr_tensor(l_v_tensor)
    c_vt_tensor = torch.mul(c_v_tensor, torch.div(l_vt_tensor, l_v_tensor))

    rgb_scale_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                l_vt_tensor,
                torch.mul(units_ab_tensor[0, :, :], c_vt_tensor),
                torch.mul(units_ab_tensor[1, :, :], c_vt_tensor),
            ]
        )
    )

    scale_l_tensor = torch.pow(
        torch.div(1.0, torch.max(rgb_scale_tensor.max(0).values, torch.zeros(rgb_scale_tensor.shape[1:]))), 1.0 / 3.0
    )

    lab_tensor[0, :, :] = torch.div(lab_tensor[0, :, :], scale_l_tensor)
    c_tensor = torch.div(c_tensor, scale_l_tensor)

    c_tensor = torch.mul(c_tensor, torch.div(ok_l_r_from_l_tensor(lab_tensor[0, :, :]), lab_tensor[0, :, :]))
    lab_tensor[0, :, :] = ok_l_r_from_l_tensor(lab_tensor[0, :, :])

    v_tensor = torch.div(lab_tensor[0, :, :], l_v_tensor)
    s_tensor = torch.div(
        torch.mul(torch.add(s_0_tensor, st_max_tensor[1, :, :]), c_v_tensor),
        torch.add(
            torch.mul(st_max_tensor[1, :, :], s_0_tensor),
            torch.mul(st_max_tensor[1, :, :], torch.mul(k_tensor, c_v_tensor)),
        ),
    )

    hsv_tensor = torch.stack([h_tensor, s_tensor, v_tensor])
    return torch.where(torch.isnan(hsv_tensor), 0.0, hsv_tensor).clamp(0.0, 1.0)


def get_st_mid_tensor(units_ab_tensor: torch.Tensor):
    """Returns a smooth approximation of cusp, where st_mid < st_max"""

    return torch.stack(
        [
            torch.add(
                torch.div(
                    1.0,
                    torch.add(
                        torch.add(
                            torch.mul(units_ab_tensor[1, :, :], 4.15901240),
                            torch.mul(
                                units_ab_tensor[0, :, :],
                                torch.add(
                                    torch.add(
                                        torch.mul(units_ab_tensor[1, :, :], 1.75198401),
                                        torch.mul(
                                            units_ab_tensor[0, :, :],
                                            torch.add(
                                                torch.add(
                                                    torch.mul(units_ab_tensor[1, :, :], -10.02301043),
                                                    torch.mul(
                                                        units_ab_tensor[0, :, :],
                                                        torch.add(
                                                            torch.add(
                                                                torch.mul(units_ab_tensor[1, :, :], 5.38770819),
                                                                torch.mul(units_ab_tensor[0, :, :], 4.69891013),
                                                            ),
                                                            -4.24894561,
                                                        ),
                                                    ),
                                                ),
                                                -2.13704948,
                                            ),
                                        ),
                                    ),
                                    -2.19557347,
                                ),
                            ),
                        ),
                        7.44778970,
                    ),
                ),
                0.11516993,
            ),
            torch.add(
                torch.div(
                    1.0,
                    torch.add(
                        torch.add(
                            torch.mul(units_ab_tensor[1, :, :], -0.68124379),
                            torch.mul(
                                units_ab_tensor[0, :, :],
                                torch.add(
                                    torch.add(
                                        torch.mul(units_ab_tensor[1, :, :], 0.90148123),
                                        torch.mul(
                                            units_ab_tensor[0, :, :],
                                            torch.add(
                                                torch.add(
                                                    torch.mul(units_ab_tensor[1, :, :], 0.61223990),
                                                    torch.mul(
                                                        units_ab_tensor[0, :, :],
                                                        torch.add(
                                                            torch.add(
                                                                torch.mul(units_ab_tensor[1, :, :], -0.45399568),
                                                                torch.mul(units_ab_tensor[0, :, :], -0.14661872),
                                                            ),
                                                            0.00299215,
                                                        ),
                                                    ),
                                                ),
                                                -0.27087943,
                                            ),
                                        ),
                                    ),
                                    0.40370612,
                                ),
                            ),
                        ),
                        1.61320320,
                    ),
                ),
                0.11239642,
            ),
        ]
    )


def get_cs_tensor(
    l_tensor: torch.Tensor, units_ab_tensor: torch.Tensor, steps: int = 1, steps_outer: int = 1
):  # -> [C_0, C_mid, C_max]
    """Arrange minimum, midpoint, and max chroma values from tensors of luminance and ab unit vectors"""

    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)

    c_max_tensor = find_gamut_intersection_tensor(
        units_ab_tensor,
        l_tensor,
        torch.ones(l_tensor.shape),
        l_tensor,
        lc_cusps_tensor=lc_cusps_tensor,
        steps=steps,
        steps_outer=steps_outer,
    )
    st_max_tensor = st_cusps_from_lc(lc_cusps_tensor)

    k_tensor = torch.div(
        c_max_tensor,
        torch.min(
            torch.mul(l_tensor, st_max_tensor[0, :, :]),
            torch.mul(torch.add(torch.mul(l_tensor, -1.0), 1.0), st_max_tensor[1, :, :]),
        ),
    )

    st_mid_tensor = get_st_mid_tensor(units_ab_tensor)
    c_a_tensor = torch.mul(l_tensor, st_mid_tensor[0, :, :])
    c_b_tensor = torch.mul(torch.add(torch.mul(l_tensor, -1.0), 1.0), st_mid_tensor[1, :, :])
    c_mid_tensor = torch.mul(
        torch.mul(
            k_tensor,
            torch.sqrt(
                torch.sqrt(
                    torch.div(
                        1.0,
                        torch.add(
                            torch.div(1.0, torch.pow(c_a_tensor, 4.0)), torch.div(1.0, torch.pow(c_b_tensor, 4.0))
                        ),
                    )
                )
            ),
        ),
        0.9,
    )

    c_a_tensor = torch.mul(l_tensor, 0.4)
    c_b_tensor = torch.mul(torch.add(torch.mul(l_tensor, -1.0), 1.0), 0.8)
    c_0_tensor = torch.sqrt(
        torch.div(
            1.0, torch.add(torch.div(1.0, torch.pow(c_a_tensor, 2.0)), torch.div(1.0, torch.pow(c_b_tensor, 2.0)))
        )
    )

    return torch.stack([c_0_tensor, c_mid_tensor, c_max_tensor])


def srgb_from_okhsl(hsl_tensor: torch.Tensor, alpha: float = 0.05, steps: int = 1, steps_outer: int = 1):
    """Get gamma-corrected sRGB from an Okhsl image tensor"""

    hsl_tensor = hsl_tensor.clamp(0.0, 1.0)

    l_ones_mask = torch.eq(hsl_tensor[2, :, :], 1.0)
    l_zeros_mask = torch.eq(hsl_tensor[2, :, :], 0.0)
    l_ones_mask = l_ones_mask.expand(hsl_tensor.shape)
    l_zeros_mask = l_zeros_mask.expand(hsl_tensor.shape)
    calc_rgb_mask = torch.logical_not(torch.logical_or(l_ones_mask, l_zeros_mask))

    rgb_tensor = torch.empty(hsl_tensor.shape)
    rgb_tensor = torch.where(l_ones_mask, 1.0, torch.where(l_zeros_mask, 0.0, rgb_tensor))

    units_ab_tensor = torch.stack(
        [torch.cos(torch.mul(hsl_tensor[0, :, :], 2.0 * PI)), torch.sin(torch.mul(hsl_tensor[0, :, :], 2.0 * PI))]
    )
    l_tensor = ok_l_from_lr_tensor(hsl_tensor[2, :, :])

    # {C_0, C_mid, C_max}
    cs_tensor = get_cs_tensor(l_tensor, units_ab_tensor, steps=steps, steps_outer=steps_outer)

    mid = 0.8
    mid_inv = 1.25

    s_lt_mid_mask = torch.lt(hsl_tensor[1, :, :], mid)
    t_tensor = torch.where(
        s_lt_mid_mask,
        torch.mul(hsl_tensor[1, :, :], mid_inv),
        torch.div(torch.sub(hsl_tensor[1, :, :], mid), 1.0 - mid),
    )
    k_1_tensor = torch.where(
        s_lt_mid_mask,
        torch.mul(cs_tensor[0, :, :], mid),
        torch.div(
            torch.mul(torch.mul(torch.pow(cs_tensor[1, :, :], 2.0), mid_inv**2.0), 1.0 - mid), cs_tensor[0, :, :]
        ),
    )
    k_2_tensor = torch.where(
        s_lt_mid_mask,
        torch.add(torch.mul(torch.div(k_1_tensor, cs_tensor[1, :, :]), -1.0), 1.0),
        torch.add(torch.mul(torch.div(k_1_tensor, torch.sub(cs_tensor[2, :, :], cs_tensor[1, :, :])), -1.0), 1.0),
    )

    c_tensor = torch.div(
        torch.mul(t_tensor, k_1_tensor), torch.add(torch.mul(torch.mul(k_2_tensor, t_tensor), -1.0), 1.0)
    )
    c_tensor = torch.where(s_lt_mid_mask, c_tensor, torch.add(cs_tensor[1, :, :], c_tensor))

    rgb_tensor = torch.where(
        calc_rgb_mask,
        linear_srgb_from_oklab(
            torch.stack(
                [l_tensor, torch.mul(c_tensor, units_ab_tensor[0, :, :]), torch.mul(c_tensor, units_ab_tensor[1, :, :])]
            )
        ),
        rgb_tensor,
    )

    rgb_tensor = srgb_from_linear_srgb(rgb_tensor, alpha=alpha, steps=steps)
    return torch.where(torch.isnan(rgb_tensor), 0.0, rgb_tensor).clamp(0.0, 1.0)


def okhsl_from_srgb(rgb_tensor: torch.Tensor, steps: int = 1, steps_outer: int = 1):
    """Get an Okhsl image tensor from gamma-corrected sRGB"""

    lab_tensor = oklab_from_linear_srgb(linear_srgb_from_srgb(rgb_tensor))

    c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1, :, :], 2.0), torch.pow(lab_tensor[2, :, :], 2.0)))
    units_ab_tensor = torch.stack([torch.div(lab_tensor[1, :, :], c_tensor), torch.div(lab_tensor[2, :, :], c_tensor)])

    h_tensor = torch.add(
        torch.div(
            torch.mul(torch.atan2(torch.mul(lab_tensor[2, :, :], -1.0), torch.mul(lab_tensor[1, :, :], -1.0)), 0.5), PI
        ),
        0.5,
    )

    # {C_0, C_mid, C_max}
    cs_tensor = get_cs_tensor(lab_tensor[0, :, :], units_ab_tensor, steps=1, steps_outer=1)

    mid = 0.8
    mid_inv = 1.25

    c_lt_c_mid_mask = torch.lt(c_tensor, cs_tensor[1, :, :])
    k_1_tensor = torch.where(
        c_lt_c_mid_mask,
        torch.mul(cs_tensor[0, :, :], mid),
        torch.div(torch.mul(torch.mul(torch.pow(cs_tensor[1, :, :], 2.0), mid_inv**2), 1.0 - mid), cs_tensor[0, :, :]),
    )
    k_2_tensor = torch.where(
        c_lt_c_mid_mask,
        torch.add(torch.mul(torch.div(k_1_tensor, cs_tensor[1, :, :]), -1.0), 1.0),
        torch.add(torch.mul(torch.div(k_1_tensor, torch.sub(cs_tensor[2, :, :], cs_tensor[1, :, :])), -1.0), 1.0),
    )
    t_tensor = torch.where(
        c_lt_c_mid_mask,
        torch.div(c_tensor, torch.add(k_1_tensor, torch.mul(k_2_tensor, c_tensor))),
        torch.div(
            torch.sub(c_tensor, cs_tensor[1, :, :]),
            torch.add(k_1_tensor, torch.mul(k_2_tensor, torch.sub(c_tensor, cs_tensor[1, :, :]))),
        ),
    )

    s_tensor = torch.where(c_lt_c_mid_mask, torch.mul(t_tensor, mid), torch.add(torch.mul(t_tensor, 1.0 - mid), mid))
    l_tensor = ok_l_r_from_l_tensor(lab_tensor[0, :, :])

    hsl_tensor = torch.stack([h_tensor, s_tensor, l_tensor])
    return torch.where(torch.isnan(hsl_tensor), 0.0, hsl_tensor).clamp(0.0, 1.0)


def xyz_from_srgb(rgb_l_tensor: torch.Tensor):
    conversion_matrix = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    return torch.einsum("zc, cwh -> zwh", conversion_matrix, rgb_l_tensor)


def lab_from_xyz_helper(channel_illuminant_quotient_matrix: torch.Tensor):
    delta = 6.0 / 29.0

    return torch.where(
        torch.gt(channel_illuminant_quotient_matrix, delta**3.0),
        torch.pow(channel_illuminant_quotient_matrix, 1.0 / 3.0),
        torch.add(torch.div(channel_illuminant_quotient_matrix, 3.0 * (delta**2.0)), 4.0 / 29.0),
    )


def lab_from_xyz(xyz_tensor: torch.Tensor, reference_illuminant: Literal["D65", "D50"] = "D65"):
    illuminant = {"D65": [95.0489, 100.0, 108.8840], "D50": [96.4212, 100.0, 82.5188]}[reference_illuminant]
    l_tensor = torch.sub(torch.mul(lab_from_xyz_helper(torch.div(xyz_tensor[1, :, :], illuminant[1])), 116.0), 16.0)
    a_tensor = torch.mul(
        torch.sub(
            lab_from_xyz_helper(torch.div(xyz_tensor[0, :, :], illuminant[0])),
            lab_from_xyz_helper(torch.div(xyz_tensor[1, :, :], illuminant[1])),
        ),
        500.0,
    )
    b_tensor = torch.mul(
        torch.sub(
            lab_from_xyz_helper(torch.div(xyz_tensor[1, :, :], illuminant[1])),
            lab_from_xyz_helper(torch.div(xyz_tensor[2, :, :], illuminant[2])),
        ),
        200.0,
    )

    return torch.stack([l_tensor, a_tensor, b_tensor])


######################################################################################\
# HSL Code derived from CPython colorsys source code [license text below]
def hsl_from_srgb(rgb_tensor: torch.Tensor):
    """Get HSL image tensor from standard gamma-corrected sRGB"""
    c_max_tensor = rgb_tensor.max(0).values
    c_min_tensor = rgb_tensor.min(0).values
    c_sum_tensor = torch.add(c_max_tensor, c_min_tensor)
    c_range_tensor = torch.sub(c_max_tensor, c_min_tensor)
    l_tensor = torch.div(c_sum_tensor, 2.0)
    s_tensor = torch.where(
        torch.eq(c_max_tensor, c_min_tensor),
        0.0,
        torch.where(
            torch.lt(l_tensor, 0.5),
            torch.div(c_range_tensor, c_sum_tensor),
            torch.div(c_range_tensor, torch.add(torch.mul(torch.add(c_max_tensor, c_min_tensor), -1.0), 2.0)),
        ),
    )
    rgb_c_tensor = torch.div(
        torch.sub(c_max_tensor.expand(rgb_tensor.shape), rgb_tensor), c_range_tensor.expand(rgb_tensor.shape)
    )
    h_tensor = torch.where(
        torch.eq(c_max_tensor, c_min_tensor),
        0.0,
        torch.where(
            torch.eq(rgb_tensor[0, :, :], c_max_tensor),
            torch.sub(rgb_c_tensor[2, :, :], rgb_c_tensor[1, :, :]),
            torch.where(
                torch.eq(rgb_tensor[1, :, :], c_max_tensor),
                torch.add(torch.sub(rgb_c_tensor[0, :, :], rgb_c_tensor[2, :, :]), 2.0),
                torch.add(torch.sub(rgb_c_tensor[1, :, :], rgb_c_tensor[0, :, :]), 4.0),
            ),
        ),
    )
    h_tensor = torch.remainder(torch.div(h_tensor, 6.0), 1.0)
    return torch.stack([h_tensor, s_tensor, l_tensor])


def srgb_from_hsl(hsl_tensor: torch.Tensor):
    """Get gamma-corrected sRGB from an HSL image tensor"""
    hsl_tensor = hsl_tensor.clamp(0.0, 1.0)
    rgb_tensor = torch.empty(hsl_tensor.shape)
    s_0_mask = torch.eq(hsl_tensor[1, :, :], 0.0)
    rgb_tensor = torch.where(
        s_0_mask.expand(rgb_tensor.shape), hsl_tensor[2, :, :].expand(hsl_tensor.shape), rgb_tensor
    )
    m2_tensor = torch.where(
        torch.le(hsl_tensor[2, :, :], 0.5),
        torch.mul(hsl_tensor[2, :, :], torch.add(hsl_tensor[1, :, :], 1.0)),
        torch.sub(
            torch.add(hsl_tensor[2, :, :], hsl_tensor[1, :, :]), torch.mul(hsl_tensor[2, :, :], hsl_tensor[1, :, :])
        ),
    )
    m1_tensor = torch.sub(torch.mul(hsl_tensor[2, :, :], 2.0), m2_tensor)

    def hsl_values(m1_tensor: torch.Tensor, m2_tensor: torch.Tensor, h_tensor: torch.Tensor):
        """Helper for computing output components"""

        h_tensor = torch.remainder(h_tensor, 1.0)
        result_tensor = m1_tensor.clone()
        result_tensor = torch.where(
            torch.lt(h_tensor, 1.0 / 6.0),
            torch.add(m1_tensor, torch.mul(torch.sub(m2_tensor, m1_tensor), torch.mul(h_tensor, 6.0))),
            torch.where(
                torch.lt(h_tensor, 0.5),
                m2_tensor,
                torch.where(
                    torch.lt(h_tensor, 2.0 / 3.0),
                    torch.add(
                        m1_tensor,
                        torch.mul(
                            torch.sub(m2_tensor, m1_tensor),
                            torch.mul(torch.add(torch.mul(h_tensor, -1.0), 2.0 / 3.0), 6.0),
                        ),
                    ),
                    result_tensor,
                ),
            ),
        )
        return result_tensor

    return torch.stack(
        [
            hsl_values(m1_tensor, m2_tensor, torch.add(hsl_tensor[0, :, :], 1.0 / 3.0)),
            hsl_values(m1_tensor, m2_tensor, hsl_tensor[0, :, :]),
            hsl_values(m1_tensor, m2_tensor, torch.sub(hsl_tensor[0, :, :], 1.0 / 3.0)),
        ]
    )


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

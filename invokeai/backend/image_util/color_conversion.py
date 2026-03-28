from math import pi as PI

import torch

MAX_FLOAT = torch.finfo(torch.tensor(1.0).dtype).max
_SRGB_TO_LINEAR_THRESHOLD = 0.0404482362771082
_LINEAR_TO_SRGB_THRESHOLD = 0.0031308
_SRGB_TO_XYZ_D65_MATRIX = (
    (0.4124, 0.3576, 0.1805),
    (0.2126, 0.7152, 0.0722),
    (0.0193, 0.1192, 0.9505),
)
_XYZ_D65_TO_SRGB_MATRIX = (
    (3.2406255, -1.5372080, -0.4986286),
    (-0.9689307, 1.8757561, 0.0415175),
    (0.0557101, -0.2040211, 1.0569959),
)
_BRADFORD_MATRIX = (
    (0.8951, 0.2664, -0.1614),
    (-0.7502, 1.7135, 0.0367),
    (0.0389, -0.0685, 1.0296),
)
_BRADFORD_INVERSE_MATRIX = (
    (0.9869929, -0.1470543, 0.1599627),
    (0.4323053, 0.5183603, 0.0492912),
    (-0.0085287, 0.0400428, 0.9684867),
)
_REFERENCE_ILLUMINANTS = {
    "D65": (0.950489, 1.0, 1.088840),
    "D50": (0.964212, 1.0, 0.825188),
}
_LINEAR_SRGB_TO_OKLAB_LMS_MATRIX = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_LMS_CUBE_ROOT_TO_OKLAB_MATRIX = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
_OKLAB_TO_LMS_CUBE_ROOT_MATRIX = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
_LMS_TO_LINEAR_SRGB_MATRIX = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)


def _require_color_tensor(color_tensor: torch.Tensor) -> torch.Tensor:
    if color_tensor.ndim != 3 or color_tensor.shape[0] != 3:
        raise ValueError("color_tensor must be a 3xHxW tensor")
    return color_tensor


def _require_reference_illuminant(reference_illuminant: str) -> str:
    normalized = reference_illuminant.upper()
    if normalized not in _REFERENCE_ILLUMINANTS:
        raise ValueError(f"Unsupported reference_illuminant: {reference_illuminant}")
    return normalized


def _full_like_spatial(reference_tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
    return torch.full(
        reference_tensor.shape[1:], fill_value, dtype=reference_tensor.dtype, device=reference_tensor.device
    )


def _degrees_from_unit_hue(unit_hue_tensor: torch.Tensor) -> torch.Tensor:
    return torch.remainder(unit_hue_tensor * 360.0, 360.0)


def _unit_hue_from_degrees(hue_tensor: torch.Tensor) -> torch.Tensor:
    return torch.remainder(hue_tensor, 360.0) / 360.0


def _matrix_tensor(matrix: tuple[tuple[float, ...], ...], reference_tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(matrix, dtype=reference_tensor.dtype, device=reference_tensor.device)


def _apply_matrix(matrix: tuple[tuple[float, ...], ...], color_tensor: torch.Tensor) -> torch.Tensor:
    return torch.einsum("rc,cwh->rwh", _matrix_tensor(matrix, color_tensor), color_tensor)


def _reference_white_tensor(reference_illuminant: str, reference_tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        _REFERENCE_ILLUMINANTS[reference_illuminant], dtype=reference_tensor.dtype, device=reference_tensor.device
    ).view(3, 1, 1)


def _adapt_xyz(xyz_tensor: torch.Tensor, source_illuminant: str, target_illuminant: str) -> torch.Tensor:
    xyz_tensor = _require_color_tensor(xyz_tensor)
    source_illuminant = _require_reference_illuminant(source_illuminant)
    target_illuminant = _require_reference_illuminant(target_illuminant)
    if source_illuminant == target_illuminant:
        return xyz_tensor
    source_white_lms = _apply_matrix(_BRADFORD_MATRIX, _reference_white_tensor(source_illuminant, xyz_tensor))
    target_white_lms = _apply_matrix(_BRADFORD_MATRIX, _reference_white_tensor(target_illuminant, xyz_tensor))
    xyz_lms = _apply_matrix(_BRADFORD_MATRIX, xyz_tensor)
    adapted_lms = xyz_lms * (target_white_lms / source_white_lms)
    return _apply_matrix(_BRADFORD_INVERSE_MATRIX, adapted_lms)


def srgb_from_linear_srgb(linear_srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW linear-light sRGB tensor in [0, 1] to gamma-corrected sRGB."""

    linear_srgb_tensor = _require_color_tensor(linear_srgb_tensor)
    linear_srgb_tensor = linear_srgb_tensor.clamp(0.0, 1.0)
    return torch.where(
        linear_srgb_tensor <= _LINEAR_TO_SRGB_THRESHOLD,
        linear_srgb_tensor * 12.92,
        1.055 * torch.pow(linear_srgb_tensor, 1.0 / 2.4) - 0.055,
    )


def linear_srgb_from_srgb(srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor in [0, 1] to linear-light sRGB."""

    srgb_tensor = _require_color_tensor(srgb_tensor)
    return torch.where(
        srgb_tensor <= _SRGB_TO_LINEAR_THRESHOLD,
        srgb_tensor / 12.92,
        torch.pow((srgb_tensor + 0.055) / 1.055, 2.4),
    )


def xyz_from_linear_srgb(linear_srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW linear-light sRGB tensor to normalized XYZ, where D65 white is approximately 1.0."""

    linear_srgb_tensor = _require_color_tensor(linear_srgb_tensor)
    return _apply_matrix(_SRGB_TO_XYZ_D65_MATRIX, linear_srgb_tensor)


def linear_srgb_from_xyz(xyz_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW normalized XYZ tensor, where D65 white is approximately 1.0, to linear-light sRGB."""

    xyz_tensor = _require_color_tensor(xyz_tensor)
    return _apply_matrix(_XYZ_D65_TO_SRGB_MATRIX, xyz_tensor)


def xyz_from_srgb(srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor to normalized XYZ, where D65 white is approximately 1.0."""

    srgb_tensor = _require_color_tensor(srgb_tensor)
    return xyz_from_linear_srgb(linear_srgb_from_srgb(srgb_tensor))


def srgb_from_xyz(xyz_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW normalized XYZ tensor, where D65 white is approximately 1.0, to gamma-corrected sRGB."""

    xyz_tensor = _require_color_tensor(xyz_tensor)
    return srgb_from_linear_srgb(linear_srgb_from_xyz(xyz_tensor))


def xyz_d65_to_d50(xyz_tensor: torch.Tensor) -> torch.Tensor:
    """Adapt a 3xHxW normalized XYZ tensor from D65 to D50 using Bradford chromatic adaptation."""

    return _adapt_xyz(xyz_tensor, source_illuminant="D65", target_illuminant="D50")


def xyz_d50_to_d65(xyz_tensor: torch.Tensor) -> torch.Tensor:
    """Adapt a 3xHxW normalized XYZ tensor from D50 to D65 using Bradford chromatic adaptation."""

    return _adapt_xyz(xyz_tensor, source_illuminant="D50", target_illuminant="D65")


def _lab_from_xyz_helper(channel_illuminant_quotient_tensor: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(
        torch.gt(channel_illuminant_quotient_tensor, delta**3.0),
        torch.pow(channel_illuminant_quotient_tensor, 1.0 / 3.0),
        torch.add(torch.div(channel_illuminant_quotient_tensor, 3.0 * (delta**2.0)), 4.0 / 29.0),
    )


def _xyz_from_lab_helper(channel_tensor: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(
        torch.gt(channel_tensor, delta),
        torch.pow(channel_tensor, 3.0),
        torch.mul(3.0 * (delta**2.0), torch.sub(channel_tensor, 4.0 / 29.0)),
    )


def lab_from_xyz(xyz_tensor: torch.Tensor, reference_illuminant: str = "D65") -> torch.Tensor:
    """Convert a 3xHxW normalized XYZ tensor to CIELAB using the given reference illuminant."""

    xyz_tensor = _require_color_tensor(xyz_tensor)
    reference_illuminant = _require_reference_illuminant(reference_illuminant)
    illuminant = _reference_white_tensor(reference_illuminant, xyz_tensor)
    l_tensor = torch.sub(torch.mul(_lab_from_xyz_helper(torch.div(xyz_tensor[1, :, :], illuminant[1])), 116.0), 16.0)
    a_tensor = torch.mul(
        torch.sub(
            _lab_from_xyz_helper(torch.div(xyz_tensor[0, :, :], illuminant[0])),
            _lab_from_xyz_helper(torch.div(xyz_tensor[1, :, :], illuminant[1])),
        ),
        500.0,
    )
    b_tensor = torch.mul(
        torch.sub(
            _lab_from_xyz_helper(torch.div(xyz_tensor[1, :, :], illuminant[1])),
            _lab_from_xyz_helper(torch.div(xyz_tensor[2, :, :], illuminant[2])),
        ),
        200.0,
    )
    return torch.stack([l_tensor, a_tensor, b_tensor])


def xyz_from_lab(lab_tensor: torch.Tensor, reference_illuminant: str = "D65") -> torch.Tensor:
    """Convert a 3xHxW CIELAB tensor to normalized XYZ using the given reference illuminant."""

    lab_tensor = _require_color_tensor(lab_tensor)
    reference_illuminant = _require_reference_illuminant(reference_illuminant)
    illuminant = _reference_white_tensor(reference_illuminant, lab_tensor)
    fy_tensor = (lab_tensor[0, :, :] + 16.0) / 116.0
    fx_tensor = fy_tensor + (lab_tensor[1, :, :] / 500.0)
    fz_tensor = fy_tensor - (lab_tensor[2, :, :] / 200.0)
    return torch.stack(
        [
            illuminant[0] * _xyz_from_lab_helper(fx_tensor),
            illuminant[1] * _xyz_from_lab_helper(fy_tensor),
            illuminant[2] * _xyz_from_lab_helper(fz_tensor),
        ]
    )


def lab_from_linear_srgb(linear_srgb_tensor: torch.Tensor, reference_illuminant: str = "D65") -> torch.Tensor:
    """Convert a 3xHxW linear-light sRGB tensor to CIELAB using the given reference illuminant."""

    linear_srgb_tensor = _require_color_tensor(linear_srgb_tensor)
    reference_illuminant = _require_reference_illuminant(reference_illuminant)
    xyz_tensor = xyz_from_linear_srgb(linear_srgb_tensor)
    if reference_illuminant != "D65":
        xyz_tensor = _adapt_xyz(xyz_tensor, source_illuminant="D65", target_illuminant=reference_illuminant)
    return lab_from_xyz(xyz_tensor, reference_illuminant=reference_illuminant)


def linear_srgb_from_lab(lab_tensor: torch.Tensor, reference_illuminant: str = "D65") -> torch.Tensor:
    """Convert a 3xHxW CIELAB tensor to linear-light sRGB using the given reference illuminant."""

    lab_tensor = _require_color_tensor(lab_tensor)
    reference_illuminant = _require_reference_illuminant(reference_illuminant)
    xyz_tensor = xyz_from_lab(lab_tensor, reference_illuminant=reference_illuminant)
    if reference_illuminant != "D65":
        xyz_tensor = _adapt_xyz(xyz_tensor, source_illuminant=reference_illuminant, target_illuminant="D65")
    return linear_srgb_from_xyz(xyz_tensor)


def lab_from_srgb(srgb_tensor: torch.Tensor, reference_illuminant: str = "D65") -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor to CIELAB using the given reference illuminant."""

    srgb_tensor = _require_color_tensor(srgb_tensor)
    return lab_from_linear_srgb(linear_srgb_from_srgb(srgb_tensor), reference_illuminant=reference_illuminant)


def srgb_from_lab(lab_tensor: torch.Tensor, reference_illuminant: str = "D65") -> torch.Tensor:
    """Convert a 3xHxW CIELAB tensor to gamma-corrected sRGB using the given reference illuminant."""

    lab_tensor = _require_color_tensor(lab_tensor)
    return srgb_from_linear_srgb(linear_srgb_from_lab(lab_tensor, reference_illuminant=reference_illuminant))


def oklab_from_linear_srgb(linear_srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW linear-light sRGB tensor to Oklab."""

    linear_srgb_tensor = _require_color_tensor(linear_srgb_tensor)
    lms_tensor = _apply_matrix(_LINEAR_SRGB_TO_OKLAB_LMS_MATRIX, linear_srgb_tensor)
    lms_cbrt_tensor = torch.sign(lms_tensor) * torch.pow(torch.abs(lms_tensor), 1.0 / 3.0)
    return _apply_matrix(_LMS_CUBE_ROOT_TO_OKLAB_MATRIX, lms_cbrt_tensor)


def linear_srgb_from_oklab(oklab_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklab tensor to linear-light sRGB."""

    oklab_tensor = _require_color_tensor(oklab_tensor)
    lms_cbrt_tensor = _apply_matrix(_OKLAB_TO_LMS_CUBE_ROOT_MATRIX, oklab_tensor)
    lms_tensor = lms_cbrt_tensor**3
    return _apply_matrix(_LMS_TO_LINEAR_SRGB_MATRIX, lms_tensor)


def oklab_from_srgb(srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor to Oklab."""

    srgb_tensor = _require_color_tensor(srgb_tensor)
    return oklab_from_linear_srgb(linear_srgb_from_srgb(srgb_tensor))


def srgb_from_oklab(oklab_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklab tensor to gamma-corrected sRGB."""

    oklab_tensor = _require_color_tensor(oklab_tensor)
    return srgb_from_linear_srgb(linear_srgb_from_oklab(oklab_tensor))


def oklab_from_xyz(xyz_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW normalized XYZ tensor, where D65 white is approximately 1.0, to Oklab."""

    xyz_tensor = _require_color_tensor(xyz_tensor)
    return oklab_from_linear_srgb(linear_srgb_from_xyz(xyz_tensor))


def xyz_from_oklab(oklab_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklab tensor to normalized XYZ, where D65 white is approximately 1.0."""

    oklab_tensor = _require_color_tensor(oklab_tensor)
    return xyz_from_linear_srgb(linear_srgb_from_oklab(oklab_tensor))


def oklch_from_oklab(oklab_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklab tensor to Oklch, with hue in degrees."""

    oklab_tensor = _require_color_tensor(oklab_tensor)
    lightness = oklab_tensor[0, ...]
    chroma = torch.sqrt(oklab_tensor[1, ...] ** 2 + oklab_tensor[2, ...] ** 2)
    hue = torch.remainder(torch.rad2deg(torch.atan2(oklab_tensor[2, ...], oklab_tensor[1, ...])), 360.0)
    return torch.stack([lightness, chroma, hue])


def oklab_from_oklch(oklch_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklch tensor, with hue in degrees, to Oklab."""

    oklch_tensor = _require_color_tensor(oklch_tensor)
    hue_radians = torch.deg2rad(oklch_tensor[2, ...])
    a_channel = oklch_tensor[1, ...] * torch.cos(hue_radians)
    b_channel = oklch_tensor[1, ...] * torch.sin(hue_radians)
    return torch.stack([oklch_tensor[0, ...], a_channel, b_channel])


def linear_srgb_from_oklch(oklch_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklch tensor directly to linear-light sRGB."""

    oklch_tensor = _require_color_tensor(oklch_tensor)
    return linear_srgb_from_oklab(oklab_from_oklch(oklch_tensor))


def oklch_from_linear_srgb(linear_srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW linear-light sRGB tensor directly to Oklch."""

    linear_srgb_tensor = _require_color_tensor(linear_srgb_tensor)
    return oklch_from_oklab(oklab_from_linear_srgb(linear_srgb_tensor))


def oklch_from_srgb(srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor directly to Oklch."""

    srgb_tensor = _require_color_tensor(srgb_tensor)
    return oklch_from_linear_srgb(linear_srgb_from_srgb(srgb_tensor))


def srgb_from_oklch(oklch_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklch tensor directly to gamma-corrected sRGB."""

    oklch_tensor = _require_color_tensor(oklch_tensor)
    return srgb_from_linear_srgb(linear_srgb_from_oklch(oklch_tensor))


def oklch_from_xyz(xyz_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW normalized XYZ tensor, where D65 white is approximately 1.0, to Oklch."""

    xyz_tensor = _require_color_tensor(xyz_tensor)
    return oklch_from_oklab(oklab_from_xyz(xyz_tensor))


def xyz_from_oklch(oklch_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklch tensor to normalized XYZ, where D65 white is approximately 1.0."""

    oklch_tensor = _require_color_tensor(oklch_tensor)
    return xyz_from_oklab(oklab_from_oklch(oklch_tensor))


def _max_srgb_saturation_tensor(units_ab_tensor: torch.Tensor, steps: int = 1) -> torch.Tensor:
    rgb_k_matrix = torch.tensor(
        [
            [1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245],
            [0.73956515, -0.45954494, 0.08285427, 0.12541070, 0.14503204],
            [1.35733652, -0.00915799, -1.15130210, -0.50559606, 0.00692167],
        ],
        dtype=units_ab_tensor.dtype,
        device=units_ab_tensor.device,
    )
    rgb_w_matrix = _matrix_tensor(_LMS_TO_LINEAR_SRGB_MATRIX, units_ab_tensor)
    cond_r_tensor = torch.add(
        torch.mul(-1.88170328, units_ab_tensor[0, :, :]), torch.mul(-0.80936493, units_ab_tensor[1, :, :])
    )
    cond_g_tensor = torch.add(
        torch.mul(1.81444104, units_ab_tensor[0, :, :]), torch.mul(-1.19445276, units_ab_tensor[1, :, :])
    )
    terms_tensor = torch.stack(
        [
            torch.ones(units_ab_tensor.shape[1:], dtype=units_ab_tensor.dtype, device=units_ab_tensor.device),
            units_ab_tensor[0, :, :],
            units_ab_tensor[1, :, :],
            torch.pow(units_ab_tensor[0, :, :], 2.0),
            torch.mul(units_ab_tensor[0, :, :], units_ab_tensor[1, :, :]),
        ]
    )
    s_tensor = torch.where(
        torch.gt(cond_r_tensor, 1.0),
        torch.einsum("twh, t -> wh", terms_tensor, rgb_k_matrix[0]),
        torch.where(
            torch.gt(cond_g_tensor, 1.0),
            torch.einsum("twh, t -> wh", terms_tensor, rgb_k_matrix[1]),
            torch.einsum("twh, t -> wh", terms_tensor, rgb_k_matrix[2]),
        ),
    )
    k_lms_matrix = _matrix_tensor(tuple(row[1:] for row in _OKLAB_TO_LMS_CUBE_ROOT_MATRIX), units_ab_tensor)
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


def _find_cusp_tensor(units_ab_tensor: torch.Tensor, steps: int = 1) -> torch.Tensor:
    s_cusp_tensor = _max_srgb_saturation_tensor(units_ab_tensor, steps=steps)
    oklab_tensor = torch.stack(
        [
            torch.ones(s_cusp_tensor.shape, dtype=s_cusp_tensor.dtype, device=s_cusp_tensor.device),
            torch.mul(s_cusp_tensor, units_ab_tensor[0, :, :]),
            torch.mul(s_cusp_tensor, units_ab_tensor[1, :, :]),
        ]
    )
    rgb_at_max_tensor = linear_srgb_from_oklab(oklab_tensor)
    l_cusp_tensor = torch.pow(torch.div(1.0, rgb_at_max_tensor.max(0).values), 1.0 / 3.0)
    c_cusp_tensor = torch.mul(l_cusp_tensor, s_cusp_tensor)
    return torch.stack([l_cusp_tensor, c_cusp_tensor])


def _find_gamut_intersection_tensor(
    units_ab_tensor: torch.Tensor,
    l_1_tensor: torch.Tensor,
    c_1_tensor: torch.Tensor,
    l_0_tensor: torch.Tensor,
    steps: int = 1,
    steps_outer: int = 1,
    lc_cusps_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    if lc_cusps_tensor is None:
        lc_cusps_tensor = _find_cusp_tensor(units_ab_tensor, steps=steps)
    cond_tensor = torch.sub(
        torch.mul(torch.sub(l_1_tensor, l_0_tensor), lc_cusps_tensor[1, :, :]),
        torch.mul(torch.sub(lc_cusps_tensor[0, :, :], l_0_tensor), c_1_tensor),
    )
    t_tensor = torch.where(
        torch.le(cond_tensor, 0.0),
        torch.div(
            torch.mul(lc_cusps_tensor[1, :, :], l_0_tensor),
            torch.add(
                torch.mul(c_1_tensor, lc_cusps_tensor[0, :, :]),
                torch.mul(lc_cusps_tensor[1, :, :], torch.sub(l_0_tensor, l_1_tensor)),
            ),
        ),
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
        k_lms_matrix = _matrix_tensor(tuple(row[1:] for row in _OKLAB_TO_LMS_CUBE_ROOT_MATRIX), units_ab_tensor)
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
            rgb_matrix = _matrix_tensor(_LMS_TO_LINEAR_SRGB_MATRIX, units_ab_tensor)
            rgb_tensor = torch.sub(torch.einsum("qt, twh -> qwh", rgb_matrix, lms_tensor), 1.0)
            rgb_tensor_1 = torch.einsum("qt, twh -> qwh", rgb_matrix, lms_dt_tensor_1)
            rgb_tensor_2 = torch.einsum("qt, twh -> qwh", rgb_matrix, lms_dt2_tensor)
            u_rgb_tensor = torch.div(
                rgb_tensor_1,
                torch.sub(torch.pow(rgb_tensor_1, 2.0), torch.mul(torch.mul(rgb_tensor, rgb_tensor_2), 0.5)),
            )
            t_rgb_tensor = torch.mul(torch.mul(rgb_tensor, -1.0), u_rgb_tensor)
            max_floats = torch.mul(
                MAX_FLOAT, torch.ones(t_rgb_tensor.shape, dtype=t_rgb_tensor.dtype, device=t_rgb_tensor.device)
            )
            t_rgb_tensor = torch.where(torch.lt(u_rgb_tensor, 0.0), max_floats, t_rgb_tensor)
            t_tensor = torch.where(
                torch.gt(cond_tensor, 0.0), torch.add(t_tensor, t_rgb_tensor.min(0).values), t_tensor
            )
    return t_tensor


def gamut_clip_tensor(
    rgb_l_tensor: torch.Tensor, alpha: float = 0.05, steps: int = 1, steps_outer: int = 1
) -> torch.Tensor:
    rgb_l_tensor = _require_color_tensor(rgb_l_tensor)
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
    t_tensor = _find_gamut_intersection_tensor(
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


def _st_cusps_from_lc(lc_cusps_tensor: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.div(lc_cusps_tensor[1, :, :], lc_cusps_tensor[0, :, :]),
            torch.div(lc_cusps_tensor[1, :, :], torch.add(torch.mul(lc_cusps_tensor[0, :, :], -1.0), 1)),
        ]
    )


def _ok_l_r_from_l_tensor(x_tensor: torch.Tensor) -> torch.Tensor:
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1.0 + k_1) / (1.0 + k_2)
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


def _ok_l_from_lr_tensor(x_tensor: torch.Tensor) -> torch.Tensor:
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1.0 + k_1) / (1.0 + k_2)
    return torch.div(
        torch.add(torch.pow(x_tensor, 2.0), torch.mul(x_tensor, k_1)), torch.mul(torch.add(x_tensor, k_2), k_3)
    )


def srgb_from_okhsv(okhsv_tensor: torch.Tensor, alpha: float = 0.05, steps: int = 1) -> torch.Tensor:
    """Convert a 3xHxW Okhsv tensor, with hue in degrees, to gamma-corrected sRGB."""

    okhsv_tensor = _require_color_tensor(okhsv_tensor)
    okhsv_tensor = okhsv_tensor.clone()
    okhsv_tensor[1:, ...] = okhsv_tensor[1:, ...].clamp(0.0, 1.0)
    unit_hue_tensor = _unit_hue_from_degrees(okhsv_tensor[0, :, :])
    units_ab_tensor = torch.stack(
        [torch.cos(torch.mul(unit_hue_tensor, 2.0 * PI)), torch.sin(torch.mul(unit_hue_tensor, 2.0 * PI))]
    )
    lc_cusps_tensor = _find_cusp_tensor(units_ab_tensor, steps=steps)
    st_max_tensor = _st_cusps_from_lc(lc_cusps_tensor)
    s_0_tensor = _full_like_spatial(st_max_tensor, 0.5)
    k_tensor = torch.add(torch.mul(torch.div(s_0_tensor, st_max_tensor[0, :, :]), -1.0), 1)
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
    l_vt_tensor = _ok_l_from_lr_tensor(lc_v_tensor[0, :, :])
    c_vt_tensor = torch.mul(lc_v_tensor[1, :, :], torch.div(l_vt_tensor, lc_v_tensor[0, :, :]))
    l_new_tensor = _ok_l_from_lr_tensor(lc_tensor[0, :, :])
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
        torch.div(
            1.0,
            torch.max(
                rgb_scale_tensor.max(0).values,
                torch.zeros(rgb_scale_tensor.shape[1:], dtype=rgb_scale_tensor.dtype, device=rgb_scale_tensor.device),
            ),
        ),
        1.0 / 3.0,
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
    rgb_tensor = srgb_from_linear_srgb(gamut_clip_tensor(rgb_tensor, alpha=alpha, steps=steps))
    return torch.where(torch.isnan(rgb_tensor), 0.0, rgb_tensor).clamp(0.0, 1.0)


def okhsv_from_srgb(srgb_tensor: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor to Okhsv, with hue in degrees."""

    srgb_tensor = _require_color_tensor(srgb_tensor)
    lab_tensor = oklab_from_linear_srgb(linear_srgb_from_srgb(srgb_tensor))
    c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1, :, :], 2.0), torch.pow(lab_tensor[2, :, :], 2.0)))
    units_ab_tensor = torch.div(lab_tensor[1:, :, :], c_tensor)
    h_tensor = torch.add(
        torch.div(
            torch.mul(torch.atan2(torch.mul(lab_tensor[2, :, :], -1.0), torch.mul(lab_tensor[1, :, :], -1.0)), 0.5), PI
        ),
        0.5,
    )
    lc_cusps_tensor = _find_cusp_tensor(units_ab_tensor, steps=steps)
    st_max_tensor = _st_cusps_from_lc(lc_cusps_tensor)
    s_0_tensor = _full_like_spatial(st_max_tensor, 0.5)
    k_tensor = torch.add(torch.mul(torch.div(s_0_tensor, st_max_tensor[0, :, :]), -1.0), 1)
    t_tensor = torch.div(
        st_max_tensor[1, :, :], torch.add(c_tensor, torch.mul(lab_tensor[0, :, :], st_max_tensor[1, :, :]))
    )
    l_v_tensor = torch.mul(t_tensor, lab_tensor[0, :, :])
    c_v_tensor = torch.mul(t_tensor, c_tensor)
    l_vt_tensor = _ok_l_from_lr_tensor(l_v_tensor)
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
        torch.div(
            1.0,
            torch.max(
                rgb_scale_tensor.max(0).values,
                torch.zeros(rgb_scale_tensor.shape[1:], dtype=rgb_scale_tensor.dtype, device=rgb_scale_tensor.device),
            ),
        ),
        1.0 / 3.0,
    )
    lab_tensor[0, :, :] = torch.div(lab_tensor[0, :, :], scale_l_tensor)
    c_tensor = torch.div(c_tensor, scale_l_tensor)
    c_tensor = torch.mul(c_tensor, torch.div(_ok_l_r_from_l_tensor(lab_tensor[0, :, :]), lab_tensor[0, :, :]))
    lab_tensor[0, :, :] = _ok_l_r_from_l_tensor(lab_tensor[0, :, :])
    v_tensor = torch.div(lab_tensor[0, :, :], l_v_tensor)
    s_tensor = torch.div(
        torch.mul(torch.add(s_0_tensor, st_max_tensor[1, :, :]), c_v_tensor),
        torch.add(
            torch.mul(st_max_tensor[1, :, :], s_0_tensor),
            torch.mul(st_max_tensor[1, :, :], torch.mul(k_tensor, c_v_tensor)),
        ),
    )
    hsv_tensor = torch.stack([_degrees_from_unit_hue(h_tensor), s_tensor, v_tensor])
    hsv_tensor = torch.where(torch.isnan(hsv_tensor), 0.0, hsv_tensor)
    hsv_tensor[1:, ...] = hsv_tensor[1:, ...].clamp(0.0, 1.0)
    return hsv_tensor


def _get_st_mid_tensor(units_ab_tensor: torch.Tensor) -> torch.Tensor:
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


def _get_cs_tensor(
    l_tensor: torch.Tensor, units_ab_tensor: torch.Tensor, steps: int = 1, steps_outer: int = 1
) -> torch.Tensor:
    lc_cusps_tensor = _find_cusp_tensor(units_ab_tensor, steps=steps)
    c_max_tensor = _find_gamut_intersection_tensor(
        units_ab_tensor,
        l_tensor,
        torch.ones(l_tensor.shape, dtype=l_tensor.dtype, device=l_tensor.device),
        l_tensor,
        lc_cusps_tensor=lc_cusps_tensor,
        steps=steps,
        steps_outer=steps_outer,
    )
    st_max_tensor = _st_cusps_from_lc(lc_cusps_tensor)
    k_tensor = torch.div(
        c_max_tensor,
        torch.min(
            torch.mul(l_tensor, st_max_tensor[0, :, :]),
            torch.mul(torch.add(torch.mul(l_tensor, -1.0), 1.0), st_max_tensor[1, :, :]),
        ),
    )
    st_mid_tensor = _get_st_mid_tensor(units_ab_tensor)
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


def srgb_from_okhsl(
    hsl_tensor: torch.Tensor, alpha: float = 0.05, steps: int = 1, steps_outer: int = 1
) -> torch.Tensor:
    """Convert a 3xHxW Okhsl tensor, with hue in degrees, to gamma-corrected sRGB."""

    hsl_tensor = _require_color_tensor(hsl_tensor)
    hsl_tensor = hsl_tensor.clone()
    hsl_tensor[1:, ...] = hsl_tensor[1:, ...].clamp(0.0, 1.0)
    l_ones_mask = torch.eq(hsl_tensor[2, :, :], 1.0)
    l_zeros_mask = torch.eq(hsl_tensor[2, :, :], 0.0)
    l_ones_mask = l_ones_mask.expand(hsl_tensor.shape)
    l_zeros_mask = l_zeros_mask.expand(hsl_tensor.shape)
    calc_rgb_mask = torch.logical_not(torch.logical_or(l_ones_mask, l_zeros_mask))
    rgb_tensor = torch.empty_like(hsl_tensor)
    rgb_tensor = torch.where(l_ones_mask, 1.0, torch.where(l_zeros_mask, 0.0, rgb_tensor))
    unit_hue_tensor = _unit_hue_from_degrees(hsl_tensor[0, :, :])
    units_ab_tensor = torch.stack(
        [torch.cos(torch.mul(unit_hue_tensor, 2.0 * PI)), torch.sin(torch.mul(unit_hue_tensor, 2.0 * PI))]
    )
    l_tensor = _ok_l_from_lr_tensor(hsl_tensor[2, :, :])
    cs_tensor = _get_cs_tensor(l_tensor, units_ab_tensor, steps=steps, steps_outer=steps_outer)
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
    rgb_tensor = srgb_from_linear_srgb(gamut_clip_tensor(rgb_tensor, alpha=alpha, steps=steps, steps_outer=steps_outer))
    return torch.where(torch.isnan(rgb_tensor), 0.0, rgb_tensor).clamp(0.0, 1.0)


def okhsl_from_srgb(rgb_tensor: torch.Tensor, steps: int = 1, steps_outer: int = 1) -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor to Okhsl, with hue in degrees."""

    rgb_tensor = _require_color_tensor(rgb_tensor)
    lab_tensor = oklab_from_linear_srgb(linear_srgb_from_srgb(rgb_tensor))
    c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1, :, :], 2.0), torch.pow(lab_tensor[2, :, :], 2.0)))
    units_ab_tensor = torch.stack([torch.div(lab_tensor[1, :, :], c_tensor), torch.div(lab_tensor[2, :, :], c_tensor)])
    h_tensor = torch.add(
        torch.div(
            torch.mul(torch.atan2(torch.mul(lab_tensor[2, :, :], -1.0), torch.mul(lab_tensor[1, :, :], -1.0)), 0.5), PI
        ),
        0.5,
    )
    cs_tensor = _get_cs_tensor(lab_tensor[0, :, :], units_ab_tensor, steps=steps, steps_outer=steps_outer)
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
    l_tensor = _ok_l_r_from_l_tensor(lab_tensor[0, :, :])
    hsl_tensor = torch.stack([_degrees_from_unit_hue(h_tensor), s_tensor, l_tensor])
    hsl_tensor = torch.where(torch.isnan(hsl_tensor), 0.0, hsl_tensor)
    hsl_tensor[1:, ...] = hsl_tensor[1:, ...].clamp(0.0, 1.0)
    return hsl_tensor


def hsl_from_srgb(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor to HSL, with hue in degrees."""

    rgb_tensor = _require_color_tensor(rgb_tensor)
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
    h_tensor = _degrees_from_unit_hue(torch.remainder(torch.div(h_tensor, 6.0), 1.0))
    return torch.stack([h_tensor, s_tensor, l_tensor])


def srgb_from_hsl(hsl_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW HSL tensor, with hue in degrees, to gamma-corrected sRGB."""

    hsl_tensor = _require_color_tensor(hsl_tensor)
    hsl_tensor = hsl_tensor.clone()
    hsl_tensor[1:, ...] = hsl_tensor[1:, ...].clamp(0.0, 1.0)
    rgb_tensor = torch.empty_like(hsl_tensor)
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
    unit_hue_tensor = _unit_hue_from_degrees(hsl_tensor[0, :, :])

    def hsl_values(m1_tensor: torch.Tensor, m2_tensor: torch.Tensor, h_tensor: torch.Tensor) -> torch.Tensor:
        h_tensor = torch.remainder(h_tensor, 1.0)
        result_tensor = m1_tensor.clone()
        return torch.where(
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

    return torch.stack(
        [
            hsl_values(m1_tensor, m2_tensor, torch.add(unit_hue_tensor, 1.0 / 3.0)),
            hsl_values(m1_tensor, m2_tensor, unit_hue_tensor),
            hsl_values(m1_tensor, m2_tensor, torch.sub(unit_hue_tensor, 1.0 / 3.0)),
        ]
    )


def hsl_from_linear_srgb(linear_srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW linear-light sRGB tensor directly to HSL, with hue in degrees."""

    linear_srgb_tensor = _require_color_tensor(linear_srgb_tensor)
    return hsl_from_srgb(srgb_from_linear_srgb(linear_srgb_tensor))


def linear_srgb_from_hsl(hsl_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW HSL tensor, with hue in degrees, directly to linear-light sRGB."""

    hsl_tensor = _require_color_tensor(hsl_tensor)
    return linear_srgb_from_srgb(srgb_from_hsl(hsl_tensor))

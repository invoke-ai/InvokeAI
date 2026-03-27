import torch


def srgb_from_linear_srgb(linear_srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW linear-light sRGB tensor in [0, 1] to gamma-corrected sRGB."""

    linear_srgb_tensor = linear_srgb_tensor.clamp(0.0, 1.0)
    return torch.where(
        linear_srgb_tensor <= 0.0031308,
        linear_srgb_tensor * 12.92,
        1.055 * torch.pow(linear_srgb_tensor, 1.0 / 2.4) - 0.055,
    )


def linear_srgb_from_srgb(srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW gamma-corrected sRGB tensor in [0, 1] to linear-light sRGB."""

    return torch.where(
        srgb_tensor <= 0.0404482362771082,
        srgb_tensor / 12.92,
        torch.pow((srgb_tensor + 0.055) / 1.055, 2.4),
    )


def oklab_from_linear_srgb(linear_srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW linear-light sRGB tensor to Oklab."""

    lms_l = (
        0.4122214708 * linear_srgb_tensor[0, ...]
        + 0.5363325363 * linear_srgb_tensor[1, ...]
        + 0.0514459929 * linear_srgb_tensor[2, ...]
    )
    lms_m = (
        0.2119034982 * linear_srgb_tensor[0, ...]
        + 0.6806995451 * linear_srgb_tensor[1, ...]
        + 0.1073969566 * linear_srgb_tensor[2, ...]
    )
    lms_s = (
        0.0883024619 * linear_srgb_tensor[0, ...]
        + 0.2817188376 * linear_srgb_tensor[1, ...]
        + 0.6299787005 * linear_srgb_tensor[2, ...]
    )

    lms_l_cbrt = torch.sign(lms_l) * torch.pow(torch.abs(lms_l), 1.0 / 3.0)
    lms_m_cbrt = torch.sign(lms_m) * torch.pow(torch.abs(lms_m), 1.0 / 3.0)
    lms_s_cbrt = torch.sign(lms_s) * torch.pow(torch.abs(lms_s), 1.0 / 3.0)

    return torch.stack(
        [
            0.2104542553 * lms_l_cbrt + 0.7936177850 * lms_m_cbrt - 0.0040720468 * lms_s_cbrt,
            1.9779984951 * lms_l_cbrt - 2.4285922050 * lms_m_cbrt + 0.4505937099 * lms_s_cbrt,
            0.0259040371 * lms_l_cbrt + 0.7827717662 * lms_m_cbrt - 0.8086757660 * lms_s_cbrt,
        ]
    )


def linear_srgb_from_oklab(oklab_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklab tensor to linear-light sRGB."""

    lms_l_cbrt = oklab_tensor[0, ...] + 0.3963377774 * oklab_tensor[1, ...] + 0.2158037573 * oklab_tensor[2, ...]
    lms_m_cbrt = oklab_tensor[0, ...] - 0.1055613458 * oklab_tensor[1, ...] - 0.0638541728 * oklab_tensor[2, ...]
    lms_s_cbrt = oklab_tensor[0, ...] - 0.0894841775 * oklab_tensor[1, ...] - 1.2914855480 * oklab_tensor[2, ...]

    lms_l = lms_l_cbrt**3
    lms_m = lms_m_cbrt**3
    lms_s = lms_s_cbrt**3

    return torch.stack(
        [
            4.0767416621 * lms_l - 3.3077115913 * lms_m + 0.2309699292 * lms_s,
            -1.2684380046 * lms_l + 2.6097574011 * lms_m - 0.3413193965 * lms_s,
            -0.0041960863 * lms_l - 0.7034186147 * lms_m + 1.7076147010 * lms_s,
        ]
    )


def oklch_from_oklab(oklab_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklab tensor to Oklch, with hue in degrees."""

    lightness = oklab_tensor[0, ...]
    chroma = torch.sqrt(oklab_tensor[1, ...] ** 2 + oklab_tensor[2, ...] ** 2)
    hue = torch.remainder(torch.rad2deg(torch.atan2(oklab_tensor[2, ...], oklab_tensor[1, ...])), 360.0)
    return torch.stack([lightness, chroma, hue])


def oklab_from_oklch(oklch_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklch tensor, with hue in degrees, to Oklab."""

    hue_radians = torch.deg2rad(oklch_tensor[2, ...])
    a_channel = oklch_tensor[1, ...] * torch.cos(hue_radians)
    b_channel = oklch_tensor[1, ...] * torch.sin(hue_radians)
    return torch.stack([oklch_tensor[0, ...], a_channel, b_channel])


def linear_srgb_from_oklch(oklch_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 3xHxW Oklch tensor directly to linear-light sRGB."""

    return linear_srgb_from_oklab(oklab_from_oklch(oklch_tensor))

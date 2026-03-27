import numpy


def linear_srgb_from_srgb(srgb: numpy.ndarray) -> numpy.ndarray:
    return numpy.where(srgb <= 0.0404482362771082, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def srgb_from_linear_srgb(linear_srgb: numpy.ndarray) -> numpy.ndarray:
    linear_srgb = numpy.clip(linear_srgb, 0.0, 1.0)
    return numpy.where(
        linear_srgb <= 0.0031308,
        linear_srgb * 12.92,
        1.055 * numpy.power(linear_srgb, 1.0 / 2.4) - 0.055,
    )


def oklab_from_linear_srgb(linear_srgb: numpy.ndarray) -> numpy.ndarray:
    lms_l = 0.4122214708 * linear_srgb[..., 0] + 0.5363325363 * linear_srgb[..., 1] + 0.0514459929 * linear_srgb[..., 2]
    lms_m = 0.2119034982 * linear_srgb[..., 0] + 0.6806995451 * linear_srgb[..., 1] + 0.1073969566 * linear_srgb[..., 2]
    lms_s = 0.0883024619 * linear_srgb[..., 0] + 0.2817188376 * linear_srgb[..., 1] + 0.6299787005 * linear_srgb[..., 2]

    lms_l_cbrt = numpy.cbrt(lms_l)
    lms_m_cbrt = numpy.cbrt(lms_m)
    lms_s_cbrt = numpy.cbrt(lms_s)

    return numpy.stack(
        [
            0.2104542553 * lms_l_cbrt + 0.7936177850 * lms_m_cbrt - 0.0040720468 * lms_s_cbrt,
            1.9779984951 * lms_l_cbrt - 2.4285922050 * lms_m_cbrt + 0.4505937099 * lms_s_cbrt,
            0.0259040371 * lms_l_cbrt + 0.7827717662 * lms_m_cbrt - 0.8086757660 * lms_s_cbrt,
        ],
        axis=-1,
    )


def linear_srgb_from_oklab(oklab: numpy.ndarray) -> numpy.ndarray:
    lms_l_cbrt = oklab[..., 0] + 0.3963377774 * oklab[..., 1] + 0.2158037573 * oklab[..., 2]
    lms_m_cbrt = oklab[..., 0] - 0.1055613458 * oklab[..., 1] - 0.0638541728 * oklab[..., 2]
    lms_s_cbrt = oklab[..., 0] - 0.0894841775 * oklab[..., 1] - 1.2914855480 * oklab[..., 2]

    lms_l = lms_l_cbrt**3
    lms_m = lms_m_cbrt**3
    lms_s = lms_s_cbrt**3

    return numpy.stack(
        [
            4.0767416621 * lms_l - 3.3077115913 * lms_m + 0.2309699292 * lms_s,
            -1.2684380046 * lms_l + 2.6097574011 * lms_m - 0.3413193965 * lms_s,
            -0.0041960863 * lms_l - 0.7034186147 * lms_m + 1.7076147010 * lms_s,
        ],
        axis=-1,
    )


def oklch_from_oklab(oklab: numpy.ndarray) -> numpy.ndarray:
    lightness = oklab[..., 0]
    chroma = numpy.sqrt(oklab[..., 1] ** 2 + oklab[..., 2] ** 2)
    hue = numpy.degrees(numpy.arctan2(oklab[..., 2], oklab[..., 1])) % 360.0
    return numpy.stack([lightness, chroma, hue], axis=-1)


def oklab_from_oklch(oklch: numpy.ndarray) -> numpy.ndarray:
    hue_radians = numpy.radians(oklch[..., 2])
    a_channel = oklch[..., 1] * numpy.cos(hue_radians)
    b_channel = oklch[..., 1] * numpy.sin(hue_radians)
    return numpy.stack([oklch[..., 0], a_channel, b_channel], axis=-1)


def linear_srgb_from_oklch(oklch: numpy.ndarray) -> numpy.ndarray:
    return linear_srgb_from_oklab(oklab_from_oklch(oklch))

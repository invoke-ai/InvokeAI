import numpy

from invokeai.backend.image_util.color_conversion import (
    linear_srgb_from_oklab,
    linear_srgb_from_oklch,
    linear_srgb_from_srgb,
    oklab_from_linear_srgb,
    oklab_from_oklch,
    oklch_from_oklab,
    srgb_from_linear_srgb,
)


def test_srgb_oklab_round_trip() -> None:
    srgb = numpy.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.1, 0.6, 0.9]],
        ],
        dtype=numpy.float32,
    )

    round_tripped = srgb_from_linear_srgb(linear_srgb_from_oklab(oklab_from_linear_srgb(linear_srgb_from_srgb(srgb))))

    assert numpy.allclose(round_tripped, srgb, atol=1e-5)


def test_oklab_from_pure_srgb_red_matches_reference_value() -> None:
    srgb_red = numpy.array([[[1.0, 0.0, 0.0]]], dtype=numpy.float32)

    oklab_red = oklab_from_linear_srgb(linear_srgb_from_srgb(srgb_red))

    assert numpy.allclose(oklab_red[0, 0], [0.62795536, 0.22486306, 0.1258463], atol=1e-6)


def test_oklab_oklch_round_trip() -> None:
    oklab = numpy.array(
        [
            [[0.6, 0.2, 0.1], [0.4, -0.1, 0.05]],
        ],
        dtype=numpy.float32,
    )

    round_tripped = oklab_from_oklch(oklch_from_oklab(oklab))

    assert numpy.allclose(round_tripped, oklab, atol=1e-6)


def test_srgb_oklch_round_trip() -> None:
    srgb = numpy.array(
        [
            [[0.2, 0.4, 0.8], [0.9, 0.3, 0.1]],
        ],
        dtype=numpy.float32,
    )

    round_tripped = srgb_from_linear_srgb(
        linear_srgb_from_oklch(oklch_from_oklab(oklab_from_linear_srgb(linear_srgb_from_srgb(srgb))))
    )

    assert numpy.allclose(round_tripped, srgb, atol=1e-5)

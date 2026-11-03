import numpy as np
import pytest

from invokeai.backend.image_util.multiscale import edge_aware_band


def test_edge_aware_band_of_constant_lightness_is_zero() -> None:
    lightness = np.full((9, 13), 0.42, dtype=np.float32)

    band = edge_aware_band(lightness, inner_radius=1, outer_radius=5, epsilon=1e-3)

    np.testing.assert_allclose(band, 0, atol=1e-6)


def test_edge_aware_band_is_deterministic_and_finite() -> None:
    y, x = np.indices((19, 27))
    lightness = (0.5 + 0.2 * np.sin(x * 0.7) + 0.1 * np.cos(y * 0.4)).astype(np.float32)

    first = edge_aware_band(lightness, inner_radius=1, outer_radius=6, epsilon=1e-2)
    second = edge_aware_band(lightness, inner_radius=1, outer_radius=6, epsilon=1e-2)

    assert first.shape == lightness.shape
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(("polarity", "expected_sign"), [(1.0, 1.0), (-1.0, -1.0)])
def test_edge_aware_band_sign_tracks_bright_and_dark_impulses(polarity: float, expected_sign: float) -> None:
    lightness = np.zeros((41, 41), dtype=np.float32)
    lightness[20, 20] = polarity

    band = edge_aware_band(lightness, inner_radius=1, outer_radius=5, epsilon=0.1)

    assert expected_sign * float(band[20, 20]) > 0


def test_edge_aware_band_responds_to_the_requested_scales() -> None:
    y, x = np.indices((73, 193))

    def band_energy(period: int, inner_radius: int, outer_radius: int) -> float:
        lightness = np.sin(2 * np.pi * x / period).astype(np.float32)
        band = edge_aware_band(
            lightness,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            epsilon=0.03,
        )
        return float(np.sqrt(np.mean(band[:, 32:-32] ** 2)))

    fine_feature_at_fine_scale = band_energy(period=8, inner_radius=1, outer_radius=6)
    same_feature_at_coarser_scale = band_energy(period=8, inner_radius=3, outer_radius=10)

    assert fine_feature_at_fine_scale > 1e-3
    assert fine_feature_at_fine_scale > 5 * same_feature_at_coarser_scale


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (5, 7), (9, 13), (3, 257)])
def test_edge_aware_band_handles_tiny_odd_and_ultrawide_inputs(shape: tuple[int, int]) -> None:
    lightness = np.zeros(shape, dtype=np.float32)
    lightness[0, 0] = 1
    lightness[-1, -1] = -0.5

    band = edge_aware_band(lightness, inner_radius=2, outer_radius=7, epsilon=1e-2)

    assert band.shape == shape
    assert np.isfinite(band).all()

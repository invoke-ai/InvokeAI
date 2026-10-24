import cv2
import numpy as np
import pytest

from invokeai.backend.image_util.guided_filter import guided_filter


def test_guided_filter_preserves_constant_signal() -> None:
    guide = np.arange(35, dtype=np.float32).reshape(5, 7)
    signal = np.full((5, 7), 0.375, dtype=np.float32)

    result = guided_filter(guide, signal, radius=2, epsilon=1e-3)

    np.testing.assert_allclose(result, signal, atol=1e-6)


def test_guided_filter_clamps_negative_float32_variance() -> None:
    rng = np.random.default_rng(12)
    guide = (0.5 + rng.standard_normal((5, 7)).astype(np.float32) * 1e-6).astype(np.float32)
    signal = rng.random((5, 7), dtype=np.float32)

    mean_guide = cv2.boxFilter(guide, ddepth=-1, ksize=(5, 5), normalize=True, borderType=cv2.BORDER_REFLECT_101)
    raw_variance = (
        cv2.boxFilter(guide * guide, ddepth=-1, ksize=(5, 5), normalize=True, borderType=cv2.BORDER_REFLECT_101)
        - mean_guide * mean_guide
    )
    assert float(raw_variance.min()) < 0
    epsilon = float(-raw_variance.min() + 1e-12)

    first_mean = cv2.boxFilter(signal, ddepth=-1, ksize=(5, 5), normalize=True, borderType=cv2.BORDER_REFLECT_101)
    expected = cv2.boxFilter(first_mean, ddepth=-1, ksize=(5, 5), normalize=True, borderType=cv2.BORDER_REFLECT_101)
    result = guided_filter(guide, signal, radius=2, epsilon=epsilon)

    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_guided_filter_smooths_noise_without_blurring_a_step_edge() -> None:
    guide = np.zeros((15, 15), dtype=np.float32)
    guide[:, 7:] = 1.0
    noise = ((np.indices(guide.shape).sum(axis=0) % 2) * 2 - 1).astype(np.float32) * 0.2
    signal = guide + noise

    result = guided_filter(guide, signal, radius=2, epsilon=1e-2)

    assert np.mean((result - guide) ** 2) < np.mean((signal - guide) ** 2)
    assert float(result[:, 7].mean() - result[:, 6].mean()) > 0.7


def test_guided_filter_uses_reflect_101_at_borders() -> None:
    guide = np.ones((2, 3), dtype=np.float32)
    signal = np.array([[0.0, 1.0, 4.0], [2.0, 8.0, 3.0]], dtype=np.float32)

    first_mean = cv2.boxFilter(signal, ddepth=-1, ksize=(3, 3), normalize=True, borderType=cv2.BORDER_REFLECT_101)
    expected = cv2.boxFilter(first_mean, ddepth=-1, ksize=(3, 3), normalize=True, borderType=cv2.BORDER_REFLECT_101)
    result = guided_filter(guide, signal, radius=1, epsilon=1e-3)

    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (6, 1), (2, 3), (5, 7)])
def test_guided_filter_handles_degenerate_and_odd_shapes(shape: tuple[int, int]) -> None:
    guide = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    signal = np.flip(guide, axis=1).copy() / max(1, guide.size)

    result = guided_filter(guide, signal, radius=2, epsilon=1e-3)

    assert result.shape == shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_guided_filter_radius_zero_is_identity() -> None:
    guide = np.array([[0.0, 1.0], [3.0, -2.0]], dtype=np.float32)
    signal = np.array([[4.0, 0.5], [-1.0, 7.0]], dtype=np.float32)

    result = guided_filter(guide, signal, radius=0, epsilon=0.25)

    np.testing.assert_array_equal(result, signal)


@pytest.mark.parametrize(
    ("radius", "epsilon"),
    [(-1, 1e-3), (1, 0.0), (1, -1e-3)],
)
def test_guided_filter_rejects_out_of_range_parameters(radius: int, epsilon: float) -> None:
    image = np.ones((3, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        guided_filter(image, image, radius=radius, epsilon=epsilon)

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.backend.image_util.clarity import clarity
from invokeai.backend.image_util.color_conversion import oklab_from_srgb


def _gray_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L")


def _gray_pattern(level: int, amplitude: int, shape: tuple[int, int] = (129, 129)) -> np.ndarray:
    y, x = np.indices(shape)
    pattern = ((x + y) % 2) * 2 - 1
    return np.clip(level + amplitude * pattern, 0, 255).astype(np.uint8)


def _gray_sinusoid(level: int, amplitude: int, wavelength: int, shape: tuple[int, int] = (129, 129)) -> np.ndarray:
    _, x = np.indices(shape)
    pattern = np.sin(2 * np.pi * x / wavelength)
    return np.clip(np.rint(level + amplitude * pattern), 0, 255).astype(np.uint8)


def test_zero_amount_returns_an_exact_copy() -> None:
    source = Image.fromarray(
        np.array([[[255, 0, 10, 0], [2, 128, 254, 73]], [[8, 9, 10, 255], [11, 12, 13, 1]]], dtype=np.uint8),
        mode="RGBA",
    )

    result = clarity(source, amount=0)

    assert result is not source
    assert result.mode == source.mode
    assert result.size == source.size
    assert result.tobytes() == source.tobytes()


def test_zero_amount_skips_color_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Image.new("RGB", (3, 2), (24, 128, 222))

    def fail_if_called(_: Image.Image) -> tuple[torch.Tensor, np.ndarray, np.ndarray | None]:
        raise AssertionError("zero amount must skip image color conversion")

    monkeypatch.setattr("invokeai.backend.image_util.clarity.prepare_lightness", fail_if_called)

    result = clarity(source, amount=0)

    assert result is not source
    assert result.tobytes() == source.tobytes()


def test_constant_color_is_preserved() -> None:
    source = Image.new("RGB", (37, 29), (74, 121, 193))

    result = clarity(source, amount=100, size=40)

    assert result.mode == source.mode
    delta = np.abs(np.asarray(result, dtype=np.int16) - np.asarray(source, dtype=np.int16))
    assert int(delta.max()) <= 1


@pytest.mark.parametrize("rgb", [(0, 17, 166), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 128, 255)])
def test_flat_color_on_the_gamut_boundary_is_unchanged(rgb: tuple[int, int, int]) -> None:
    """Float32 round-trip noise must not push a boundary color into the gamut clip, which would move it."""
    source = Image.new("RGB", (4, 4), rgb)

    result = clarity(source, amount=100, size=40)

    assert result.tobytes() == source.tobytes()


def test_positive_and_negative_amount_separate_midtone_detail_in_opposite_directions() -> None:
    source = _gray_image(_gray_sinusoid(level=128, amplitude=16, wavelength=16))
    center = (slice(32, -32), slice(32, -32))

    positive = np.asarray(clarity(source, amount=100, size=12), dtype=np.float32)
    negative = np.asarray(clarity(source, amount=-100, size=12), dtype=np.float32)
    original = np.asarray(source, dtype=np.float32)

    positive_contrast = float(positive[center].std())
    negative_contrast = float(negative[center].std())
    source_contrast = float(original[center].std())
    assert positive_contrast > source_contrast + 1.0
    assert negative_contrast < source_contrast - 1.0


def test_extended_amounts_produce_stronger_clarity_adjustments() -> None:
    source = _gray_image(_gray_sinusoid(level=128, amplitude=16, wavelength=16))
    center = (slice(32, -32), slice(32, -32))

    original = np.asarray(source, dtype=np.float32)
    source_contrast = float(original[center].std())
    positive_100 = np.asarray(clarity(source, amount=100, size=12), dtype=np.float32)
    positive_200 = np.asarray(clarity(source, amount=200, size=12), dtype=np.float32)
    negative_100 = np.asarray(clarity(source, amount=-100, size=12), dtype=np.float32)
    negative_200 = np.asarray(clarity(source, amount=-200, size=12), dtype=np.float32)

    assert float(positive_200[center].std()) > float(positive_100[center].std()) > source_contrast
    assert float(negative_200[center].std()) < float(negative_100[center].std()) < source_contrast


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf"), -200.01, 200.01, True])
def test_invalid_amount_is_rejected(amount: float) -> None:
    with pytest.raises(ValueError):
        clarity(Image.new("RGB", (3, 2), (120, 130, 140)), amount=amount)


def test_tone_gate_reduces_effect_near_black_and_white() -> None:
    regions = np.concatenate(
        [
            _gray_sinusoid(level=5, amplitude=16, wavelength=16),
            _gray_sinusoid(level=128, amplitude=16, wavelength=16),
            _gray_sinusoid(level=250, amplitude=16, wavelength=16),
        ],
        axis=1,
    )
    source = _gray_image(regions)

    result = np.asarray(clarity(source, amount=100, size=12), dtype=np.float32)
    original = regions.astype(np.float32)
    dark_delta = float(np.mean(np.abs(result[32:-32, 32:97] - original[32:-32, 32:97])))
    mid_delta = float(np.mean(np.abs(result[32:-32, 161:226] - original[32:-32, 161:226])))
    bright_delta = float(np.mean(np.abs(result[32:-32, 290:355] - original[32:-32, 290:355])))

    assert mid_delta > 1.0
    assert dark_delta < mid_delta * 0.2
    assert bright_delta < mid_delta * 0.2


def test_size_is_measured_in_source_pixels_independent_of_canvas_dimensions() -> None:
    small = np.zeros((129, 129), dtype=np.uint8)
    large = np.zeros((257, 257), dtype=np.uint8)
    small[64, 64] = 160
    large[128, 128] = 160

    result_small = np.asarray(clarity(_gray_image(small), amount=100, size=20), dtype=np.int16)
    result_large = np.asarray(clarity(_gray_image(large), amount=100, size=20), dtype=np.int16)

    np.testing.assert_allclose(result_small[44:85, 44:85], result_large[108:149, 108:149], atol=1)


def test_clarity_changes_lightness_without_introducing_chroma_detail() -> None:
    _, x = np.indices((129, 129))
    levels = (112.0 + 32.0 * np.sin(2 * np.pi * x / 24)).astype(np.float32) / 255.0
    # Keep a single hue while varying lightness so the test can detect color fringes.
    rgb = np.stack((levels, levels * 0.62, levels * 0.43), axis=-1)
    source = Image.fromarray(np.round(rgb * 255).astype(np.uint8), mode="RGB")

    result = clarity(source, amount=100, size=12)
    source_tensor = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1).float() / 255.0
    result_tensor = torch.from_numpy(np.asarray(result).copy()).permute(2, 0, 1).float() / 255.0
    source_oklab = oklab_from_srgb(source_tensor)
    result_oklab = oklab_from_srgb(result_tensor)
    interior = (slice(32, -32), slice(32, -32))
    chroma_delta = torch.linalg.vector_norm(
        result_oklab[1:, interior[0], interior[1]] - source_oklab[1:, interior[0], interior[1]], dim=0
    )
    lightness_delta = result_oklab[0, interior[0], interior[1]] - source_oklab[0, interior[0], interior[1]]

    assert float(lightness_delta.abs().mean()) > 0.005
    assert float(chroma_delta.max()) < 0.015


def test_alpha_and_hidden_rgb_are_preserved() -> None:
    rgba = np.zeros((65, 65, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.asarray(_gray_pattern(128, 6, (65, 65)))[..., None]
    rgba[:, :, 3] = 255
    rgba[20:25, 20:25, :3] = (255, 0, 255)
    rgba[20:25, 20:25, 3] = 0
    source = Image.fromarray(rgba, mode="RGBA")

    result = np.asarray(clarity(source, amount=100, size=12))

    np.testing.assert_array_equal(result[:, :, 3], rgba[:, :, 3])
    np.testing.assert_array_equal(result[rgba[:, :, 3] == 0], rgba[rgba[:, :, 3] == 0])


def test_sharp_edge_has_less_overshoot_than_large_radius_gaussian_unsharp_mask() -> None:
    edge = np.full((129, 129), 112, dtype=np.uint8)
    edge[:, 64:] = 144
    source = _gray_image(edge)
    source_float = edge.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(source_float, (0, 0), sigmaX=20, sigmaY=20, borderType=cv2.BORDER_REFLECT_101)
    gaussian_unsharp = source_float + (source_float - blurred)

    clarity_result = np.asarray(clarity(source, amount=100, size=40), dtype=np.float32) / 255.0
    interior = (slice(32, -32), slice(32, -32))
    lower, upper = float(source_float.min()), float(source_float.max())
    clarity_overshoot = max(
        lower - float(clarity_result[interior].min()),
        float(clarity_result[interior].max()) - upper,
        0.0,
    )
    gaussian_overshoot = max(
        lower - float(gaussian_unsharp[interior].min()),
        float(gaussian_unsharp[interior].max()) - upper,
        0.0,
    )

    assert gaussian_overshoot > 0.02
    assert clarity_overshoot < gaussian_overshoot * 0.5

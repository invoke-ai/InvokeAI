from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from invokeai.backend.image_util.clarity import clarity
from invokeai.backend.image_util.texture import texture


def _gray_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L")


def _gray_sinusoid(
    wavelength: int,
    amplitude: float = 20,
    level: float = 128,
    shape: tuple[int, int] = (257, 257),
) -> np.ndarray:
    _, x = np.indices(shape)
    pattern = np.sin(2 * np.pi * x / wavelength)
    return np.clip(np.rint(level + amplitude * pattern), 0, 255).astype(np.uint8)


def _interior(values: np.ndarray, border: int = 48) -> np.ndarray:
    return values[border:-border, border:-border]


def _sinusoid_amplitude(values: np.ndarray, wavelength: int, border: int = 48) -> float:
    """Measure one repeated spatial frequency, ignoring the image border."""
    region = _interior(np.asarray(values, dtype=np.float32), border=border)
    _, x = np.indices(region.shape)
    angle = 2 * np.pi * x / wavelength
    design = np.column_stack((np.ones(region.size), np.sin(angle).ravel(), np.cos(angle).ravel()))
    coefficients = np.linalg.lstsq(design, region.ravel(), rcond=None)[0]
    return float(np.hypot(coefficients[1], coefficients[2]))


def test_zero_amount_returns_an_exact_copy() -> None:
    source = Image.fromarray(
        np.array([[[255, 0, 10, 0], [2, 128, 254, 73]], [[8, 9, 10, 255], [11, 12, 13, 1]]], dtype=np.uint8),
        mode="RGBA",
    )

    result = texture(source, amount=0)

    assert result is not source
    assert result.mode == source.mode
    assert result.size == source.size
    assert result.tobytes() == source.tobytes()


def test_zero_amount_copies_without_requiring_a_supported_filter_mode() -> None:
    source = Image.new("CMYK", (3, 2), (10, 20, 30, 40))

    result = texture(source, amount=0)

    assert result is not source
    assert result.mode == source.mode
    assert result.tobytes() == source.tobytes()


def test_constant_color_is_preserved() -> None:
    source = Image.new("RGB", (37, 29), (74, 121, 193))

    result = texture(source, amount=100, size=8)

    assert result.mode == source.mode
    delta = np.abs(np.asarray(result, dtype=np.int16) - np.asarray(source, dtype=np.int16))
    assert int(delta.max()) <= 1


def test_positive_texture_gates_low_level_noise_in_smooth_regions() -> None:
    fine_texture = _gray_sinusoid(wavelength=24, amplitude=20, shape=(257, 257))
    smooth_noise = _gray_sinusoid(wavelength=24, amplitude=1, shape=(257, 257))
    source = _gray_image(np.concatenate((fine_texture, smooth_noise), axis=1))

    result = np.asarray(texture(source, amount=100, size=8), dtype=np.float32)
    original = np.asarray(source, dtype=np.float32)
    texture_delta = float(np.mean(np.abs(result[48:-48, 48:209] - original[48:-48, 48:209])))
    noise_delta = float(np.mean(np.abs(result[48:-48, 305:466] - original[48:-48, 305:466])))

    assert texture_delta > 1.0
    assert noise_delta < texture_delta * 0.2


def test_negative_texture_softens_fine_detail_without_blurring_broad_detail() -> None:
    _, x = np.indices((512, 512))
    values = np.rint(128 + 20 * np.sin(2 * np.pi * x / 24) + 24 * np.sin(2 * np.pi * x / 128))
    source = _gray_image(np.clip(values, 0, 255).astype(np.uint8))

    result = np.asarray(texture(source, amount=-100, size=8), dtype=np.float32)
    original = np.asarray(source, dtype=np.float32)
    fine_before = _sinusoid_amplitude(original, wavelength=24, border=64)
    fine_after = _sinusoid_amplitude(result, wavelength=24, border=64)
    broad_before = _sinusoid_amplitude(original, wavelength=128, border=64)
    broad_after = _sinusoid_amplitude(result, wavelength=128, border=64)

    assert fine_after < fine_before - 1.0
    assert abs(broad_after - broad_before) < (fine_before - fine_after) * 0.5


def test_amount_sign_enhances_or_softens_the_same_fine_detail() -> None:
    source = _gray_image(_gray_sinusoid(wavelength=24, amplitude=20))

    positive = np.asarray(texture(source, amount=100, size=8), dtype=np.float32)
    negative = np.asarray(texture(source, amount=-100, size=8), dtype=np.float32)
    original = np.asarray(source, dtype=np.float32)

    original_contrast = float(_interior(original).std())
    positive_contrast = float(_interior(positive).std())
    negative_contrast = float(_interior(negative).std())
    assert positive_contrast > original_contrast + 1.0
    assert negative_contrast < original_contrast - 1.0


def test_extended_amounts_produce_stronger_texture_adjustments() -> None:
    source = _gray_image(_gray_sinusoid(wavelength=24, amplitude=20))

    source_amplitude = _sinusoid_amplitude(np.asarray(source), wavelength=24)
    positive_100 = texture(source, amount=100, size=8)
    positive_200 = texture(source, amount=200, size=8)
    negative_100 = texture(source, amount=-100, size=8)
    negative_200 = texture(source, amount=-200, size=8)

    assert (
        _sinusoid_amplitude(np.asarray(positive_200), wavelength=24)
        > _sinusoid_amplitude(np.asarray(positive_100), wavelength=24)
        > source_amplitude
    )
    assert (
        _sinusoid_amplitude(np.asarray(negative_200), wavelength=24)
        < _sinusoid_amplitude(np.asarray(negative_100), wavelength=24)
        < source_amplitude
    )


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf"), -200.01, 200.01, True])
def test_invalid_amount_is_rejected(amount: float) -> None:
    with pytest.raises(ValueError):
        texture(Image.new("RGB", (3, 2), (120, 130, 140)), amount=amount)


def test_texture_has_a_measurable_response_to_reproducible_sinusoidal_detail() -> None:
    source = _gray_image(_gray_sinusoid(wavelength=24, amplitude=20))

    result = np.asarray(texture(source, amount=100, size=8), dtype=np.float32)
    original = np.asarray(source, dtype=np.float32)

    assert float(np.sqrt(np.mean((_interior(result) - _interior(original)) ** 2))) > 0.5


def test_texture_targets_smaller_wavelengths_than_medium_scale_clarity() -> None:
    # Size 8 targets fine detail; Clarity size 40 is calibrated for medium scales.
    # Formula-based calibration shows Texture size 8 responds around 24–32 px, while
    # Clarity size 40 responds around 96–128 px. The 64 px crop leaves 384 px, an
    # integer number of periods at every wavelength, while a 2x margin avoids snapshots.
    # These response ranges are provisional until rerun against the implementation.
    wavelengths = (8, 12, 16, 24, 32, 48, 64, 96, 128)
    texture_response: list[float] = []
    clarity_response: list[float] = []

    for wavelength in wavelengths:
        source = _gray_image(_gray_sinusoid(wavelength=wavelength, amplitude=20, shape=(512, 512)))
        original = np.asarray(source, dtype=np.float32)
        interior_original = _interior(original, border=64)
        texture_result = np.asarray(texture(source, amount=100, size=8), dtype=np.float32)
        clarity_result = np.asarray(clarity(source, amount=100, size=40), dtype=np.float32)
        texture_response.append(
            float(np.sqrt(np.mean((_interior(texture_result, border=64) - interior_original) ** 2)))
        )
        clarity_response.append(
            float(np.sqrt(np.mean((_interior(clarity_result, border=64) - interior_original) ** 2)))
        )

    texture_peak_wavelength = wavelengths[int(np.argmax(texture_response))]
    clarity_peak_wavelength = wavelengths[int(np.argmax(clarity_response))]
    assert max(texture_response) > 0.5
    assert max(clarity_response) > 1.0
    assert texture_peak_wavelength * 2 <= clarity_peak_wavelength


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
def test_supported_modes_and_alpha_are_preserved(mode: str) -> None:
    gray = _gray_sinusoid(wavelength=24, amplitude=20, shape=(65, 65))
    if mode == "L":
        array = gray
    elif mode == "LA":
        array = np.dstack((gray, np.full_like(gray, 255)))
        array[20:25, 20:25, 1] = 0
        array[20:25, 20:25, 0] = 37
    elif mode == "RGB":
        array = np.repeat(gray[..., None], 3, axis=2)
    else:
        array = np.dstack((gray, gray, gray, np.full_like(gray, 255)))
        array[20:25, 20:25, :3] = (255, 0, 170)
        array[20:25, 20:25, 3] = 0
    source = Image.fromarray(array.astype(np.uint8), mode=mode)

    result = texture(source, amount=100, size=8)

    assert result.mode == mode
    result_array = np.asarray(result)
    if mode in {"LA", "RGBA"}:
        source_array = np.asarray(source)
        np.testing.assert_array_equal(result_array[..., -1], source_array[..., -1])
        np.testing.assert_array_equal(
            result_array[source_array[..., -1] == 0], source_array[source_array[..., -1] == 0]
        )

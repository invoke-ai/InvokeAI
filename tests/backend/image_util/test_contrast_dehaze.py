from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.backend.image_util.color_conversion import oklab_from_srgb
from invokeai.backend.image_util.contrast_dehaze import contrast_dehaze


def _veiled_gray_bands() -> Image.Image:
    """Broad image regions with reduced luminance separation, as from a light veil."""
    levels = np.array([112, 122, 132, 142], dtype=np.uint8)
    luminance = np.broadcast_to(np.repeat(levels, 32)[None, :], (96, 128))
    return Image.fromarray(np.repeat(luminance[..., None], 3, axis=2), mode="RGB")


def _broad_luminance_spread(image: Image.Image) -> float:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    return float(np.percentile(grayscale, 90) - np.percentile(grayscale, 10))


def _mean_oklab_chroma(image: Image.Image) -> float:
    srgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
    oklab = oklab_from_srgb(srgb)
    return float(torch.linalg.vector_norm(oklab[1:], dim=0).mean())


def test_zero_amount_returns_an_exact_copy() -> None:
    source = Image.fromarray(
        np.array([[[255, 0, 10, 0], [2, 128, 254, 73]], [[8, 9, 10, 255], [11, 12, 13, 1]]], dtype=np.uint8),
        mode="RGBA",
    )

    result = contrast_dehaze(source, amount=0)

    assert result is not source
    assert result.mode == source.mode
    assert result.size == source.size
    assert result.tobytes() == source.tobytes()


def test_positive_amount_increases_broad_scale_luminance_separation() -> None:
    source = _veiled_gray_bands()

    result = contrast_dehaze(source, amount=60)

    assert _broad_luminance_spread(result) > _broad_luminance_spread(source) + 2.0


def test_negative_amount_reduces_broad_scale_separation_like_a_veil() -> None:
    source = _veiled_gray_bands()

    result = contrast_dehaze(source, amount=-60)

    assert _broad_luminance_spread(result) < _broad_luminance_spread(source) - 2.0


def test_extended_amounts_produce_stronger_contrast_dehaze_adjustments() -> None:
    source = _veiled_gray_bands()
    source_spread = _broad_luminance_spread(source)

    positive_100 = _broad_luminance_spread(contrast_dehaze(source, amount=100))
    positive_200 = _broad_luminance_spread(contrast_dehaze(source, amount=200))
    negative_100 = _broad_luminance_spread(contrast_dehaze(source, amount=-100))
    negative_200 = _broad_luminance_spread(contrast_dehaze(source, amount=-200))

    assert positive_200 > positive_100 > source_spread
    assert negative_200 < negative_100 < source_spread


def test_amount_moves_chroma_in_the_same_direction_on_a_safe_color_patch() -> None:
    source = Image.new("RGB", (24, 24), (130, 90, 70))
    source_chroma = _mean_oklab_chroma(source)

    positive_chroma = _mean_oklab_chroma(contrast_dehaze(source, amount=60))
    negative_chroma = _mean_oklab_chroma(contrast_dehaze(source, amount=-60))

    assert positive_chroma > source_chroma
    assert negative_chroma < source_chroma


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA"])
def test_supported_modes_keep_mode_size_and_exact_alpha(mode: str) -> None:
    gray = np.broadcast_to(np.tile(np.array([90, 120, 150], dtype=np.uint8), 11)[:32], (24, 32))
    if mode in {"RGB", "RGBA"}:
        rgb = np.repeat(gray[..., None], 3, axis=2)
        if mode == "RGB":
            source = Image.fromarray(rgb, mode=mode)
        else:
            alpha = np.tile(np.array([0, 64, 255], dtype=np.uint8), (24, 11))[:, :32]
            source = Image.fromarray(np.dstack((rgb, alpha)), mode=mode)
    elif mode == "L":
        source = Image.fromarray(gray, mode=mode)
    else:
        alpha = np.tile(np.array([0, 64, 255], dtype=np.uint8), (24, 11))[:, :32]
        source = Image.fromarray(np.stack((gray, alpha), axis=-1), mode=mode)

    source_bytes = source.tobytes()
    result = contrast_dehaze(source, amount=50)

    assert source.tobytes() == source_bytes
    assert result.mode == source.mode
    assert result.size == source.size
    if "A" in source.getbands():
        np.testing.assert_array_equal(np.asarray(result.getchannel("A")), np.asarray(source.getchannel("A")))


def test_transparent_pixels_keep_their_hidden_color() -> None:
    pixels = np.empty((16, 16, 4), dtype=np.uint8)
    pixels[..., :3] = (110, 125, 140)
    pixels[..., 3] = 255
    pixels[4:9, 5:10, :3] = (255, 0, 211)
    pixels[4:9, 5:10, 3] = 0
    source = Image.fromarray(pixels, mode="RGBA")

    result = np.asarray(contrast_dehaze(source, amount=70))

    np.testing.assert_array_equal(result[pixels[..., 3] == 0], pixels[pixels[..., 3] == 0])


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf"), -200.01, 200.01, True])
def test_invalid_amount_is_rejected(amount: float) -> None:
    with pytest.raises(ValueError):
        contrast_dehaze(Image.new("RGB", (3, 2), (120, 130, 140)), amount=amount)


def test_output_is_finite_and_stays_within_byte_bounds() -> None:
    source = Image.fromarray(
        np.array(
            [
                [[0, 255, 8], [255, 0, 247], [1, 128, 254]],
                [[250, 2, 0], [5, 250, 255], [128, 127, 126]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    )

    result = contrast_dehaze(source, amount=100)
    pixels = np.asarray(result)

    assert np.isfinite(pixels).all()
    assert int(pixels.min()) >= 0
    assert int(pixels.max()) <= 255


def test_core_memory_error_is_reported_as_a_clear_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_memory_error(*args: object, **kwargs: object) -> np.ndarray:
        raise MemoryError("allocation failed")

    monkeypatch.setattr("invokeai.backend.image_util.contrast_dehaze.edge_aware_band", raise_memory_error)

    with pytest.raises(RuntimeError, match="(?i)(memory|image)"):
        contrast_dehaze(_veiled_gray_bands(), amount=60)

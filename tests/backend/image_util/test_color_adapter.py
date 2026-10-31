from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.backend.image_util import color
from invokeai.backend.image_util.color_conversion import (
    gamut_clip_tensor,
    linear_srgb_from_oklab,
    linear_srgb_from_srgb,
    oklab_from_linear_srgb,
    oklab_from_srgb,
    srgb_from_linear_srgb,
)


def _image(mode: str, size: tuple[int, int] = (3, 2)) -> Image.Image:
    width, height = size
    rgb = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3) * 11
    if mode == "RGB":
        return Image.fromarray(rgb)
    if mode == "RGBA":
        alpha = np.arange(width * height, dtype=np.uint8).reshape(height, width) * 37
        return Image.fromarray(np.dstack((rgb, alpha)))
    if mode == "L":
        return Image.fromarray(rgb[..., 0])
    if mode == "LA":
        alpha = np.arange(width * height, dtype=np.uint8).reshape(height, width) * 37
        return Image.fromarray(np.dstack((rgb[..., 0], alpha)), mode="LA")
    raise AssertionError(mode)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA"])
def test_prepare_restore_round_trip_preserves_mode_size_and_alpha(mode: str) -> None:
    source = _image(mode)

    color_tensor, lightness, alpha = color.prepare_lightness(source)
    result = color.restore_lightness(source, color_tensor, lightness, alpha)

    assert result.mode == source.mode
    assert result.size == source.size
    if "A" in source.getbands():
        assert np.array_equal(np.asarray(result.getchannel("A")), np.asarray(source.getchannel("A")))
    assert (
        np.max(
            np.abs(
                np.asarray(result.convert("RGB"), dtype=np.int16) - np.asarray(source.convert("RGB"), dtype=np.int16)
            )
        )
        <= 1
    )


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA"])
def test_prepare_returns_finite_contiguous_cpu_rgb_tensor_and_lightness(mode: str) -> None:
    source = _image(mode, (7, 5))

    color_tensor, lightness, _ = color.prepare_lightness(source)

    assert color_tensor.device.type == "cpu"
    assert color_tensor.dtype == torch.float32
    assert color_tensor.shape == (3, 5, 7)
    assert color_tensor.is_contiguous()
    assert lightness.shape == (5, 7)
    assert lightness.dtype == np.float32
    assert lightness.flags.c_contiguous
    assert np.isfinite(lightness).all()
    assert np.isfinite(color_tensor.numpy()).all()


def test_prepare_uses_core_oklab_conversion_with_rgb_channel_order() -> None:
    source = Image.new("RGB", (1, 1), (11, 73, 201))
    rgb_chw = torch.tensor([11.0, 73.0, 201.0], dtype=torch.float32).view(3, 1, 1) / 255.0
    expected_oklab = oklab_from_srgb(rgb_chw)

    color_tensor, _, _ = color.prepare_lightness(source)

    assert torch.allclose(color_tensor, expected_oklab, atol=1e-6)


@pytest.mark.parametrize("size", [(1, 1), (5, 3), (257, 1), (1, 257)])
def test_prepare_restore_supports_tiny_odd_ultrawide_and_tall_images(size: tuple[int, int]) -> None:
    source = _image("RGBA", size)

    color_tensor, lightness, alpha = color.prepare_lightness(source)
    result = color.restore_lightness(source, color_tensor, lightness, alpha)

    assert result.size == source.size
    assert np.isfinite(lightness).all()
    assert np.array_equal(np.asarray(result.getchannel("A")), np.asarray(source.getchannel("A")))
    source_rgba = np.asarray(source)
    result_rgba = np.asarray(result)
    visible = source_rgba[..., 3] > 0
    hidden = ~visible
    if visible.any():
        assert (
            np.max(np.abs(result_rgba[visible, :3].astype(np.int16) - source_rgba[visible, :3].astype(np.int16))) <= 1
        )
    assert np.array_equal(result_rgba[hidden], source_rgba[hidden])


def test_restore_isolates_hidden_rgba_rgb_from_visible_pixels() -> None:
    visible = np.array([[[220, 30, 10, 255], [0, 0, 0, 0], [25, 100, 210, 255]]], dtype=np.uint8)
    hidden_variant = visible.copy()
    hidden_variant[0, 1, :3] = [255, 0, 255]
    first = Image.fromarray(visible, mode="RGBA")
    second = Image.fromarray(hidden_variant, mode="RGBA")

    prepared = []
    restored = []
    for source in (first, second):
        color_tensor, lightness, alpha = color.prepare_lightness(source)
        prepared.append(lightness)
        restored.append(color.restore_lightness(source, color_tensor, lightness, alpha))

    assert np.array_equal(prepared[0], prepared[1])
    assert np.array_equal(np.asarray(restored[0])[0, [0, 2]], np.asarray(restored[1])[0, [0, 2]])
    assert np.array_equal(np.asarray(restored[0])[0, 1], visible[0, 1])
    assert np.array_equal(np.asarray(restored[1])[0, 1], hidden_variant[0, 1])


def test_prepare_prefills_transparent_lightness_from_nearest_visible_pixel() -> None:
    transparent_black = np.array([[[255, 255, 255, 255], [0, 0, 0, 0], [255, 255, 255, 255]]], dtype=np.uint8)
    transparent_magenta = transparent_black.copy()
    transparent_magenta[0, 1, :3] = [255, 0, 255]

    first = color.prepare_lightness(Image.fromarray(transparent_black, mode="RGBA"))[1]
    second = color.prepare_lightness(Image.fromarray(transparent_magenta, mode="RGBA"))[1]

    assert first[0, 1] == pytest.approx(first[0, 0], abs=1e-6)
    assert np.array_equal(first, second)


def test_restore_uses_core_gamut_mapper_and_measures_chroma_shift() -> None:
    source = Image.new("RGB", (1, 1), (255, 80, 20))
    color_tensor, lightness, alpha = color.prepare_lightness(source)
    adjusted = lightness.copy()
    adjusted[0, 0] = 0.5

    expected_oklab = color_tensor.clone()
    expected_oklab[0, 0, 0] = 0.5
    out_of_gamut = linear_srgb_from_oklab(expected_oklab)
    assert torch.any((out_of_gamut < 0.0) | (out_of_gamut > 1.0))
    gamut_mapped = gamut_clip_tensor(out_of_gamut)
    expected_rgb = (srgb_from_linear_srgb(gamut_mapped).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    result = color.restore_lightness(source, color_tensor, adjusted, alpha)

    pixels = np.asarray(result.convert("RGB"), dtype=np.float32) / 255.0
    actual_rgb = torch.from_numpy(pixels.copy()).permute(2, 0, 1).contiguous()
    actual_oklab = oklab_from_srgb(actual_rgb)
    raw_chroma = torch.linalg.vector_norm(expected_oklab[1:, 0, 0])
    mapped_chroma = torch.linalg.vector_norm(actual_oklab[1:, 0, 0])
    raw_hue = torch.atan2(expected_oklab[2, 0, 0], expected_oklab[1, 0, 0])
    mapped_hue = torch.atan2(actual_oklab[2, 0, 0], actual_oklab[1, 0, 0])
    hue_delta = torch.remainder(mapped_hue - raw_hue + torch.pi, 2.0 * torch.pi) - torch.pi

    assert result.mode == "RGB"
    assert np.isfinite(pixels).all()
    assert np.all((pixels >= 0.0) & (pixels <= 1.0))
    assert torch.equal(torch.from_numpy(np.asarray(result).copy()).permute(2, 0, 1), expected_rgb)
    assert mapped_chroma < raw_chroma
    assert raw_chroma - mapped_chroma == pytest.approx(0.0508, abs=0.002)
    assert abs(hue_delta.item()) < 0.003


def test_restore_clips_out_of_range_lightness_before_color_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Image.new("RGB", (2, 1), (128, 128, 128))
    color_tensor, lightness, alpha = color.prepare_lightness(source)
    lightness[:] = [-0.25, 1.25]
    converted_lightness: list[torch.Tensor] = []
    convert = color.linear_srgb_from_oklab

    def capture_conversion(oklab: torch.Tensor) -> torch.Tensor:
        converted_lightness.append(oklab[0].clone())
        return convert(oklab)

    monkeypatch.setattr(color, "linear_srgb_from_oklab", capture_conversion)

    color.restore_lightness(source, color_tensor, lightness, alpha)

    assert len(converted_lightness) == 1
    assert torch.equal(converted_lightness[0], torch.tensor([[0.0, 1.0]], dtype=torch.float32))


def test_restore_converts_large_images_in_bounded_row_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Image.new("RGB", (8, 600), (128, 128, 128))
    color_tensor, lightness, alpha = color.prepare_lightness(source)
    original_color_tensor = color_tensor.clone()
    converted_shapes: list[tuple[int, ...]] = []
    convert = color.linear_srgb_from_oklab

    def capture_conversion(oklab: torch.Tensor) -> torch.Tensor:
        converted_shapes.append(tuple(oklab.shape))
        return convert(oklab)

    monkeypatch.setattr(color, "linear_srgb_from_oklab", capture_conversion)

    result = color.restore_lightness(source, color_tensor, lightness, alpha)

    assert result.size == source.size
    assert converted_shapes
    assert all(shape[1] <= 256 for shape in converted_shapes)
    assert sum(shape[1] for shape in converted_shapes) == source.height
    assert torch.equal(color_tensor, original_color_tensor)


@pytest.mark.parametrize("invalid_input", ["color_tensor", "lightness"])
def test_restore_rejects_non_finite_tile_inputs(invalid_input: str) -> None:
    source = Image.new("RGB", (2, 300), (128, 128, 128))
    color_tensor, lightness, alpha = color.prepare_lightness(source)
    if invalid_input == "color_tensor":
        color_tensor[0, 257, 0] = float("nan")
    else:
        lightness[257, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        color.restore_lightness(source, color_tensor, lightness, alpha)


def test_core_oklab_gamut_clip_reduces_out_of_gamut_chroma_without_nonfinite_values() -> None:
    red = torch.tensor([[[1.0]], [[0.0]], [[0.0]]], dtype=torch.float32)
    oklab = oklab_from_linear_srgb(linear_srgb_from_srgb(red))
    oklab[1:] *= 2.0
    out_of_gamut = linear_srgb_from_oklab(oklab)

    clipped = gamut_clip_tensor(out_of_gamut)
    clipped_oklab = oklab_from_linear_srgb(clipped)

    assert torch.any((out_of_gamut < 0.0) | (out_of_gamut > 1.0))
    assert torch.isfinite(clipped).all()
    assert torch.all((clipped >= 0.0) & (clipped <= 1.0))
    assert torch.linalg.vector_norm(clipped_oklab[1:, 0, 0]) < torch.linalg.vector_norm(oklab[1:, 0, 0])


def test_prepare_rejects_16_bit_gray_without_saturating_through_pillow_rgb_conversion() -> None:
    values = np.array([[0, 32768, 65535]], dtype=np.uint16)
    source = Image.fromarray(values)
    assert source.mode == "I;16"

    # Reject rather than treating Pillow's saturating RGB conversion as a faithful decode.
    with pytest.raises(ValueError, match="16-bit|I;16|unsupported mode"):
        color.prepare_lightness(source)


@pytest.mark.parametrize("mode", ["CMYK", "F", "P"])
def test_prepare_rejects_unusual_modes_explicitly(mode: str) -> None:
    source = Image.new(mode, (2, 2))

    with pytest.raises(ValueError, match="unsupported|mode"):
        color.prepare_lightness(source)


def test_prepare_reports_clear_error_when_memory_allocation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _image("RGB")

    def fail_allocation(*args: object, **kwargs: object) -> None:
        raise MemoryError

    monkeypatch.setattr(color.torch, "from_numpy", fail_allocation)

    with pytest.raises(RuntimeError, match="memory|allocate|image"):
        color.prepare_lightness(source)

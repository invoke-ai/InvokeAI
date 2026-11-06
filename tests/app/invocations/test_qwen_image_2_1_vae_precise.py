"""Qwen-Image-2.1 `qwen_image_2_1_l2i` and `qwen_image_2_1_i2l` with precise image storage.

The VAE is a stand-in that answers with decodes the test builds (the real one is far too large for CPU tests), so
every expected value comes from the test, not from the node.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from invokeai.app.invocations.vae.qwen_image_2_1_image_to_latents import QwenImage21ImageToLatentsInvocation
from invokeai.app.invocations.vae.qwen_image_2_1_latents_to_image import QwenImage21LatentsToImageInvocation
from invokeai.backend.image_util.raster.descriptor import RasterChannels

_CHANNELS = 4


class _VAE:
    """Answers decodes with `decode_output` and records the pixels it is asked to encode."""

    dtype = torch.float32

    def __init__(self, decode_output: torch.Tensor | None = None) -> None:
        self.config = SimpleNamespace(latents_mean=[0.0] * _CHANNELS, latents_std=[1.0] * _CHANNELS)
        self.decode_output = decode_output
        self.pixels: torch.Tensor | None = None

    def parameters(self):
        return iter([torch.zeros(1, dtype=self.dtype)])

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        return (self.decode_output,)

    def encode(self, pixels: torch.Tensor):
        self.pixels = pixels
        z = torch.zeros(1, _CHANNELS, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))


@contextmanager
def _no_tiling(*_args):
    yield


@pytest.fixture(autouse=True)
def _untiled(monkeypatch) -> None:
    for module in ("latents_to_image", "image_to_latents"):
        monkeypatch.setattr(
            f"invokeai.app.invocations.vae.qwen_image_2_1_{module}.patch_qwen_image_vae_tiling", _no_tiling
        )


def _use(precise_world, vae: _VAE) -> None:
    info = MagicMock()
    info.model = vae
    info.compute_device = torch.device("cpu")

    @contextmanager
    def model_on_device(working_mem_bytes=None):
        yield (None, vae)

    info.model_on_device = model_on_device
    precise_world.use_vae(info)
    precise_world.context.config = SimpleNamespace(
        get=lambda: SimpleNamespace(force_tiled_decode=False, auto_tiled_decode=False)
    )
    precise_world.context.logger = MagicMock()


def _decode_output(transparent: bool, dtype: torch.dtype = torch.float32, stray: bool = False) -> torch.Tensor:
    """A `(1, 4, 1, 32, 48)` decode in [-1, 1]; opaque alpha, or alpha -1 on the left half when `transparent`, or on
    a single pixel when `stray` (within the share of transparent pixels the legacy image still calls opaque)."""
    generator = torch.Generator().manual_seed(1)
    output = torch.rand(1, 4, 1, 32, 48, generator=generator) * 2 - 1
    output[:, 3] = 1.0
    if transparent:
        output[:, 3, :, :, :24] = -1.0
    if stray:
        output[:, 3, :, 5, 5] = -1.0
    return output.to(dtype)


def _decode(precise_world, vae: _VAE) -> str:
    _use(precise_world, vae)
    precise_world.tensors["latents"] = torch.zeros(1, _CHANNELS, 2, 3)
    node = QwenImage21LatentsToImageInvocation.model_construct(
        id="l2i",
        latents=SimpleNamespace(latents_name="latents"),
        vae=SimpleNamespace(vae="vae"),
        tiled=False,
        tile_size=0,
    )
    return node.invoke(precise_world.context).image.image_name


@pytest.mark.parametrize("transparent", [False, True])
def test_precise_decode_stores_the_decoder_output_in_the_channels_the_gallery_image_has(
    precise_world, transparent: bool
) -> None:
    output = _decode_output(transparent)
    vae = _VAE(output)
    precise_world.configuration.precise_image_storage = False
    disabled = _decode(precise_world, vae)
    precise_world.configuration.precise_image_storage = True

    name = _decode(precise_world, vae)

    channels = RasterChannels.RGBA if transparent else RasterChannels.RGB
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == precise_world.descriptor(channels)
    expected = ((output[0, : len(channels.value), 0].float() + 1.0) / 2).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(raster.view(), expected)
    assert precise_world.stored_bytes(name) == precise_world.stored_bytes(disabled)
    assert Image.open(precise_world.storage.get_path(name)).mode == ("RGBA" if transparent else "RGB")


def test_a_few_transparent_pixels_keep_the_legacy_rgb_image_and_an_rgb_raster(precise_world) -> None:
    output = _decode_output(False, stray=True)

    name = _decode(precise_world, _VAE(output))

    raster = precise_world.images.get_raster(name)
    assert raster.descriptor == precise_world.descriptor(RasterChannels.RGB)
    np.testing.assert_array_equal(raster.view(), ((output[0, :3, 0] + 1.0) / 2).permute(1, 2, 0).numpy())
    assert Image.open(precise_world.storage.get_path(name)).mode == "RGB"


def test_disabled_decode_saves_an_ordinary_8_bit_image(precise_world) -> None:
    precise_world.configuration.precise_image_storage = False

    name = _decode(precise_world, _VAE(_decode_output(False)))

    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.database.queries.image_precise_assets.get(name) is None


def test_a_bf16_decode_is_widened_before_the_raster_is_mapped_to_unit_range(precise_world) -> None:
    output = _decode_output(False, torch.bfloat16)
    name = _decode(precise_world, _VAE(output))

    widened = ((output[0, :3, 0].float() + 1.0) / 2).permute(1, 2, 0).numpy()
    in_bf16 = ((output[0, :3, 0] + 1.0) / 2).float().permute(1, 2, 0).numpy()
    assert (widened != in_bf16).any()  # otherwise the case could not tell the two orders apart
    np.testing.assert_array_equal(precise_world.images.get_raster(name).view(), widened)


def test_non_finite_decoder_output_falls_back_to_the_ordinary_image_and_warns(precise_world) -> None:
    output = _decode_output(False)
    output[0, 1, 0, 5, 5] = float("nan")

    name = _decode(precise_world, _VAE(output))

    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.database.queries.image_precise_assets.get(name) is None
    precise_world.context.logger.warning.assert_called_once()
    assert "not finite" in precise_world.context.logger.warning.call_args.args[0]


# --- encode ---------------------------------------------------------------------------------------------------------


def _encode(precise_world, image_name: str, reference: bool) -> tuple[_VAE, SimpleNamespace]:
    vae = _VAE()
    _use(precise_world, vae)
    node = QwenImage21ImageToLatentsInvocation.model_construct(
        id="i2l",
        image=SimpleNamespace(image_name=image_name),
        vae=SimpleNamespace(vae="vae"),
        tiled=False,
        tile_size=0,
        reference=reference,
    )
    return vae, node.invoke(precise_world.context)


def _rgba(height: int, width: int) -> np.ndarray:
    """Float RGBA with colour outside [0, 1] and varied straight alpha."""
    array = np.random.default_rng(3).uniform(-0.4, 1.6, size=(height, width, 4)).astype(np.float32)
    array[..., 3] = np.random.default_rng(4).uniform(0.1, 0.9, size=(height, width))
    return array


def _chw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _resized(array: np.ndarray, height: int, width: int) -> torch.Tensor:
    return F.interpolate(_chw(array)[None], size=(height, width), mode="bicubic", align_corners=False, antialias=True)


def _alpha_weighted(array: np.ndarray, height: int, width: int) -> torch.Tensor:
    """The expected RGBA resample: colour * alpha and alpha resized, colour divided back where alpha is at least
    Pillow's smallest alpha step and kept within the source colour range, 0 elsewhere."""
    alpha = np.clip(array[..., 3:], 0.0, 1.0)
    resized = _resized(np.concatenate([array[..., :3] * alpha, alpha], axis=2), height, width)[0]
    opacity = resized[3].clamp(0, 1)
    color = torch.zeros_like(resized[:3])
    visible = opacity >= 0.5 / 255
    color[:, visible] = (resized[:3][:, visible] / opacity[visible]).clamp(array[..., :3].min(), array[..., :3].max())
    return torch.cat([color, opacity[None]])


def _unbounded(array: np.ndarray, height: int, width: int) -> torch.Tensor:
    """The colour of a plain alpha-weighted resample (no threshold, no bound): the guard that the fix is needed."""
    alpha = np.clip(array[..., 3:], 0.0, 1.0)
    resized = _resized(np.concatenate([array[..., :3] * alpha, alpha], axis=2), height, width)[0]
    return resized[:3] / resized[3].clamp_min(1e-12)


def _shape(height: int, width: int, antialiased_edge: bool) -> np.ndarray:
    """A disc of flat blue over noisy colour under alpha 0, with a hard edge or a 1-px antialiased one."""
    yy, xx = np.mgrid[:height, :width]
    distance = np.hypot(yy - height / 2, xx - width / 2) - min(height, width) / 4
    alpha = np.clip(0.5 - distance, 0.0, 1.0) if antialiased_edge else (distance < 0).astype(np.float32)
    array = np.random.default_rng(6).uniform(0.0, 1.0, size=(height, width, 4)).astype(np.float32)
    array[..., :3] = np.where(alpha[..., None] > 0, [0.2, 0.4, 0.8], array[..., :3])
    array[..., 3] = alpha
    return array


@pytest.mark.parametrize("size", [(32, 64), (45, 70)])
def test_an_image_without_a_precise_asset_encodes_as_the_eight_bit_path_did(precise_world, size) -> None:
    pixels = np.random.default_rng(2).integers(0, 256, size=(*size, 4), dtype=np.uint8)
    name = precise_world.store_png(pixels, "RGBA")
    width, height = size[1] // 32 * 32, size[0] // 32 * 32
    image = Image.fromarray(pixels, "RGBA")
    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.LANCZOS)
    expected = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1) * 2.0 - 1.0
    expected[3] = 1.0

    vae, output = _encode(precise_world, name, reference=False)

    assert (output.width, output.height) == (width, height)
    torch.testing.assert_close(vae.pixels[0, :, 0], expected, rtol=0, atol=0)


def test_a_precise_image_on_the_grid_encodes_its_unclamped_colour_with_alpha_forced_to_one(precise_world) -> None:
    array = _rgba(32, 64)
    name = precise_world.store_precise(array, RasterChannels.RGBA)

    vae, output = _encode(precise_world, name, reference=False)

    expected = _chw(array) * 2.0 - 1.0
    expected[3] = 1.0
    assert (output.width, output.height) == (64, 32)
    assert expected[:3].min() < -1.0 and expected[:3].max() > 1.0
    torch.testing.assert_close(vae.pixels[0, :, 0], expected, rtol=0, atol=0)


def test_a_precise_image_off_the_grid_is_resized_without_a_crop(precise_world) -> None:
    # 48 x 80 snaps to 32 x 64 on the 32 grid; a 16 grid would keep it at 48 x 80.
    array = _rgba(48, 80)
    name = precise_world.store_precise(array, RasterChannels.RGBA)

    vae, output = _encode(precise_world, name, reference=False)

    expected = _alpha_weighted(array, 32, 64) * 2.0 - 1.0
    expected[3] = 1.0
    assert (output.width, output.height) == (64, 32)
    torch.testing.assert_close(vae.pixels[0, :, 0], expected)


@pytest.mark.parametrize("reference", [False, True])
def test_a_resize_does_not_bleed_the_colour_under_transparent_pixels(precise_world, reference: bool) -> None:
    # Red under fully transparent pixels next to opaque blue, as Pillow's premultiplied RGBA resize treats them.
    height, width = (48, 80) if not reference else (40, 100)
    array = np.zeros((height, width, 4), dtype=np.float32)
    array[:, : width // 2] = [1.0, 0.0, 0.0, 0.0]
    array[:, width // 2 :] = [0.0, 0.0, 1.0, 1.0]
    name = precise_world.store_precise(array, RasterChannels.RGBA)

    vae, output = _encode(precise_world, name, reference=reference)

    pixels = vae.pixels[0, :, 0]
    unit_color = (pixels[:3] + 1.0) / 2
    assert pixels.shape[-2:] == (output.height, output.width)
    assert (output.height, output.width) != (height, width)
    # Red exists only under alpha 0, so no output pixel carries any of it.
    assert unit_color[0].abs().max() < 1e-6
    margin = output.width // 10  # the resample blurs the edge by a few source pixels
    # The transparent half comes back with colour 0; the opaque half keeps its blue.
    assert unit_color[:, :, : output.width // 2 - margin].abs().max() < 1e-6
    torch.testing.assert_close(
        unit_color[2, :, output.width // 2 + margin :],
        torch.ones(output.height, output.width - output.width // 2 - margin),
    )
    # An unweighted resample would put red into the opaque pixels next to the edge.
    assert _resized(array, output.height, output.width)[0, 0, :, output.width // 2 :].max() > 1e-3
    weighted = _alpha_weighted(array, output.height, output.width)
    weighted[3] = weighted[3].clamp(0, 1) if reference else 1.0
    torch.testing.assert_close(pixels, weighted * 2.0 - 1.0)


def test_a_precise_image_below_the_grid_is_refused(precise_world) -> None:
    name = precise_world.store_precise(_rgba(20, 70), RasterChannels.RGBA)

    with pytest.raises(ValueError, match="at least 32x32"):
        _encode(precise_world, name, reference=False)


def test_a_precise_reference_is_resized_to_its_reading_size_and_keeps_a_clamped_alpha(precise_world) -> None:
    array = _rgba(300, 500)
    array[::7, ::5, 3] = 1.0
    array[3::7, 2::5, 3] = 0.0  # hard alpha edges overshoot under bicubic
    name = precise_world.store_precise(array, RasterChannels.RGBA)

    vae, output = _encode(precise_world, name, reference=True)

    assert (output.width, output.height) == (1312, 800)
    resampled = _resized(array, 800, 1312)[0]
    assert resampled[3].min() < 0.0 or resampled[3].max() > 1.0  # otherwise the clamp is not exercised
    expected = _alpha_weighted(array, 800, 1312)
    expected[3] = expected[3].clamp(0, 1)
    torch.testing.assert_close(vae.pixels[0, :, 0], expected * 2.0 - 1.0)
    assert not torch.equal(vae.pixels[0, 3], torch.ones(800, 1312))


@pytest.mark.parametrize("antialiased_edge", [False, True])
@pytest.mark.parametrize(("reference", "size"), [(False, (90, 150)), (True, (90, 150)), (True, (1500, 1700))])
def test_a_resize_keeps_the_colour_within_the_source_range_at_hard_and_one_pixel_alpha_edges(
    precise_world, antialiased_edge: bool, reference: bool, size
) -> None:
    array = _shape(*size, antialiased_edge)
    name = precise_world.store_precise(array, RasterChannels.RGBA)

    vae, output = _encode(precise_world, name, reference=reference)

    pixels = vae.pixels[0, :, 0]
    assert (output.height, output.width) != size
    color = (pixels[:3] + 1.0) / 2
    low, high = array[..., :3].min(), array[..., :3].max()
    weighted = _alpha_weighted(array, output.height, output.width)
    # Where the resampled alpha rounds to nothing the colour is 0, whatever noise sat under the source's alpha 0;
    # everywhere else it stays within the source colour range.
    transparent = weighted[3] < 0.5 / 255
    assert transparent.any()
    assert (color[:, transparent] == 0).all()
    visible = color[:, ~transparent]
    assert visible.min() >= low - 1e-6 and visible.max() <= high + 1e-6
    # Without the bound, ringing at the edge would leave the range (the bug this pins).
    unbounded = _unbounded(array, output.height, output.width)
    assert unbounded.min() < low - 0.05 or unbounded.max() > high + 0.05
    weighted[3] = weighted[3] if reference else 1.0
    torch.testing.assert_close(pixels, weighted * 2.0 - 1.0)


def test_source_alpha_outside_zero_to_one_is_clamped_before_it_weights_the_colour(precise_world) -> None:
    array = _rgba(48, 80)
    array[::3, ::2, 3] = 1.3
    array[1::3, 1::2, 3] = -0.2
    name = precise_world.store_precise(array, RasterChannels.RGBA)

    vae, _ = _encode(precise_world, name, reference=False)

    weighted = _alpha_weighted(array, 32, 64)
    weighted[3] = 1.0
    torch.testing.assert_close(vae.pixels[0, :, 0], weighted * 2.0 - 1.0)
    # Weighting by the raw alpha would give another result, so the clamp is what the expectation pins.
    raw = np.concatenate([array[..., :3] * array[..., 3:], array[..., 3:]], axis=2)
    raw_color = _resized(raw, 32, 64)[0, :3] / _resized(raw, 32, 64)[0, 3].clamp_min(1e-3)
    assert not torch.allclose(raw_color, weighted[:3], atol=1e-3)

"""`esrgan` with precise image storage: a tiny real RRDBNet (random weights) through the real RealESRGAN code on CPU.

Only the model download is replaced; the network, padding, tiling and rounding are the shipped code."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.app.invocations import upscale
from invokeai.app.invocations.upscale import ESRGANInvocation
from invokeai.backend.image_util.basicsr.rrdbnet_arch import RRDBNet
from invokeai.backend.image_util.raster.descriptor import RasterChannels
from invokeai.backend.image_util.realesrgan.realesrgan import RealESRGAN

SCALE = 4


def _tiny_net(**_kwargs) -> RRDBNet:
    torch.manual_seed(0)
    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=SCALE, num_feat=4, num_block=1, num_grow_ch=4)
    with torch.no_grad():
        net.conv_last.weight.mul_(8.0)  # push the output beyond [0, 1] so clamping would be visible
        net.conv_last.bias.add_(0.4)
    return net


@pytest.fixture
def esrgan(precise_world, monkeypatch):
    """Runs the node on an image; `inputs` collects what reached the network (the padded NCHW tensors)."""
    monkeypatch.setattr(upscale, "RRDBNet", _tiny_net)
    monkeypatch.setattr(
        "invokeai.backend.image_util.realesrgan.realesrgan.TorchDevice.choose_torch_device",
        staticmethod(lambda: torch.device("cpu")),
    )
    state = {"params": _tiny_net().state_dict()}

    @contextmanager
    def loaded(**_kwargs):
        yield state

    context = precise_world.context
    context.models = SimpleNamespace(load_remote_model=loaded)
    context.logger = MagicMock()
    inputs: list[torch.Tensor] = []
    real_forward = RRDBNet.forward

    def spying_forward(self, x):
        inputs.append(x.clone())
        return real_forward(self, x)

    monkeypatch.setattr(RRDBNet, "forward", spying_forward)

    def run(image_name: str, tile_size: int = 0) -> str:
        node = ESRGANInvocation.model_construct(
            id="esrgan",
            image=SimpleNamespace(image_name=image_name),
            model_name="RealESRGAN_x4plus.pth",
            tile_size=tile_size,
        )
        return node.invoke(context).image.image_name

    return SimpleNamespace(run=run, inputs=inputs, context=context, weights=state)


def _rgb8(height: int = 12, width: int = 17, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _legacy_upscale(rgb8: np.ndarray, tile_size: int) -> np.ndarray:
    """What the node produced before this change: the vendored `upscale` on the BGR uint8 image (an independent call
    of the unchanged legacy function)."""
    scaler = RealESRGAN(scale=SCALE, loadnet={"params": _tiny_net().state_dict()}, model=_tiny_net(), tile=tile_size)
    bgr = np.ascontiguousarray(rgb8[:, :, ::-1])
    return np.asarray(Image.fromarray(scaler.upscale(bgr)[:, :, ::-1]).convert("RGBA"))


def _png_of(precise_world, image: Image.Image) -> bytes:
    was = precise_world.configuration.precise_image_storage
    precise_world.configuration.precise_image_storage = False
    try:
        return precise_world.stored_bytes(precise_world.images.save(image).image_name)
    finally:
        precise_world.configuration.precise_image_storage = was


@pytest.mark.parametrize("tile_size", [0, 8])
def test_switch_off_saves_the_legacy_png(precise_world, esrgan, tile_size) -> None:
    precise_world.configuration.precise_image_storage = False
    rgb8 = _rgb8()
    name = esrgan.run(precise_world.store_png(rgb8, "RGB"), tile_size)

    expected = Image.fromarray(_legacy_upscale(rgb8, tile_size), "RGBA")
    assert precise_world.stored_bytes(name) == _png_of(precise_world, expected)
    assert precise_world.images.get_raster(name).dtype == np.uint8


def test_switch_on_without_asset_stores_the_float_output_behind_the_legacy_png(precise_world, esrgan) -> None:
    rgb8 = _rgb8()
    name = esrgan.run(precise_world.store_png(rgb8, "RGB"))

    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == precise_world.descriptor(RasterChannels.RGBA)
    pixels = raster.view()
    assert pixels.shape == (12 * SCALE, 17 * SCALE, 4)
    assert (pixels[:, :, 3] == 1.0).all()
    # The float output really leaves [0, 1], so the raster is not just the clamped legacy result.
    assert pixels[:, :, :3].min() < 0.0 and pixels[:, :, :3].max() > 1.0
    # The network's own output: the input padded as RealESRGAN does, cropped back after the forward pass.
    with torch.no_grad():
        scaled = torch.from_numpy(np.ascontiguousarray(rgb8.transpose(2, 0, 1) / np.float32(255)))[None]
        padded = torch.nn.functional.pad(scaled.float(), (0, 10, 0, 10), "reflect")
        out = _tiny_net()(padded)[0, :, : 12 * SCALE, : 17 * SCALE].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(pixels[:, :, :3], out)
    expected = Image.fromarray(_legacy_upscale(rgb8, 0), "RGBA")
    assert precise_world.stored_bytes(name) == _png_of(precise_world, expected)


def test_switch_on_with_tiling_keeps_the_float_output_behind_the_legacy_png(precise_world, esrgan) -> None:
    rgb8 = _rgb8()
    name = esrgan.run(precise_world.store_png(rgb8, "RGB"), tile_size=8)

    legacy = _legacy_upscale(rgb8, 8)
    pixels = precise_world.images.get_raster(name).view()
    quantized = (np.clip(pixels[:, :, :3], 0, 1) * 255.0).round().astype(np.uint8)
    np.testing.assert_array_equal(quantized, legacy[:, :, :3])
    assert pixels[:, :, :3].min() < 0.0 and pixels[:, :, :3].max() > 1.0
    assert precise_world.stored_bytes(name) == _png_of(precise_world, Image.fromarray(legacy, "RGBA"))


@pytest.mark.parametrize("channels", [RasterChannels.RGB, RasterChannels.RGBA])
def test_a_precise_input_reaches_the_network_unquantized(precise_world, esrgan, channels) -> None:
    rng = np.random.default_rng(5)
    # Off the 1/255 grid and overshooting [0, 1]; alpha (when present) must be dropped.
    stored = rng.random((12, 17, len(channels.value)), dtype=np.float32) * 1.2 - 0.1
    name = esrgan.run(precise_world.store_precise(stored, channels))

    assert len(esrgan.inputs) == 1
    reached = esrgan.inputs[0][0, :, :12, :17].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(reached, stored[:, :, :3])
    assert (reached * 255 != np.round(reached * 255)).any()
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32 and raster.descriptor == precise_world.descriptor(RasterChannels.RGBA)


def test_a_precise_input_with_the_switch_off_still_feeds_the_floats_and_saves_8_bit(precise_world, esrgan) -> None:
    stored = np.random.default_rng(6).random((12, 17, 3), dtype=np.float32)
    name = precise_world.store_precise(stored, RasterChannels.RGB)
    on = esrgan.run(name)
    precise_world.configuration.precise_image_storage = False
    off = esrgan.run(name)

    for reached in esrgan.inputs:
        np.testing.assert_array_equal(reached[0, :, :12, :17].permute(1, 2, 0).numpy(), stored)
    assert precise_world.images.get_raster(off).dtype == np.uint8
    assert precise_world.stored_bytes(off) == precise_world.stored_bytes(on)


def test_non_finite_output_saves_the_8_bit_image_and_warns(precise_world, esrgan) -> None:
    esrgan.weights["params"]["conv_last.bias"][1] = float("nan")
    name = esrgan.run(precise_world.store_png(_rgb8(), "RGB"))

    assert precise_world.images.get_raster(name).dtype == np.uint8
    esrgan.context.logger.warning.assert_called_once()


def _forbid_float_path(monkeypatch) -> None:
    monkeypatch.setattr(RealESRGAN, "upscale_rgb_unit", lambda *a, **k: pytest.fail("the float path must not run"))
    monkeypatch.setattr(upscale, "float_output_raster", lambda *a: pytest.fail("a float raster was built"))


def test_switch_off_without_asset_never_takes_the_float_path(precise_world, esrgan, monkeypatch) -> None:
    precise_world.configuration.precise_image_storage = False
    _forbid_float_path(monkeypatch)

    name = esrgan.run(precise_world.store_png(_rgb8(), "RGB"))

    assert precise_world.images.get_raster(name).dtype == np.uint8


def test_an_output_beyond_the_asset_bounds_runs_the_legacy_path_without_a_float_buffer(
    precise_world, esrgan, monkeypatch
) -> None:
    rgb8 = _rgb8()
    source = precise_world.store_png(rgb8, "RGB")
    monkeypatch.setattr("invokeai.backend.image_util.raster.precise_asset.MAX_PIXEL_BYTES", 1000)
    _forbid_float_path(monkeypatch)

    name = esrgan.run(source)

    esrgan.context.logger.warning.assert_called_once()
    assert precise_world.images.get_raster(name).dtype == np.uint8
    expected = Image.fromarray(_legacy_upscale(rgb8, 0), "RGBA")
    assert precise_world.stored_bytes(name) == _png_of(precise_world, expected)

"""`spandrel_image_to_image` and its autoscale variant with precise image storage, on CPU with a tiny real Compact
(Real-ESRGAN SRVGG) model."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.app.invocations import spandrel_image_to_image as spandrel_node
from invokeai.app.invocations.spandrel_image_to_image import (
    SpandrelImageToImageAutoscaleInvocation,
    SpandrelImageToImageInvocation,
)
from invokeai.backend.image_util.raster.descriptor import RasterChannels
from invokeai.backend.spandrel_image_to_image_model import SpandrelImageToImageModel
from invokeai.backend.util.devices import TorchDevice
from tests.fixtures.spandrel_models import write_compact_upscaler


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))


class _Model(SpandrelImageToImageModel):
    """A real model that records every tensor it is run on and every tensor it returns."""

    def __init__(self, model: SpandrelImageToImageModel, output_transform=None) -> None:
        super().__init__(model._spandrel_model)
        self.real = model
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.output_transform = output_transform

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        self.inputs.append(tensor.clone())
        output = self.real.run(tensor)
        if self.output_transform is not None:
            output = self.output_transform(output)
        self.outputs.append(output.clone())
        return output


@pytest.fixture
def make_model(tmp_path):
    def build(scale: int, output_transform=None) -> _Model:
        torch.manual_seed(0)
        path = write_compact_upscaler(tmp_path / f"compact-{scale}.safetensors", scale)
        return _Model(SpandrelImageToImageModel.load_from_file(path), output_transform)

    return build


def _wire(precise_world, model: _Model) -> MagicMock:
    @contextmanager
    def load(_field):
        yield model

    precise_world.context.models = SimpleNamespace(load=load)
    precise_world.context.util = SimpleNamespace(is_canceled=lambda: False, signal_progress=lambda *a, **k: None)
    precise_world.context.logger = MagicMock()
    return precise_world.context.logger


def _rgb(width: int, height: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _float_source(width: int, height: int, seed: int = 2) -> np.ndarray:
    """A float32 image with values past 1 and a step far finer than 1/255."""
    rng = np.random.default_rng(seed)
    return (rng.random((height, width, 3), dtype=np.float32) * 1.25 - 0.1).astype(np.float32)


def _node(cls, name: str, tile_size: int, **fields):
    return cls.model_construct(
        id="spandrel",
        image=SimpleNamespace(image_name=name),
        image_to_image_model=SimpleNamespace(),
        tile_size=tile_size,
        **fields,
    )


def _hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].permute(1, 2, 0).numpy()


def _legacy_tiled(precise_world, model: _Model, name: str, tile_size: int) -> Image.Image:
    """The pre-change result for an 8-bit image: `upscale_image` without a float buffer."""
    image = precise_world.images.get_pil(name, mode="RGB")
    return SpandrelImageToImageInvocation.upscale_image(image, tile_size, model, lambda: False, lambda *_: None)


def _display_of(unit: np.ndarray) -> np.ndarray:
    return (np.clip(unit, 0, 1) * 255).astype(np.uint8)


@pytest.mark.parametrize("tile_size", [0, 32])
def test_without_asset_and_switch_off_the_result_is_the_legacy_png(precise_world, make_model, tile_size) -> None:
    model = make_model(2)
    _wire(precise_world, model)
    precise_world.configuration.precise_image_storage = False
    name = precise_world.store_png(_rgb(45, 41), "RGB")
    expected = _legacy_tiled(precise_world, model, name, tile_size)

    out = _node(SpandrelImageToImageInvocation, name, tile_size).invoke(precise_world.context).image.image_name

    stored = precise_world.images.get_raster(out)
    assert stored.dtype == np.uint8
    assert np.array_equal(stored.view(), np.asarray(expected))


@pytest.mark.parametrize("tile_size", [0, 32])
def test_precise_output_is_the_models_float_output_and_the_png_stays_legacy(
    precise_world, make_model, tile_size
) -> None:
    # Stretch the output so it leaves [0, 1] the way a real upscaler's does.
    model = make_model(2, output_transform=lambda output: output * 1.6 - 0.3)
    _wire(precise_world, model)
    name = precise_world.store_png(_rgb(45, 41), "RGB")
    expected = _legacy_tiled(precise_world, model, name, tile_size)
    tiles = len(model.inputs)
    model.inputs.clear()
    model.outputs.clear()

    out = _node(SpandrelImageToImageInvocation, name, tile_size).invoke(precise_world.context).image.image_name

    raster = precise_world.images.get_raster(out)
    assert raster.dtype == np.float32 and raster.view().shape == (82, 90, 3)
    # The model ran once per tile, not twice.
    assert len(model.outputs) == tiles
    values = raster.view()
    # The float raster has overshoot the gallery PNG clips, and it was assembled with the legacy seams: quantized
    # the legacy way, it is the legacy image, byte for byte.
    assert values.min() < 0 or values.max() > 1
    assert np.array_equal(_display_of(values), np.asarray(expected))
    assert np.array_equal(np.asarray(precise_world.images.get_pil(out)), np.asarray(expected))
    if tile_size == 0:
        assert np.array_equal(values, _hwc(model.outputs[0]))


def test_precise_input_reaches_the_model_unquantized(precise_world, make_model) -> None:
    model = make_model(1)
    _wire(precise_world, model)
    source = _float_source(21, 17)
    name = precise_world.store_precise(source, RasterChannels.RGB)

    out = _node(SpandrelImageToImageInvocation, name, 0).invoke(precise_world.context).image.image_name

    assert source.max() > 1
    assert np.array_equal(_hwc(model.inputs[0]), source)
    assert np.array_equal(precise_world.images.get_raster(out).view(), _hwc(model.outputs[0]))


def test_precise_input_with_the_switch_off_feeds_floats_and_saves_the_legacy_png(precise_world, make_model) -> None:
    model = make_model(2)
    _wire(precise_world, model)
    source = _float_source(21, 17)
    name = precise_world.store_precise(source, RasterChannels.RGB)
    precise_world.configuration.precise_image_storage = False

    out = _node(SpandrelImageToImageInvocation, name, 0).invoke(precise_world.context).image.image_name

    assert np.array_equal(_hwc(model.inputs[0]), source)
    stored = precise_world.images.get_raster(out)
    assert stored.dtype == np.uint8
    assert np.array_equal(stored.view(), _display_of(_hwc(model.outputs[0])))


def test_non_finite_output_saves_the_ordinary_image_with_a_warning(precise_world, make_model) -> None:
    def poison(output: torch.Tensor) -> torch.Tensor:
        output = output.clone()
        output[0, 0, 0, 0] = float("nan")
        return output

    model = make_model(2, output_transform=poison)
    logger = _wire(precise_world, model)
    name = precise_world.store_png(_rgb(16, 16), "RGB")

    out = _node(SpandrelImageToImageInvocation, name, 0).invoke(precise_world.context).image.image_name

    assert precise_world.images.get_raster(out).dtype == np.uint8
    assert "not finite" in logger.warning.call_args.args[0]


def test_autoscale_without_asset_and_switch_off_is_the_legacy_chain(precise_world, make_model) -> None:
    model = make_model(2)
    _wire(precise_world, model)
    precise_world.configuration.precise_image_storage = False
    name = precise_world.store_png(_rgb(35, 33), "RGB")
    # 2x model, 3x target: two passes (4x), then the Lanczos resize down.
    first = _legacy_tiled(precise_world, model, name, 0)
    second = SpandrelImageToImageInvocation.upscale_image(first, 0, model, lambda: False, lambda *_: None)
    expected = second.resize((105, 99), resample=Image.Resampling.LANCZOS)

    node = _node(SpandrelImageToImageAutoscaleInvocation, name, 0, scale=3.0, fit_to_multiple_of_8=False)
    out = node.invoke(precise_world.context).image.image_name

    assert np.array_equal(precise_world.images.get_raster(out).view(), np.asarray(expected))


@pytest.mark.parametrize(("scale", "size"), [(3.0, (105, 99)), (4.0, (140, 132)), (1.5, (52, 49))])
def test_autoscale_precise_result_keeps_the_last_pass_in_float_and_the_png_stays_legacy(
    precise_world, make_model, scale, size
) -> None:
    model = make_model(2)
    _wire(precise_world, model)
    name = precise_world.store_png(_rgb(35, 33), "RGB")
    node = _node(SpandrelImageToImageAutoscaleInvocation, name, 32, scale=scale, fit_to_multiple_of_8=False)
    precise_world.configuration.precise_image_storage = False
    legacy = np.asarray(precise_world.images.get_pil(node.invoke(precise_world.context).image.image_name))
    precise_world.configuration.precise_image_storage = True

    out = node.invoke(precise_world.context).image.image_name

    raster = precise_world.images.get_raster(out)
    assert raster.dtype == np.float32
    assert raster.view().shape == (size[1], size[0], 3)
    assert np.array_equal(np.asarray(precise_world.images.get_pil(out)), legacy)
    # Only the final resize can differ from the 8-bit image: it filters unclamped values (and unquantized ones), so
    # it differs near clipped extremes but stays the same picture. With no resize the two agree exactly.
    difference = np.abs(_display_of(raster.view()).astype(int) - legacy.astype(int))
    assert difference.max() == 0 if scale == 4.0 else difference.mean() < 0.5


def test_autoscale_precise_input_feeds_the_first_pass_unquantized(precise_world, make_model) -> None:
    model = make_model(2)
    _wire(precise_world, model)
    source = _float_source(35, 33)
    name = precise_world.store_precise(source, RasterChannels.RGB)

    node = _node(SpandrelImageToImageAutoscaleInvocation, name, 0, scale=2.0, fit_to_multiple_of_8=False)
    out = node.invoke(precise_world.context).image.image_name

    assert len(model.outputs) == 1
    assert np.array_equal(_hwc(model.inputs[0]), source)
    # A 2x target with a 2x model needs no resize, so the stored result is exactly the model's output.
    assert np.array_equal(precise_world.images.get_raster(out).view(), _hwc(model.outputs[0]))


def _forbid_float_buffers(monkeypatch) -> None:
    """Any float buffer the nodes would allocate comes from `_float_output_buffer`; no raster may be built either."""
    real = SpandrelImageToImageInvocation._float_output_buffer

    def no_buffer(*args):
        assert real(*args) is None, "a float buffer was allocated"

    monkeypatch.setattr(SpandrelImageToImageInvocation, "_float_output_buffer", staticmethod(no_buffer))
    monkeypatch.setattr(spandrel_node, "float_output_raster", lambda *a: pytest.fail("a float raster was built"))


@pytest.mark.parametrize("cls", [SpandrelImageToImageInvocation, SpandrelImageToImageAutoscaleInvocation])
def test_switch_off_without_asset_never_allocates_a_float_buffer(precise_world, make_model, monkeypatch, cls) -> None:
    model = make_model(2)
    _wire(precise_world, model)
    precise_world.configuration.precise_image_storage = False
    _forbid_float_buffers(monkeypatch)
    name = precise_world.store_png(_rgb(35, 33), "RGB")
    fields = {"scale": 3.0, "fit_to_multiple_of_8": True} if cls is SpandrelImageToImageAutoscaleInvocation else {}

    out = _node(cls, name, 0, **fields).invoke(precise_world.context).image.image_name

    assert precise_world.images.get_raster(out).dtype == np.uint8


@pytest.mark.parametrize("cls", [SpandrelImageToImageInvocation, SpandrelImageToImageAutoscaleInvocation])
def test_an_output_beyond_the_asset_bounds_runs_the_legacy_path_without_a_float_buffer(
    precise_world, make_model, monkeypatch, cls
) -> None:
    model = make_model(2)
    logger = _wire(precise_world, model)
    name = precise_world.store_png(_rgb(35, 33), "RGB")
    fields = {"scale": 3.0, "fit_to_multiple_of_8": False} if cls is SpandrelImageToImageAutoscaleInvocation else {}
    precise_world.configuration.precise_image_storage = False
    legacy = precise_world.stored_bytes(_node(cls, name, 0, **fields).invoke(precise_world.context).image.image_name)
    precise_world.configuration.precise_image_storage = True
    monkeypatch.setattr("invokeai.backend.image_util.raster.precise_asset.MAX_PIXEL_BYTES", 1000)
    _forbid_float_buffers(monkeypatch)

    out = _node(cls, name, 0, **fields).invoke(precise_world.context).image.image_name

    logger.warning.assert_called()
    assert precise_world.images.get_raster(out).dtype == np.uint8
    assert precise_world.stored_bytes(out) == legacy


def test_autoscale_precise_input_reaches_only_the_first_of_several_passes(precise_world, make_model) -> None:
    model = make_model(2)
    _wire(precise_world, model)
    source = _float_source(35, 33)
    name = precise_world.store_precise(source, RasterChannels.RGB)

    node = _node(SpandrelImageToImageAutoscaleInvocation, name, 0, scale=4.0, fit_to_multiple_of_8=False)
    out = node.invoke(precise_world.context).image.image_name

    assert len(model.inputs) == 2
    assert np.array_equal(_hwc(model.inputs[0]), source)
    second = _hwc(model.inputs[1]) * 255  # the first pass's 8-bit result
    assert np.array_equal(second, np.round(second))
    # Four times the size needs no resize, so the stored result is the last pass's output.
    assert np.array_equal(precise_world.images.get_raster(out).view(), _hwc(model.outputs[1]))


def test_autoscale_fit_to_multiple_of_8_resizes_the_float_result_to_the_fitted_size(precise_world, make_model) -> None:
    model = make_model(2)
    _wire(precise_world, model)
    name = precise_world.store_png(_rgb(35, 33), "RGB")

    node = _node(SpandrelImageToImageAutoscaleInvocation, name, 0, scale=3.0, fit_to_multiple_of_8=True)
    out = node.invoke(precise_world.context).image.image_name

    raster = precise_world.images.get_raster(out)
    assert raster.dtype == np.float32 and raster.view().shape == (96, 104, 3)
    assert precise_world.images.get_pil(out).size == (104, 96)


def test_autoscale_with_a_model_that_does_not_upscale_stores_the_float_output(precise_world, make_model) -> None:
    model = make_model(1)
    logger = _wire(precise_world, model)
    source = _float_source(35, 33)
    name = precise_world.store_precise(source, RasterChannels.RGB)

    node = _node(SpandrelImageToImageAutoscaleInvocation, name, 0, scale=4.0, fit_to_multiple_of_8=False)
    out = node.invoke(precise_world.context).image.image_name

    logger.warning.assert_called_once()  # the "will not be scaled" notice, nothing else
    assert len(model.outputs) == 1
    assert np.array_equal(precise_world.images.get_raster(out).view(), _hwc(model.outputs[0]))


def test_autoscale_with_a_model_that_does_not_upscale_ignores_the_requested_target_for_the_bounds(
    precise_world, make_model, monkeypatch
) -> None:
    model = make_model(1)
    _wire(precise_world, model)
    name = precise_world.store_precise(_float_source(35, 33), RasterChannels.RGB)
    # Room for the 35x33 output (12 bytes per pixel) but not for the 4x target.
    monkeypatch.setattr("invokeai.backend.image_util.raster.precise_asset.MAX_PIXEL_BYTES", 35 * 33 * 12)

    node = _node(SpandrelImageToImageAutoscaleInvocation, name, 0, scale=4.0, fit_to_multiple_of_8=False)
    out = node.invoke(precise_world.context).image.image_name

    assert precise_world.images.get_raster(out).dtype == np.float32

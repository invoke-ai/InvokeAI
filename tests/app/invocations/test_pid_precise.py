"""PiD nodes with precise image storage: `save_pid_image` and the PiD Upscale input."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

import invokeai.app.invocations.pid.pid_upscale as pid_upscale_module
import invokeai.app.invocations.vae.precise_io as precise_io
from invokeai.app.invocations.pid.pid_output import save_pid_image
from invokeai.app.invocations.pid.pid_upscale import PiDUpscaleInvocation
from invokeai.app.invocations.vae.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.backend.image_util.raster.descriptor import RasterChannels
from invokeai.backend.model_manager.taxonomy import BaseModelType

_PID_DIR = Path(__file__).parents[3] / "invokeai" / "app" / "invocations" / "pid"
# SD3 precise output is only on demand, so its node keeps the 8-bit tail.
_8BIT_ONLY = {"sd3_pid_decode"}
_NODES = sorted(
    p
    for p in _PID_DIR.parent.rglob("*.py")
    if "PiDDecodeConfig(" in p.read_text(encoding="utf-8") and p.stem not in _8BIT_ONLY
)
assert _NODES, "no PiD nodes found - has the invocations directory moved?"


def _x0(batch: int = 1) -> torch.Tensor:
    """A decoder-shaped output in [-1, 1] whose values fall between 8-bit levels, so truncation and rounding differ."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(batch, 3, 6, 7, generator=generator) * 2 - 1


def _legacy(x0: torch.Tensor) -> Image.Image:
    """The quantizer the nodes used before `save_pid_image`."""
    chw = x0[0].clamp(-1, 1)
    return Image.fromarray((127.5 * (chw.permute(1, 2, 0) + 1.0)).byte().numpy())


def _plain_png(precise_world, image: Image.Image) -> bytes:
    was = precise_world.configuration.precise_image_storage
    precise_world.configuration.precise_image_storage = False
    try:
        return precise_world.stored_bytes(precise_world.images.save(image).image_name)
    finally:
        precise_world.configuration.precise_image_storage = was


def test_precise_save_stores_the_unit_float32_output_and_keeps_the_legacy_png(precise_world) -> None:
    x0 = _x0(batch=2)  # a second sample must be ignored, as before

    name = save_pid_image(precise_world.context, x0).image_name

    legacy = _legacy(x0)
    # The setup must separate truncation from rounding, or the PNG check would not pin the quantizer.
    unit = (x0[0].float() + 1.0) / 2
    assert (np.asarray(legacy) != (unit.permute(1, 2, 0) * 255).round().byte().numpy()).any()
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == precise_world.descriptor(RasterChannels.RGB)
    np.testing.assert_array_equal(raster.view(), unit.permute(1, 2, 0).numpy())
    assert precise_world.stored_bytes(name) == _plain_png(precise_world, legacy)


def test_disabled_saves_the_legacy_image_and_builds_no_raster(precise_world, monkeypatch) -> None:
    precise_world.configuration.precise_image_storage = False
    monkeypatch.setattr(precise_io, "decoded_unit_raster", MagicMock(side_effect=AssertionError("raster built")))
    monkeypatch.setattr(precise_world.images, "save_raster", MagicMock(side_effect=AssertionError("precise save")))
    x0 = _x0()

    name = save_pid_image(precise_world.context, x0).image_name

    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.stored_bytes(name) == _plain_png(precise_world, _legacy(x0))


def test_non_finite_output_falls_back_to_the_ordinary_png_and_warns(precise_world) -> None:
    precise_world.context.logger = MagicMock()
    x0 = _x0()
    x0[0, 1, 2, 3] = float("nan")

    name = save_pid_image(precise_world.context, x0).image_name

    precise_world.context.logger.warning.assert_called_once()
    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.stored_bytes(name) == _plain_png(precise_world, _legacy(x0))


@pytest.mark.parametrize("node", _NODES, ids=lambda p: p.stem)
def test_every_pid_node_saves_through_the_helper(node: Path) -> None:
    """Running a node needs Gemma-2 and a PidNet; the structural check keeps a node from saving 8-bit only."""
    tree = ast.parse(node.read_text(encoding="utf-8"))
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "save_pid_image" in called
    assert "save" not in attrs


class _Encoded(Exception):
    def __init__(self, image_tensor: torch.Tensor) -> None:
        self.image_tensor = image_tensor


def _encoded_input(precise_world, monkeypatch, image_name: str) -> torch.Tensor:
    """The tensor PiD Upscale hands the FLUX VAE encode for `image_name` (the node stops there)."""
    context = precise_world.context
    context.models.get_config = lambda _model: SimpleNamespace(base=BaseModelType.Flux)
    precise_world.use_vae(SimpleNamespace(model=object()))
    monkeypatch.setattr(pid_upscale_module, "is_flux_family_vae", lambda _model: True)

    def vae_encode(vae_info, image_tensor):
        raise _Encoded(image_tensor)

    monkeypatch.setattr(FluxVaeEncodeInvocation, "vae_encode", staticmethod(vae_encode))
    node = PiDUpscaleInvocation.model_construct(
        id="pid_upscale",
        image=SimpleNamespace(image_name=image_name),
        vae=SimpleNamespace(vae="vae"),
        gemma2_encoder=SimpleNamespace(),
        pid_decoder=SimpleNamespace(decoder="decoder"),
        prompt="",
    )
    with pytest.raises(_Encoded) as caught:
        node.invoke(context)
    return caught.value.image_tensor


def test_upscale_encodes_the_precise_pixels_of_an_image_that_has_them(precise_world, monkeypatch) -> None:
    pixels = np.random.default_rng(0).uniform(0.0, 1.0, (8, 16, 3)).astype(np.float32)
    pixels[2, 3, 0] = 1.5
    name = precise_world.store_precise(pixels, RasterChannels.RGB)

    tensor = _encoded_input(precise_world, monkeypatch, name)

    assert tensor.shape == (1, 3, 8, 16)
    np.testing.assert_allclose(tensor[0].permute(1, 2, 0).numpy(), pixels * 2 - 1, atol=1e-6)
    assert tensor.max() > 1.0


def test_upscale_encodes_exactly_the_8_bit_pixels_of_an_ordinary_image(precise_world, monkeypatch) -> None:
    pixels = np.random.default_rng(1).integers(0, 256, (8, 16, 3), dtype=np.uint8)
    name = precise_world.store_png(pixels, "RGB")

    tensor = _encoded_input(precise_world, monkeypatch, name)

    expected = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255 * 2 - 1
    assert tensor.shape == (1, 3, 8, 16)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor[0], expected, rtol=0, atol=1e-6)

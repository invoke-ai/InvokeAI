"""MiniMax H3 `minimax_h3_latents_to_image` with precise image storage. The real video VAE is heavy, so
`decode_video_latents` is stubbed with a known [C, T, H, W] clip in [0, 1]."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.app.invocations.vae.minimax_h3_latents_to_image import MiniMaxH3LatentsToImageInvocation
from invokeai.backend.image_util.raster.descriptor import RasterChannels

FRAMES = 3


@pytest.fixture
def clip(monkeypatch) -> torch.Tensor:
    """A seeded clip whose values are not multiples of 1/255, so the precise raster differs from the PNG."""
    value = torch.rand(3, FRAMES, 8, 12, generator=torch.Generator().manual_seed(4))
    monkeypatch.setattr(
        "invokeai.app.invocations.vae.minimax_h3_latents_to_image.decode_video_latents",
        lambda context, vae_field, latents: value,
    )
    return value


def _run(precise_world, frame_index: int = 0) -> str:
    precise_world.tensors["latents"] = torch.zeros(1, 1, 1, 1, 1)
    node = MiniMaxH3LatentsToImageInvocation.model_construct(
        id="h3_l2i",
        video_latents=SimpleNamespace(latents_name="latents"),
        vae=SimpleNamespace(vae="vae"),
        frame_index=frame_index,
    )
    return node.invoke(precise_world.context).image.image_name


def _legacy_png(precise_world, frame: torch.Tensor) -> bytes:
    pixels = (255.0 * frame.permute(1, 2, 0)).round().byte().numpy()
    was = precise_world.configuration.precise_image_storage
    precise_world.configuration.precise_image_storage = False
    try:
        return precise_world.stored_bytes(precise_world.images.save(Image.fromarray(pixels)).image_name)
    finally:
        precise_world.configuration.precise_image_storage = was


def test_disabled_saves_an_ordinary_png(precise_world, clip) -> None:
    precise_world.configuration.precise_image_storage = False

    name = _run(precise_world)

    assert precise_world.stored_bytes(name) == _legacy_png(precise_world, clip[:, 0])
    assert precise_world.images.get_raster(name).dtype == np.uint8


@pytest.mark.parametrize("frame_index", [0, 2])
def test_enabled_stores_the_selected_frame_as_float32_and_the_png_matches_the_disabled_one(
    precise_world, clip, frame_index: int
) -> None:
    name = _run(precise_world, frame_index)

    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == precise_world.descriptor(RasterChannels.RGB)
    np.testing.assert_array_equal(raster.view(), clip[:, frame_index].permute(1, 2, 0).numpy())
    assert precise_world.stored_bytes(name) == _legacy_png(precise_world, clip[:, frame_index])
    if frame_index:
        assert precise_world.stored_bytes(name) != _legacy_png(precise_world, clip[:, 0])


def test_a_frame_index_past_the_clip_is_rejected_before_saving(precise_world, clip) -> None:
    with pytest.raises(ValueError, match="out of range"):
        _run(precise_world, FRAMES)


def test_non_finite_frame_falls_back_to_the_ordinary_png_and_warns(precise_world, clip) -> None:
    clip[1, 1, 0, 0] = float("nan")
    precise_world.context.logger = MagicMock()
    name = _run(precise_world, 1)

    precise_world.context.logger.warning.assert_called_once()
    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.database.queries.image_precise_assets.get(name) is None

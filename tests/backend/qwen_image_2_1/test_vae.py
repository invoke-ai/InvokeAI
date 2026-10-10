"""Qwen-Image-2.1's RGBA VAE: how a decode becomes an image, what an encode reads, and the tile geometry it runs in."""

import pytest
import torch
from diffusers import AutoencoderKLQwenImage21
from PIL import Image

from invokeai.backend.qwen_image_2_1 import vae as vae_module
from invokeai.backend.qwen_image_2_1.vae import (
    MIN_TILE_SIZE,
    SPATIAL_SCALE,
    choose_tile_size,
    to_image,
    to_vae_input,
    working_memory_bytes,
)
from invokeai.backend.util.qwen_image_vae import patch_qwen_image_vae_tiling


def _decode(alpha: torch.Tensor) -> torch.Tensor:
    """A (4, H, W) decode in [-1, 1]: mid-grey colour and the given alpha in 0..255."""
    rgb = torch.zeros(3, *alpha.shape)
    return torch.cat([rgb, (alpha.float() / 127.5 - 1)[None]])


def test_an_opaque_decode_is_saved_as_rgb() -> None:
    image = to_image(_decode(torch.full((64, 64), 252)))
    assert image.mode == "RGB"


def test_tiling_seams_in_alpha_do_not_make_an_opaque_image_transparent() -> None:
    alpha = torch.full((100, 100), 252)
    alpha[0, :5] = 181  # 0.05 % of pixels, as 256px tiles leave them
    assert to_image(_decode(alpha)).mode == "RGB"


def test_a_transparent_background_is_kept() -> None:
    alpha = torch.full((64, 64), 255)
    alpha[:, :32] = 0
    image = to_image(_decode(alpha))
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0))[3] == 0 and image.getpixel((63, 0))[3] == 255


def test_values_map_like_the_pipelines_postprocess() -> None:
    # (x / 2 + 0.5) * 255, rounded: -1 -> 0, 0 -> 128 (127.5 rounds to even), 1 -> 255.
    decoded = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3).expand(4, 1, 3).clone()
    decoded[3] = 1.0
    image = to_image(decoded)
    assert [image.getpixel((x, 0)) for x in range(3)] == [(0, 0, 0), (128, 128, 128), (255, 255, 255)]


def test_an_encode_ignores_the_images_alpha() -> None:
    image = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
    pixels = to_vae_input(image)
    assert pixels.shape == (1, 4, 32, 32)
    assert torch.equal(pixels[0, 3], torch.ones(32, 32))
    assert pixels[0, 0].unique().tolist() == [1.0] and pixels[0, 1].unique().tolist() == [-1.0]


@pytest.fixture
def vae() -> AutoencoderKLQwenImage21:
    # Tiling is geometry on the module; a minimal network is enough to hold it.
    return AutoencoderKLQwenImage21(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 1], num_res_blocks=1, temperal_downsample=[False]
    )


def test_tiles_stride_on_the_16px_grid_and_the_modules_geometry_is_restored(vae: AutoencoderKLQwenImage21) -> None:
    before = (vae.use_tiling, vae.tile_sample_min_height, vae.tile_sample_stride_height)
    with patch_qwen_image_vae_tiling(vae, 1008, SPATIAL_SCALE):
        assert vae.use_tiling
        assert (vae.tile_sample_min_height, vae.tile_sample_min_width) == (1008, 1008)
        # 3/4 of 1008 is 756, which the 8x Qwen-Image grid would keep; at 16x it must drop to 752.
        assert (vae.tile_sample_stride_height, vae.tile_sample_stride_width) == (752, 752)
    assert (vae.use_tiling, vae.tile_sample_min_height, vae.tile_sample_stride_height) == before


def test_the_geometry_is_restored_when_the_decode_fails(vae: AutoencoderKLQwenImage21) -> None:
    before = (vae.use_tiling, vae.tile_sample_min_height)
    with pytest.raises(RuntimeError), patch_qwen_image_vae_tiling(vae, 512, SPATIAL_SCALE):
        raise RuntimeError("out of memory")
    assert (vae.use_tiling, vae.tile_sample_min_height) == before


CPU = torch.device("cpu")


def test_a_float32_vae_is_priced_at_twice_the_bfloat16_memory() -> None:
    fixed = working_memory_bytes(0, 0, None, 2)
    assert working_memory_bytes(1024, 1024, None, 4) - fixed == 2 * (working_memory_bytes(1024, 1024, None, 2) - fixed)


def test_nothing_is_tiled_unless_asked_or_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stand in for the device: it holds 12 GiB of working memory.
    monkeypatch.setattr(vae_module, "should_pretile_vae_decode", lambda device, needed: needed > 12 * 1024**3)

    def choose(size: int, **kwargs) -> int | None:
        return choose_tile_size(size, size, 2, CPU, **({"tiled": False, "tile_size": 0, "auto_tile": True} | kwargs))

    assert choose(1024) is None
    # 2048x2048 untiled needs ~32 GiB: tiled, at the device's tile size, encode and decode alike.
    assert choose(2048) == 1024
    assert choose(2048, auto_tile=False) is None
    assert choose(1024, tiled=True) == 1024


def test_a_tile_size_field_below_the_floor_is_raised_to_it() -> None:
    assert choose_tile_size(2048, 2048, 2, CPU, tiled=True, tile_size=16, auto_tile=False) == MIN_TILE_SIZE
    assert choose_tile_size(2048, 2048, 2, CPU, tiled=True, tile_size=768, auto_tile=False) == 768

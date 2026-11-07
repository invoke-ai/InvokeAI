"""The vendored TAESD decoders: key layout against the released checkpoints, and the IO contract."""

import pytest
import torch

from invokeai.backend.tiny_autoencoder.taesd import Decoder, F16Decoder, TaesdDecoder


def _conv(index: int, n_out: int, n_in: int, bias: bool = True) -> dict[str, tuple[int, ...]]:
    shapes = {f"{index}.weight": (n_out, n_in, 3, 3)}
    if bias:
        shapes[f"{index}.bias"] = (n_out,)
    return shapes


def _block(index: int, n: int) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for conv in (0, 2, 4):
        shapes |= {f"{index}.conv.{conv}.weight": (n, n, 3, 3), f"{index}.conv.{conv}.bias": (n,)}
    return shapes


def _pool(index: int) -> dict[str, tuple[int, ...]]:
    return {
        f"{index}.pool.0.weight": (256, 64, 1, 1),
        f"{index}.pool.1.weight": (256,),
        f"{index}.pool.1.bias": (256,),
        f"{index}.pool.3.weight": (64, 256, 1, 1),
    }


def _stages(widths: tuple[int, int, int], last: int, latent: int) -> dict[str, tuple[int, ...]]:
    """Transcribed from the published headers (native indices; diffusers files are one lower)."""
    w0, w1, w2 = widths
    shapes = _conv(1, w0, latent)
    for i in (3, 4, 5):
        shapes |= _block(i, w0)
    shapes |= _conv(7, w1, w0, bias=False)
    for i in (8, 9, 10):
        shapes |= _block(i, w1)
    shapes |= _conv(12, w2, w1, bias=False)
    for i in (13, 14, 15):
        shapes |= _block(i, w2)
    shapes |= _conv(17, 64, w2, bias=False)
    shapes |= _block(18, 64)
    shapes |= _conv(19, last, 64)
    return shapes


# One representative released file per architecture, at its pinned revision.
RELEASED_LAYOUTS = {
    # madebyollin/taesd@614f768 taesd_decoder.safetensors: 67 tensors, native layout.
    "taesd": (Decoder(4), _stages((64, 64, 64), last=3, latent=4)),
    # madebyollin/taef2@bd244eb taef2.safetensors: pooling on decoder.layers.2-4, i.e. native 3-5.
    "taef2": (
        Decoder(32, use_midblock_gn=True),
        _stages((64, 64, 64), last=3, latent=32) | _pool(3) | _pool(4) | _pool(5),
    ),
    # madebyollin/taeqi2_1@5d62f11 taeqi2_1.safetensors: 256/128/64 stages, 16 = 4 RGBA x 2x2 pixel shuffle.
    "taeqi2_1": (F16Decoder(64, image_channels=4), _stages((256, 128, 64), last=16, latent=64)),
}


@pytest.mark.parametrize("name", RELEASED_LAYOUTS)
def test_decoder_keys_and_shapes_match_the_released_checkpoint(name: str) -> None:
    decoder, expected = RELEASED_LAYOUTS[name]

    actual = {key: tuple(value.shape) for key, value in decoder.state_dict().items()}

    assert actual == expected


@pytest.mark.parametrize(
    ("decoder", "latent_channels", "image_channels", "compression"),
    [
        (Decoder(4), 4, 3, 8),
        (Decoder(32, use_midblock_gn=True), 32, 3, 8),
        (F16Decoder(64, image_channels=4), 64, 4, 16),
    ],
)
def test_decoder_upscales_by_its_declared_compression(
    decoder: TaesdDecoder, latent_channels: int, image_channels: int, compression: int
) -> None:
    latents = torch.randn(1, latent_channels, 3, 5, generator=torch.Generator().manual_seed(0))

    with torch.inference_mode():
        image = decoder(latents)

    assert decoder.latent_channels == latent_channels
    assert decoder.spatial_compression == compression
    assert image.shape == (1, image_channels, 3 * compression, 5 * compression)


def test_latents_are_clamped_before_the_first_convolution() -> None:
    """Upstream soft-clamps latents to (-3, 3); far beyond that, scaling them changes nothing."""
    torch.manual_seed(0)
    decoder = Decoder(4).eval()
    latents = torch.randn(1, 4, 2, 2, generator=torch.Generator().manual_seed(1)).sign()

    with torch.inference_mode():
        torch.testing.assert_close(decoder(latents * 1e3), decoder(latents * 1e4))


def test_midblock_pooling_takes_part_in_the_flux_2_decode() -> None:
    torch.manual_seed(0)
    decoder = Decoder(32, use_midblock_gn=True).eval()
    latents = torch.randn(1, 32, 4, 4, generator=torch.Generator().manual_seed(1))

    with torch.inference_mode():
        pooled = decoder(latents)
        for index in (3, 4, 5):
            decoder[index].pool[3].weight.zero_()
        unpooled = decoder(latents)

    assert not torch.allclose(pooled, unpooled)

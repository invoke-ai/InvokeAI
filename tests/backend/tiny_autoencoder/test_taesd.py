"""The vendored TAESD and TAEHV decoders: key layout against the released checkpoints, and the IO contract."""

import pytest
import torch

from invokeai.backend.tiny_autoencoder.checkpoint import (
    build_taesd_decoder,
    detect_taesd_layout,
    taesd_decoder_state_dict,
)
from invokeai.backend.tiny_autoencoder.taehv import ImageDecoder
from invokeai.backend.tiny_autoencoder.taesd import Decoder, F16Decoder, TaesdDecoder
from tests.fixtures.taehv_upstream import UpstreamTaehvDecoder


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


def _taehv_stages(latent: int) -> dict[str, tuple[int, ...]]:
    """Transcribed from madebyollin/taehv@011dfc2 taew2_1.safetensors, `decoder.` stripped: 64 decoder tensors.

    The file's `TGrow`s 13 and 19 are stride 2, `(2 n, n, 1, 1)`; the image decoder holds one group of 19.
    """
    shapes = _conv(1, 256, latent)
    for stage, (n_in, n_out) in enumerate(((256, 128), (128, 64), (64, 64))):
        start = 3 + 6 * stage
        for i in range(start, start + 3):
            for conv in (0, 2, 4):
                shapes[f"{i}.conv.{conv}.weight"] = (n_in, 2 * n_in if conv == 0 else n_in, 3, 3)
                shapes[f"{i}.conv.{conv}.bias"] = (n_in,)
        shapes[f"{start + 4}.conv.weight"] = (2 * n_in if stage == 1 else n_in, n_in, 1, 1)
        shapes |= _conv(start + 5, n_out, n_in, bias=False)
    return shapes | _conv(22, 3, 64)


RELEASED_LAYOUTS["taew2_1"] = (ImageDecoder(16), _taehv_stages(latent=16))


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
        (ImageDecoder(16), 16, 3, 8),
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


def test_a_taehv_image_decode_is_upstreams_last_decoded_frame_for_each_latent_of_a_batch() -> None:
    """From an upstream-format state dict through identification and a strict load, against upstream's own decoder."""
    torch.manual_seed(0)
    upstream = UpstreamTaehvDecoder().eval()
    checkpoint = {key: value.clone() for key, value in upstream.state_dict().items()}
    state_dict = taesd_decoder_state_dict(checkpoint)
    layout = detect_taesd_layout(state_dict)
    assert layout is not None
    decoder = build_taesd_decoder(layout).eval()
    decoder.load_state_dict(state_dict, strict=True)
    latents = torch.randn(3, 16, 3, 5, generator=torch.Generator().manual_seed(1))

    with torch.inference_mode():
        expected = upstream.decode_video(latents[:, None])
        actual = decoder(latents).clamp(0, 1)

    assert expected.shape == (3, 1, 3, 24, 40)
    torch.testing.assert_close(actual, expected[:, -1])

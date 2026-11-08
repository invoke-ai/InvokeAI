"""The vendored TAEHV decoder: key layout against the released checkpoints, and single-frame decoding."""

import pytest
import torch

from invokeai.backend.tiny_autoencoder.checkpoint import TAELTX2_3, TAEW2_1, TAEW2_2, build_taehv_decoder
from invokeai.backend.tiny_autoencoder.taehv import TaehvDecoder


def _conv(index: int, n_out: int, n_in: int, bias: bool = True) -> dict[str, tuple[int, ...]]:
    shapes = {f"decoder.{index}.weight": (n_out, n_in, 3, 3)}
    if bias:
        shapes[f"decoder.{index}.bias"] = (n_out,)
    return shapes


def _memblock(index: int, n: int) -> dict[str, tuple[int, ...]]:
    """A MemBlock's first conv also reads the previous frame, so it takes twice the channels."""
    return {
        f"decoder.{index}.conv.0.weight": (n, 2 * n, 3, 3),
        f"decoder.{index}.conv.0.bias": (n,),
        f"decoder.{index}.conv.2.weight": (n, n, 3, 3),
        f"decoder.{index}.conv.2.bias": (n,),
        f"decoder.{index}.conv.4.weight": (n, n, 3, 3),
        f"decoder.{index}.conv.4.bias": (n,),
    }


def _published(latent: int, out: int, tgrow: tuple[int, int, int]) -> dict[str, tuple[int, ...]]:
    """Transcribed from the published headers: 256/128/64 stages, TGrow 1x1 convs (out channels `tgrow`) at 7, 13, 19."""
    shapes = _conv(1, 256, latent)
    for stage, (first, width, next_width) in enumerate(((3, 256, 128), (9, 128, 64), (15, 64, 64))):
        for i in range(first, first + 3):
            shapes |= _memblock(i, width)
        shapes[f"decoder.{first + 4}.conv.weight"] = (tgrow[stage], width, 1, 1)
        shapes |= _conv(first + 5, next_width, width, bias=False)
    return shapes | _conv(22, out, 64)


# madebyollin/taehv@62f7591 safetensors/<name>.safetensors, decoder half (64 of 128 tensors each).
RELEASED_LAYOUTS = {
    "taew2_1": (TAEW2_1, _published(latent=16, out=3, tgrow=(256, 256, 128))),
    "taew2_2": (TAEW2_2, _published(latent=48, out=12, tgrow=(256, 256, 128))),
    "taeltx2_3": (TAELTX2_3, _published(latent=128, out=48, tgrow=(512, 256, 128))),
}


@pytest.mark.parametrize("name", RELEASED_LAYOUTS)
def test_decoder_keys_and_shapes_match_the_released_checkpoint(name: str) -> None:
    layout, expected = RELEASED_LAYOUTS[name]

    actual = {key: tuple(value.shape) for key, value in build_taehv_decoder(layout).state_dict().items()}

    assert len(expected) == 64
    assert actual == expected


def _decoder(layout, seed: int = 0) -> TaehvDecoder:
    torch.manual_seed(seed)
    return build_taehv_decoder(layout).eval()


def _latent(*shape: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize(
    ("layout", "compression"),
    [(TAEW2_1, 8), (TAEW2_2, 16), (TAELTX2_3, 32)],
    ids=["taew2_1", "taew2_2", "taeltx"],
)
def test_a_latent_decodes_to_one_frame_at_the_declared_compression(layout, compression: int) -> None:
    decoder = _decoder(layout)

    with torch.inference_mode():
        image = decoder(_latent(1, layout.latent_channels, 2, 3))

    assert decoder.spatial_compression == compression
    assert image.shape == (1, 3, 2 * compression, 3 * compression)


@pytest.mark.parametrize(("layout", "frames"), [(TAEW2_1, 2 * 4 - 3), (TAELTX2_3, 2 * 8 - 7)], ids=["taew2", "taeltx"])
def test_a_video_drops_the_causal_warm_up_frames(layout, frames: int) -> None:
    """Upstream `decode_video`: T latents give T * t_upscale frames, less the first t_upscale - 1."""
    with torch.inference_mode():
        video = _decoder(layout).decode_video(_latent(1, 2, layout.latent_channels, 1, 1))

    assert video.shape[:3] == (1, frames, 3)


@pytest.mark.parametrize("layout", [TAEW2_1, TAELTX2_3], ids=["taew2", "taeltx"])
def test_the_single_frame_is_the_last_of_the_untrimmed_decode(layout) -> None:
    """The kept frame is decoded after, and with the memory of, the warm-up frames trimmed before it."""
    decoder = _decoder(layout)
    latent = _latent(1, layout.latent_channels, 2, 2)

    with torch.inference_mode():
        frame = decoder(latent)
        untrimmed = decoder._decode_frames(latent.unsqueeze(1))

    assert untrimmed.shape[1] == decoder.t_upscale
    torch.testing.assert_close(frame, untrimmed[:, -1], rtol=0, atol=0)
    assert not torch.allclose(frame, untrimmed[:, 0])

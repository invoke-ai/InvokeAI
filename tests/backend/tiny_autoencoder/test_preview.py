"""Rendering a preview with a TAE, and choosing the installed TAE that renders a latent space."""

import numpy as np
import pytest
import torch

from invokeai.backend.architectures import resolve_latent_space
from invokeai.backend.model_manager.configs.tiny_autoencoder import (
    TinyAutoencoder_Checkpoint_Config_Base,
    TinyAutoencoder_Checkpoint_Flux2_Config,
    TinyAutoencoder_Checkpoint_FLUX_Config,
    TinyAutoencoder_Checkpoint_LTX2_Config,
    TinyAutoencoder_Checkpoint_SD1_Config,
    TinyAutoencoder_Checkpoint_SD3_Config,
    TinyAutoencoder_Checkpoint_SDXL_Config,
    TinyAutoencoder_Checkpoint_Wan_Config,
    is_tiny_autoencoder_supported,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.tiny_autoencoder.checkpoint import TAELTX2_3, TAEW2_1, TAEW2_2, build_taehv_decoder
from invokeai.backend.tiny_autoencoder.decoder import TinyAutoencoderDecoder
from invokeai.backend.tiny_autoencoder.preview import decode_preview, select_tiny_autoencoder
from invokeai.backend.tiny_autoencoder.taesd import Decoder, F16Decoder, TaesdDecoder


@pytest.fixture(scope="module")
def taesd() -> TaesdDecoder:
    torch.manual_seed(0)
    return Decoder(4).eval()


def _latent(*shape: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_a_latent_with_or_without_batch_decodes_to_the_same_rgb_image(taesd: TaesdDecoder) -> None:
    latent = _latent(4, 3, 5)

    unbatched = decode_preview(taesd, latent)
    batched = decode_preview(taesd, latent.unsqueeze(0))

    assert unbatched.mode == "RGB"
    assert unbatched.size == (40, 24)
    assert np.array_equal(np.asarray(unbatched), np.asarray(batched))


def test_only_the_first_latent_of_a_batch_is_decoded(taesd: TaesdDecoder) -> None:
    first = _latent(1, 4, 3, 3, seed=1)
    batch = torch.cat([first, _latent(1, 4, 3, 3, seed=2)])

    assert np.array_equal(np.asarray(decode_preview(taesd, batch)), np.asarray(decode_preview(taesd, first)))


def test_a_latent_of_another_width_is_refused(taesd: TaesdDecoder) -> None:
    with pytest.raises(ValueError, match="4 latent channels"):
        decode_preview(taesd, _latent(16, 3, 3))


def test_an_oversized_latent_is_area_downsampled_to_the_cap(taesd: TaesdDecoder, monkeypatch) -> None:
    """A latent of 2x2-constant blocks, halved by the cap, decodes exactly as the latent it was made from."""
    monkeypatch.setattr(taesd, "max_preview_edge", 48)
    small = _latent(4, 4, 6)
    large = small.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    capped = decode_preview(taesd, large)

    assert capped.size == (48, 32)
    reference = np.asarray(decode_preview(taesd, small)).astype(int)
    assert np.abs(np.asarray(capped).astype(int) - reference).max() <= 1  # the mean of four equal floats may round


def test_the_cap_bounds_the_longest_edge_of_any_aspect(taesd: TaesdDecoder, monkeypatch) -> None:
    monkeypatch.setattr(taesd, "max_preview_edge", 64)

    image = decode_preview(taesd, _latent(4, 6, 16))

    assert image.size == (64, 24)


@pytest.fixture(scope="module")
def taeqi() -> TaesdDecoder:
    torch.manual_seed(0)
    return F16Decoder(64, image_channels=4).eval()


def test_the_cap_is_applied_at_the_decoders_own_compression(taeqi: TaesdDecoder, monkeypatch) -> None:
    """An 8 x 4 latent fits a 96 cap at 8x (64 x 32) but not at TAEQI2.1's 16x, where it is scaled by 3/4."""
    monkeypatch.setattr(taeqi, "max_preview_edge", 96)

    image = decode_preview(taeqi, _latent(64, 8, 4))

    assert image.size == (48, 96)


def test_the_decode_runs_on_the_device_passed_not_where_the_weights_sit(taesd: TaesdDecoder) -> None:
    """A partially loaded decoder's first parameter can be off the compute device the cache names."""

    class _Stop(Exception):
        pass

    seen: list[torch.device] = []

    def record(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        seen.append(args[0].device)
        raise _Stop

    handle = taesd.register_forward_pre_hook(record)
    try:
        with pytest.raises(_Stop):
            decode_preview(taesd, _latent(4, 2, 2), device=torch.device("meta"))
    finally:
        handle.remove()

    assert seen == [torch.device("meta")]


def test_alpha_is_dropped_from_an_rgba_decoder(taeqi: TaesdDecoder) -> None:
    decoder = taeqi
    latent = _latent(64, 2, 2)

    image = decode_preview(decoder, latent)

    with torch.inference_mode():
        rgb = decoder(latent.unsqueeze(0))[0, :3].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0)
    assert image.mode == "RGB"
    assert np.array_equal(np.asarray(image), rgb.numpy())


@pytest.mark.parametrize(("bias", "expected"), [(10.0, 255), (-10.0, 0)])
def test_out_of_range_output_saturates_instead_of_wrapping(bias: float, expected: int) -> None:
    torch.manual_seed(0)
    decoder = Decoder(4).eval()
    with torch.no_grad():
        decoder[19].weight.zero_()
        decoder[19].bias.fill_(bias)

    image = decode_preview(decoder, _latent(4, 2, 2))

    assert (np.asarray(image) == expected).all()


def test_a_sample_follows_the_decoders_dtype(taesd: TaesdDecoder) -> None:
    latent = _latent(4, 3, 3)

    from_double = decode_preview(taesd, latent.double())

    assert np.array_equal(np.asarray(from_double), np.asarray(decode_preview(taesd, latent)))


def _record(config_class: type[TinyAutoencoder_Checkpoint_Config_Base], channels: int, name: str = "tae", key="k"):
    return config_class.model_construct(name=name, key=key, latent_channels=channels)


def _space(base: BaseModelType, channels: int):
    return resolve_latent_space(base, torch.zeros(1, channels, 2, 2))


SD1_TAE = _record(TinyAutoencoder_Checkpoint_SD1_Config, 4, "taesd")
SDXL_TAE = _record(TinyAutoencoder_Checkpoint_SDXL_Config, 4, "taesdxl")
SD3_TAE = _record(TinyAutoencoder_Checkpoint_SD3_Config, 16, "taesd3")
FLUX_TAE = _record(TinyAutoencoder_Checkpoint_FLUX_Config, 16, "taef1")
FLUX2_TAE = _record(TinyAutoencoder_Checkpoint_Flux2_Config, 32, "taef2")
WAN16_TAE = _record(TinyAutoencoder_Checkpoint_Wan_Config, 16, "taew2_1")
WAN48_TAE = _record(TinyAutoencoder_Checkpoint_Wan_Config, 48, "taew2_2")
LTX2_TAE = _record(TinyAutoencoder_Checkpoint_LTX2_Config, 128, "taeltx2_3")
INSTALLED = [SD1_TAE, SDXL_TAE, SD3_TAE, FLUX_TAE, FLUX2_TAE, WAN16_TAE, WAN48_TAE, LTX2_TAE]


@pytest.mark.parametrize(
    ("base", "channels", "expected"),
    [
        (BaseModelType.StableDiffusion2, 4, SD1_TAE),
        (BaseModelType.StableDiffusionXLRefiner, 4, SDXL_TAE),
        (BaseModelType.ZImage, 16, FLUX_TAE),
        (BaseModelType.StableDiffusion3, 16, SD3_TAE),
        (BaseModelType.ErnieImage, 32, FLUX2_TAE),
        (BaseModelType.CogView4, 16, None),
        # The Wan 2.1 VAE's space, shared by four architectures; Wan's 48-channel space is TI2V-5B's alone.
        (BaseModelType.Wan, 16, WAN16_TAE),
        (BaseModelType.QwenImage, 16, WAN16_TAE),
        (BaseModelType.Krea2, 16, WAN16_TAE),
        (BaseModelType.Anima, 16, WAN16_TAE),
        (BaseModelType.Wan, 48, WAN48_TAE),
        (BaseModelType.LTX2, 128, LTX2_TAE),
        (BaseModelType.MiniMaxH3, 24, None),
    ],
)
def test_a_tae_serves_every_architecture_sharing_its_latent_space(
    base: BaseModelType, channels: int, expected: TinyAutoencoder_Checkpoint_Config_Base | None
) -> None:
    assert select_tiny_autoencoder(INSTALLED, _space(base, channels)) is expected


def test_the_choice_among_fitting_taes_is_by_name_then_key_whatever_the_order() -> None:
    b = _record(TinyAutoencoder_Checkpoint_FLUX_Config, 16, "b", "1")
    a2 = _record(TinyAutoencoder_Checkpoint_FLUX_Config, 16, "a", "2")
    a1 = _record(TinyAutoencoder_Checkpoint_FLUX_Config, 16, "a", "1")
    space = _space(BaseModelType.Flux, 16)

    assert select_tiny_autoencoder([b, a2, a1], space) is a1
    assert select_tiny_autoencoder([a1, b, a2], space) is a1


@pytest.mark.parametrize(
    ("base", "channels", "supported"),
    [
        (BaseModelType.StableDiffusion2, 4, True),
        (BaseModelType.ZImage, 16, True),
        (BaseModelType.Ideogram4, 32, True),
        (BaseModelType.QwenImage21, 64, True),
        (BaseModelType.QwenImage, 16, True),
        (BaseModelType.Wan, 48, True),
        (BaseModelType.LTX2, 128, True),
        (BaseModelType.CogView4, 16, False),
        (BaseModelType.MiniMaxH3, 24, False),
    ],
)
def test_whether_any_tae_decodes_a_space(base: BaseModelType, channels: int, supported: bool) -> None:
    assert is_tiny_autoencoder_supported(_space(base, channels)) is supported


@pytest.mark.parametrize(
    ("layout", "latent", "image_size"),
    [
        # 8x: an 8 x 6 latent is 64 x 48 pixels, inside the cap.
        (TAEW2_1, (16, 8, 6), (48, 64)),
        # 16x and 32x: the same latent edge is 128 and 256 pixels, so the latent is halved and quartered.
        (TAEW2_2, (48, 8, 6), (48, 64)),
        (TAELTX2_3, (128, 8, 4), (32, 64)),
    ],
    ids=["taew2_1", "taew2_2", "taeltx"],
)
def test_a_taehv_preview_is_one_rgb_frame_capped_at_its_compression(
    layout, latent: tuple[int, int, int], image_size: tuple[int, int]
) -> None:
    torch.manual_seed(0)
    decoder = build_taehv_decoder(layout).eval()
    decoder.max_preview_edge = 64

    image = decode_preview(decoder, _latent(*latent))

    assert image.mode == "RGB"
    assert image.size == image_size


class _Stop(Exception):
    pass


def _first_conv_input(decoder: TinyAutoencoderDecoder, latent: torch.Tensor) -> tuple[int, ...]:
    """The latent shape the decoder's first convolution receives, without paying for the decode."""
    seen: list[tuple[int, ...]] = []

    def record(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        seen.append(tuple(args[0].shape))
        raise _Stop

    first_conv = next(m for m in decoder.modules() if isinstance(m, torch.nn.Conv2d))
    handle = first_conv.register_forward_pre_hook(record)
    try:
        with pytest.raises(_Stop):
            decode_preview(decoder, latent)
    finally:
        handle.remove()
    return seen[0]


@pytest.mark.parametrize(
    ("layout", "latent", "decoded"),
    [
        # 1024 pixels at 8x: taew2_1 caps at 512, so the latent is halved.
        (TAEW2_1, (16, 128, 96), (1, 16, 64, 48)),
        # 1024 pixels at 16x: taew2_2 decodes it as it is.
        (TAEW2_2, (48, 64, 48), (1, 48, 64, 48)),
    ],
    ids=["taew2_1", "taew2_2"],
)
def test_each_decoder_brings_its_own_preview_edge(layout, latent: tuple[int, ...], decoded: tuple[int, ...]) -> None:
    decoder = build_taehv_decoder(layout).eval()

    assert _first_conv_input(decoder, torch.zeros(latent)) == decoded


@pytest.fixture(scope="module")
def taew2_1() -> TinyAutoencoderDecoder:
    torch.manual_seed(0)
    return build_taehv_decoder(TAEW2_1).eval()


def test_a_latent_window_decodes_to_the_last_frame_of_its_video(taew2_1: TinyAutoencoderDecoder) -> None:
    window = _latent(1, 16, 2, 3, 4)

    image = decode_preview(taew2_1, window)

    with torch.inference_mode():
        last = taew2_1.decode_video(window.transpose(1, 2))[0, -1]
    expected = last.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    assert np.array_equal(np.asarray(image), expected)


def test_a_one_frame_window_decodes_like_the_single_latent(taew2_1: TinyAutoencoderDecoder) -> None:
    latent = _latent(1, 16, 3, 4)

    assert np.array_equal(
        np.asarray(decode_preview(taew2_1, latent.unsqueeze(2))), np.asarray(decode_preview(taew2_1, latent))
    )


def test_a_capped_window_keeps_its_frames(taew2_1: TinyAutoencoderDecoder, monkeypatch) -> None:
    """Area pooling a 5-D window pools in time too, unless the size keeps T."""
    seen: list[tuple[int, ...]] = []

    def record(window: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(window.shape))
        raise _Stop

    monkeypatch.setattr(taew2_1, "max_preview_edge", 16)
    monkeypatch.setattr(taew2_1, "decode_window", record)

    with pytest.raises(_Stop):
        decode_preview(taew2_1, _latent(1, 16, 2, 4, 4))

    assert seen == [(1, 16, 2, 2, 2)]


def test_a_taesd_decoder_refuses_a_latent_window(taesd: TaesdDecoder) -> None:
    with pytest.raises(ValueError, match="not a \\(B, C, T, h, w\\) window"):
        decode_preview(taesd, _latent(1, 4, 2, 3, 3))

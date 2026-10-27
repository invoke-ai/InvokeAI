"""Rendering a preview with a TAE, and choosing the installed TAE that renders a latent space."""

import copy
import math

import numpy as np
import pytest
import torch

from invokeai.backend.architectures import resolve_latent_space
from invokeai.backend.model_manager.configs.tiny_autoencoder import (
    TinyAutoencoder_Checkpoint_Config_Base,
    TinyAutoencoder_Checkpoint_Flux2_Config,
    TinyAutoencoder_Checkpoint_FLUX_Config,
    TinyAutoencoder_Checkpoint_SD1_Config,
    TinyAutoencoder_Checkpoint_SD3_Config,
    TinyAutoencoder_Checkpoint_SDXL_Config,
    is_tiny_autoencoder_supported,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.tiny_autoencoder import preview
from invokeai.backend.tiny_autoencoder.preview import (
    decode_preview,
    decode_working_memory_bytes,
    select_tiny_autoencoder,
)
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
    monkeypatch.setattr(preview, "MAX_PREVIEW_EDGE", 48)
    small = _latent(4, 4, 6)
    large = small.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    capped = decode_preview(taesd, large)

    assert capped.size == (48, 32)
    reference = np.asarray(decode_preview(taesd, small)).astype(int)
    assert np.abs(np.asarray(capped).astype(int) - reference).max() <= 1  # the mean of four equal floats may round


def test_the_cap_bounds_the_longest_edge_of_any_aspect(taesd: TaesdDecoder, monkeypatch) -> None:
    monkeypatch.setattr(preview, "MAX_PREVIEW_EDGE", 64)

    image = decode_preview(taesd, _latent(4, 6, 16))

    assert image.size == (64, 24)


def _area_pool(latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Area downsampling written out: output cell i averages input cells floor(i * n / m) to ceil((i + 1) * n / m)."""

    def windows(n: int, m: int) -> list[tuple[int, int]]:
        return [(math.floor(i * n / m), math.ceil((i + 1) * n / m)) for i in range(m)]

    rows = windows(latent.shape[-2], height)
    cols = windows(latent.shape[-1], width)
    return torch.stack(
        [torch.stack([latent[..., r0:r1, c0:c1].mean(dim=(-2, -1)) for c0, c1 in cols], dim=-1) for r0, r1 in rows],
        dim=-2,
    )


@pytest.mark.parametrize("pooled_on_cpu", [False, True], ids=["pooled on its device", "pooled on the CPU, as on MPS"])
def test_a_latent_the_cap_does_not_divide_is_area_downsampled(
    taesd: TaesdDecoder, monkeypatch, pooled_on_cpu: bool
) -> None:
    """Scaled by 2/3, as a 1536px SDXL latent (192) is to the 1024 cap (128): no output cell covers whole input cells.

    MPS pools such a size on the CPU. The decoder here is float64, which that detour pools at float32 and must hand
    back at the decoder's own width.
    """
    monkeypatch.setattr(preview, "MAX_PREVIEW_EDGE", 48)
    if pooled_on_cpu:
        monkeypatch.setattr(preview, "_POOLING_NEEDS_DIVISIBLE_SIZES", frozenset({"cpu"}))
    decoder = copy.deepcopy(taesd).double()
    large = _latent(4, 6, 9).double()

    capped = decode_preview(decoder, large)

    assert capped.size == (48, 32)
    reference = np.asarray(decode_preview(decoder, _area_pool(large, 4, 6))).astype(int)
    assert np.abs(np.asarray(capped).astype(int) - reference).max() <= 1


def test_a_decode_reserves_its_full_resolution_activations_at_the_capped_size(taesd: TaesdDecoder, monkeypatch) -> None:
    monkeypatch.setattr(preview, "MAX_PREVIEW_EDGE", 48)
    half = copy.deepcopy(taesd).half()

    # Five 64-channel activations per output pixel, at the decoder's element size.
    assert decode_working_memory_bytes(taesd, _latent(4, 3, 5)) == 5 * 64 * (24 * 40) * 4
    assert decode_working_memory_bytes(half, _latent(4, 3, 5)) == 5 * 64 * (24 * 40) * 2
    # 12 x 18 latents exceed the cap and are decoded at 4 x 6.
    assert decode_working_memory_bytes(taesd, _latent(1, 4, 12, 18)) == 5 * 64 * (32 * 48) * 4


@pytest.fixture(scope="module")
def taeqi() -> TaesdDecoder:
    torch.manual_seed(0)
    return F16Decoder(64, image_channels=4).eval()


def test_the_cap_is_applied_at_the_decoders_own_compression(taeqi: TaesdDecoder, monkeypatch) -> None:
    """An 8 x 4 latent fits a 96 cap at 8x (64 x 32) but not at TAEQI2.1's 16x, where it is scaled by 3/4."""
    monkeypatch.setattr(preview, "MAX_PREVIEW_EDGE", 96)

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
INSTALLED = [SD1_TAE, SDXL_TAE, SD3_TAE, FLUX_TAE, FLUX2_TAE]


@pytest.mark.parametrize(
    ("base", "channels", "expected"),
    [
        (BaseModelType.StableDiffusion2, 4, SD1_TAE),
        (BaseModelType.StableDiffusionXLRefiner, 4, SDXL_TAE),
        (BaseModelType.ZImage, 16, FLUX_TAE),
        (BaseModelType.StableDiffusion3, 16, SD3_TAE),
        (BaseModelType.ErnieImage, 32, FLUX2_TAE),
        (BaseModelType.CogView4, 16, None),
        (BaseModelType.QwenImage, 16, None),
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
        (BaseModelType.CogView4, 16, False),
        (BaseModelType.QwenImage, 16, False),
        (BaseModelType.LTX2, 128, False),
    ],
)
def test_whether_any_tae_decodes_a_space(base: BaseModelType, channels: int, supported: bool) -> None:
    assert is_tiny_autoencoder_supported(_space(base, channels)) is supported

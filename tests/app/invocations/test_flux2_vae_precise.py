"""FLUX.2 `flux2_vae_decode` and `flux2_vae_encode` with precise image storage, on CPU with a tiny real `AutoencoderKLFlux2`."""

from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from PIL import Image

from invokeai.app.invocations.vae.flux2_vae_decode import Flux2VaeDecodeInvocation
from invokeai.app.invocations.vae.flux2_vae_encode import Flux2VaeEncodeInvocation
from invokeai.backend.image_util.raster.descriptor import RasterChannels


@pytest.fixture
def flux2_vae() -> Callable[..., AutoencoderKLFlux2]:
    """A tiny seeded FLUX.2 VAE (32 latent channels, 8x); `overshoot` scales the decoder's last conv so its output
    leaves [-1, 1] and the unclamped path has something to keep."""

    def build(overshoot: float = 1.0) -> AutoencoderKLFlux2:
        torch.manual_seed(0)
        vae = AutoencoderKLFlux2(
            block_out_channels=(4, 4, 4, 4), layers_per_block=1, latent_channels=32, norm_num_groups=2, sample_size=64
        ).eval()
        with torch.no_grad():
            vae.decoder.conv_out.weight.mul_(overshoot)
            vae.decoder.conv_out.bias.add_(0.3 * overshoot)
        return vae

    return build


def _loaded(vae: AutoencoderKLFlux2) -> MagicMock:
    info = MagicMock()
    info.model = vae
    info.compute_device = torch.device("cpu")

    @contextmanager
    def model_on_device(working_memory=None):
        yield (None, vae)

    info.model_on_device = model_on_device
    return info


def _use(precise_world, vae: AutoencoderKLFlux2) -> None:
    precise_world.use_vae(_loaded(vae))
    precise_world.context.config = SimpleNamespace(
        get=lambda: SimpleNamespace(force_tiled_decode=False, auto_tiled_decode=False)
    )
    precise_world.context.logger = MagicMock()


def _decode_node() -> Flux2VaeDecodeInvocation:
    return Flux2VaeDecodeInvocation.model_construct(
        id="flux2_vae_decode",
        latents=SimpleNamespace(latents_name="latents"),
        vae=SimpleNamespace(vae="vae"),
        tiled=False,
        tile_size=0,
    )


def _latents(seed: int = 1) -> torch.Tensor:
    return torch.randn(1, 32, 2, 2, generator=torch.Generator().manual_seed(seed))


class _DecodeSpy:
    """Records what the VAE's decoder returned, so expected values come from the decoder, not from the node.

    `dtype` casts the output the node receives: a bf16 decode without a bf16 forward pass (the CPU VAE stays fp32).
    """

    def __init__(self, vae: AutoencoderKLFlux2, dtype: torch.dtype = torch.float32) -> None:
        self.outputs: list[torch.Tensor] = []
        real = vae.decode

        def decode(*args, **kwargs):
            output = real(*args, **kwargs)[0].detach().to(dtype)
            self.outputs.append(output.clone())
            return (output,)

        vae.decode = decode


def _run_decode(precise_world, vae: AutoencoderKLFlux2, latents: torch.Tensor) -> str:
    _use(precise_world, vae)
    precise_world.tensors["latents"] = latents
    return _decode_node().invoke(precise_world.context).image.image_name


def _legacy(decoded: torch.Tensor) -> Image.Image:
    """The pre-change quantizer: `decoded / 2 + 0.5` clamped to [0, 1] in the decode dtype, `* 255`, `.byte()`."""
    chw = (decoded[0] / 2 + 0.5).clamp(0, 1)
    return Image.fromarray((chw.permute(1, 2, 0) * 255).byte().numpy())


def _png_of(precise_world, image: Image.Image) -> bytes:
    """What the gallery PNG of `image` looks like when saved ordinarily (precise storage off)."""
    was = precise_world.configuration.precise_image_storage
    precise_world.configuration.precise_image_storage = False
    try:
        return precise_world.stored_bytes(precise_world.images.save(image).image_name)
    finally:
        precise_world.configuration.precise_image_storage = was


# --- decode ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("overshoot", [1.0, 8.0])
def test_decode_with_precise_storage_disabled_saves_the_truncating_legacy_png(
    flux2_vae, precise_world, monkeypatch, overshoot
) -> None:
    vae = flux2_vae(overshoot)
    spy = _DecodeSpy(vae)
    precise_world.configuration.precise_image_storage = False
    calls = []
    monkeypatch.setattr(precise_world.images, "save_raster", lambda *a, **k: calls.append((a, k)))

    name = _run_decode(precise_world, vae, _latents())

    legacy = _legacy(spy.outputs[0])
    # Truncation, not rounding: a rounding quantizer would differ from `.byte()` somewhere in a random image.
    unit = (spy.outputs[0][0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0) * 255
    assert (np.asarray(legacy) != unit.round().byte().numpy()).any()
    assert calls == []
    assert precise_world.stored_bytes(name) == _png_of(precise_world, legacy)
    assert precise_world.images.get_raster(name).dtype == np.uint8


def test_precise_decode_keeps_the_unclamped_output_and_the_gallery_png_stays_legacy(flux2_vae, precise_world) -> None:
    vae = flux2_vae(overshoot=8.0)
    spy = _DecodeSpy(vae)

    name = _run_decode(precise_world, vae, _latents())

    decoded = spy.outputs[0]
    # The setup must actually overshoot, or the unclamped path would be indistinguishable from the clamped one.
    assert decoded.min() < -1.0 and decoded.max() > 1.0
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == precise_world.descriptor(RasterChannels.RGB)
    expected = ((decoded[0].float() + 1.0) / 2).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(raster.view(), expected)
    assert raster.view().min() < 0.0 and raster.view().max() > 1.0
    assert precise_world.stored_bytes(name) == _png_of(precise_world, _legacy(decoded))


@pytest.mark.parametrize("precise", [False, True])
def test_a_bf16_decode_keeps_the_legacy_png_and_widens_the_raster(flux2_vae, precise_world, precise: bool) -> None:
    vae = flux2_vae(overshoot=4.0)
    spy = _DecodeSpy(vae, torch.bfloat16)
    precise_world.configuration.precise_image_storage = precise

    name = _run_decode(precise_world, vae, _latents())

    decoded = spy.outputs[0]
    assert decoded.dtype == torch.bfloat16
    legacy = _legacy(decoded)
    # The legacy quantizer runs in bf16. Deriving the PNG from the float32 unit tensor would differ, so this pins it.
    from_float32_unit = Image.fromarray(
        (((decoded[0].float() + 1.0) / 2).clamp(0, 1).permute(1, 2, 0) * 255).byte().numpy()
    )
    assert np.asarray(legacy).tolist() != np.asarray(from_float32_unit).tolist()
    assert precise_world.stored_bytes(name) == _png_of(precise_world, legacy)
    if not precise:
        assert precise_world.images.get_raster(name).dtype == np.uint8
        return
    widened = ((decoded[0].float() + 1.0) / 2).permute(1, 2, 0).numpy()
    in_bf16 = ((decoded[0] + 1.0) / 2).float().permute(1, 2, 0).numpy()
    assert (widened != in_bf16).any()  # otherwise the case could not tell the two orders apart
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    np.testing.assert_array_equal(raster.view(), widened)


@pytest.mark.parametrize("poisoned", ["all", "one-channel"])
def test_non_finite_decoder_output_falls_back_to_the_ordinary_png_and_warns(
    flux2_vae, precise_world, poisoned: str
) -> None:
    vae = flux2_vae()
    with torch.no_grad():
        if poisoned == "all":
            vae.decoder.conv_out.bias.fill_(float("nan"))
        else:
            vae.decoder.conv_out.bias[1] = float("nan")
    latents = _latents()

    name = _run_decode(precise_world, vae, latents)
    warning = precise_world.context.logger.warning
    precise_world.configuration.precise_image_storage = False
    disabled = _run_decode(precise_world, vae, latents)

    assert precise_world.stored_bytes(name) == precise_world.stored_bytes(disabled)
    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.database.queries.image_precise_assets.get(name) is None
    warning.assert_called_once()
    assert "not finite" in warning.call_args.args[0]


# --- encode ---------------------------------------------------------------------------------------------------------


def _encode(precise_world, vae: AutoencoderKLFlux2, image_name: str) -> torch.Tensor:
    _use(precise_world, vae)
    node = Flux2VaeEncodeInvocation.model_construct(
        id="flux2_vae_encode",
        image=SimpleNamespace(image_name=image_name),
        vae=SimpleNamespace(vae="vae"),
        tiled=False,
        tile_size=0,
    )
    return precise_world.tensors[node.invoke(precise_world.context).latents.latents_name]


def _chw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


def _noise(shape: tuple[int, ...], dtype, low: float, high: float) -> np.ndarray:
    return np.random.default_rng(3).uniform(low, high, size=shape).astype(dtype)


def _reference_latents(vae: AutoencoderKLFlux2, chw_unit: torch.Tensor) -> torch.Tensor:
    """Encodes an explicitly built [0, 1] CHW tensor through the static entry point (its sampler is seeded)."""
    with torch.no_grad():
        return Flux2VaeEncodeInvocation.vae_encode(
            context=MagicMock(), vae_info=_loaded(vae), image_tensor=(chw_unit * 2.0 - 1.0)[None]
        )


def test_an_image_without_a_precise_asset_encodes_exactly_as_the_eight_bit_path_did(flux2_vae, precise_world) -> None:
    vae = flux2_vae()
    pixels = np.random.default_rng(2).integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
    name = precise_world.store_png(pixels, "RGB")
    assert not precise_world.images.has_precise_asset(name)

    latents = _encode(precise_world, vae, name)

    np.testing.assert_array_equal(latents.numpy(), _reference_latents(vae, _chw(pixels) / 255.0).numpy())


def test_an_image_with_a_precise_asset_encodes_its_unclamped_values(flux2_vae, precise_world) -> None:
    vae = flux2_vae()
    array = _noise((16, 24, 3), np.float32, low=-0.4, high=1.6)
    name = precise_world.store_precise(array, RasterChannels.RGB)

    latents = _encode(precise_world, vae, name)

    np.testing.assert_array_equal(latents.numpy(), _reference_latents(vae, _chw(array)).numpy())
    clamped = _reference_latents(vae, _chw(np.clip(array, 0, 1)))
    assert not torch.allclose(latents, clamped)

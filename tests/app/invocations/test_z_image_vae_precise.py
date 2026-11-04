"""Z-Image `z_image_l2i` and `z_image_i2l` with precise image storage, on CPU with a tiny real `AutoencoderKL`."""

from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image

from invokeai.app.invocations.vae.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.app.invocations.vae.z_image_latents_to_image import ZImageLatentsToImageInvocation
from invokeai.backend.image_util.raster.descriptor import RasterChannels


@pytest.fixture
def z_image_vae(flux_shaped_vae) -> Callable[..., AutoencoderKL]:
    """Builds the conftest 16-channel VAE (Z-Image's shape) with seeded weights; `overshoot` scales the decoder's last
    conv so its output leaves [-1, 1] and the unclamped path has something to keep."""

    def build(overshoot: float = 1.0) -> AutoencoderKL:
        torch.manual_seed(0)
        vae = flux_shaped_vae().eval()
        with torch.no_grad():
            vae.decoder.conv_out.weight.mul_(overshoot)
            vae.decoder.conv_out.bias.add_(0.3 * overshoot)
        return vae

    return build


def _loaded(vae: AutoencoderKL) -> MagicMock:
    info = MagicMock()
    info.model = vae
    info.compute_device = torch.device("cpu")

    @contextmanager
    def model_on_device(working_memory=None):
        yield (None, vae)

    info.model_on_device = model_on_device
    return info


def _use(precise_world, vae: AutoencoderKL) -> None:
    precise_world.use_vae(_loaded(vae))
    precise_world.context.config = SimpleNamespace(
        get=lambda: SimpleNamespace(force_tiled_decode=False, auto_tiled_decode=False)
    )
    precise_world.context.logger = MagicMock()


def _decode_node() -> ZImageLatentsToImageInvocation:
    return ZImageLatentsToImageInvocation.model_construct(
        id="z_image_l2i",
        latents=SimpleNamespace(latents_name="latents"),
        vae=SimpleNamespace(vae="vae", seamless_axes=[]),
        tiled=False,
        tile_size=0,
    )


def _latents(seed: int = 1) -> torch.Tensor:
    return torch.randn(1, 16, 2, 2, generator=torch.Generator().manual_seed(seed))


class _DecodeSpy:
    """Records what the VAE's decoder returned, so expected values come from the decoder, not from the node.

    `dtype` casts the output the node receives: a bf16 decode without a bf16 forward pass (the CPU VAE stays fp32).
    """

    def __init__(self, vae: AutoencoderKL, dtype: torch.dtype = torch.float32) -> None:
        self.outputs: list[torch.Tensor] = []
        real = vae.decode

        def decode(*args, **kwargs):
            output = real(*args, **kwargs)[0].detach().to(dtype)
            self.outputs.append(output.clone())
            return (output,)

        vae.decode = decode


def _run_decode(precise_world, vae: AutoencoderKL, latents: torch.Tensor) -> str:
    _use(precise_world, vae)
    precise_world.tensors["latents"] = latents
    return _decode_node().invoke(precise_world.context).image.image_name


def _legacy(decoded: torch.Tensor) -> Image.Image:
    """The pre-change quantizer: clamp, `127.5 * (x + 1)` in the decode dtype, `.byte()` truncation."""
    chw = decoded[0].clamp(-1, 1)
    return Image.fromarray((127.5 * (chw.permute(1, 2, 0) + 1.0)).byte().numpy())


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
    z_image_vae, precise_world, monkeypatch, overshoot
) -> None:
    vae = z_image_vae(overshoot)
    spy = _DecodeSpy(vae)
    precise_world.configuration.precise_image_storage = False
    calls = []
    monkeypatch.setattr(precise_world.images, "save_raster", lambda *a, **k: calls.append((a, k)))

    name = _run_decode(precise_world, vae, _latents())

    legacy = _legacy(spy.outputs[0])
    # Truncation, not rounding: a rounding quantizer would differ from `.byte()` somewhere in a random image.
    unit = (spy.outputs[0][0].clamp(-1, 1).permute(1, 2, 0) + 1.0) * 127.5
    assert (np.asarray(legacy) != unit.round().byte().numpy()).any()
    assert calls == []
    assert precise_world.stored_bytes(name) == _png_of(precise_world, legacy)
    assert precise_world.images.get_raster(name).dtype == np.uint8


def test_precise_decode_keeps_the_unclamped_output_and_the_gallery_png_stays_legacy(z_image_vae, precise_world) -> None:
    vae = z_image_vae(overshoot=8.0)
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
def test_a_bf16_decode_keeps_the_legacy_png_and_widens_the_raster(z_image_vae, precise_world, precise: bool) -> None:
    vae = z_image_vae(overshoot=4.0)
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


def test_the_oom_retry_still_saves_a_precise_decode(z_image_vae, precise_world) -> None:
    vae = z_image_vae(overshoot=8.0)
    spy = _DecodeSpy(vae)
    inner = vae.decode
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return inner(*args, **kwargs)

    vae.decode = fail_once

    name = _run_decode(precise_world, vae, _latents())

    assert len(attempts) == 2
    expected = ((spy.outputs[0][0].float() + 1.0) / 2).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(precise_world.images.get_raster(name).view(), expected)


@pytest.mark.parametrize("poisoned", ["all", "one-channel"])
def test_non_finite_decoder_output_falls_back_to_the_ordinary_png_and_warns(
    z_image_vae, precise_world, poisoned: str
) -> None:
    vae = z_image_vae()
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


def _encode(precise_world, vae: AutoencoderKL, image_name: str) -> torch.Tensor:
    _use(precise_world, vae)
    node = ZImageImageToLatentsInvocation.model_construct(
        id="z_image_i2l",
        image=SimpleNamespace(image_name=image_name),
        vae=SimpleNamespace(vae="vae"),
        tiled=False,
        tile_size=0,
    )
    torch.manual_seed(5)  # the encode samples its latent distribution
    return precise_world.tensors[node.invoke(precise_world.context).latents.latents_name]


def _chw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


def _noise(shape: tuple[int, ...], dtype, low: float, high: float) -> np.ndarray:
    return np.random.default_rng(3).uniform(low, high, size=shape).astype(dtype)


def _reference_latents(vae: AutoencoderKL, chw_unit: torch.Tensor) -> torch.Tensor:
    """Encodes an explicitly built [0, 1] CHW tensor through the static entry point (seeded: the encode samples)."""
    with torch.no_grad():
        torch.manual_seed(5)
        return ZImageImageToLatentsInvocation.vae_encode(
            vae_info=_loaded(vae), image_tensor=(chw_unit * 2.0 - 1.0)[None]
        )


def test_an_image_without_a_precise_asset_encodes_exactly_as_the_eight_bit_path_did(z_image_vae, precise_world) -> None:
    vae = z_image_vae()
    pixels = np.random.default_rng(2).integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
    name = precise_world.store_png(pixels, "RGB")
    assert not precise_world.images.has_precise_asset(name)

    latents = _encode(precise_world, vae, name)

    np.testing.assert_array_equal(latents.numpy(), _reference_latents(vae, _chw(pixels) / 255.0).numpy())


def test_an_image_with_a_precise_asset_encodes_its_unclamped_values(z_image_vae, precise_world) -> None:
    vae = z_image_vae()
    array = _noise((16, 24, 3), np.float32, low=-0.4, high=1.6)
    name = precise_world.store_precise(array, RasterChannels.RGB)

    latents = _encode(precise_world, vae, name)

    np.testing.assert_array_equal(latents.numpy(), _reference_latents(vae, _chw(array)).numpy())
    clamped = _reference_latents(vae, _chw(np.clip(array, 0, 1)))
    assert not torch.allclose(latents, clamped)

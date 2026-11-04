"""Qwen Image `qwen_image_l2i` and `qwen_image_i2l` with precise image storage, on CPU with a tiny real VAE."""

from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKLQwenImage
from PIL import Image

from invokeai.app.invocations.vae.qwen_image_image_to_latents import (
    QwenImageImageToLatentsInvocation,
    reference_image_size,
)
from invokeai.app.invocations.vae.qwen_image_latents_to_image import QwenImageLatentsToImageInvocation
from invokeai.backend.image_util.raster.descriptor import RasterChannels
from invokeai.backend.util.devices import TorchDevice


@pytest.fixture(autouse=True)
def _cpu_untiled(monkeypatch) -> None:
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        "invokeai.app.invocations.vae.qwen_image_latents_to_image.should_pretile_vae_decode", lambda *a, **k: False
    )


@pytest.fixture
def qwen_vae() -> Callable[..., AutoencoderKLQwenImage]:
    """Builds a tiny Qwen Image VAE (16 latent channels, 8x) with seeded weights; `overshoot` scales the decoder's
    last conv so more of its output saturates at the clamp."""

    def build(overshoot: float = 1.0) -> AutoencoderKLQwenImage:
        torch.manual_seed(0)
        vae = AutoencoderKLQwenImage(
            base_dim=2,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
        ).eval()
        with torch.no_grad():
            vae.decoder.conv_out.weight.mul_(overshoot)
            vae.decoder.conv_out.bias.add_(0.3 * overshoot)
        return vae

    return build


def _loaded(vae: AutoencoderKLQwenImage) -> MagicMock:
    info = MagicMock()
    info.model = vae
    info.compute_device = torch.device("cpu")

    @contextmanager
    def model_on_device(working_mem_bytes=None):
        yield (None, vae)

    info.model_on_device = model_on_device
    return info


def _use(precise_world, vae: AutoencoderKLQwenImage) -> None:
    precise_world.use_vae(_loaded(vae))
    precise_world.context.config = SimpleNamespace(
        get=lambda: SimpleNamespace(force_tiled_decode=False, auto_tiled_decode=False)
    )
    precise_world.context.logger = MagicMock()


def _decode_node() -> QwenImageLatentsToImageInvocation:
    return QwenImageLatentsToImageInvocation.model_construct(
        id="qwen_image_l2i",
        latents=SimpleNamespace(latents_name="latents"),
        vae=SimpleNamespace(vae="vae", seamless_axes=[]),
        tiled=False,
        tile_size=0,
    )


def _latents(seed: int = 1) -> torch.Tensor:
    return torch.randn(1, 16, 1, 2, 2, generator=torch.Generator().manual_seed(seed))


class _DecodeSpy:
    """Records what the VAE's decoder returned, so expected values come from the decoder, not from the node.

    `dtype` casts the output the node receives: a bf16 decode without a bf16 forward pass (the CPU VAE stays fp32).
    """

    def __init__(self, vae: AutoencoderKLQwenImage, dtype: torch.dtype = torch.float32) -> None:
        self.outputs: list[torch.Tensor] = []
        real = vae.decode

        def decode(*args, **kwargs):
            output = real(*args, **kwargs)[0].detach().to(dtype)
            self.outputs.append(output[:, :, 0].clone())  # the node drops the temporal dim
            return (output,)

        vae.decode = decode


def _run_decode(precise_world, vae: AutoencoderKLQwenImage, latents: torch.Tensor) -> str:
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
    qwen_vae, precise_world, monkeypatch, overshoot
) -> None:
    vae = qwen_vae(overshoot)
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


def test_precise_decode_stores_the_decoder_output_as_a_float_raster_and_the_gallery_png_stays_legacy(
    qwen_vae, precise_world
) -> None:
    # The decoder clamps its own output to [-1, 1]; the raster is that output mapped to unit range in float32.
    vae = qwen_vae()
    spy = _DecodeSpy(vae)

    name = _run_decode(precise_world, vae, _latents())

    decoded = spy.outputs[0]
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == precise_world.descriptor(RasterChannels.RGB)
    expected = ((decoded[0].float() + 1.0) / 2).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(raster.view(), expected)
    assert precise_world.stored_bytes(name) == _png_of(precise_world, _legacy(decoded))


@pytest.mark.parametrize("precise", [False, True])
def test_a_bf16_decode_keeps_the_legacy_png_and_widens_the_raster(qwen_vae, precise_world, precise: bool) -> None:
    vae = qwen_vae(overshoot=4.0)
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
    qwen_vae, precise_world, poisoned: str
) -> None:
    vae = qwen_vae()
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


def _encode(precise_world, vae: AutoencoderKLQwenImage, image_name: str, **fields) -> torch.Tensor:
    _use(precise_world, vae)
    node = QwenImageImageToLatentsInvocation.model_construct(
        id="qwen_image_i2l",
        image=SimpleNamespace(image_name=image_name),
        vae=SimpleNamespace(vae="vae"),
        tiled=False,
        tile_size=0,
        width=fields.get("width"),
        height=fields.get("height"),
        reference=fields.get("reference", False),
    )
    return precise_world.tensors[node.invoke(precise_world.context).latents.latents_name]


def _encode_input(precise_world, vae, monkeypatch, image_name: str, **fields) -> torch.Tensor:
    """The pixel tensor the node hands to the VAE."""
    seen: list[torch.Tensor] = []
    real = QwenImageImageToLatentsInvocation.vae_encode

    def spy(vae_info, image_tensor, tiled=False, tile_size=0):
        seen.append(image_tensor.clone())
        return real(vae_info=vae_info, image_tensor=image_tensor, tiled=tiled, tile_size=tile_size)

    monkeypatch.setattr(QwenImageImageToLatentsInvocation, "vae_encode", staticmethod(spy))
    _encode(precise_world, vae, image_name, **fields)
    return seen[0]


def _chw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


def _noise(shape: tuple[int, ...], low: float, high: float) -> np.ndarray:
    return np.random.default_rng(3).uniform(low, high, size=shape).astype(np.float32)


def _bicubic(chw: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(chw[None], size=(height, width), mode="bicubic", align_corners=False, antialias=True)[0]


def test_an_image_without_a_precise_asset_encodes_exactly_as_the_eight_bit_path_did(qwen_vae, precise_world) -> None:
    vae = qwen_vae()
    pixels = np.random.default_rng(2).integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    name = precise_world.store_png(pixels, "RGB")
    assert not precise_world.images.has_precise_asset(name)

    latents = _encode(precise_world, vae, name)

    expected = QwenImageImageToLatentsInvocation.vae_encode(
        vae_info=_loaded(vae), image_tensor=(_chw(pixels) / 255.0 * 2.0 - 1.0)[None]
    )
    np.testing.assert_array_equal(latents.numpy(), expected.numpy())


def test_a_precise_asset_on_the_grid_encodes_its_unclamped_values(qwen_vae, precise_world, monkeypatch) -> None:
    vae = qwen_vae()
    array = _noise((32, 48, 3), low=-0.4, high=1.6)
    name = precise_world.store_precise(array, RasterChannels.RGB)

    tensor = _encode_input(precise_world, vae, monkeypatch, name)

    expected = (_chw(array) * 2.0 - 1.0)[None]
    assert tensor.shape == (1, 3, 32, 48)
    np.testing.assert_array_equal(tensor.numpy(), expected.numpy())
    assert tensor.min() < -1.0 and tensor.max() > 1.0


def test_a_precise_asset_off_the_grid_is_trimmed_by_antialiased_bicubic_resize(
    qwen_vae, precise_world, monkeypatch
) -> None:
    vae = qwen_vae()
    array = _noise((40, 52, 3), low=-0.2, high=1.2)
    name = precise_world.store_precise(array, RasterChannels.RGB)

    tensor = _encode_input(precise_world, vae, monkeypatch, name)

    expected = _bicubic(_chw(array), 32, 48)[None] * 2.0 - 1.0
    assert tensor.shape == (1, 3, 32, 48)
    torch.testing.assert_close(tensor, expected)


def test_a_precise_asset_with_a_target_size_is_resized_bicubic_then_snapped_to_the_grid(
    qwen_vae, precise_world, monkeypatch
) -> None:
    vae = qwen_vae()
    array = _noise((32, 48, 3), low=-0.2, high=1.2)
    name = precise_world.store_precise(array, RasterChannels.RGB)

    # 30 x 50 is off the 16 grid, so the resize is followed by a trim to 16 x 48.
    tensor = _encode_input(precise_world, vae, monkeypatch, name, width=50, height=30)

    expected = _bicubic(_bicubic(_chw(array), 30, 50), 16, 48)[None] * 2.0 - 1.0
    assert tensor.shape == (1, 3, 16, 48)
    torch.testing.assert_close(tensor, expected)

    # An on-grid target is resized once.
    tensor = _encode_input(precise_world, vae, monkeypatch, name, width=64, height=16)
    torch.testing.assert_close(tensor, _bicubic(_chw(array), 16, 64)[None] * 2.0 - 1.0)


def test_a_precise_reference_is_resized_bilinearly_to_the_reference_size(qwen_vae, precise_world, monkeypatch) -> None:
    vae = qwen_vae()
    array = _noise((20, 36, 3), low=-0.3, high=1.3)
    name = precise_world.store_precise(array, RasterChannels.RGB)
    assert reference_image_size(20, 36) == (16, 32)  # the size changes, so the resize is exercised

    tensor = _encode_input(precise_world, vae, monkeypatch, name, reference=True)

    expected = F.interpolate(_chw(array)[None], size=(16, 32), mode="bilinear") * 2.0 - 1.0
    assert tensor.shape == (1, 3, 16, 32)
    torch.testing.assert_close(tensor, expected)


def test_a_precise_reference_with_a_target_size_is_refused(qwen_vae, precise_world) -> None:
    vae = qwen_vae()
    name = precise_world.store_precise(_noise((32, 32, 3), low=0, high=1), RasterChannels.RGB)

    with pytest.raises(ValueError, match="sized automatically"):
        _encode(precise_world, vae, name, reference=True, width=32, height=32)

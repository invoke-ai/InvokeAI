"""`ideogram4_l2i` with precise image storage, on CPU with a tiny real `AutoencoderKLFlux2`."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image

from invokeai.app.invocations.vae import ideogram4_latents_to_image as module
from invokeai.app.invocations.vae.ideogram4_latents_to_image import Ideogram4LatentsToImageInvocation
from invokeai.backend.image_util.raster.descriptor import RasterChannels


def _vae(overshoot: float = 1.0) -> AutoencoderKLFlux2:
    """Four stages (8x) and 32 latent channels. `overshoot` scales the last conv so the output leaves [-1, 1] and the
    unclamped path has something to keep."""
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        block_out_channels=(32, 64, 64, 64), latent_channels=32, layers_per_block=1, norm_num_groups=32
    ).eval()
    with torch.no_grad():
        vae.decoder.conv_out.weight.mul_(overshoot)
        vae.decoder.conv_out.bias.add_(0.3 * overshoot)
    return vae


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


def _run(precise_world, vae: AutoencoderKLFlux2, monkeypatch) -> str:
    # The estimate is about CUDA memory; this test is about what the node does with the decoded tensor.
    monkeypatch.setattr(module, "estimate_vae_working_memory_flux2", lambda **_: None)
    info = MagicMock()
    info.compute_device = torch.device("cpu")

    @contextmanager
    def model_on_device(working_memory=None):
        yield (None, vae)

    info.model_on_device = model_on_device
    precise_world.use_vae(info)
    precise_world.context.logger = MagicMock()
    precise_world.tensors["latents"] = torch.randn(1, 128, 1, 1, generator=torch.Generator().manual_seed(1))
    node = Ideogram4LatentsToImageInvocation.model_construct(
        id="ideogram4_l2i", latents=SimpleNamespace(latents_name="latents"), vae=SimpleNamespace(vae="vae")
    )
    return node.invoke(precise_world.context).image.image_name


def _legacy(decoded: torch.Tensor) -> Image.Image:
    """The pre-change quantizer: float32 clamp to [-1, 1], `127.5 * (x + 1)` and `.byte()` truncation."""
    chw = decoded[0].float().clamp(-1, 1)
    return Image.fromarray((127.5 * (chw.permute(1, 2, 0) + 1.0)).byte().numpy())


def _png_of(precise_world, image: Image.Image) -> bytes:
    """What the gallery PNG of `image` looks like when saved ordinarily (precise storage off)."""
    was = precise_world.configuration.precise_image_storage
    precise_world.configuration.precise_image_storage = False
    try:
        return precise_world.stored_bytes(precise_world.images.save(image).image_name)
    finally:
        precise_world.configuration.precise_image_storage = was


def test_decode_with_precise_storage_disabled_saves_the_legacy_png_through_save(precise_world, monkeypatch) -> None:
    vae = _vae(overshoot=8.0)
    spy = _DecodeSpy(vae)
    precise_world.configuration.precise_image_storage = False
    calls = []
    monkeypatch.setattr(precise_world.images, "save_raster", lambda *a, **k: calls.append((a, k)))

    name = _run(precise_world, vae, monkeypatch)

    assert calls == []
    assert precise_world.stored_bytes(name) == _png_of(precise_world, _legacy(spy.outputs[0]))
    assert precise_world.images.get_raster(name).dtype == np.uint8


def test_precise_decode_keeps_the_unclamped_output_and_the_gallery_png_stays_legacy(precise_world, monkeypatch) -> None:
    vae = _vae(overshoot=8.0)
    spy = _DecodeSpy(vae)

    name = _run(precise_world, vae, monkeypatch)

    decoded = spy.outputs[0]
    # The setup must actually overshoot, or the unclamped path would be indistinguishable from the clamped one.
    assert decoded.min() < -1.0 and decoded.max() > 1.0
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == precise_world.descriptor(RasterChannels.RGB)
    np.testing.assert_array_equal(raster.view(), ((decoded[0] + 1.0) / 2).permute(1, 2, 0).numpy())
    assert raster.view().min() < 0.0 and raster.view().max() > 1.0
    assert precise_world.stored_bytes(name) == _png_of(precise_world, _legacy(decoded))


@pytest.mark.parametrize("precise", [False, True])
def test_a_bf16_decode_keeps_the_legacy_png_and_widens_the_raster(precise_world, monkeypatch, precise: bool) -> None:
    vae = _vae(overshoot=4.0)
    spy = _DecodeSpy(vae, torch.bfloat16)
    precise_world.configuration.precise_image_storage = precise

    name = _run(precise_world, vae, monkeypatch)

    decoded = spy.outputs[0]
    assert decoded.dtype == torch.bfloat16
    # The legacy quantizer widens before it clamps and scales; one that stayed in bf16 would differ, so this pins it.
    in_bf16 = Image.fromarray((127.5 * (decoded[0].clamp(-1, 1).permute(1, 2, 0) + 1.0)).byte().numpy())
    legacy = _legacy(decoded)
    assert np.asarray(legacy).tolist() != np.asarray(in_bf16).tolist()
    assert precise_world.stored_bytes(name) == _png_of(precise_world, legacy)
    if not precise:
        assert precise_world.images.get_raster(name).dtype == np.uint8
        return
    widened = ((decoded[0].float() + 1.0) / 2).permute(1, 2, 0).numpy()
    assert (widened != ((decoded[0] + 1.0) / 2).float().permute(1, 2, 0).numpy()).any()
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    np.testing.assert_array_equal(raster.view(), widened)


def test_non_finite_decoder_output_falls_back_to_the_ordinary_png_and_warns(precise_world, monkeypatch) -> None:
    vae = _vae()
    with torch.no_grad():
        vae.decoder.conv_out.bias[1] = float("nan")

    name = _run(precise_world, vae, monkeypatch)
    warning = precise_world.context.logger.warning
    precise_world.configuration.precise_image_storage = False
    disabled = _run(precise_world, vae, monkeypatch)

    assert precise_world.stored_bytes(name) == precise_world.stored_bytes(disabled)
    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.database.queries.image_precise_assets.get(name) is None
    warning.assert_called_once()
    assert "not finite" in warning.call_args.args[0]

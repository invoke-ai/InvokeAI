"""`ernie_image_vae_decode` with precise image storage, on CPU with a tiny real `AutoencoderKLFlux2`."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image

from invokeai.app.invocations.vae.ernie_image_vae_decode import ErnieImageVaeDecodeInvocation
from invokeai.backend.image_util.raster.descriptor import RasterChannels


def _vae(overshoot: float = 1.0) -> AutoencoderKLFlux2:
    """Four stages (8x), 32 latent channels, hence a `bn` over the 128 patched channels. `overshoot` scales the last
    conv so the output leaves [-1, 1] and the unclamped path has something to keep."""
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


def _run(precise_world, vae: AutoencoderKLFlux2) -> str:
    info = MagicMock()
    info.model = vae
    info.compute_device = torch.device("cpu")
    precise_world.context.config = SimpleNamespace(
        get=lambda: SimpleNamespace(force_tiled_decode=False, auto_tiled_decode=False)
    )

    @contextmanager
    def model_on_device(working_memory=None):
        yield (None, vae)

    info.model_on_device = model_on_device
    precise_world.use_vae(info)
    precise_world.context.logger = MagicMock()
    precise_world.tensors["latents"] = torch.randn(1, 128, 1, 1, generator=torch.Generator().manual_seed(1))
    node = ErnieImageVaeDecodeInvocation.model_construct(
        id="ernie_image_vae_decode", latents=SimpleNamespace(latents_name="latents"), vae=SimpleNamespace(vae="vae")
    )
    return node.invoke(precise_world.context).image.image_name


def _legacy(decoded: torch.Tensor) -> Image.Image:
    """The pre-change quantizer: `/ 2 + 0.5`, clamp, `* 255` and `.byte()` truncation in the decode dtype."""
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


def test_decode_with_precise_storage_disabled_saves_the_legacy_png_through_save(precise_world, monkeypatch) -> None:
    vae = _vae(overshoot=8.0)
    spy = _DecodeSpy(vae)
    precise_world.configuration.precise_image_storage = False
    calls = []
    monkeypatch.setattr(precise_world.images, "save_raster", lambda *a, **k: calls.append((a, k)))

    name = _run(precise_world, vae)

    assert calls == []
    assert precise_world.stored_bytes(name) == _png_of(precise_world, _legacy(spy.outputs[0]))
    assert precise_world.images.get_raster(name).dtype == np.uint8


def test_precise_decode_keeps_the_unclamped_output_and_the_gallery_png_stays_legacy(precise_world) -> None:
    vae = _vae(overshoot=8.0)
    spy = _DecodeSpy(vae)

    name = _run(precise_world, vae)

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
def test_a_bf16_decode_keeps_the_legacy_png_and_widens_the_raster(precise_world, precise: bool) -> None:
    vae = _vae(overshoot=4.0)
    spy = _DecodeSpy(vae, torch.bfloat16)
    precise_world.configuration.precise_image_storage = precise

    name = _run(precise_world, vae)

    decoded = spy.outputs[0]
    assert decoded.dtype == torch.bfloat16
    legacy = _legacy(decoded)
    # The legacy quantizer runs in bf16. Deriving the PNG from the float32 unit tensor would differ, so this pins it.
    from_float32 = Image.fromarray((((decoded[0].float() + 1.0) / 2).clamp(0, 1).permute(1, 2, 0) * 255).byte().numpy())
    assert np.asarray(legacy).tolist() != np.asarray(from_float32).tolist()
    assert precise_world.stored_bytes(name) == _png_of(precise_world, legacy)
    if not precise:
        assert precise_world.images.get_raster(name).dtype == np.uint8
        return
    widened = ((decoded[0].float() + 1.0) / 2).permute(1, 2, 0).numpy()
    assert (widened != ((decoded[0] + 1.0) / 2).float().permute(1, 2, 0).numpy()).any()
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    np.testing.assert_array_equal(raster.view(), widened)


def test_non_finite_decoder_output_falls_back_to_the_ordinary_png_and_warns(precise_world) -> None:
    vae = _vae()
    with torch.no_grad():
        vae.decoder.conv_out.bias[1] = float("nan")

    name = _run(precise_world, vae)
    warning = precise_world.context.logger.warning
    precise_world.configuration.precise_image_storage = False
    disabled = _run(precise_world, vae)

    assert precise_world.stored_bytes(name) == precise_world.stored_bytes(disabled)
    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.database.queries.image_precise_assets.get(name) is None
    warning.assert_called_once()
    assert "not finite" in warning.call_args.args[0]


def test_an_untiled_decode_that_runs_out_of_memory_is_retried_tiled_and_the_tiling_state_restored(
    precise_world,
) -> None:
    vae = _vae()
    real = vae.decode
    tiling_seen: list[bool] = []

    def decode(*args, **kwargs):
        tiling_seen.append(vae.use_tiling)
        if len(tiling_seen) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(*args, **kwargs)

    vae.decode = decode

    _run(precise_world, vae)

    assert tiling_seen == [False, True]
    assert vae.use_tiling is False
    precise_world.context.util.signal_progress.assert_any_call("VAE decode ran out of memory, retrying tiled")

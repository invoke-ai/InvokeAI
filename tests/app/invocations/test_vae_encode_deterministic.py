"""The image-to-latents encoders take the mean of the VAE's posterior, so an image encodes to one latent.

A sample of the posterior adds noise of its own that no seed controls, so img2img and inpaint from the same image,
seed and settings would not repeat. Each cell runs the real encode twice against a small, randomly initialised VAE of
the real class and expects the same bits.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from PIL import Image

from invokeai.app.invocations.vae.anima_image_to_latents import AnimaImageToLatentsInvocation
from invokeai.app.invocations.vae.cogview4_image_to_latents import CogView4ImageToLatentsInvocation
from invokeai.app.invocations.vae.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.vae.sd3_image_to_latents import SD3ImageToLatentsInvocation
from invokeai.app.invocations.vae.wan_image_to_latents import WanImageToLatentsInvocation
from invokeai.app.invocations.vae.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.wan.extensions.wan_ref_image_extension import (
    encode_reference_image_to_condition,
    encode_reference_image_to_ti2v_condition,
    encode_reference_image_to_video_condition,
)


def _wan_vae() -> AutoencoderKLWan:
    """A real Wan VAE with the A14B geometry (8x spatial, 4x temporal, 16 channels) and next to no weight."""
    return AutoencoderKLWan(
        base_dim=4,
        z_dim=16,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
    )


def _vae_info(vae: torch.nn.Module) -> MagicMock:
    vae_info = MagicMock()
    vae_info.model = vae
    vae_info.compute_device = torch.device("cpu")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device.return_value = cm
    return vae_info


def _pixels(frames: int | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    shape = (1, 3, 64, 64) if frames is None else (1, 3, frames, 64, 64)
    return torch.rand(shape, generator=generator) * 2 - 1


def _image() -> Image.Image:
    return Image.fromarray((_pixels()[0].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype("uint8"))


# Each builder takes the `flux_shaped_vae` factory and returns a callable that encodes the same image once.
def _kl_encode(node_encode, estimator: str | None = None):
    """`estimator` names a working-memory estimator with no fp32 line, which the CPU VAE here computes in."""

    def build(flux_shaped_vae):
        vae = flux_shaped_vae()

        def encode():
            if estimator is None:
                return node_encode(_vae_info(vae), _pixels())
            with patch(estimator, return_value=0):
                return node_encode(_vae_info(vae), _pixels())

        return encode

    return build


def _sd_encode(flux_shaped_vae):
    vae = flux_shaped_vae()
    return lambda: ImageToLatentsInvocation._encode_to_tensor(vae, _pixels())


def _anima_encode(_):
    vae_info = _vae_info(_wan_vae())
    return lambda: AnimaImageToLatentsInvocation.vae_encode(vae_info, _pixels())


def _wan_encode(_):
    vae_info = _vae_info(_wan_vae())
    return lambda: WanImageToLatentsInvocation.vae_encode(vae_info, _pixels())


def _wan_ref(encode):
    def build(_):
        vae = _wan_vae()
        return lambda: encode(vae)

    return build


_CPU = {"device": torch.device("cpu"), "dtype": torch.float32}

ENCODERS = {
    "sd1-sdxl": _sd_encode,
    "sd3": _kl_encode(
        SD3ImageToLatentsInvocation.vae_encode,
        "invokeai.app.invocations.vae.sd3_image_to_latents.estimate_vae_working_memory_sd3",
    ),
    "cogview4": _kl_encode(
        CogView4ImageToLatentsInvocation.vae_encode,
        "invokeai.app.invocations.vae.cogview4_image_to_latents.estimate_vae_working_memory_cogview4",
    ),
    "z-image": _kl_encode(ZImageImageToLatentsInvocation.vae_encode),
    "anima": _anima_encode,
    "wan": _wan_encode,
    "wan-ref-i2v": _wan_ref(lambda vae: encode_reference_image_to_condition(_image(), vae, 64, 64, **_CPU)),
    "wan-ref-video": _wan_ref(
        lambda vae: encode_reference_image_to_video_condition(_image(), vae, 64, 64, num_frames=5, **_CPU)
    ),
    "wan-ref-ti2v": _wan_ref(lambda vae: encode_reference_image_to_ti2v_condition(_image(), vae, 64, 64, **_CPU)),
}


@pytest.mark.parametrize("family", ENCODERS)
def test_encoding_the_same_image_twice_gives_identical_latents(family, flux_shaped_vae):
    encode_once = ENCODERS[family](flux_shaped_vae)

    with patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")):
        first, second = encode_once(), encode_once()

    assert first.numel() > 0
    assert torch.equal(first, second)

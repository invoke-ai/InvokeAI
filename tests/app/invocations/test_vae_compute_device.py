from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers import AutoencoderKL, AutoencoderKLFlux2

from invokeai.app.invocations.fields import LatentsField
from invokeai.app.invocations.vae.cogview4_image_to_latents import CogView4ImageToLatentsInvocation
from invokeai.app.invocations.vae.ernie_image_vae_decode import ErnieImageVaeDecodeInvocation
from invokeai.app.invocations.vae.sd3_image_to_latents import SD3ImageToLatentsInvocation
from invokeai.backend.util.devices import TorchDevice


@pytest.fixture
def cpu_vae_with_accelerator_session(monkeypatch):
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("meta"))

    def load_vae(vae, operation):
        run_vae = getattr(vae, operation)

        def check_input(tensor, **kwargs):
            assert tensor.device == torch.device("cpu")
            assert tensor.dtype == vae.dtype
            return run_vae(tensor, **kwargs)

        monkeypatch.setattr(vae, operation, check_input)
        return SimpleNamespace(
            model=vae,
            compute_device=torch.device("cpu"),
            model_on_device=lambda **_: nullcontext((None, vae)),
        )

    return load_vae


@pytest.mark.parametrize(
    "invocation", [SD3ImageToLatentsInvocation, CogView4ImageToLatentsInvocation], ids=["sd3", "cogview4"]
)
def test_image_encode_uses_the_cpu_vae_in_an_accelerator_session(
    invocation, flux_shaped_vae, cpu_vae_with_accelerator_session
):
    vae = flux_shaped_vae()
    if invocation is CogView4ImageToLatentsInvocation:
        # CogView4's estimator only accepts BF16, and its VAE has no mid-block attention.
        vae = AutoencoderKL.from_config(vae.config, mid_block_add_attention=False).to(dtype=torch.bfloat16)
    vae_info = cpu_vae_with_accelerator_session(vae, "encode")
    latents = invocation.vae_encode(vae_info, torch.zeros(1, 3, 32, 32, dtype=torch.float64))

    assert latents.shape == (1, 16, 4, 4)
    assert latents.device == torch.device("cpu")


def test_ernie_decode_uses_the_cpu_vae_in_an_accelerator_session(cpu_vae_with_accelerator_session):
    vae = AutoencoderKLFlux2(
        block_out_channels=(32, 32, 32, 32), latent_channels=32, layers_per_block=1, norm_num_groups=32
    ).eval()
    vae_info = cpu_vae_with_accelerator_session(vae, "decode")
    context = SimpleNamespace(
        models=SimpleNamespace(load=lambda _: vae_info),
        tensors=SimpleNamespace(load=lambda _: torch.zeros(1, 128, 2, 2, dtype=torch.float64)),
        images=SimpleNamespace(
            save=lambda image: SimpleNamespace(image_name="decoded.png", width=image.width, height=image.height)
        ),
    )
    invocation = ErnieImageVaeDecodeInvocation.model_construct(
        latents=LatentsField(latents_name="latents"), vae=SimpleNamespace(vae=None)
    )

    result = invocation.invoke(context)

    assert (result.width, result.height) == (32, 32)

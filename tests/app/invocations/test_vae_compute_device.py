from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers import AutoencoderKL, AutoencoderKLFlux2, AutoencoderKLWan

from invokeai.app.invocations.fields import LatentsField
from invokeai.app.invocations.vae.cogview4_image_to_latents import CogView4ImageToLatentsInvocation
from invokeai.app.invocations.vae.ernie_image_vae_decode import ErnieImageVaeDecodeInvocation
from invokeai.app.invocations.vae.qwen_image_image_to_latents import QwenImageImageToLatentsInvocation
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
    invocation, flux_shaped_vae, cpu_vae_with_accelerator_session, monkeypatch, fixed_working_memory
):
    vae = flux_shaped_vae()
    if invocation is CogView4ImageToLatentsInvocation:
        # CogView4's calibrated estimator only accepts BF16; device routing needs no low-precision forward.
        vae = AutoencoderKL.from_config(vae.config, mid_block_add_attention=False)
        monkeypatch.setattr(
            "invokeai.app.invocations.vae.cogview4_image_to_latents.estimate_vae_working_memory_cogview4",
            lambda **_: fixed_working_memory(1024),
        )
    vae_info = cpu_vae_with_accelerator_session(vae, "encode")
    latents = invocation.vae_encode(vae_info, torch.zeros(1, 3, 32, 32, dtype=torch.float64))

    assert latents.shape == (1, 16, 4, 4)
    assert latents.device == torch.device("cpu")
    assert latents.dtype == torch.float32


def test_encode_uses_compute_device_when_the_unused_decoder_is_offloaded(flux_shaped_vae):
    vae = flux_shaped_vae().eval()
    # Encode never calls the decoder. Offload its real weights and enumerate it first: module
    # registration order is not an execution-device contract. The encoder remains entirely on CPU.
    decoder_weights = vae.decoder.state_dict()
    vae.decoder.to("meta")
    vae._modules = {"decoder": vae.decoder, **vae._modules}
    assert next(vae.parameters()).device == torch.device("meta")
    assert all(parameter.device == torch.device("cpu") for parameter in vae.encoder.parameters())
    vae_info = SimpleNamespace(
        model=vae,
        compute_device=torch.device("cpu"),
        model_on_device=lambda **_: nullcontext((None, vae)),
    )

    latents = SD3ImageToLatentsInvocation.vae_encode(vae_info, torch.zeros(1, 3, 32, 32))

    assert latents.device == vae_info.compute_device
    assert latents.shape == (1, 16, 4, 4)
    assert torch.isfinite(latents).all()
    assert next(vae.decoder.parameters()).device == torch.device("meta")
    vae.decoder.load_state_dict(decoder_weights, assign=True)


@pytest.mark.parametrize(
    ("invocation", "estimator"),
    [
        (SD3ImageToLatentsInvocation, "sd3"),
        (CogView4ImageToLatentsInvocation, "cogview4"),
        (QwenImageImageToLatentsInvocation, "qwen_image"),
    ],
    ids=["sd3", "cogview4", "qwen-image"],
)
def test_encode_estimate_uses_the_vae_compute_device(
    invocation, estimator, flux_shaped_vae, cpu_vae_with_accelerator_session, fixed_working_memory, monkeypatch
):
    if invocation is QwenImageImageToLatentsInvocation:
        vae = AutoencoderKLWan(
            base_dim=2,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
        ).eval()
    else:
        vae = flux_shaped_vae().eval()
    reservation = fixed_working_memory(123456)
    # Qwen's existing estimator/cache boundary still takes bytes rather than WorkingMemoryEstimate.
    expected = reservation.reservation_bytes if estimator == "qwen_image" else reservation
    estimate = Mock(return_value=expected)
    monkeypatch.setattr(f"{invocation.__module__}.estimate_vae_working_memory_{estimator}", estimate)
    vae_info = cpu_vae_with_accelerator_session(vae, "encode")
    on_device = Mock(wraps=vae_info.model_on_device)
    vae_info.model_on_device = on_device
    pixels = torch.zeros(1, 3, 32, 32)

    latents = invocation.vae_encode(vae_info, pixels)

    estimate.assert_called_once()
    assert estimate.call_args.kwargs["device"] == vae_info.compute_device
    assert estimate.call_args.kwargs["operation"] == "encode"
    assert estimate.call_args.kwargs["image_tensor"] is pixels
    assert estimate.call_args.kwargs["vae"] is vae
    argument = "working_mem_bytes" if estimator == "qwen_image" else "working_memory"
    on_device.assert_called_once_with(**{argument: expected})
    assert latents.device == vae_info.compute_device
    assert torch.isfinite(latents).all()


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

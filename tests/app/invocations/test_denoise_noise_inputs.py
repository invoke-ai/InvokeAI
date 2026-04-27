import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.cogview4_denoise import CogView4DenoiseInvocation
from invokeai.app.invocations.flux2_denoise import Flux2DenoiseInvocation
from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.metadata_linked import FluxDenoiseLatentsMetaInvocation, ZImageDenoiseMetaInvocation
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.sd3_denoise import SD3DenoiseInvocation
from invokeai.app.invocations.z_image_denoise import ZImageDenoiseInvocation
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional, get_schedule
from invokeai.backend.flux.schedulers import ANIMA_SCHEDULER_MAP, FLUX_SCHEDULER_MAP, ZIMAGE_SCHEDULER_MAP
from invokeai.backend.flux2.sampling_utils import get_schedule_flux2
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


def test_flux_prepare_noise_uses_external_noise():
    invocation = FluxDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch("invokeai.app.invocations.flux_denoise.get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_flux_prepare_noise_rejects_invalid_shape():
    invocation = FluxDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 15, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))


def test_flux_add_noise_false_ignores_connected_noise():
    invocation = FluxDenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        noise=MagicMock(latents_name="noise"),
        add_noise=False,
        width=64,
        height=64,
        num_steps=4,
        denoising_start=0.25,
        denoising_end=0.25,
        positive_text_conditioning=MagicMock(conditioning_name="positive"),
        transformer=MagicMock(transformer="transformer"),
        seed=123,
    )
    init_latents = torch.full((1, 16, 8, 8), 2.0)
    dummy_conditioning = SimpleNamespace(
        t5_embeds=torch.zeros(1, 4, 16),
        clip_embeds=torch.zeros(1, 768),
        to=lambda **_: dummy_conditioning,
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    mock_context.conditioning.load.return_value = SimpleNamespace(conditionings=[dummy_conditioning])
    mock_context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Flux, type=ModelType.Main, variant=None
    )

    with (
        patch(
            "invokeai.app.invocations.flux_denoise.TorchDevice.choose_torch_device", return_value=torch.device("cpu")
        ),
        patch("invokeai.app.invocations.flux_denoise.FLUXConditioningInfo", object),
        patch(
            "invokeai.app.invocations.flux_denoise.RegionalPromptingExtension.from_text_conditioning",
            return_value=MagicMock(),
        ),
        patch.object(invocation, "_prepare_noise_tensor", side_effect=AssertionError("noise should be ignored")),
        patch.object(invocation, "_load_redux_conditioning", return_value=[]),
        patch("invokeai.app.invocations.flux_denoise.get_schedule", return_value=[0.75]),
    ):
        result = invocation._run_diffusion(mock_context)

    assert torch.equal(result, init_latents)


def test_flux2_prepare_noise_uses_external_noise():
    invocation = Flux2DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 32, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch("invokeai.app.invocations.flux2_denoise.get_noise_flux2") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_flux2_prepare_noise_rejects_invalid_shape():
    invocation = Flux2DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))


def test_sd3_prepare_noise_uses_external_noise():
    invocation = SD3DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch.object(invocation, "_get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, 16, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_sd3_prepare_noise_rejects_invalid_shape():
    invocation = SD3DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 8, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, 16, torch.bfloat16, torch.device("cpu"))


def test_cogview4_prepare_noise_uses_external_noise():
    invocation = CogView4DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch.object(invocation, "_get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, 16, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_cogview4_prepare_noise_rejects_invalid_shape():
    invocation = CogView4DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 4, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, 16, torch.bfloat16, torch.device("cpu"))


def test_z_image_prepare_noise_uses_external_noise():
    invocation = ZImageDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch.object(invocation, "_get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_z_image_prepare_noise_rejects_invalid_shape():
    invocation = ZImageDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 8, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))


def test_z_image_add_noise_false_ignores_connected_noise():
    invocation = ZImageDenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        noise=MagicMock(latents_name="noise"),
        add_noise=False,
        width=64,
        height=64,
        steps=4,
        denoising_start=0.0,
        denoising_end=1.0,
        positive_conditioning=SimpleNamespace(conditioning_name="positive", mask=None),
        transformer=MagicMock(transformer="transformer"),
        seed=123,
        scheduler="euler",
    )
    init_latents = torch.full((1, 16, 8, 8), 2.0)
    dummy_conditioning = SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16))
    dummy_conditioning.to = lambda **_: dummy_conditioning
    regional_extension = SimpleNamespace(
        regional_text_conditioning=SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16))
    )
    loaded_text_conditioning = [SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16), mask=None)]
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    mock_context.conditioning.load.return_value = SimpleNamespace(conditionings=[dummy_conditioning])

    with (
        patch(
            "invokeai.app.invocations.z_image_denoise.TorchDevice.choose_torch_device", return_value=torch.device("cpu")
        ),
        patch(
            "invokeai.app.invocations.z_image_denoise.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.bfloat16,
        ),
        patch("invokeai.app.invocations.z_image_denoise.ZImageConditioningInfo", object),
        patch(
            "invokeai.app.invocations.z_image_denoise.ZImageRegionalPromptingExtension.from_text_conditionings",
            return_value=regional_extension,
        ),
        patch.object(invocation, "_load_text_conditioning", return_value=loaded_text_conditioning),
        patch.object(invocation, "_prepare_noise_tensor", side_effect=AssertionError("noise should be ignored")),
        patch.object(invocation, "_get_sigmas", return_value=[0.75]),
    ):
        result = invocation._run_diffusion(mock_context)

    assert torch.equal(result, init_latents)


def test_anima_prepare_noise_uses_external_noise():
    invocation = AnimaDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 1, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch.object(invocation, "_get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_anima_prepare_noise_rejects_invalid_rank():
    invocation = AnimaDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))


def test_anima_add_noise_false_ignores_connected_noise():
    invocation = AnimaDenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        noise=MagicMock(latents_name="noise"),
        add_noise=False,
        width=64,
        height=64,
        steps=4,
        denoising_start=0.0,
        denoising_end=1.0,
        positive_conditioning=SimpleNamespace(conditioning_name="positive", mask=None),
        transformer=MagicMock(transformer="transformer"),
        seed=123,
        scheduler="euler",
    )
    init_latents = torch.full((1, 16, 8, 8), 2.0)
    loaded_text_conditioning = [SimpleNamespace(mask=None)]
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    mock_context.models.load.return_value = MagicMock()

    with (
        patch(
            "invokeai.app.invocations.anima_denoise.TorchDevice.choose_torch_device", return_value=torch.device("cpu")
        ),
        patch(
            "invokeai.app.invocations.anima_denoise.TorchDevice.choose_bfloat16_safe_dtype", return_value=torch.bfloat16
        ),
        patch.object(invocation, "_load_text_conditionings", return_value=loaded_text_conditioning),
        patch.object(invocation, "_prepare_noise_tensor", side_effect=AssertionError("noise should be ignored")),
        patch.object(invocation, "_get_sigmas", return_value=[0.75]),
    ):
        result = invocation._run_diffusion(mock_context)

    assert torch.equal(result, init_latents)


def test_flux2_add_noise_false_ignores_connected_noise():
    invocation = Flux2DenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        noise=MagicMock(latents_name="noise"),
        add_noise=False,
        width=64,
        height=64,
        num_steps=4,
        denoising_start=0.25,
        denoising_end=0.25,
        positive_text_conditioning=MagicMock(conditioning_name="positive"),
        transformer=MagicMock(transformer="transformer"),
        vae=MagicMock(vae="vae"),
        seed=123,
    )
    init_latents = torch.full((1, 32, 8, 8), 2.0)
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    mock_context.conditioning.load.return_value = SimpleNamespace(
        conditionings=[
            SimpleNamespace(
                t5_embeds=torch.zeros(1, 4, 16), to=lambda **_: SimpleNamespace(t5_embeds=torch.zeros(1, 4, 16))
            )
        ]
    )
    mock_context.models.get_config.return_value = SimpleNamespace(base=BaseModelType.Flux2, type=ModelType.Main)

    with (
        patch(
            "invokeai.app.invocations.flux2_denoise.TorchDevice.choose_torch_device", return_value=torch.device("cpu")
        ),
        patch("invokeai.app.invocations.flux2_denoise.FLUXConditioningInfo", object),
        patch.object(invocation, "_get_bn_stats", return_value=None),
        patch.object(invocation, "_prepare_noise_tensor", side_effect=AssertionError("noise should be ignored")),
    ):
        result = invocation._run_diffusion(mock_context)

    assert torch.equal(result, init_latents)


def test_flux_metadata_ignores_external_noise_seed_when_noise_not_used():
    invocation = FluxDenoiseLatentsMetaInvocation.model_construct(
        width=64,
        height=64,
        num_steps=4,
        guidance=3.5,
        denoising_start=0.0,
        denoising_end=1.0,
        latents=MagicMock(latents_name="latents"),
        transformer=MagicMock(transformer="transformer", loras=[]),
        noise=MagicMock(seed=123),
        seed=999,
        add_noise=False,
    )
    mock_context = MagicMock()
    output = LatentsOutput.build("latents", torch.zeros(1, 16, 8, 8), seed=None)

    with patch("invokeai.app.invocations.metadata_linked.FluxDenoiseInvocation.invoke", return_value=output):
        result = invocation.invoke(mock_context)

    assert result.metadata.root["seed"] == 999


def test_z_image_metadata_ignores_external_noise_seed_when_noise_not_used():
    invocation = ZImageDenoiseMetaInvocation.model_construct(
        width=64,
        height=64,
        steps=8,
        guidance_scale=1.0,
        denoising_start=0.0,
        denoising_end=1.0,
        scheduler="euler",
        latents=MagicMock(latents_name="latents"),
        transformer=MagicMock(transformer="transformer", loras=[]),
        noise=MagicMock(seed=123),
        seed=999,
        add_noise=False,
    )
    mock_context = MagicMock()
    output = LatentsOutput.build("latents", torch.zeros(1, 16, 8, 8), seed=None)

    with patch("invokeai.app.invocations.metadata_linked.ZImageDenoiseInvocation.invoke", return_value=output):
        result = invocation.invoke(mock_context)

    assert result.metadata.root["seed"] == 999


def _get_first_scheduler_sigma(
    scheduler, *, scheduler_name: str, sigmas: list[float], mu: float | None = None
) -> float:
    set_timesteps_signature = inspect.signature(scheduler.set_timesteps)
    if scheduler_name != "lcm" and "sigmas" in set_timesteps_signature.parameters:
        kwargs: dict[str, object] = {"sigmas": sigmas, "device": "cpu"}
        if mu is not None and "mu" in set_timesteps_signature.parameters:
            kwargs["mu"] = mu
        scheduler.set_timesteps(**kwargs)
    else:
        kwargs = {"num_inference_steps": len(sigmas) - 1, "device": "cpu"}
        if mu is not None and "mu" in set_timesteps_signature.parameters:
            kwargs["mu"] = mu
        scheduler.set_timesteps(**kwargs)
    return float(scheduler.sigmas[0])


@pytest.mark.parametrize(
    "scheduler_name",
    [
        "euler",
        pytest.param(
            "heun",
            marks=pytest.mark.xfail(
                reason="Known img2img preblend mismatch for FLUX with scheduler-defined first step.",
                strict=True,
            ),
        ),
        pytest.param(
            "lcm",
            marks=pytest.mark.xfail(
                reason="Known img2img preblend mismatch for FLUX with scheduler-defined first step.",
                strict=True,
            ),
        ),
    ],
)
def test_flux_img2img_preblend_matches_scheduler_first_sigma(scheduler_name: str):
    sigmas = clip_timestep_schedule_fractional(get_schedule(num_steps=4, image_seq_len=16, shift=True), 0.25, 1.0)
    scheduler_class = FLUX_SCHEDULER_MAP[scheduler_name]
    scheduler = scheduler_class(num_train_timesteps=1000)

    assert sigmas[0] == pytest.approx(
        _get_first_scheduler_sigma(scheduler, scheduler_name=scheduler_name, sigmas=sigmas)
    )


def test_flux2_partial_denoise_short_circuit_uses_first_clipped_timestep():
    invocation = Flux2DenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        width=64,
        height=64,
        num_steps=4,
        denoising_start=0.25,
        denoising_end=0.25,
        positive_text_conditioning=MagicMock(conditioning_name="positive"),
        transformer=MagicMock(transformer="transformer"),
        vae=MagicMock(vae="vae"),
        seed=0,
        scheduler="lcm",
    )
    init_latents = torch.full((1, 32, 8, 8), 2.0)
    noise = torch.full((1, 32, 8, 8), 10.0)
    dummy_conditioning = SimpleNamespace(t5_embeds=torch.zeros(1, 4, 16))
    dummy_conditioning.to = lambda **_: dummy_conditioning
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    mock_context.conditioning.load.return_value = SimpleNamespace(conditionings=[dummy_conditioning])
    mock_context.models.get_config.return_value = SimpleNamespace(base=BaseModelType.Flux2, type=ModelType.Main)

    with (
        patch(
            "invokeai.app.invocations.flux2_denoise.TorchDevice.choose_torch_device", return_value=torch.device("cpu")
        ),
        patch("invokeai.app.invocations.flux2_denoise.FLUXConditioningInfo", object),
        patch.object(invocation, "_get_bn_stats", return_value=None),
        patch.object(invocation, "_prepare_noise_tensor", return_value=noise),
    ):
        result = invocation._run_diffusion(mock_context)

    timesteps = clip_timestep_schedule_fractional(get_schedule_flux2(num_steps=4, image_seq_len=16), 0.25, 0.25)
    expected = timesteps[0] * noise + (1.0 - timesteps[0]) * init_latents
    assert torch.equal(result, expected)


def test_flux2_lcm_scheduler_setup_passes_mu():
    from invokeai.backend.flux2.denoise import denoise

    class DummyScheduler:
        def __init__(self) -> None:
            self.received_mu = None
            self.timesteps = torch.tensor([750.0, 500.0], dtype=torch.float32)
            self.sigmas = torch.tensor([0.75, 0.5, 0.0], dtype=torch.float32)
            self.config = SimpleNamespace(num_train_timesteps=1000)

        def set_timesteps(self, num_inference_steps: int, device: str | torch.device, mu: float | None = None) -> None:
            self.received_mu = mu

        def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor):
            return SimpleNamespace(prev_sample=sample)

    class DummyModel(torch.nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            txt_ids: torch.Tensor,
            guidance: torch.Tensor,
            return_dict: bool = False,
        ):
            return (torch.zeros_like(hidden_states),)

    scheduler = DummyScheduler()
    denoise(
        model=DummyModel(),
        img=torch.zeros(1, 4, 8),
        img_ids=torch.zeros(1, 4, 4, dtype=torch.long),
        txt=torch.zeros(1, 4, 8),
        txt_ids=torch.zeros(1, 4, 4, dtype=torch.long),
        timesteps=[0.75, 0.5, 0.0],
        step_callback=lambda _: None,
        guidance=1.0,
        cfg_scale=[1.0, 1.0],
        scheduler=scheduler,
        mu=0.42,
    )

    assert scheduler.received_mu == pytest.approx(0.42)


@pytest.mark.parametrize(
    "scheduler_name",
    [
        "euler",
        pytest.param(
            "heun",
            marks=pytest.mark.xfail(
                reason="Known img2img preblend mismatch for Z-Image with scheduler-defined first step.",
                strict=True,
            ),
        ),
        pytest.param(
            "lcm",
            marks=pytest.mark.xfail(
                reason="Known img2img preblend mismatch for Z-Image with scheduler-defined first step.",
                strict=True,
            ),
        ),
    ],
)
def test_z_image_img2img_preblend_matches_scheduler_first_sigma(scheduler_name: str):
    invocation = ZImageDenoiseInvocation.model_construct(steps=8, width=1024, height=1024)
    img_seq_len = (invocation.height // 8 // 2) * (invocation.width // 8 // 2)
    shift = invocation._calculate_shift(img_seq_len)
    sigmas = invocation._get_sigmas(shift, invocation.steps)
    sigmas = sigmas[int(0.25 * (len(sigmas) - 1)) :]
    scheduler_class = ZIMAGE_SCHEDULER_MAP[scheduler_name]
    scheduler = scheduler_class(num_train_timesteps=1000, shift=1.0)

    assert sigmas[0] == pytest.approx(
        _get_first_scheduler_sigma(scheduler, scheduler_name=scheduler_name, sigmas=sigmas)
    )


@pytest.mark.parametrize(
    "scheduler_name",
    [
        "euler",
        pytest.param(
            "heun",
            marks=pytest.mark.xfail(
                reason="Known img2img preblend mismatch for Anima with scheduler-defined first step.",
                strict=True,
            ),
        ),
        pytest.param(
            "lcm",
            marks=pytest.mark.xfail(
                reason="Known img2img preblend mismatch for Anima with scheduler-defined first step.",
                strict=True,
            ),
        ),
    ],
)
def test_anima_img2img_preblend_matches_scheduler_first_sigma(scheduler_name: str):
    invocation = AnimaDenoiseInvocation.model_construct(steps=30)
    sigmas = invocation._get_sigmas(invocation.steps)
    sigmas = sigmas[int(0.25 * (len(sigmas) - 1)) :]
    scheduler_class = ANIMA_SCHEDULER_MAP[scheduler_name]
    scheduler = scheduler_class(num_train_timesteps=1000, shift=1.0)

    assert sigmas[0] == pytest.approx(
        _get_first_scheduler_sigma(scheduler, scheduler_name=scheduler_name, sigmas=sigmas)
    )


def test_sd3_partial_denoise_short_circuit_uses_first_clipped_timestep():
    invocation = SD3DenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        width=64,
        height=64,
        steps=4,
        denoising_start=0.25,
        denoising_end=0.25,
        positive_conditioning=MagicMock(conditioning_name="positive"),
        negative_conditioning=MagicMock(conditioning_name="negative"),
        transformer=MagicMock(transformer="transformer"),
        seed=0,
    )
    init_latents = torch.full((1, 16, 8, 8), 2.0)
    noise = torch.full((1, 16, 8, 8), 10.0)
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    mock_context.models.load.return_value = MagicMock(
        model=MagicMock(config=MagicMock(in_channels=16, joint_attention_dim=4096))
    )

    with (
        patch("invokeai.app.invocations.sd3_denoise.TorchDevice.choose_torch_device", return_value=torch.device("cpu")),
        patch("invokeai.app.invocations.sd3_denoise.TorchDevice.choose_torch_dtype", return_value=torch.float32),
        patch.object(invocation, "_prepare_noise_tensor", return_value=noise),
        patch.object(invocation, "_load_text_conditioning", return_value=(torch.zeros(1, 1, 1), torch.zeros(1, 1))),
    ):
        result = invocation._run_diffusion(mock_context)

    timesteps = clip_timestep_schedule_fractional(torch.linspace(1, 0, invocation.steps + 1).tolist(), 0.25, 0.25)
    expected = timesteps[0] * noise + (1.0 - timesteps[0]) * init_latents
    assert torch.equal(result, expected)


def test_cogview4_partial_denoise_short_circuit_uses_first_clipped_sigma():
    invocation = CogView4DenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        width=64,
        height=64,
        steps=4,
        denoising_start=0.25,
        denoising_end=0.25,
        positive_conditioning=MagicMock(conditioning_name="positive"),
        negative_conditioning=MagicMock(conditioning_name="negative"),
        transformer=MagicMock(transformer="transformer"),
        seed=0,
    )
    init_latents = torch.full((1, 16, 8, 8), 2.0)
    noise = torch.full((1, 16, 8, 8), 10.0)
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    transformer_model = MagicMock(config=MagicMock(in_channels=16, patch_size=2))
    mock_context.models.load.return_value = MagicMock(model=transformer_model)

    with (
        patch("invokeai.app.invocations.cogview4_denoise.CogView4Transformer2DModel", object),
        patch(
            "invokeai.app.invocations.cogview4_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch.object(invocation, "_prepare_noise_tensor", return_value=noise),
        patch.object(invocation, "_load_text_conditioning", return_value=torch.zeros(1, 1, 1)),
    ):
        result = invocation._run_diffusion(mock_context)

    timesteps = clip_timestep_schedule_fractional(torch.linspace(1, 0, invocation.steps + 1).tolist(), 0.25, 0.25)
    sigmas = invocation._convert_timesteps_to_sigmas(
        image_seq_len=((invocation.height // 8) * (invocation.width // 8)) // (2**2),
        timesteps=torch.tensor(timesteps),
    )
    expected = sigmas[0] * noise + (1.0 - sigmas[0]) * init_latents
    assert torch.allclose(result, expected, atol=2e-3, rtol=0)

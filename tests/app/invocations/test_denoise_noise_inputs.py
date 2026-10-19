import inspect
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.cogview4.cogview4_denoise import CogView4DenoiseInvocation
from invokeai.app.invocations.flux.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.flux2.flux2_denoise import Flux2DenoiseInvocation
from invokeai.app.invocations.metadata_linked import (
    AnimaDenoiseMetaInvocation,
    DenoiseLatentsMetaInvocation,
    FluxDenoiseLatentsMetaInvocation,
    ZImageDenoiseMetaInvocation,
)
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.sd3.sd3_denoise import SD3DenoiseInvocation
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional, get_schedule
from invokeai.backend.flux.schedulers import (
    ANIMA_SCHEDULER_MAP,
    FLUX_SCHEDULER_MAP,
    ZIMAGE_SCHEDULER_MAP,
    set_heun_timesteps_from_sigmas,
)
from invokeai.backend.flux2.sampling_utils import get_schedule_flux2
from invokeai.backend.model_manager.taxonomy import BaseModelType, FluxVariantType, ModelFormat, ModelType


def test_flux_prepare_noise_uses_external_noise():
    invocation = FluxDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch("invokeai.app.invocations.flux.flux_denoise.get_noise") as mock_get_noise:
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
            "invokeai.app.invocations.flux.flux_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch("invokeai.app.invocations.flux.flux_denoise.FLUXConditioningInfo", object),
        patch(
            "invokeai.app.invocations.flux.flux_denoise.RegionalPromptingExtension.from_text_conditioning",
            return_value=MagicMock(),
        ),
        patch.object(invocation, "_prepare_noise_tensor", side_effect=AssertionError("noise should be ignored")),
        patch.object(invocation, "_load_redux_conditioning", return_value=[]),
        patch("invokeai.app.invocations.flux.flux_denoise.get_schedule", return_value=[0.75]),
    ):
        result = invocation._run_diffusion(mock_context)

    assert torch.equal(result, init_latents)


def test_flux_add_noise_false_denoises_incoming_latents_without_noise():
    """A second stage that continues from latents with add_noise=False runs its remaining steps."""
    invocation = FluxDenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        add_noise=False,
        width=64,
        height=64,
        num_steps=4,
        denoising_start=0.5,
        denoising_end=1.0,
        positive_text_conditioning=MagicMock(conditioning_name="positive"),
        transformer=MagicMock(transformer="transformer"),
        seed=123,
    )
    init_latents = (torch.arange(16 * 8 * 8) % 8).float().reshape(1, 16, 8, 8)
    dummy_conditioning = SimpleNamespace(
        t5_embeds=torch.zeros(1, 4, 16),
        clip_embeds=torch.zeros(1, 768),
        to=lambda **_: dummy_conditioning,
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    mock_context.conditioning.load.return_value = SimpleNamespace(conditionings=[dummy_conditioning])
    mock_context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Flux, type=ModelType.Main, variant=FluxVariantType.Dev, format=ModelFormat.GGUFQuantized
    )
    mock_context.models.load.return_value.model_on_device.return_value = nullcontext((None, MagicMock()))
    denoise_calls: list[dict[str, object]] = []

    def fake_denoise(**kwargs):
        denoise_calls.append(kwargs)
        return kwargs["img"]

    with (
        patch(
            "invokeai.app.invocations.flux.flux_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch("invokeai.app.invocations.flux.flux_denoise.FLUXConditioningInfo", object),
        patch("invokeai.app.invocations.flux.flux_denoise.Flux", object),
        patch(
            "invokeai.app.invocations.flux.flux_denoise.RegionalPromptingExtension.from_text_conditioning",
            return_value=MagicMock(),
        ),
        patch.object(invocation, "_prepare_noise_tensor", side_effect=AssertionError("noise should be ignored")),
        patch.object(invocation, "_load_redux_conditioning", return_value=[]),
        patch.object(invocation, "_prep_controlnet_extensions", return_value=[]),
        patch.object(invocation, "_prep_ip_adapter_extensions", return_value=([], [])),
        patch("invokeai.app.invocations.flux.flux_denoise.peak_dequant_transient_bytes", return_value=0),
        patch(
            "invokeai.app.invocations.flux.flux_denoise.LayerPatcher.apply_smart_model_patches",
            return_value=nullcontext(),
        ),
        patch("invokeai.app.invocations.flux.flux_denoise.denoise", side_effect=fake_denoise),
    ):
        result = invocation._run_diffusion(mock_context)

    assert len(denoise_calls) == 1
    assert len(denoise_calls[0]["timesteps"]) > 1
    assert denoise_calls[0]["inpaint_extension"] is None
    assert torch.equal(result, init_latents)


@pytest.mark.parametrize(
    ["variant", "expects_shift"],
    [(FluxVariantType.Schnell, False), (FluxVariantType.Dev, True)],
)
def test_flux_derives_the_schedule_shift_and_token_count_from_the_model(variant: FluxVariantType, expects_shift: bool):
    """Two wirings that live only at the node, and that cells on `get_schedule` cannot see.

    `shift=not is_schnell` is the only thing deciding whether FLUX.1 shifts its schedule at all.
    And `image_seq_len` is a *packed* token count -- 4096 for a 1024x1024 frame -- which is the unit
    the shift's fit and its clamp are expressed in; passing latent dimensions instead would be four
    times too large and, above one megapixel, now invisible because both values clamp alike.
    """
    invocation = FluxDenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        add_noise=False,
        width=1024,
        height=1024,
        num_steps=4,
        denoising_start=0.25,
        denoising_end=0.25,
        positive_text_conditioning=MagicMock(conditioning_name="positive"),
        transformer=MagicMock(transformer="transformer"),
        seed=123,
    )
    init_latents = torch.full((1, 16, 128, 128), 2.0)
    dummy_conditioning = SimpleNamespace(
        t5_embeds=torch.zeros(1, 4, 16),
        clip_embeds=torch.zeros(1, 768),
        to=lambda **_: dummy_conditioning,
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = init_latents
    mock_context.conditioning.load.return_value = SimpleNamespace(conditionings=[dummy_conditioning])
    mock_context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Flux, type=ModelType.Main, variant=variant
    )
    schedule_calls: list[dict[str, object]] = []

    with (
        patch(
            "invokeai.app.invocations.flux.flux_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch("invokeai.app.invocations.flux.flux_denoise.FLUXConditioningInfo", object),
        patch(
            "invokeai.app.invocations.flux.flux_denoise.RegionalPromptingExtension.from_text_conditioning",
            return_value=MagicMock(),
        ),
        patch.object(invocation, "_load_redux_conditioning", return_value=[]),
        patch(
            "invokeai.app.invocations.flux.flux_denoise.get_schedule",
            side_effect=lambda **kwargs: schedule_calls.append(kwargs) or [0.75],
        ),
    ):
        invocation._run_diffusion(mock_context)

    assert len(schedule_calls) == 1
    assert schedule_calls[0]["shift"] is expects_shift
    assert schedule_calls[0]["image_seq_len"] == 4096


def test_flux2_prepare_noise_uses_external_noise():
    invocation = Flux2DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 32, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch("invokeai.app.invocations.flux2.flux2_denoise.get_noise_flux2") as mock_get_noise:
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
            "invokeai.app.invocations.z_image.z_image_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch(
            "invokeai.app.invocations.z_image.z_image_denoise.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.bfloat16,
        ),
        patch("invokeai.app.invocations.z_image.z_image_denoise.ZImageConditioningInfo", object),
        patch(
            "invokeai.app.invocations.z_image.z_image_denoise.ZImageRegionalPromptingExtension.from_text_conditionings",
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
            "invokeai.app.invocations.anima.anima_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch(
            "invokeai.app.invocations.anima.anima_denoise.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.bfloat16,
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
        positive_text_conditioning=MagicMock(conditioning_name="positive", mask=None),
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
            "invokeai.app.invocations.flux2.flux2_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch("invokeai.app.invocations.flux2.flux2_denoise.FLUXConditioningInfo", object),
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


def test_anima_metadata_ignores_external_noise_seed_when_noise_not_used():
    invocation = AnimaDenoiseMetaInvocation.model_construct(
        width=64,
        height=64,
        steps=30,
        guidance_scale=4.5,
        denoising_start=0.0,
        denoising_end=1.0,
        scheduler="euler",
        latents=MagicMock(latents_name="latents"),
        transformer=MagicMock(transformer="transformer", loras=[]),
        noise=MagicMock(seed=123),
        seed=999,
        add_noise=False,
        vae=None,
        qwen3_encoder=None,
    )
    mock_context = MagicMock()
    output = LatentsOutput.build("latents", torch.zeros(1, 16, 8, 8), seed=None)

    with patch("invokeai.app.invocations.metadata_linked.AnimaDenoiseInvocation.invoke", return_value=output):
        result = invocation.invoke(mock_context)

    assert result.metadata.root["seed"] == 999


def test_anima_metadata_matches_the_recall_contract():
    """The keys this node writes are what the gallery's recall handlers read.

    Anima's CFG must land in `cfg_scale` (not `guidance`, which is FLUX's guidance-embeds scale and is
    recalled into a different param), and the standalone VAE / Qwen3 encoder must be recorded, since
    unlike SD or FLUX they cannot be derived from the transformer field.
    """
    invocation = AnimaDenoiseMetaInvocation.model_construct(
        width=64,
        height=64,
        steps=30,
        guidance_scale=4.5,
        denoising_start=0.0,
        denoising_end=1.0,
        scheduler="dpmpp_2m",
        latents=None,
        transformer=SimpleNamespace(transformer="anima-main-model", loras=[]),
        noise=MagicMock(seed=123),
        seed=999,
        add_noise=True,
        vae=SimpleNamespace(vae="anima-vae"),
        qwen3_encoder=SimpleNamespace(text_encoder="anima-qwen3-encoder"),
    )
    mock_context = MagicMock()
    output = LatentsOutput.build("latents", torch.zeros(1, 16, 8, 8), seed=None)

    with patch("invokeai.app.invocations.metadata_linked.AnimaDenoiseInvocation.invoke", return_value=output):
        result = invocation.invoke(mock_context)

    md = result.metadata.root
    assert md["cfg_scale"] == 4.5
    assert "guidance" not in md
    assert md["model"] == "anima-main-model"
    assert md["vae"] == "anima-vae"
    assert md["qwen3_encoder"] == "anima-qwen3-encoder"
    assert md["scheduler"] == "dpmpp_2m"
    assert md["steps"] == 30
    assert md["seed"] == 123
    assert md["width"] == 64
    assert md["height"] == 64


def test_anima_metadata_omits_unconnected_standalone_components():
    """Both component inputs are optional - an unconnected one must not write a null into metadata,
    which the recall handlers would then try (and fail) to resolve as a model."""
    invocation = AnimaDenoiseMetaInvocation.model_construct(
        width=64,
        height=64,
        steps=30,
        guidance_scale=4.5,
        denoising_start=0.0,
        denoising_end=1.0,
        scheduler="euler",
        latents=None,
        transformer=SimpleNamespace(transformer="anima-main-model", loras=[]),
        noise=MagicMock(seed=123),
        seed=999,
        add_noise=True,
        vae=None,
        qwen3_encoder=None,
    )
    mock_context = MagicMock()
    output = LatentsOutput.build("latents", torch.zeros(1, 16, 8, 8), seed=None)

    with patch("invokeai.app.invocations.metadata_linked.AnimaDenoiseInvocation.invoke", return_value=output):
        result = invocation.invoke(mock_context)

    assert "vae" not in result.metadata.root
    assert "qwen3_encoder" not in result.metadata.root


def test_denoise_metadata_persists_hidiffusion_fields_when_disabled():
    invocation = DenoiseLatentsMetaInvocation.model_construct(
        width=64,
        height=64,
        steps=10,
        cfg_scale=7.5,
        cfg_rescale_multiplier=0,
        denoising_start=0.0,
        denoising_end=1.0,
        scheduler="euler",
        unet=SimpleNamespace(unet="main-model", loras=[]),
        hidiffusion=False,
        hidiffusion_raunet=False,
        hidiffusion_window_attn=False,
        hidiffusion_t1_ratio=0.25,
        hidiffusion_t2_ratio=0.1,
    )
    mock_context = MagicMock()
    output = LatentsOutput.build("latents", torch.zeros(1, 4, 8, 8), seed=None)

    with patch("invokeai.app.invocations.metadata_linked.DenoiseLatentsInvocation.invoke", return_value=output):
        result = invocation.invoke(mock_context)

    assert result.metadata.root["hidiffusion"] is False
    assert result.metadata.root["hidiffusion_raunet"] is False
    assert result.metadata.root["hidiffusion_window_attn"] is False
    assert result.metadata.root["hidiffusion_t1_ratio"] == 0.25
    assert result.metadata.root["hidiffusion_t2_ratio"] == 0.1


def _get_first_scheduler_sigma(
    scheduler,
    *,
    scheduler_name: str,
    sigmas: list[float],
    mu: float | None = None,
    use_flux_heun_schedule: bool = False,
) -> float:
    if scheduler_name == "heun" and use_flux_heun_schedule:
        set_heun_timesteps_from_sigmas(scheduler, sigmas[:-1], sigmas[-1], "cpu")
        return float(scheduler.sigmas[0])

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
        "heun",
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
        _get_first_scheduler_sigma(
            scheduler,
            scheduler_name=scheduler_name,
            sigmas=sigmas,
            use_flux_heun_schedule=True,
        )
    )


def test_flux2_partial_denoise_short_circuit_uses_first_clipped_timestep():
    invocation = Flux2DenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        width=64,
        height=64,
        num_steps=4,
        denoising_start=0.25,
        denoising_end=0.25,
        positive_text_conditioning=MagicMock(conditioning_name="positive", mask=None),
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
            "invokeai.app.invocations.flux2.flux2_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch("invokeai.app.invocations.flux2.flux2_denoise.FLUXConditioningInfo", object),
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

        def step(
            self,
            model_output: torch.Tensor,
            timestep: torch.Tensor,
            sample: torch.Tensor,
            generator: torch.Generator | None = None,
        ):
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
            joint_attention_kwargs: dict | None = None,
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


@pytest.mark.skipif(FLUX_SCHEDULER_MAP.get("lcm") is None, reason="FlowMatchLCMScheduler not available")
def test_flux2_lcm_step_noise_follows_the_generator():
    """LCM re-noises the sample on every step, so FLUX.2 must draw that noise from the seeded
    generator instead of the global RNG."""
    from invokeai.backend.flux2.denoise import denoise

    class DummyModel(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, **kwargs: object):
            del kwargs
            return (0.1 * hidden_states,)

    img = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(7))

    def run(seed: int) -> torch.Tensor:
        # Advance the global RNG between runs, like a long-lived server process does.
        torch.rand(64)
        return denoise(
            model=DummyModel(),
            img=img.clone(),
            img_ids=torch.zeros(1, 4, 4, dtype=torch.long),
            txt=torch.zeros(1, 4, 8),
            txt_ids=torch.zeros(1, 4, 4, dtype=torch.long),
            timesteps=[1.0, 0.75, 0.5, 0.25, 0.0],
            step_callback=lambda _: None,
            guidance=1.0,
            cfg_scale=[1.0] * 4,
            scheduler=FLUX_SCHEDULER_MAP["lcm"](num_train_timesteps=1000),
            generator=torch.Generator().manual_seed(seed),
        )

    assert torch.equal(run(1234), run(1234))
    assert not torch.equal(run(1), run(2))


class _StopAtStep(Exception):
    """Raised by a recording stub so `_run_diffusion` stops once the step generator has been captured."""


# Mirrors the nodes: the step noise is seeded from `seed`, XORed to stay apart from the initial noise.
_LCM_STEP_SEEDS = [(0, 0xFFFFFFFF), (1234, 1234 ^ 0xFFFFFFFF), (0xFFFFFFFF, 0)]


def _assert_step_generator(generator: object, expected_seed: int) -> None:
    assert isinstance(generator, torch.Generator)
    assert generator.device == torch.device("cpu")
    assert generator.initial_seed() == expected_seed


def _run_until_backend_denoise(invocation, module: str, context: MagicMock, *extra_patches) -> dict[str, object]:
    """Drive the real `_run_diffusion` up to the backend `denoise()` call and return its kwargs."""
    captured: dict[str, object] = {}

    def recording_denoise(**kwargs: object):
        captured.update(kwargs)
        raise _StopAtStep

    transformer_info = MagicMock()
    transformer_info.model_on_device.return_value = nullcontext((None, transformer_info.model))
    context.models.load.return_value = transformer_info

    with ExitStack() as stack:
        stack.enter_context(patch(f"{module}.TorchDevice.choose_torch_device", return_value=torch.device("cpu")))
        stack.enter_context(patch(f"{module}.peak_dequant_transient_bytes", return_value=0))
        stack.enter_context(patch(f"{module}.requires_sidecar_patching", return_value=False))
        stack.enter_context(patch(f"{module}.LayerPatcher.apply_smart_model_patches", return_value=nullcontext()))
        stack.enter_context(patch(f"{module}.denoise", side_effect=recording_denoise))
        for extra_patch in extra_patches:
            stack.enter_context(extra_patch)
        with pytest.raises(_StopAtStep):
            invocation._run_diffusion(context)

    return captured


@pytest.mark.parametrize(("seed", "expected_seed"), _LCM_STEP_SEEDS)
def test_flux_invocation_passes_a_seeded_step_generator_to_denoise(seed: int, expected_seed: int):
    """Without `generator=`, FLUX.1 LCM draws its per-step noise from the global RNG again (#9791)."""
    module = "invokeai.app.invocations.flux.flux_denoise"
    invocation = FluxDenoiseInvocation.model_construct(
        latents=None,
        add_noise=True,
        width=64,
        height=64,
        num_steps=2,
        denoising_start=0.0,
        denoising_end=1.0,
        positive_text_conditioning=MagicMock(conditioning_name="positive"),
        transformer=MagicMock(transformer="transformer", loras=[]),
        scheduler="lcm",
        seed=seed,
    )
    dummy_conditioning = SimpleNamespace(
        t5_embeds=torch.zeros(1, 4, 16),
        clip_embeds=torch.zeros(1, 768),
        to=lambda **_: dummy_conditioning,
    )
    context = MagicMock()
    context.conditioning.load.return_value = SimpleNamespace(conditionings=[dummy_conditioning])
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Flux, type=ModelType.Main, variant=FluxVariantType.Dev, format=ModelFormat.Checkpoint
    )

    captured = _run_until_backend_denoise(
        invocation,
        module,
        context,
        patch(f"{module}.FLUXConditioningInfo", object),
        patch(f"{module}.RegionalPromptingExtension.from_text_conditioning", return_value=MagicMock()),
        patch(f"{module}.get_schedule", return_value=[1.0, 0.5, 0.0]),
        patch.object(invocation, "_load_redux_conditioning", return_value=[]),
        patch.object(invocation, "_prepare_noise_tensor", return_value=torch.zeros(1, 16, 8, 8)),
        # `_run_diffusion` asserts the loaded transformer is a `Flux`.
        patch(f"{module}.Flux", object),
    )

    _assert_step_generator(captured["generator"], expected_seed)


@pytest.mark.parametrize(("seed", "expected_seed"), _LCM_STEP_SEEDS)
def test_flux2_invocation_passes_a_seeded_step_generator_to_denoise(seed: int, expected_seed: int):
    """Without `generator=`, FLUX.2 LCM draws its per-step noise from the global RNG again (#9791)."""
    module = "invokeai.app.invocations.flux2.flux2_denoise"
    invocation = Flux2DenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        add_noise=False,
        width=64,
        height=64,
        num_steps=2,
        denoising_start=0.0,
        denoising_end=1.0,
        positive_text_conditioning=MagicMock(conditioning_name="positive", mask=None),
        transformer=MagicMock(transformer="transformer", loras=[]),
        vae=MagicMock(vae="vae"),
        scheduler="lcm",
        seed=seed,
    )
    context = MagicMock()
    context.tensors.load.return_value = torch.zeros(1, 32, 8, 8)
    context.conditioning.load.return_value = SimpleNamespace(
        conditionings=[
            SimpleNamespace(
                t5_embeds=torch.zeros(1, 4, 16), to=lambda **_: SimpleNamespace(t5_embeds=torch.zeros(1, 4, 16))
            )
        ]
    )
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.Checkpoint, variant=None
    )

    captured = _run_until_backend_denoise(
        invocation,
        module,
        context,
        patch(f"{module}.FLUXConditioningInfo", object),
        patch.object(invocation, "_get_bn_stats", return_value=None),
    )

    _assert_step_generator(captured["generator"], expected_seed)


@pytest.mark.skipif(ZIMAGE_SCHEDULER_MAP.get("lcm") is None, reason="FlowMatchLCMScheduler not available")
@pytest.mark.parametrize(("seed", "expected_seed"), _LCM_STEP_SEEDS)
def test_z_image_invocation_steps_lcm_with_a_seeded_generator(seed: int, expected_seed: int):
    """Z-Image steps the scheduler inline, so check the generator that reaches `scheduler.step()`."""
    module = "invokeai.app.invocations.z_image.z_image_denoise"
    lcm_class = ZIMAGE_SCHEDULER_MAP["lcm"]
    step_generators: list[object] = []

    class _RecordingLCM(lcm_class):
        def step(self, *args: object, **kwargs: object):
            step_generators.append(kwargs.get("generator"))
            raise _StopAtStep

    class _ZeroTransformer(torch.nn.Module):
        def forward(self, x: list[torch.Tensor], t: torch.Tensor, cap_feats: list[torch.Tensor]):
            del t, cap_feats
            return ([torch.zeros_like(item) for item in x],)

    transformer = _ZeroTransformer()
    transformer_info = MagicMock()
    transformer_info.model = transformer
    transformer_info.model_on_device.return_value = nullcontext((None, transformer))

    invocation = ZImageDenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        noise=None,
        denoise_mask=None,
        add_noise=False,
        width=64,
        height=64,
        steps=2,
        denoising_start=0.0,
        denoising_end=1.0,
        positive_conditioning=SimpleNamespace(conditioning_name="positive", mask=None),
        negative_conditioning=None,
        guidance_scale=1.0,
        transformer=MagicMock(transformer="transformer", loras=[]),
        control=None,
        shift=None,
        scheduler="lcm",
        seed=seed,
    )
    regional_extension = SimpleNamespace(
        regional_text_conditioning=SimpleNamespace(prompt_embeds=torch.zeros(4, 16)),
        regional_attn_mask=None,
    )
    context = MagicMock()
    context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)
    context.models.load.return_value = transformer_info
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.Checkpoint
    )

    with (
        patch(f"{module}.TorchDevice.choose_torch_device", return_value=torch.device("cpu")),
        patch(f"{module}.TorchDevice.choose_bfloat16_safe_dtype", return_value=torch.bfloat16),
        patch(f"{module}.ZImageRegionalPromptingExtension.from_text_conditionings", return_value=regional_extension),
        patch.object(invocation, "_load_text_conditioning", return_value=[]),
        patch.object(invocation, "_estimate_working_memory", return_value=0),
        patch.dict(ZIMAGE_SCHEDULER_MAP, {"lcm": _RecordingLCM}),
        pytest.raises(_StopAtStep),
    ):
        invocation._run_diffusion(context)

    assert len(step_generators) == 1
    _assert_step_generator(step_generators[0], expected_seed)


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
    scheduler_class, scheduler_kwargs = ANIMA_SCHEDULER_MAP[scheduler_name]
    scheduler = scheduler_class(num_train_timesteps=1000, **scheduler_kwargs)

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
        patch(
            "invokeai.app.invocations.sd3.sd3_denoise.TorchDevice.choose_torch_device", return_value=torch.device("cpu")
        ),
        patch("invokeai.app.invocations.sd3.sd3_denoise.TorchDevice.choose_torch_dtype", return_value=torch.float32),
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
        patch("invokeai.app.invocations.cogview4.cogview4_denoise.CogView4Transformer2DModel", object),
        patch(
            "invokeai.app.invocations.cogview4.cogview4_denoise.TorchDevice.choose_torch_device",
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

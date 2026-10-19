"""Regression tests for how ``Flux2DenoiseInvocation._run_diffusion`` wires the schedule and the
start latents together.

``_prepare_normalized_start_latents`` and ``time_shift_flux2`` are covered on their own elsewhere;
what those tests cannot see is the code that *calls* them. Each of these one-line breakages used to
pass the whole suite:

* dropping the ``time_shift_flux2`` call, so img2img runs the unshifted linear schedule again;
* shifting unconditionally, which shifts txt2img twice (the scheduler already applies ``mu``);
* re-normalizing ``noise_packed`` before the inpaint extension is built, bringing back the half of
  the noise-scaling bug that lives in the unmasked region;
* handing ``_prepare_normalized_start_latents`` a ``t_0`` from the unclipped schedule;
* skipping the respacing of the clipped schedule, so img2img at low strength runs one or two steps
  instead of the requested number and the final Euler jump flattens the source image's detail.

These tests drive the real ``_run_diffusion`` up to the transformer load and assert on what the
wiring actually produced.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

import invokeai.app.invocations.flux2_denoise as flux2_denoise
from invokeai.app.invocations.flux2_denoise import Flux2DenoiseInvocation
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.flux2.sampling_utils import (
    compute_empirical_mu,
    get_schedule_flux2,
    pack_flux2,
    redensify_schedule_flux2,
    time_shift_flux2,
    unpack_flux2,
)
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension

# Measured on the BFL FLUX.2 VAE; a value far enough from 1.0 that a stray normalization is obvious.
BN_STD_VALUE = 1.7676
BN_MEAN_VALUE = 0.05
PACKED_CHANNELS = 128

# 128px -> a 16x16 latent -> an 8x8 packed grid -> 64 image tokens.
SIZE = 128
LATENT = SIZE // 8
IMAGE_SEQ_LEN = (LATENT // 2) ** 2
NUM_STEPS = 4


class _StopBeforeLoad(Exception):
    """Raised in place of entering the transformer's device context, to end _run_diffusion early."""


def _bn_stats() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((PACKED_CHANNELS,), BN_MEAN_VALUE),
        torch.full((PACKED_CHANNELS,), BN_STD_VALUE),
    )


def _latents(seed: int) -> torch.Tensor:
    # _run_diffusion pins inference_dtype to bfloat16, so feed it bf16 and compute the expectations in
    # bf16 too -- an fp32 expectation differs from the node's result by ~1e-3 for precision reasons alone.
    return torch.randn(1, 32, LATENT, LATENT, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _drive(
    *,
    denoising_start: float = 0.0,
    denoising_end: float = 1.0,
    with_init: bool = True,
    with_mask: bool = False,
    scheduler: str = "euler",
    bn: bool = True,
):
    """Run `_run_diffusion` far enough to build the schedule, start latents and inpaint extension."""
    from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux2VariantType, ModelFormat, ModelType
    from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
        ConditioningFieldData,
        FLUXConditioningInfo,
    )

    noise = _latents(7)
    init_latents = _latents(8)
    mask = torch.full_like(noise, 0.5)
    captured: dict = {"shift_calls": [], "start_kwargs": [], "inpaint": [], "redensify_calls": []}

    transformer_info = MagicMock()
    transformer_info.model_on_device = MagicMock(side_effect=_StopBeforeLoad)

    context = MagicMock()
    context.models.load.return_value = transformer_info
    context.models.get_config.return_value = MagicMock(
        base=BaseModelType.Flux2,
        type=ModelType.Main,
        format=ModelFormat.Checkpoint,
        variant=Flux2VariantType("klein_9b"),
    )
    context.conditioning.load.return_value = ConditioningFieldData(
        conditionings=[FLUXConditioningInfo(clip_embeds=torch.zeros(1, 768), t5_embeds=torch.zeros(1, 64, 12288))]
    )
    context.tensors.load.return_value = init_latents

    invocation = Flux2DenoiseInvocation.model_construct(
        latents=MagicMock(latents_name="init") if with_init else None,
        noise=None,
        denoise_mask=MagicMock(mask_name="mask") if with_mask else None,
        denoising_start=denoising_start,
        denoising_end=denoising_end,
        add_noise=True,
        transformer=MagicMock(transformer=MagicMock(), loras=[]),
        positive_text_conditioning=MagicMock(conditioning_name="pos", mask=None),
        negative_text_conditioning=None,
        guidance=4.0,
        cfg_scale=1.0,
        width=SIZE,
        height=SIZE,
        num_steps=NUM_STEPS,
        scheduler=scheduler,
        seed=0,
        vae=MagicMock(vae=MagicMock()),
        kontext_conditioning=None,
    )

    real_shift = flux2_denoise.time_shift_flux2

    def recording_shift(timesteps, mu):
        captured["shift_calls"].append((list(timesteps), mu))
        return real_shift(timesteps, mu)

    real_redensify = flux2_denoise.redensify_schedule_flux2

    def recording_redensify(timesteps, num_steps, mu):
        result = real_redensify(timesteps, num_steps, mu)
        captured["redensify_calls"].append({"clipped": list(timesteps), "num_steps": num_steps, "result": result})
        return result

    real_prepare = Flux2DenoiseInvocation._prepare_normalized_start_latents

    def recording_prepare(self, **kwargs):
        captured["start_kwargs"].append(kwargs)
        return real_prepare(self, **kwargs)

    class _RecordingInpaintExtension(RectifiedFlowInpaintExtension):
        def __init__(self, *, init_latents, inpaint_mask, noise):
            captured["inpaint"].append({"init_latents": init_latents, "inpaint_mask": inpaint_mask, "noise": noise})
            super().__init__(init_latents, inpaint_mask, noise)

    with (
        patch.object(Flux2DenoiseInvocation, "_get_bn_stats", return_value=_bn_stats() if bn else None),
        patch.object(Flux2DenoiseInvocation, "_prepare_noise_tensor", return_value=noise),
        patch.object(Flux2DenoiseInvocation, "_prep_inpaint_mask", return_value=mask if with_mask else None),
        patch.object(Flux2DenoiseInvocation, "_prepare_normalized_start_latents", recording_prepare),
        patch("invokeai.backend.util.devices.TorchDevice.choose_torch_device", return_value=torch.device("cpu")),
        patch.object(flux2_denoise, "time_shift_flux2", recording_shift),
        patch.object(flux2_denoise, "redensify_schedule_flux2", recording_redensify),
        patch.object(flux2_denoise, "RectifiedFlowInpaintExtension", _RecordingInpaintExtension),
    ):
        try:
            result = invocation._run_diffusion(context)
        except _StopBeforeLoad:
            result = None

    captured["result"] = result
    captured["noise"] = noise
    captured["init_latents"] = init_latents
    captured["reached_transformer"] = transformer_info.model_on_device.called
    return captured


def _expected_schedule(denoising_start: float, denoising_end: float = 1.0, shifted: bool = True) -> list[float]:
    linear = get_schedule_flux2(num_steps=NUM_STEPS, image_seq_len=IMAGE_SEQ_LEN)
    mu = compute_empirical_mu(image_seq_len=IMAGE_SEQ_LEN, num_steps=NUM_STEPS)
    timesteps = time_shift_flux2(linear, mu) if shifted else linear
    clipped = clip_timestep_schedule_fractional(timesteps, denoising_start, denoising_end)
    return redensify_schedule_flux2(clipped, NUM_STEPS, mu) if shifted else clipped


def test_txt2img_does_not_shift_the_schedule() -> None:
    """The txt2img scheduler applies mu itself; shifting here too would shift twice."""
    captured = _drive(denoising_start=0.0, with_init=False)

    assert captured["shift_calls"] == []
    assert captured["reached_transformer"]


@pytest.mark.parametrize(("denoising_start", "with_mask"), [(0.5, False), (0.0, True)])
def test_manual_euler_paths_shift_the_schedule_once(denoising_start: float, with_mask: bool) -> None:
    """img2img and inpainting step the schedule themselves, so they must apply the shift."""
    captured = _drive(denoising_start=denoising_start, with_mask=with_mask)

    assert len(captured["shift_calls"]) == 1
    shifted_timesteps, mu = captured["shift_calls"][0]
    assert mu == pytest.approx(compute_empirical_mu(image_seq_len=IMAGE_SEQ_LEN, num_steps=NUM_STEPS))
    # Shifted before clipping, so the call still sees the full linear schedule.
    assert shifted_timesteps == pytest.approx(get_schedule_flux2(num_steps=NUM_STEPS, image_seq_len=IMAGE_SEQ_LEN))


def test_a_scheduler_that_falls_back_to_manual_euler_still_shifts() -> None:
    """Manual Euler also runs when the requested scheduler is unavailable (e.g. lcm without LCM support).

    Keying the shift on denoise_mask/denoising_start alone left that path on the unshifted schedule.
    """
    captured = _drive(denoising_start=0.0, with_init=False, scheduler="not_a_real_scheduler")

    assert len(captured["shift_calls"]) == 1


def test_start_latents_use_the_first_sigma_of_the_clipped_schedule() -> None:
    """A t_0 taken from the unclipped schedule would noise the init latents to the wrong level."""
    captured = _drive(denoising_start=0.5)

    assert len(captured["start_kwargs"]) == 1
    expected = _expected_schedule(denoising_start=0.5)
    assert captured["start_kwargs"][0]["t_0"] == pytest.approx(expected[0])
    # Sanity: the clip actually moved t_0, so the assertion above can fail.
    assert expected[0] != pytest.approx(1.0)


def test_the_inpaint_extension_gets_unit_scale_noise_and_normalized_init() -> None:
    """The extension mixes noise into the unmasked region, so its noise must match the timestep.

    Normalizing it (again) here would divide it by bn_std and reintroduce the bug for the region
    outside the mask.
    """
    captured = _drive(denoising_start=0.0, with_mask=True)

    assert len(captured["inpaint"]) == 1
    recorded = captured["inpaint"][0]
    expected_noise = pack_flux2(captured["noise"])
    assert torch.equal(recorded["noise"], expected_noise)

    bn_mean, bn_std = (t.to(torch.bfloat16) for t in _bn_stats())
    expected_init = (pack_flux2(captured["init_latents"]) - bn_mean) / bn_std
    assert torch.allclose(recorded["init_latents"], expected_init, atol=1e-2)


def test_a_degenerate_schedule_returns_denormalized_start_latents() -> None:
    """denoising_start == denoising_end leaves nothing to step.

    The early return must still hand back latents in the same raw space the full path returns, or a
    downstream FLUX.2 denoise that normalizes them divides the noise term by bn_std a second time.
    """
    captured = _drive(denoising_start=0.5, denoising_end=0.5)

    timesteps = _expected_schedule(denoising_start=0.5, denoising_end=0.5)
    assert len(timesteps) == 1, "this test only means something while the schedule is degenerate"
    assert not captured["reached_transformer"]

    bn_mean, bn_std = (t.to(torch.bfloat16) for t in _bn_stats())
    t_0 = timesteps[0]
    normalized_init = (pack_flux2(captured["init_latents"]) - bn_mean) / bn_std
    expected_packed = t_0 * pack_flux2(captured["noise"]) + (1.0 - t_0) * normalized_init
    expected = unpack_flux2((expected_packed * bn_std + bn_mean).float(), SIZE, SIZE)

    assert captured["result"] is not None
    assert torch.allclose(captured["result"], expected, atol=1e-2)


def test_the_degenerate_return_is_not_the_raw_space_blend() -> None:
    """Guard against a regression to returning the un-denormalized mixture."""
    captured = _drive(denoising_start=0.5, denoising_end=0.5)

    t_0 = _expected_schedule(denoising_start=0.5, denoising_end=0.5)[0]
    raw_blend = (t_0 * captured["noise"] + (1.0 - t_0) * captured["init_latents"]).float()

    # The two differ by the bn_std scaling of the noise term, far beyond bf16 precision.
    assert not torch.allclose(captured["result"], raw_blend, atol=1e-1)


def test_img2img_still_takes_the_requested_number_of_steps() -> None:
    """Clipping the shifted schedule dropped most of the steps; the node has to space them back in.

    At 4 steps and denoising_start 0.5 the clipped window holds a single step, from sigma 0.5 straight
    to 0. One Euler jump that large destroys the source image's fine detail, which is what users saw
    as a loss of skin texture.
    """
    captured = _drive(denoising_start=0.5)

    assert len(captured["redensify_calls"]) == 1
    call = captured["redensify_calls"][0]
    assert call["num_steps"] == NUM_STEPS
    assert len(call["clipped"]) - 1 < NUM_STEPS, "precondition: clipping dropped steps"
    assert len(call["result"]) - 1 == NUM_STEPS
    # The respacing must not move the window: t_0 feeds the img2img preblend.
    assert call["result"][0] == call["clipped"][0]
    assert call["result"][-1] == call["clipped"][-1]


def test_txt2img_is_not_respaced() -> None:
    """txt2img hands the full schedule to the scheduler; there is nothing clipped to space out."""
    captured = _drive(denoising_start=0.0, with_init=False)

    assert captured["redensify_calls"] == []


def test_respacing_leaves_full_strength_img2img_alone() -> None:
    """denoising_start == 0 with init latents keeps every sigma of the shifted schedule.

    This is the endpoint that must stay identical to txt2img, so respacing may not touch it.
    """
    captured = _drive(denoising_start=0.0, with_mask=True)

    assert len(captured["redensify_calls"]) == 1
    call = captured["redensify_calls"][0]
    assert call["result"] == call["clipped"]

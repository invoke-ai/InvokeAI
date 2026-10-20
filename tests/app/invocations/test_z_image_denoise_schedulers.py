"""Z-Image's Heun scheduler runs the node's shifted, clipped schedule.

Heun used to rebuild its own schedule from the step count with a shift of 1.0: it ran linear sigmas
instead of the shifted ones, started img2img at sigma 1.0 whatever the img2img blend used, ignored
`denoising_end`, and its progress stopped one step short of the end.

These tests drive the real `_run_diffusion` through the denoising loop with a stand-in transformer.
The node's own manual Euler path is the reference for the schedule: it steps the clipped sigmas
directly, so Heun has to evaluate the same sigmas in Diffusers' Heun layout. An interval of zero length,
which an explicit shift of 0 leaves at sigma 0, is the exception: Heun would divide by zero there.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.app.util.step_callback import calc_percentage
from invokeai.backend.flux.schedulers import ZIMAGE_SCHEDULER_MAP
from invokeai.backend.model_manager.taxonomy import ModelFormat

SIZE = 128  # a 16x16 latent
STEPS = 8
VELOCITY = 0.25


class _ConstantVelocityTransformer(torch.nn.Module):
    """Predicts a constant velocity and records each evaluation's sigma and sample level.

    With a constant velocity every flow-matching solver moves the sample by exactly
    (sigma_end - sigma_start) * velocity, so the result shows where the schedule ended.
    """

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.sigmas: list[float] = []
        self.sample_levels: list[float] = []

    def forward(self, x: list[torch.Tensor], t: torch.Tensor, cap_feats: list[torch.Tensor]):
        del cap_feats
        # The model's t runs from 0 (noise) to 1 (clean); the node negates the prediction.
        self.sigmas.append(1.0 - float(t[0]))
        self.sample_levels.append(float(torch.stack(x).mean()))
        return ([torch.full_like(sample, -VELOCITY) for sample in x],)


def _drive(scheduler: str, denoising_start: float = 0.0, denoising_end: float = 1.0, shift: float | None = None):
    """Run the node with zero init latents and unit noise, so a blended sample's value is its sigma."""
    transformer = _ConstantVelocityTransformer()
    transformer_info = MagicMock(model=transformer)
    transformer_info.model_on_device.return_value = nullcontext((None, transformer))

    context = MagicMock()
    context.models.load.return_value = transformer_info
    context.models.get_config.return_value = SimpleNamespace(format=ModelFormat.Diffusers)
    context.tensors.load.return_value = torch.zeros(1, 16, SIZE // 8, SIZE // 8)

    node = ZImageDenoiseInvocation.model_construct(
        id="test",
        latents=MagicMock(latents_name="latents") if denoising_start > 0 else None,
        add_noise=True,
        width=SIZE,
        height=SIZE,
        steps=STEPS,
        denoising_start=denoising_start,
        denoising_end=denoising_end,
        shift=shift,
        guidance_scale=1.0,
        positive_conditioning=SimpleNamespace(conditioning_name="positive", mask=None),
        negative_conditioning=None,
        transformer=MagicMock(transformer="transformer", loras=[]),
        control=None,
        denoise_mask=None,
        seed=0,
        scheduler=scheduler,
    )
    regional_extension = SimpleNamespace(
        regional_text_conditioning=SimpleNamespace(prompt_embeds=torch.zeros(4, 16)),
        regional_attn_mask=None,
    )
    callbacks = []

    with (
        patch(
            "invokeai.app.invocations.z_image.z_image_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch(
            "invokeai.app.invocations.z_image.z_image_denoise.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.float32,
        ),
        patch(
            "invokeai.app.invocations.z_image.z_image_denoise.ZImageRegionalPromptingExtension.from_text_conditionings",
            return_value=regional_extension,
        ),
        patch.object(node, "_load_text_conditioning", return_value=[]),
        patch.object(node, "_prepare_noise_tensor", return_value=torch.ones(1, 16, SIZE // 8, SIZE // 8)),
        patch.object(node, "_estimate_working_memory", return_value=0),
        patch.object(node, "_build_step_callback", return_value=callbacks.append),
    ):
        result = node._run_diffusion(context)

    return transformer, result, callbacks


def _heun_layout(sigmas: list[float]) -> list[float]:
    """Diffusers' Heun evaluates the first sigma once and every later one twice."""
    return [sigmas[0]] + [sigma for sigma in sigmas[1:] for _ in range(2)]


@pytest.mark.parametrize(
    ("denoising_start", "denoising_end"),
    [
        pytest.param(0.0, 1.0, id="txt2img"),
        pytest.param(0.5, 1.0, id="img2img"),
        pytest.param(0.0, 0.5, id="partial"),
    ],
)
def test_heun_evaluates_the_sigmas_euler_steps(denoising_start: float, denoising_end: float):
    euler, _, _ = _drive("euler", denoising_start, denoising_end)
    heun, _, _ = _drive("heun", denoising_start, denoising_end)

    assert heun.sigmas == pytest.approx(_heun_layout(euler.sigmas), abs=1e-6)


def test_heun_img2img_starts_at_the_sigma_the_blend_used():
    heun, _, _ = _drive("heun", denoising_start=0.5)

    # Zero init latents and unit noise: the blended sample's value is the sigma it was blended at.
    assert heun.sigmas[0] == pytest.approx(heun.sample_levels[0], abs=1e-6)
    assert heun.sigmas[0] < 1.0


@pytest.mark.parametrize("scheduler", ["euler", "heun"])
def test_a_pass_ending_at_denoising_end_stops_where_the_continuing_pass_starts(scheduler: str):
    _, result, _ = _drive(scheduler, denoising_end=0.5)
    continuing, _, _ = _drive(scheduler, denoising_start=0.5)

    # From unit noise at sigma 1.0, a constant velocity leaves the sample at 1 + (sigma_end - 1) * v.
    end_sigma = 1.0 + (float(result.mean()) - 1.0) / VELOCITY
    assert end_sigma == pytest.approx(continuing.sample_levels[0], abs=1e-5)
    assert end_sigma > 0.0


@pytest.mark.skipif("lcm" not in ZIMAGE_SCHEDULER_MAP, reason="FlowMatchLCMScheduler unavailable")
def test_lcm_keeps_its_own_schedule():
    """LCM builds its schedule from the step count; only Heun is handed the node's sigmas."""
    lcm, _, callbacks = _drive("lcm")

    assert len(lcm.sigmas) == STEPS
    assert lcm.sigmas[0] == pytest.approx(1.0)
    assert [callback.step for callback in callbacks] == list(range(1, STEPS + 1))


def test_heun_progress_reaches_every_step():
    _, _, callbacks = _drive("heun")

    assert [callback.step for callback in callbacks] == list(range(1, STEPS + 1))
    assert [calc_percentage(callback) for callback in callbacks] == pytest.approx(
        [step / STEPS for step in range(1, STEPS + 1)]
    )


def test_heun_at_a_shift_of_zero_takes_the_single_step_to_zero_that_euler_takes():
    # A shift of 0 schedules [1, 0, ..., 0]: one interval, then intervals of zero length at sigma 0, where Heun
    # would divide 0 by 0.
    heun, result, callbacks = _drive("heun", shift=0.0)

    # One step from unit noise at sigma 1 to sigma 0 at a constant velocity.
    assert torch.allclose(result, torch.full_like(result, 1.0 - VELOCITY))
    assert heun.sigmas == [1.0]
    assert [callback.step for callback in callbacks] == [1]
    assert calc_percentage(callbacks[-1]) == pytest.approx(1.0)


def test_heun_img2img_at_a_shift_of_zero_returns_the_blend_when_only_sigma_zero_is_left():
    # Past the first step a shift of 0 leaves only sigma 0: the blend uses it, and no interval remains to denoise.
    heun, result, callbacks = _drive("heun", denoising_start=0.5, shift=0.0)

    # Zero init latents blended at sigma 0 stay zero.
    assert torch.equal(result, torch.zeros_like(result))
    assert heun.sigmas == []
    assert callbacks == []

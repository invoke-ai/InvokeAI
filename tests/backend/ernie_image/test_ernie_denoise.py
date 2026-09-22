import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler

from invokeai.backend.ernie_image.denoise import denoise
from invokeai.backend.ernie_image.sampling_utils import get_schedule
from invokeai.backend.flux.schedulers import ERNIE_IMAGE_SCHEDULER_MAP
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState

# LCM is the stochastic option in the dropdown, but it is not in every diffusers release.
_LCM = ERNIE_IMAGE_SCHEDULER_MAP.get("lcm")
_HAS_LCM = _LCM is not None


class _StubTransformer(torch.nn.Module):
    """Stands in for `ErnieImageTransformer2DModel`; returns a velocity of the right shape."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


def _run(
    scheduler,
    steps: int,
    denoising_start: float = 0.0,
    denoising_end: float = 1.0,
    img: torch.Tensor | None = None,
    init_latents: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
):
    states: list[PipelineIntermediateState] = []
    # 128 channels is the patched channel count; the spatial dims are tiny to keep this fast.
    img = torch.zeros(1, 128, 2, 2) if img is None else img
    text_bth = torch.zeros(1, 4, 8)
    text_lens = torch.tensor([4])
    sigmas = get_schedule(steps, denoising_start=denoising_start, denoising_end=denoising_end)

    out = denoise(
        model=_StubTransformer(),
        img=img,
        text_bth=text_bth,
        text_lens=text_lens,
        timesteps=sigmas.tolist(),
        step_callback=states.append,
        cfg_scale=[1.0] * (len(sigmas) - 1),
        scheduler=scheduler,
        init_latents=init_latents,
        generator=generator,
    )
    return states, sigmas, out


def test_euler_progress_matches_requested_steps() -> None:
    states, _, _ = _run(FlowMatchEulerDiscreteScheduler(), steps=8)

    assert len(states) == 8
    # Progress must never overrun: the last emitted step equals the reported total.
    assert states[-1].step == states[-1].total_steps == 8


def test_heun_progress_does_not_overrun_total_steps() -> None:
    # Heun is 2nd order: `set_timesteps(8)` yields 15 timesteps, so the loop runs 15 times.
    # `total_steps` must reflect the real iteration count, otherwise progress runs past 100%.
    states, _, _ = _run(FlowMatchHeunDiscreteScheduler(), steps=8)

    assert len(states) > 8
    assert all(s.step <= s.total_steps for s in states)
    assert states[-1].step == states[-1].total_steps == len(states)


def test_heun_rejects_a_partial_denoise_range() -> None:
    # Heun's `set_timesteps` takes only a step count, so it cannot honor a custom sigma window.
    # Silently running a full denoise instead would be worse than refusing.
    with pytest.raises(ValueError, match="denoising_start/denoising_end"):
        _run(FlowMatchHeunDiscreteScheduler(), steps=8, denoising_start=0.5)


@pytest.mark.parametrize("denoising_end", [1.0, 0.75, 0.5])
def test_denoising_end_stops_at_the_requested_sigma(denoising_end: float) -> None:
    """The scheduler must terminate at the requested end sigma, not at 0.

    Every FlowMatch scheduler appends its own terminal 0 sigma in `set_timesteps`. If the driver
    hands it the window minus its last entry, that appended 0 silently replaces the requested end
    sigma -- so `denoising_end < 1.0` would run a *full* denoise in fewer, coarser steps instead of
    stopping early, which breaks every multi-stage handoff built on this field.
    """
    scheduler = FlowMatchEulerDiscreteScheduler()
    states, sigmas, _ = _run(scheduler, steps=8, denoising_end=denoising_end)

    scheduler_sigmas = scheduler.sigmas
    assert scheduler_sigmas is not None
    assert float(scheduler_sigmas[-1]) == pytest.approx(float(sigmas[-1]), abs=1e-6)
    # One step per adjacent sigma pair, and no step is dropped or duplicated.
    assert len(states) == len(sigmas) - 1


def test_denoising_end_keeps_the_schedulers_own_shift() -> None:
    """The terminal sigma must be shifted by the scheduler, not spliced in raw.

    Truncating the scheduler's appended zero (rather than overwriting the last sigma by hand) is
    what keeps this true: with `shift=3`, the requested end sigma 0.25 has to land on
    `3 * 0.25 / (1 + 2 * 0.25) = 0.5`, matching the shift applied to every other sigma.
    """
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    _run(scheduler, steps=8, denoising_end=0.75)

    scheduler_sigmas = scheduler.sigmas
    assert scheduler_sigmas is not None
    assert float(scheduler_sigmas[-1]) == pytest.approx(0.5, abs=1e-6)


def test_degenerate_denoising_window_is_rejected() -> None:
    """A window that rounds down to a single sigma yields zero steps.

    The loop would then return its input untouched and the graph would decode raw noise with no
    error at all, so `get_schedule` has to refuse rather than hand back a one-entry schedule.
    """
    with pytest.raises(ValueError, match="rounds to zero steps"):
        get_schedule(8, denoising_end=0.1)


@pytest.mark.skipif(not _HAS_LCM, reason="FlowMatchLCMScheduler not available in this diffusers")
def test_stochastic_scheduler_honors_the_seed() -> None:
    """LCM re-noises the sample on every step, so it needs an explicit generator.

    Without one, `scheduler.step` draws from the global RNG. The seeded initial latent then only
    fixes the starting point, and the same seed produces a different image on every run in a
    long-lived server process -- silently breaking seed recall from gallery metadata.
    """
    assert _LCM is not None
    img = torch.randn(1, 128, 2, 2, generator=torch.Generator().manual_seed(7))

    def once() -> torch.Tensor:
        # Advance the global RNG between runs, exactly as a long-lived process does.
        torch.rand(64)
        _, _, out = _run(_LCM(), steps=4, img=img.clone(), generator=torch.Generator().manual_seed(1234))
        return out

    assert torch.equal(once(), once())


@pytest.mark.skipif(not _HAS_LCM, reason="FlowMatchLCMScheduler not available in this diffusers")
def test_stochastic_scheduler_respects_different_seeds() -> None:
    """Guard against the previous test passing because the generator is ignored entirely."""
    assert _LCM is not None
    img = torch.randn(1, 128, 2, 2, generator=torch.Generator().manual_seed(7))

    _, _, a = _run(_LCM(), steps=4, img=img.clone(), generator=torch.Generator().manual_seed(1))
    _, _, b = _run(_LCM(), steps=4, img=img.clone(), generator=torch.Generator().manual_seed(2))

    assert not torch.equal(a, b)


def test_deterministic_scheduler_is_unaffected_by_the_generator() -> None:
    """Euler is deterministic here (`s_churn` is 0), so the generator must not perturb it."""
    img = torch.randn(1, 128, 2, 2, generator=torch.Generator().manual_seed(7))

    _, _, a = _run(FlowMatchEulerDiscreteScheduler(), steps=4, img=img.clone(), generator=None)
    _, _, b = _run(
        FlowMatchEulerDiscreteScheduler(), steps=4, img=img.clone(), generator=torch.Generator().manual_seed(99)
    )

    assert torch.equal(a, b)


@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_init_latents_are_blended_at_the_post_shift_sigma(shift: float) -> None:
    """The blend must use the sigma the loop actually opens at, not the raw schedule value.

    `get_schedule` emits raw linspace values; the scheduler applies `shift` inside `set_timesteps`.
    Blending at the raw value builds the sample at one sigma and then tells the first model call it
    is at another -- exact only when `shift == 1` or `denoising_start == 0`.
    """
    noise = torch.ones(1, 128, 2, 2)
    init = torch.zeros(1, 128, 2, 2)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)

    captured: list[torch.Tensor] = []

    class _CapturingTransformer(_StubTransformer):
        def forward(self, hidden_states, timestep, text_bth, text_lens, return_dict=False):
            captured.append(hidden_states.clone())
            return torch.zeros_like(hidden_states)

    states: list[PipelineIntermediateState] = []
    sigmas = get_schedule(8, denoising_start=0.5)
    denoise(
        model=_CapturingTransformer(),
        img=noise,
        text_bth=torch.zeros(1, 4, 8),
        text_lens=torch.tensor([4]),
        timesteps=sigmas.tolist(),
        step_callback=states.append,
        cfg_scale=[1.0] * (len(sigmas) - 1),
        scheduler=scheduler,
        init_latents=init,
    )

    scheduler_sigmas = scheduler.sigmas
    assert scheduler_sigmas is not None
    expected_sigma = float(scheduler_sigmas[0])
    # noise is all ones and init is all zeros, so the blend collapses to the sigma itself.
    assert captured[0].flatten()[0].item() == pytest.approx(expected_sigma, abs=1e-6)
    if shift != 1.0:
        # Pin the actual bug: the raw schedule value would have been 0.5.
        assert expected_sigma != pytest.approx(float(sigmas[0]), abs=1e-6)

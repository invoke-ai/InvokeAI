"""The LTX-2 noise schedules and the step that walks them."""

import math

import pytest
import torch
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

from invokeai.backend.ltx2.constants import (
    LTX2_DISTILLED_SIGMAS,
    LTX2_MAX_SEQ_LEN,
    LTX2_STAGE_2_NOISE_SCALE,
)
from invokeai.backend.ltx2.packing import video_sequence_length
from invokeai.backend.ltx2.sampling import (
    LTX2_SHIFT_TERMINAL,
    build_refine_sigmas,
    build_sigmas,
    calculate_shift,
    flow_step,
)


def test_the_distilled_schedule_is_the_released_one_with_a_terminal_zero() -> None:
    sigmas = build_sigmas(distilled=True, num_steps=99, video_seq_len=1)
    assert sigmas.tolist() == pytest.approx([*LTX2_DISTILLED_SIGMAS, 0.0], abs=1e-6)


@pytest.mark.parametrize("num_steps", [2, 8, 30, 50])
@pytest.mark.parametrize("video_seq_len", [320, 6144, 13728])
def test_a_dev_schedule_starts_at_one_descends_and_ends_on_the_terminal(num_steps: int, video_seq_len: int) -> None:
    """The terminal stretch is load-bearing: `mu` is extrapolated well past its 4096-token anchor,
    and without it a 14k-token clip never drops below sigma ~0.9 and comes back as noise."""
    sigmas = build_sigmas(distilled=False, num_steps=num_steps, video_seq_len=video_seq_len)

    assert sigmas.numel() == num_steps + 1
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == 0.0
    assert (sigmas[:-1] - sigmas[1:] > 0).all()
    assert sigmas[-2] == pytest.approx(LTX2_SHIFT_TERMINAL)


def test_the_refine_pass_reproduces_the_released_second_stage_exactly() -> None:
    """The reason the refine pass truncates a schedule instead of carrying its own.

    Upstream publishes the distilled second stage as a literal list. Entering the distilled
    schedule at ``LTX2_STAGE_2_NOISE_SCALE`` has to *be* that list -- if this ever stops holding,
    the truncation rule has stopped describing the release and the dev half of it, which has no
    published list to check against, loses its justification too.
    """
    refine = build_sigmas(distilled=True, num_steps=8, video_seq_len=4096, start_sigma=LTX2_STAGE_2_NOISE_SCALE)

    # Compared against upstream's own list rather than our re-export of it: a constant we copied
    # would make this a check that two of our own lines agree.
    assert refine.tolist() == pytest.approx([*STAGE_2_DISTILLED_SIGMA_VALUES, 0.0])


@pytest.mark.parametrize("start_sigma", [0.909375, 0.7, 0.42, 0.11])
def test_a_truncated_dev_schedule_is_the_tail_of_the_whole_one(start_sigma: float) -> None:
    """The shape of the run is a property of the whole schedule, so the stretch onto the terminal
    level happens before truncation: a refine pass samples levels the base pass would have, not a
    schedule rescaled onto a shorter interval."""
    seq_len = video_sequence_length(121, 1024, 1792)
    whole = build_sigmas(distilled=False, num_steps=30, video_seq_len=seq_len)
    refine = build_sigmas(distilled=False, num_steps=30, video_seq_len=seq_len, start_sigma=start_sigma)

    # Compared against a hand-written slice rather than the implementation's own mask, so flipping
    # `<=` to `<` in `_truncate` cannot satisfy both sides at once.
    sampled = [float(x) for x in whole[:-1]]
    expected = [x for x in sampled if x <= start_sigma]
    assert [float(x) for x in refine[:-1]] == pytest.approx(expected)
    assert float(refine[-1]) == 0.0
    assert float(refine[0]) <= start_sigma
    # Every level the refine samples is one the whole schedule sampled, at its own spacing.
    assert {float(x) for x in refine[:-1]} <= set(sampled)


def test_a_refine_level_below_the_whole_schedule_is_refused_by_name() -> None:
    # Silently returning an empty schedule would denoise nothing and hand back the noised input.
    with pytest.raises(ValueError, match="at or below"):
        build_sigmas(distilled=False, num_steps=30, video_seq_len=4096, start_sigma=0.01)


@pytest.mark.parametrize("start_sigma", [0.0, -0.5, 1.5])
def test_a_refine_level_outside_the_unit_interval_is_refused(start_sigma: float) -> None:
    with pytest.raises(ValueError, match="re-enter the schedule"):
        build_sigmas(distilled=False, num_steps=30, video_seq_len=4096, start_sigma=start_sigma)


def test_a_single_step_schedule_is_just_the_two_endpoints() -> None:
    """The stretch divides by `1 - sigma` at the last sampled level, which is 0 for one step."""
    assert build_sigmas(distilled=False, num_steps=1, video_seq_len=4096).tolist() == [1.0, 0.0]


def test_the_shift_interpolates_between_the_anchors_and_holds_outside_them() -> None:
    """Independently: mu is 0.95 at 1024 tokens and 2.05 at 4096, linear in between, flat outside.

    Extrapolating instead is what makes the schedule degenerate at real canvases -- see below."""
    assert calculate_shift(1024) == pytest.approx(0.95)
    assert calculate_shift(4096) == pytest.approx(2.05)
    assert calculate_shift(2560) == pytest.approx(1.5)
    assert calculate_shift(320) == pytest.approx(0.95)
    assert calculate_shift(7168) == pytest.approx(2.05)
    assert calculate_shift(13728) == pytest.approx(2.05)


def test_the_schedule_keeps_spending_steps_where_the_clip_is_formed() -> None:
    """The failure the shift clamp exists to prevent: an unclamped mu at a real canvas piles every
    step into the high-noise end and then jumps to the terminal in one. Stated as a property rather
    than a pinned list -- no step may be more than a third of the whole range."""
    for video_seq_len in (320, 6144, 13728, 24576):
        sigmas = build_sigmas(distilled=False, num_steps=30, video_seq_len=video_seq_len)
        deltas = (sigmas[:-1] - sigmas[1:]).tolist()

        assert max(deltas) < 0.34, f"{video_seq_len} tokens: largest step is {max(deltas):.2f}"
        # And the schedule reaches the middle of the range rather than clinging to the top.
        assert float(sigmas[len(sigmas) // 2]) < 0.9


def test_every_canvas_the_panel_offers_samples_the_released_schedule() -> None:
    """All three presets are past the 4096-token anchor, so all three get the released pipeline's
    own schedule rather than three different extrapolations of it."""
    reference = build_sigmas(distilled=False, num_steps=30, video_seq_len=LTX2_MAX_SEQ_LEN)

    for width, height in ((928, 512), (1248, 704), (1376, 768)):
        sigmas = build_sigmas(distilled=False, num_steps=30, video_seq_len=video_sequence_length(121, height, width))
        assert torch.equal(sigmas, reference)


def test_a_deterministic_step_is_the_flow_match_euler_step_in_velocity_space() -> None:
    """The expectation is the reference's own formulation -- step the velocity prediction by the
    sigma difference -- which this module deliberately rewrites in x0 space."""
    sigmas = torch.tensor([0.9, 0.4, 0.0])
    sample = torch.randn(1, 6, 4)
    denoised = torch.randn(1, 6, 4)

    velocity = (sample - denoised) / sigmas[0]
    expected = sample + (sigmas[1] - sigmas[0]) * velocity

    assert torch.allclose(flow_step(sample, denoised, sigmas, 0), expected, atol=1e-6)


def test_the_terminal_step_returns_the_prediction_itself() -> None:
    sigmas = torch.tensor([0.4, 0.0])
    sample, denoised = torch.randn(1, 4, 2), torch.randn(1, 4, 2)
    assert torch.equal(flow_step(sample, denoised, sigmas, 0, eta=1.0, noise=torch.randn(1, 4, 2)), denoised)


def test_an_ancestral_step_needs_noise_and_a_zero_eta_does_not() -> None:
    sigmas = torch.tensor([0.9, 0.4, 0.0])
    sample, denoised = torch.randn(1, 4, 2), torch.randn(1, 4, 2)

    flow_step(sample, denoised, sigmas, 0, eta=0.0)
    with pytest.raises(ValueError, match="needs a noise tensor"):
        flow_step(sample, denoised, sigmas, 0, eta=1.0)


def test_an_ancestral_step_lands_on_the_next_noise_level() -> None:
    """The step drops to an intermediate sigma and renoises back up, rescaling the signal so the
    transition is variance preserving. With x = alpha*x0 + sigma*noise at the current level, the
    result should be the same mixture at the next one."""
    torch.manual_seed(0)
    sigmas = torch.tensor([0.9, 0.5, 0.0])
    x0 = torch.randn(1, 4096, 8)
    noise = torch.randn_like(x0)
    sample = (1 - sigmas[0]) * x0 + sigmas[0] * noise

    stepped = flow_step(sample, x0, sigmas, 0, eta=1.0, s_noise=1.0, noise=torch.randn_like(x0))

    residual = stepped - (1 - sigmas[1]) * x0
    assert float(residual.std()) == pytest.approx(float(sigmas[1]), rel=0.05)


def test_a_zero_eta_ancestral_step_is_the_deterministic_one() -> None:
    sigmas = torch.tensor([0.8, 0.3, 0.0])
    sample, denoised = torch.randn(1, 5, 3), torch.randn(1, 5, 3)
    assert torch.allclose(
        flow_step(sample, denoised, sigmas, 0, eta=0.0),
        flow_step(sample, denoised, sigmas, 0, eta=0.0, noise=torch.randn(1, 5, 3)),
    )


def test_the_shifted_schedule_matches_the_reference_transformation() -> None:
    """Independently: shift sigma -> e^mu / (e^mu + 1/sigma - 1), then stretch onto the terminal."""
    num_steps, seq_len = 10, 2048
    mu = math.exp(calculate_shift(seq_len))
    raw = [mu / (mu + 1 / s - 1) for s in torch.linspace(1.0, 0.1, num_steps).tolist()]
    scale = (1 - raw[-1]) / (1 - LTX2_SHIFT_TERMINAL)
    expected = [1 - (1 - s) / scale for s in raw]

    sigmas = build_sigmas(distilled=False, num_steps=num_steps, video_seq_len=seq_len)
    assert sigmas[:-1].tolist() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("refine_steps", [2, 3, 4, 6, 8, 12, 30])
def test_a_refine_budget_buys_that_many_steps_near_the_level_it_asks_for(refine_steps: int) -> None:
    """Truncating a schedule built *at* the budget made the entry level a function of the budget --
    not monotonically, either: 4 steps entered at 0.867 and 6 at 0.861. Worse, 2 steps entered at
    0.100, so the "refine" re-noised almost to clean and ran one step that returned its own input,
    at four times the base pass's token cost. The budget is the step count; the resolution that
    delivers it is solved for."""
    sigmas = build_refine_sigmas(
        distilled=False, refine_steps=refine_steps, video_seq_len=24576, noise_scale=LTX2_STAGE_2_NOISE_SCALE
    )

    assert sigmas.numel() - 1 == refine_steps
    assert float(sigmas[0]) <= LTX2_STAGE_2_NOISE_SCALE
    # Near the level asked for, not merely at or below it.
    assert float(sigmas[0]) > 0.75 * LTX2_STAGE_2_NOISE_SCALE
    assert float(sigmas[-1]) == 0.0


def test_a_refine_budget_too_small_to_reach_its_level_is_refused() -> None:
    with pytest.raises(ValueError, match="far below"):
        build_refine_sigmas(distilled=False, refine_steps=1, video_seq_len=24576, noise_scale=LTX2_STAGE_2_NOISE_SCALE)


def test_the_distilled_refine_ignores_the_budget_as_its_base_pass_does() -> None:
    for refine_steps in (2, 8, 30):
        sigmas = build_refine_sigmas(
            distilled=True, refine_steps=refine_steps, video_seq_len=24576, noise_scale=LTX2_STAGE_2_NOISE_SCALE
        )
        assert sigmas.tolist() == pytest.approx([*STAGE_2_DISTILLED_SIGMA_VALUES, 0.0])


@pytest.mark.parametrize("noise_scale", [1.0, 0.99999999, 0.999999999999])
def test_a_refine_entering_at_the_top_of_the_schedule_is_refused(noise_scale: float) -> None:
    """`lerp(x0, noise, 1.0)` is pure noise: the base pass and the upscale would be generated and
    then discarded. Values a hair under 1 round up to exactly 1 in the schedule's float32, so the
    guard is on the level that survives truncation rather than on the value requested."""
    with pytest.raises(ValueError, match="discard the base pass|re-enter the schedule"):
        build_sigmas(distilled=True, num_steps=8, video_seq_len=24576, start_sigma=noise_scale)

"""LTX-2 flow-matching schedule and step.

Two schedules, one step. The dev checkpoint samples a resolution-shifted linear schedule
deterministically; the guidance-distilled checkpoint samples eight fixed noise levels ancestrally.
Both are stepped by :func:`flow_step`, which is the rectified-flow ancestral Euler step of
``ltx_core.components.diffusion_steps.EulerAncestralDiffusionStep`` and reduces exactly to the
deterministic Euler step at ``eta=0``.

The step is written in x0 space (the space the guidance combine works in) rather than in velocity
space. The two are the same update: a velocity prediction is ``v = (x - x0) / sigma``, and the
flow-match Euler step ``x + (sigma_next - sigma) * v`` expands to
``(sigma_next / sigma) * x + (1 - sigma_next / sigma) * x0`` -- the deterministic branch below.
Converting back and forth around the guidance combine, as the reference pipelines do, would only
add two divisions by a sigma that is already the step's own ratio.
"""

import math

import torch

from invokeai.backend.ltx2.constants import (
    LTX2_BASE_SEQ_LEN,
    LTX2_BASE_SHIFT,
    LTX2_DISTILLED_SIGMAS,
    LTX2_MAX_SEQ_LEN,
    LTX2_MAX_SHIFT,
)

# The last sampled noise level every shifted schedule is stretched onto. Not cosmetic: at the
# max-shift anchor the raw schedule's final sigma is 0.21, and without the stretch the run would
# stop a fifth of the way from clean. Both upstream implementations apply it --
# ``LTX2Scheduler(stretch=True, terminal=0.1)`` in ltx-core, and ``shift_terminal: 0.1`` in the
# released diffusers scheduler config, which
# ``FlowMatchEulerDiscreteScheduler.stretch_shift_to_terminal`` reads.
LTX2_SHIFT_TERMINAL = 0.1

# How close to the level it asked for a refine pass has to start for the pass to mean anything.
_MIN_REFINE_ENTRY_FRACTION = 0.5


def calculate_shift(video_seq_len: int) -> float:
    """The schedule's ``mu``, interpolated from the token count and held at the anchors outside them.

    The clamp is the whole point. The line is fitted between 1024 and 4096 tokens, and every canvas
    this architecture offers is past its top: a 1248x704x121 clip is 13728 tokens, where the
    unclamped line gives mu 5.5 and the resulting schedule spends 29 of 30 steps above sigma 0.54
    before covering 0.54 -> 0.1 in one. Held at 2.05, the schedule is the released pipeline's
    exactly -- ltx-core's ``LTX2Scheduler`` is called with no latent, so it always takes the
    max-shift anchor, and the diffusers port's own default operating point is inside the range.
    """
    slope = (LTX2_MAX_SHIFT - LTX2_BASE_SHIFT) / (LTX2_MAX_SEQ_LEN - LTX2_BASE_SEQ_LEN)
    shift = video_seq_len * slope + LTX2_BASE_SHIFT - slope * LTX2_BASE_SEQ_LEN

    return min(max(shift, LTX2_BASE_SHIFT), LTX2_MAX_SHIFT)


def build_sigmas(
    *, distilled: bool, num_steps: int, video_seq_len: int, start_sigma: float | None = None
) -> torch.Tensor:
    """The noise levels to sample, terminal 0 included, as ``[steps + 1]`` float32 on the CPU.

    ``num_steps`` is ignored for the distilled schedule, whose eight levels are a property of the
    checkpoint's distillation rather than a sampling budget; the caller reports that to the user.

    ``start_sigma`` is the refine pass: the schedule is built whole and then entered partway down,
    keeping the levels at or below it. Truncating rather than carrying a second schedule is what
    makes the two stages one recipe -- and it is not an approximation of the released one. Upstream
    publishes the distilled second stage as a list, and that list is exactly this schedule's tail
    from ``LTX2_STAGE_2_NOISE_SCALE`` down, which ``test_ltx2_sampling`` pins. The dev checkpoint has
    no published second stage; entering its own shifted schedule at the same level is this
    implementation's generalization, and keeps a guided refine sampling the levels dev was trained
    on instead of three distilled jumps.

    The stretch onto the terminal level is applied to the whole schedule before truncation: it is
    what the run's shape is, not a property of the part that gets sampled.
    """
    if distilled:
        sigmas = torch.tensor(LTX2_DISTILLED_SIGMAS, dtype=torch.float32)
        return _with_terminal(_truncate(sigmas, start_sigma))

    if num_steps < 1:
        raise ValueError(f"LTX-2 needs at least one step; got {num_steps}.")

    # Built in float64 and narrowed once at the end. The shift divides mu by itself at sigma 1,
    # which in float32 is the exact mu over a rounded one -- the schedule would not start at 1.
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float64)
    mu = math.exp(calculate_shift(video_seq_len))
    sigmas = mu / (mu + (1.0 / sigmas - 1.0))

    # A single step is already the terminal one, and its `1 - sigma` is 0: there is nothing to
    # stretch onto and the scale factor would be a division by zero.
    if num_steps > 1:
        one_minus = 1.0 - sigmas
        sigmas = 1.0 - one_minus * (1.0 - LTX2_SHIFT_TERMINAL) / one_minus[-1]

    return _with_terminal(_truncate(sigmas.to(torch.float32), start_sigma))


def _truncate(sigmas: torch.Tensor, start_sigma: float | None) -> torch.Tensor:
    """The levels at or below ``start_sigma``; the whole schedule when it is ``None``.

    The comparison is made in the schedule's own float32, not in Python's float64. That is what
    admits the level the released second stage starts from: 0.909375 is not representable in
    float32, and the nearest value is *above* it, so a float64 comparison would drop the very level
    ``LTX2_STAGE_2_NOISE_SCALE`` names.
    """
    if start_sigma is None:
        return sigmas
    if not 0.0 < start_sigma < 1.0:
        raise ValueError(f"The refine pass must re-enter the schedule in (0, 1); got {start_sigma}.")

    kept = sigmas[sigmas <= start_sigma]
    if kept.numel() == 0:
        raise ValueError(
            f"No noise level of this schedule is at or below {start_sigma}; its lowest is "
            f"{float(sigmas[-1]):.4f}. Raise the refine noise level or add steps."
        )
    # Guarded on the outcome rather than the request: a `start_sigma` a hair under 1 rounds up to
    # exactly 1 in float32, and re-entering at 1.0 is not a refine -- `lerp(x0, noise, 1.0)` is pure
    # noise, so the base pass and the upscale would be generated and then thrown away.
    if float(kept[0]) >= 1.0:
        raise ValueError(
            f"A refine pass entering at noise level {float(kept[0])} would discard the base pass "
            f"entirely; choose a level below the schedule's first."
        )
    return kept


def build_refine_sigmas(*, distilled: bool, refine_steps: int, video_seq_len: int, noise_scale: float) -> torch.Tensor:
    """The refine pass's schedule: ``refine_steps`` levels, entered at ``noise_scale``.

    Truncating a schedule built at the refine budget would make the *entry level* depend on the
    budget, because which levels exist near ``noise_scale`` is a property of the whole schedule's
    resolution. That is not a knob anyone wants: at 2 steps the nearest kept level is 0.1, so the
    "refine" re-noises almost to clean and runs a single step that returns its own input, at four
    times the base pass's token cost; at 1 it raises after the base pass has already run.

    So the resolution is solved for instead. The whole schedule is built at increasing step counts
    until its tail at ``noise_scale`` is long enough, and that tail is the refine pass -- which
    makes the budget mean what it says, keeps the entry level pinned near ``noise_scale`` whatever
    the budget, and keeps every sampled level one the base pass would also have sampled.

    The distilled schedule is fixed, so its budget is ignored exactly as it is in a base pass.
    """
    if refine_steps < 1:
        raise ValueError(f"A refine pass needs at least one step; got {refine_steps}.")

    if distilled:
        return build_sigmas(
            distilled=True, num_steps=refine_steps, video_seq_len=video_seq_len, start_sigma=noise_scale
        )

    # The tail grows monotonically with the whole schedule's resolution, so the first count that
    # covers the budget is the coarsest schedule that can express it. Counted on the untruncated
    # schedule rather than by catching `_truncate`'s refusal, so a genuinely bad `noise_scale` still
    # reports itself instead of being mistaken for too coarse a schedule. The ceiling is generous; a
    # schedule is a few hundred floats and the search runs once per pass.
    for num_steps in range(refine_steps, refine_steps * 32 + 2):
        whole = build_sigmas(distilled=False, num_steps=num_steps, video_seq_len=video_seq_len)[:-1]
        if int((whole <= noise_scale).sum()) >= refine_steps:
            sigmas = build_sigmas(
                distilled=False, num_steps=num_steps, video_seq_len=video_seq_len, start_sigma=noise_scale
            )
            # A budget so small that the coarsest schedule holding it starts far below the level
            # asked for is not a short refine, it is a different one: at one step the only level at
            # or below 0.909 is 0.1, so the pass would re-noise almost to clean and return its own
            # input after a full transformer pass at four times the base pass's tokens.
            if float(sigmas[0]) < _MIN_REFINE_ENTRY_FRACTION * noise_scale:
                raise ValueError(
                    f"A {refine_steps}-step refine can only enter this schedule at "
                    f"{float(sigmas[0]):.3f}, far below the {noise_scale:.3f} asked for, so it would "
                    f"barely change the upscaled clip. Give the refine pass more steps."
                )
            return sigmas

    raise ValueError(
        f"No schedule resolution puts {refine_steps} levels at or below {noise_scale}; lower the "
        f"refine noise level or the step count."
    )


def _with_terminal(sigmas: torch.Tensor) -> torch.Tensor:
    return torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype)]).to(torch.float32)


def flow_step(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
    *,
    eta: float = 0.0,
    s_noise: float = 1.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance one sampling step, from ``sigmas[index]`` to ``sigmas[index + 1]``.

    ``denoised`` is the x0 prediction. With ``eta > 0`` the step advances deterministically to an
    intermediate ``sigma_down`` and renoises back up, rescaling the signal component by
    ``alpha_next / alpha_down`` (with ``alpha = 1 - sigma``, the rectified-flow parameterization)
    so the transition stays variance preserving. ``eta = 0`` is the plain Euler step and needs no
    noise. The terminal level returns the prediction itself.
    """
    sigma = sigmas[index].to(torch.float32)
    sigma_next = sigmas[index + 1].to(torch.float32)
    x = sample.to(torch.float32)
    x0 = denoised.to(torch.float32)

    if sigma_next == 0:
        return x0

    down_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * down_ratio

    ratio = sigma_down / sigma
    x_next = ratio * x + (1.0 - ratio) * x0

    if eta > 0:
        if noise is None:
            raise ValueError("An ancestral LTX-2 step (eta > 0) needs a noise tensor.")
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0) ** 0.5
        x_next = (alpha_next / alpha_down) * x_next + noise.to(torch.float32) * s_noise * renoise

    return x_next

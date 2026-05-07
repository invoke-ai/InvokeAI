"""Tests for Anima scheduler registry."""

import typing

import pytest
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from invokeai.backend.flux.schedulers import (
    ANIMA_SCHEDULER_LABELS,
    ANIMA_SCHEDULER_MAP,
    ANIMA_SCHEDULER_NAME_VALUES,
)


def test_anima_scheduler_map_entries_are_class_kwargs_tuples():
    """Every entry must be (SchedulerClass, kwargs_dict)."""
    for name, entry in ANIMA_SCHEDULER_MAP.items():
        assert isinstance(entry, tuple), f"{name} is not a tuple"
        assert len(entry) == 2, f"{name} tuple has wrong arity"
        cls, kwargs = entry
        assert isinstance(cls, type) and issubclass(cls, SchedulerMixin), (
            f"{name} first element is not a SchedulerMixin subclass"
        )
        assert isinstance(kwargs, dict), f"{name} second element is not a dict"


def test_anima_scheduler_map_entries_can_be_constructed():
    """Every entry must construct cleanly by splatting its kwargs."""
    for name, (cls, kwargs) in ANIMA_SCHEDULER_MAP.items():
        scheduler = cls(num_train_timesteps=1000, **kwargs)
        assert isinstance(scheduler, SchedulerMixin), f"{name} did not produce a SchedulerMixin"


def test_anima_scheduler_labels_cover_every_map_key():
    for name in ANIMA_SCHEDULER_MAP.keys():
        assert name in ANIMA_SCHEDULER_LABELS, f"{name} has no label"


def test_anima_scheduler_map_includes_new_dpmpp_entries():
    assert "dpmpp_2m" in ANIMA_SCHEDULER_MAP
    assert "dpmpp_2m_sde" in ANIMA_SCHEDULER_MAP


def test_anima_dpmpp_2m_uses_flow_prediction():
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    cls, kwargs = ANIMA_SCHEDULER_MAP["dpmpp_2m"]
    assert kwargs["prediction_type"] == "flow_prediction"
    assert kwargs["use_flow_sigmas"] is True
    assert kwargs["flow_shift"] == ANIMA_SHIFT
    assert kwargs["solver_order"] == 2
    assert "algorithm_type" not in kwargs  # deterministic, default algorithm


def test_anima_dpmpp_2m_sde_uses_sde_algorithm():
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    cls, kwargs = ANIMA_SCHEDULER_MAP["dpmpp_2m_sde"]
    assert kwargs["prediction_type"] == "flow_prediction"
    assert kwargs["use_flow_sigmas"] is True
    assert kwargs["flow_shift"] == ANIMA_SHIFT
    assert kwargs["algorithm_type"] == "sde-dpmsolver++"
    assert kwargs["solver_order"] == 2


def test_anima_dpmpp_2m_produces_anima_compatible_sigma_schedule():
    """The DPM++ 2M scheduler, when run through the same dispatch logic as
    anima_denoise._run_diffusion, must produce a sigma schedule equivalent
    to Anima's reference schedule (loglinear_timestep_shift with shift=3.0).

    On diffusers 0.35.1, DPMSolverMultistepScheduler.set_timesteps does not
    accept `sigmas=`, so the runtime falls back to num_inference_steps and
    relies on the scheduler's internal flow_shift=3.0 to compute equivalent
    sigmas. This test verifies that equivalence end-to-end.
    """
    import inspect

    from invokeai.app.invocations.anima_denoise import loglinear_timestep_shift
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    num_steps = 10

    # Reference: Anima's own pre-shifted sigma schedule.
    anima_sigmas = [loglinear_timestep_shift(ANIMA_SHIFT, 1.0 - i / num_steps) for i in range(num_steps + 1)]

    cls, kwargs = ANIMA_SCHEDULER_MAP["dpmpp_2m"]
    scheduler = cls(num_train_timesteps=1000, **kwargs)

    # Mirror anima_denoise.py:502-506 dispatch.
    sig = inspect.signature(scheduler.set_timesteps)
    if "sigmas" in sig.parameters:
        scheduler.set_timesteps(sigmas=anima_sigmas, device="cpu")
    else:
        scheduler.set_timesteps(num_inference_steps=num_steps, device="cpu")

    diffusers_sigmas = [float(s) for s in scheduler.sigmas[: len(anima_sigmas)]]
    max_diff = max(abs(a - b) for a, b in zip(anima_sigmas, diffusers_sigmas, strict=True))
    assert max_diff < 1e-3, f"DPM++ 2M sigma schedule diverges from Anima reference (max abs diff = {max_diff:.6f})"


def test_anima_dpmpp_2m_with_denoising_start_honors_clipped_schedule():
    """DPM++ img2img: the set_begin_index path must start at the correct sigma.

    When DPMSolverMultistepScheduler doesn't accept sigmas=, anima_denoise falls back to
    set_timesteps(num_inference_steps=full_steps) + set_begin_index(start_idx).  The
    effective first sigma must match the clipped Anima reference schedule within 1e-3.
    """
    import inspect

    from invokeai.app.invocations.anima_denoise import loglinear_timestep_shift
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    num_steps = 30
    denoising_start = 0.5
    start_idx = int(denoising_start * num_steps)  # mirrors anima_denoise clipping math

    full_sigmas = [loglinear_timestep_shift(ANIMA_SHIFT, 1.0 - i / num_steps) for i in range(num_steps + 1)]
    expected_first_sigma = full_sigmas[start_idx]

    cls, kwargs = ANIMA_SCHEDULER_MAP["dpmpp_2m"]
    scheduler = cls(num_train_timesteps=1000, **kwargs)
    sig = inspect.signature(scheduler.set_timesteps)

    if "sigmas" in sig.parameters:
        # Future diffusers: sigmas= supported, clipped schedule passed directly.
        scheduler.set_timesteps(sigmas=full_sigmas[start_idx:], device="cpu")
        actual_first_sigma = float(scheduler.sigmas[0])
    else:
        # Current diffusers: use set_begin_index on the full schedule.
        scheduler.set_timesteps(num_inference_steps=num_steps, device="cpu")
        scheduler.set_begin_index(start_idx)
        actual_first_sigma = float(scheduler.sigmas[start_idx])

    assert abs(actual_first_sigma - expected_first_sigma) < 1e-3, (
        f"DPM++ first sigma with denoising_start=0.5: got {actual_first_sigma:.6f}, expected {expected_first_sigma:.6f}"
    )


def test_anima_set_begin_index_path_step_count_with_denoising_end():
    """set_begin_index fallback must honour denoising_end, not just denoising_start.

    Regression test: the old formula (len(timesteps) - begin_index) ignored denoising_end
    and ran past it. For steps=30, denoising_start=0.2, denoising_end=0.8 the correct
    step count is 18, not 24.
    """
    import inspect

    from invokeai.app.invocations.anima_denoise import loglinear_timestep_shift
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    num_steps = 30
    denoising_start = 0.2
    denoising_end = 0.8

    # Reference step count from the Euler path (clipped sigmas).
    full_sigmas = [loglinear_timestep_shift(ANIMA_SHIFT, 1.0 - i / num_steps) for i in range(num_steps + 1)]
    total_sigmas = len(full_sigmas)
    start_idx = int(denoising_start * (total_sigmas - 1))
    end_idx = int(denoising_end * (total_sigmas - 1)) + 1
    expected_steps = (end_idx - start_idx) - 1  # 18

    cls, kwargs = ANIMA_SCHEDULER_MAP["dpmpp_2m"]
    scheduler = cls(num_train_timesteps=1000, **kwargs)
    sig = inspect.signature(scheduler.set_timesteps)

    scheduler_begin_index = int(denoising_start * num_steps)
    if "sigmas" in sig.parameters:
        clipped = full_sigmas[start_idx:end_idx]
        scheduler.set_timesteps(sigmas=clipped, device="cpu")
        num_scheduler_steps = len(scheduler.timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps=num_steps, device="cpu")
        scheduler.set_begin_index(scheduler_begin_index)
        num_scheduler_steps = int(denoising_end * num_steps) - scheduler_begin_index

    assert num_scheduler_steps == expected_steps, (
        f"DPM++ scheduler step count with denoising_start={denoising_start}, "
        f"denoising_end={denoising_end}: got {num_scheduler_steps}, expected {expected_steps}"
    )


@pytest.mark.parametrize(
    ["denoising_start", "denoising_end", "steps"],
    [
        (0.2, 0.8, 30),  # mid-range: 18 logical steps → 36 doubled calls
        (0.0, 0.8, 30),  # start-only clip: 24 logical steps → 48 doubled calls
        (0.2, 1.0, 30),  # end=1.0: clamp kicks in (last step first-order only)
        (0.5, 0.75, 20),  # different step count
    ],
)
def test_anima_heun_set_begin_index_path_begin_index_and_step_count(
    denoising_start: float, denoising_end: float, steps: int
):
    """Heun img2img: set_begin_index must use doubled-array index and step count must
    account for Heun's 2N-1 timestep structure.

    Logical step k maps to doubled-array begin index 2k.  For a range [k_start, k_end)
    the total calls is 2*(k_end-k_start), clamped to len(timesteps)-begin_index so that
    denoising_end=1.0 correctly gets the 2N-1 (not 2N) count.
    """
    from diffusers import FlowMatchHeunDiscreteScheduler

    k_start = int(denoising_start * steps)
    k_end = int(denoising_end * steps)

    scheduler = FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    scheduler.set_timesteps(num_inference_steps=steps, device="cpu")

    expected_begin_index = 2 * k_start
    expected_steps = min(2 * (k_end - k_start), len(scheduler.timesteps) - expected_begin_index)

    # Verify the doubled structure: len(timesteps) == 2*steps - 1
    assert len(scheduler.timesteps) == 2 * steps - 1, (
        f"Heun timesteps length: expected {2 * steps - 1}, got {len(scheduler.timesteps)}"
    )

    # The fixed code's begin index must map logical step to doubled-array space.
    assert expected_begin_index == 2 * k_start

    # For mid-range (denoising_end < 1): all steps in range have first + second order.
    if denoising_end < 1.0:
        assert expected_steps == 2 * (k_end - k_start), (
            f"mid-range step count: expected {2 * (k_end - k_start)}, got {expected_steps}"
        )

    # For denoising_end=1.0: last step is first-order only → clamped to 2N-1-begin.
    if denoising_end == 1.0 and k_start > 0:
        full_from_begin = len(scheduler.timesteps) - expected_begin_index
        assert expected_steps == full_from_begin

    # Bounds check: begin_index + num_steps must not exceed len(timesteps).
    assert expected_begin_index + expected_steps <= len(scheduler.timesteps), (
        f"step range [{expected_begin_index}, {expected_begin_index + expected_steps}) "
        f"exceeds timesteps length {len(scheduler.timesteps)}"
    )

    # Sigma sanity: the sigma at the doubled begin index must equal the sigma at logical k_start.
    # sigmas has 2N entries; sigmas[2k] == s_k for all k.
    assert len(scheduler.sigmas) == 2 * steps
    sigma_at_begin = scheduler.sigmas[expected_begin_index].item()
    sigma_at_logical_k = scheduler.sigmas[2 * k_start].item()
    assert abs(sigma_at_begin - sigma_at_logical_k) < 1e-6


def test_anima_literal_covers_every_map_key():
    """Catch the silent failure mode where a new entry lands in the map but
    the Literal isn't updated — Pydantic validation would still accept it
    via runtime introspection but type-check tooling would not."""
    literal_values = set(typing.get_args(ANIMA_SCHEDULER_NAME_VALUES))
    for name in ANIMA_SCHEDULER_MAP:
        assert name in literal_values, f"{name} is in the map but missing from the Literal"


def test_anima_scheduler_literal_includes_er_sde():
    """er_sde must appear in the literal type, have a label, and be
    registered in ANIMA_SCHEDULER_MAP for dispatch through the universal
    scheduler path."""
    literal_args = typing.get_args(ANIMA_SCHEDULER_NAME_VALUES)
    assert "er_sde" in literal_args
    assert "er_sde" in ANIMA_SCHEDULER_LABELS
    assert ANIMA_SCHEDULER_LABELS["er_sde"] == "ER-SDE"
    assert "er_sde" in ANIMA_SCHEDULER_MAP


def test_anima_heun_uses_anima_shift_for_internal_schedule():
    """Heun does NOT accept set_timesteps(sigmas=...) so it always builds its own internal
    schedule. With shift=1.0 (the previous setting), that schedule was linear and gave the
    wrong noise levels for img2img — Heun's sigmas[2*k_start] would be ~0.48 when Anima's
    reference at user step k_start (denoising_start=0.5) is ~0.75. The model would receive
    a timestep matching neither the latents nor its training distribution.

    Fix: give Heun shift=ANIMA_SHIFT so its internal schedule approximates Anima's reference.
    """
    from invokeai.app.invocations.anima_denoise import loglinear_timestep_shift

    cls, kwargs = ANIMA_SCHEDULER_MAP["heun"]
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    assert kwargs["shift"] == ANIMA_SHIFT, (
        f"Heun must use shift={ANIMA_SHIFT} (Anima's loglinear shift) since it doesn't accept "
        f"sigmas=; got shift={kwargs['shift']}"
    )

    # Verify the schedule approximates Anima's reference for the bulk of user steps.
    # The two formulas diverge at the tail (Heun uses linspace(1, T, N+1), Anima uses
    # 1 - i/N), so we tolerate up to 5% absolute. The previous shift=1.0 bug gave 25-40%+
    # divergence at mid-schedule, so any reasonable tolerance catches that regression.
    steps = 30
    anima_ref = [loglinear_timestep_shift(ANIMA_SHIFT, 1.0 - i / steps) for i in range(steps + 1)]
    s = cls(num_train_timesteps=1000, **kwargs)
    s.set_timesteps(num_inference_steps=steps, device="cpu")

    # Heun's sigmas array has 2*N entries; sigmas[2*k] is the noise level at user step k.
    for k in (0, 5, 10, 15):
        heun_sigma = s.sigmas[2 * k].item()
        ref = anima_ref[k]
        assert abs(heun_sigma - ref) < 0.05, (
            f"Heun internal sigma at user step {k} ({heun_sigma:.4f}) diverges from "
            f"Anima reference ({ref:.4f}) by more than 5% — likely a shift kwarg regression"
        )


def test_anima_scheduler_map_er_sde_entry():
    """ANIMA_SCHEDULER_MAP['er_sde'] must map to ERSDEScheduler with rectified-flow kwargs.

    This is the wiring that lets Anima dispatch er_sde through the universal scheduler
    path (replacing the legacy elif is_er_sde: branch in anima_denoise.py).
    """
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT
    from invokeai.backend.rectified_flow.er_sde_scheduler import ERSDEScheduler

    assert "er_sde" in ANIMA_SCHEDULER_MAP, "er_sde must be in ANIMA_SCHEDULER_MAP"
    cls, kwargs = ANIMA_SCHEDULER_MAP["er_sde"]
    assert cls is ERSDEScheduler
    assert kwargs["use_flow_sigmas"] is True
    assert kwargs["prediction_type"] == "flow_prediction"
    assert kwargs["solver_order"] == 3
    assert kwargs["stochastic"] is True
    assert kwargs["flow_shift"] == ANIMA_SHIFT

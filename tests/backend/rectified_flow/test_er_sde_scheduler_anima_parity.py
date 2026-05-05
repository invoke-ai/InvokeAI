"""Parity gate (Task D of universal-scheduler investigation).

Asserts that ``ERSDEScheduler`` configured for rectified flow produces
trajectories byte-identical (to ~1e-6) to the existing ``er_sde_rf_step``
helper. Both should consume the same noise tensors per step.

If this test ever fails, the universal scheduler promotion is unsafe --
do not merge any change that converts Anima to use ``ERSDEScheduler``
until it passes.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.rectified_flow.er_sde import ErSdeState, er_sde_rf_step
from invokeai.backend.rectified_flow.er_sde_scheduler import ERSDEScheduler


SHAPE = (1, 4, 8, 8)
SEED = 42
DTYPE = torch.float64


def _make_noise_supplier(noise_tensors: List[torch.Tensor]):
    """Returns a stub for ``randn_tensor`` that yields pre-generated tensors in order.

    Mirrors ``diffusers.utils.torch_utils.randn_tensor`` signature loosely so
    the scheduler call site is satisfied. The pre-generated tensors are
    deep-copied so the helper trajectory sees the exact same bytes.
    """
    iterator = iter(noise_tensors)

    def stub(shape, generator=None, device=None, dtype=None, layout=None):
        tensor = next(iterator)
        # Honour the dtype/device the scheduler asks for so downstream math
        # matches what the helper trajectory will see.
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    return stub


def _run_scheduler_trajectory(
    sigmas: List[float],
    initial_sample: torch.Tensor,
    velocities: List[torch.Tensor],
    noise_tensors: List[torch.Tensor],
    solver_order: int = 3,
    stochastic: bool = True,
) -> torch.Tensor:
    """Run the new scheduler over a fixed sigma schedule with monkey-patched noise."""
    sched = ERSDEScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        solver_order=solver_order,
        stochastic=stochastic,
    )
    sched.set_timesteps(sigmas=sigmas)

    sample = initial_sample.clone()
    with patch(
        "invokeai.backend.rectified_flow.er_sde_scheduler.randn_tensor",
        side_effect=_make_noise_supplier(noise_tensors),
    ):
        for i, ts in enumerate(sched.timesteps):
            sample = sched.step(
                model_output=velocities[i],
                timestep=ts,
                sample=sample,
            ).prev_sample
            assert torch.isfinite(sample).all(), (
                f"scheduler produced non-finite sample at step {i} (sigma={sigmas[i]} -> {sigmas[i + 1]})"
            )
    return sample


def _run_helper_trajectory(
    sigmas: List[float],
    initial_sample: torch.Tensor,
    velocities: List[torch.Tensor],
    noise_tensors: List[torch.Tensor],
    stochastic: bool = True,
) -> torch.Tensor:
    """Run the existing ground-truth helper over the same schedule with the same noise.

    For steps where ``sigma_next == 0`` (terminal) the helper's algebra zeros out
    the noise contribution regardless of the tensor passed, so we feed a zero
    tensor as a defensive placeholder to avoid running off the noise list.
    """
    state = ErSdeState()
    sample = initial_sample.clone()
    noise_idx = 0
    for i in range(len(sigmas) - 1):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        # The scheduler only samples (and so the supplier only advances) when
        # stochastic AND sigma_next > 0. Mirror that here so the noise tensors
        # align step-for-step.
        if stochastic and sigma_next > 0.0:
            n = noise_tensors[noise_idx]
            noise_idx += 1
        else:
            n = torch.zeros_like(initial_sample)
        sample = er_sde_rf_step(
            x_t=sample,
            v=velocities[i],
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
            state=state,
            noise=n,
        )
        assert torch.isfinite(sample).all(), (
            f"helper produced non-finite sample at step {i} (sigma={sigma_curr} -> {sigma_next})"
        )
    return sample


def _count_noise_tensors_needed(sigmas: List[float], stochastic: bool) -> int:
    """Number of noise tensors the scheduler will request via ``randn_tensor``."""
    if not stochastic:
        return 0
    return sum(1 for i in range(len(sigmas) - 1) if sigmas[i + 1] > 0.0)


def _generate_inputs(sigmas: List[float], stochastic: bool = True):
    """Deterministically generate (sample, velocities, noise_tensors) for a trajectory."""
    torch.manual_seed(SEED)
    initial_sample = torch.randn(SHAPE, dtype=DTYPE)
    velocities = [torch.randn(SHAPE, dtype=DTYPE) for _ in range(len(sigmas) - 1)]
    n_noise = _count_noise_tensors_needed(sigmas, stochastic)
    noise_tensors = [torch.randn(SHAPE, dtype=DTYPE) for _ in range(n_noise)]
    return initial_sample, velocities, noise_tensors


# --- Parity tests ----------------------------------------------------------------


def test_parity_standard_schedule_5_steps() -> None:
    """5-step trajectory away from any boundary -- exercises orders 1, 2, 3."""
    sigmas = [0.95, 0.80, 0.60, 0.40, 0.20, 0.05]
    initial_sample, velocities, noise_tensors = _generate_inputs(sigmas, stochastic=True)

    out_sched = _run_scheduler_trajectory(sigmas, initial_sample, velocities, noise_tensors)
    out_helper = _run_helper_trajectory(sigmas, initial_sample, velocities, noise_tensors)

    assert torch.isfinite(out_sched).all()
    assert torch.isfinite(out_helper).all()
    delta = (out_sched - out_helper).abs().max().item()
    assert delta < 1e-5, f"max abs delta = {delta:.3e} (tol 1e-5)"


def test_parity_with_sigma_one_start() -> None:
    """5-step trajectory starting at the sigma=1 boundary -- exercises the closed-form limit."""
    sigmas = [1.0, 0.85, 0.65, 0.45, 0.25, 0.05]
    initial_sample, velocities, noise_tensors = _generate_inputs(sigmas, stochastic=True)

    out_sched = _run_scheduler_trajectory(sigmas, initial_sample, velocities, noise_tensors)
    out_helper = _run_helper_trajectory(sigmas, initial_sample, velocities, noise_tensors)

    assert torch.isfinite(out_sched).all()
    assert torch.isfinite(out_helper).all()
    delta = (out_sched - out_helper).abs().max().item()
    assert delta < 1e-5, f"max abs delta = {delta:.3e} (tol 1e-5)"


def test_parity_to_terminal() -> None:
    """10-step trajectory ending at sigma_next == 0 -- exercises terminal handling."""
    sigmas = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.0]
    initial_sample, velocities, noise_tensors = _generate_inputs(sigmas, stochastic=True)

    out_sched = _run_scheduler_trajectory(sigmas, initial_sample, velocities, noise_tensors)
    out_helper = _run_helper_trajectory(sigmas, initial_sample, velocities, noise_tensors)

    assert torch.isfinite(out_sched).all()
    assert torch.isfinite(out_helper).all()
    delta = (out_sched - out_helper).abs().max().item()
    assert delta < 1e-5, f"max abs delta = {delta:.3e} (tol 1e-5)"


def test_parity_long_trajectory_30_steps() -> None:
    """Full 30-step trajectory -- mimics what Anima would actually do at inference."""
    # Anima-style schedule: shifted linear from 1.0 down to 0.0.
    n_steps = 30
    raw = torch.linspace(1.0, 1.0 / 1000, n_steps + 1).tolist()
    # Time-shift like the scheduler's default flow path (shift=1.0 is identity here).
    sigmas: List[float] = list(raw[:-1]) + [0.0]
    # Avoid the exact 1.0 boundary in the loop body to keep this distinct from
    # test_parity_with_sigma_one_start; back off a hair.
    sigmas[0] = 0.999

    initial_sample, velocities, noise_tensors = _generate_inputs(sigmas, stochastic=True)

    out_sched = _run_scheduler_trajectory(sigmas, initial_sample, velocities, noise_tensors)
    out_helper = _run_helper_trajectory(sigmas, initial_sample, velocities, noise_tensors)

    assert torch.isfinite(out_sched).all()
    assert torch.isfinite(out_helper).all()
    delta = (out_sched - out_helper).abs().max().item()
    assert delta < 1e-4, f"max abs delta = {delta:.3e} (tol 1e-4)"


def test_parity_deterministic_zero_noise() -> None:
    """Approach B (robustness check): stochastic=False vs helper with zero noise.

    The deterministic ODE companion in the scheduler should match the helper
    when the helper is fed zero noise tensors (which makes its own noise
    contribution vanish at every step).
    """
    sigmas = [0.95, 0.80, 0.60, 0.40, 0.20, 0.05]
    initial_sample, velocities, _ = _generate_inputs(sigmas, stochastic=True)
    # No noise tensors needed for the deterministic scheduler trajectory.
    sched_noise: List[torch.Tensor] = []
    # The helper trajectory will pull all-zeros placeholders internally because
    # we pass stochastic=False to _run_helper_trajectory.
    helper_noise: List[torch.Tensor] = []

    out_sched = _run_scheduler_trajectory(
        sigmas, initial_sample, velocities, sched_noise, stochastic=False
    )
    out_helper = _run_helper_trajectory(
        sigmas, initial_sample, velocities, helper_noise, stochastic=False
    )

    assert torch.isfinite(out_sched).all()
    assert torch.isfinite(out_helper).all()
    delta = (out_sched - out_helper).abs().max().item()
    assert delta < 1e-5, f"max abs delta = {delta:.3e} (tol 1e-5)"


# --- Skipped / informational ----------------------------------------------------


@pytest.mark.skip(
    reason=(
        "The helper auto-warms 1->2->3 and has no solver_order cap; comparing the "
        "scheduler with solver_order=2 to the existing helper would require a "
        "hand-rolled order-2-cap helper variant. Flag for controller decision."
    )
)
def test_parity_solver_order_2() -> None:
    """Placeholder: see skip reason. The helper would need an order-cap parameter."""

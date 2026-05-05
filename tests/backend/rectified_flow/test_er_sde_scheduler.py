"""Smoke / structural tests for ``ERSDEScheduler`` (PoC).

Parity against the existing ``er_sde_rf_step`` (Anima ground truth) is
deliberately deferred to Task D and lives in a separate test file.
"""

from __future__ import annotations

import pytest
import torch

from invokeai.backend.rectified_flow.er_sde_scheduler import ERSDEScheduler

# --- Construction ----------------------------------------------------------------


@pytest.mark.parametrize(
    "prediction_type, solver_order, use_flow_sigmas",
    [
        ("epsilon", 1, False),
        ("epsilon", 2, False),
        ("epsilon", 3, False),
        ("v_prediction", 2, False),
        ("flow_prediction", 1, True),
        ("flow_prediction", 2, True),
        ("flow_prediction", 3, True),
    ],
)
def test_construction_smoke(prediction_type: str, solver_order: int, use_flow_sigmas: bool) -> None:
    sched = ERSDEScheduler(
        prediction_type=prediction_type,
        solver_order=solver_order,
        use_flow_sigmas=use_flow_sigmas,
    )
    assert sched.config.prediction_type == prediction_type
    assert sched.config.solver_order == solver_order
    assert sched.config.use_flow_sigmas == use_flow_sigmas
    # History containers are right length.
    assert len(sched.model_outputs) == solver_order
    assert len(sched._sigma_history) == solver_order
    assert all(m is None for m in sched.model_outputs)
    assert all(s is None for s in sched._sigma_history)


def test_flow_prediction_requires_flow_sigmas() -> None:
    with pytest.raises(ValueError):
        ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=False)


# --- set_timesteps ---------------------------------------------------------------


def test_set_timesteps_accepts_user_sigmas() -> None:
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=3)
    user_sigmas = [1.0, 0.8, 0.5, 0.2, 0.0]
    sched.set_timesteps(sigmas=user_sigmas)
    # self.sigmas should match (terminal 0 already in the user list).
    assert sched.sigmas.tolist() == pytest.approx(user_sigmas, rel=0, abs=1e-6)
    assert sched.num_inference_steps == len(user_sigmas) - 1


def test_set_timesteps_resets_history() -> None:
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=3)
    # Pre-populate state.
    sched.model_outputs = [torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)]
    sched._sigma_history = [0.9, 0.5, 0.1]
    sched.lower_order_nums = 3
    sched._step_index = 5

    sched.set_timesteps(sigmas=[1.0, 0.5, 0.0])

    assert all(m is None for m in sched.model_outputs)
    assert all(s is None for s in sched._sigma_history)
    assert sched.lower_order_nums == 0
    assert sched._step_index is None


def test_set_timesteps_default_path() -> None:
    sched = ERSDEScheduler(prediction_type="epsilon", use_flow_sigmas=False, solver_order=2)
    sched.set_timesteps(num_inference_steps=10)
    assert sched.num_inference_steps == 10
    assert len(sched.sigmas) == 11  # n + terminal 0
    assert sched.sigmas[-1].item() == pytest.approx(0.0, abs=1e-7)


# --- Boundary handling -----------------------------------------------------------


def test_first_order_at_sigma_one_boundary_returns_finite() -> None:
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=1)
    sched.set_timesteps(sigmas=[1.0, 0.95, 0.0])
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    v = torch.randn_like(sample)
    out = sched.step(
        model_output=v,
        timestep=sched.timesteps[0],
        sample=sample,
        generator=torch.Generator().manual_seed(0),
    )
    prev = out.prev_sample
    assert torch.isfinite(prev).all(), "boundary step produced non-finite values"
    assert prev.shape == sample.shape


def test_no_zero_division_when_prev_sigma_is_one() -> None:
    """Regression: prior step at sigma=1 must not crash subsequent higher-order branches."""
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=3)
    sched.set_timesteps(sigmas=[1.0, 0.9, 0.7, 0.0])
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    for ts in sched.timesteps:
        v = torch.randn_like(sample)
        out = sched.step(model_output=v, timestep=ts, sample=sample, generator=gen)
        sample = out.prev_sample
        assert torch.isfinite(sample).all()


# --- Multistep ramp --------------------------------------------------------------


def test_multistep_ramp_engages_higher_orders() -> None:
    """``lower_order_nums`` ramps 0->1->2->3 and order-2 / order-3 branches actually engage."""
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=3)
    # Avoid sigma=1 boundary so all three orders engage cleanly.
    sched.set_timesteps(sigmas=[0.95, 0.7, 0.5, 0.3, 0.0])

    sample = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)

    nums_seen = [sched.lower_order_nums]
    # Run enough steps to fully ramp.
    for ts in sched.timesteps[:3]:
        v = torch.randn_like(sample)
        sched.step(model_output=v, timestep=ts, sample=sample, generator=gen)
        nums_seen.append(sched.lower_order_nums)

    assert nums_seen == [0, 1, 2, 3], f"unexpected ramp: {nums_seen}"

    # Behavioural check: deterministic (no-noise) order-3 trajectory must diverge from
    # deterministic order-1 trajectory after multiple steps — the higher-order Taylor
    # terms have to actually contribute.
    sigmas = [0.95, 0.7, 0.5, 0.3, 0.0]
    sched_d3 = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=3, stochastic=False)
    sched_d3.set_timesteps(sigmas=sigmas)
    sched_d1 = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=1, stochastic=False)
    sched_d1.set_timesteps(sigmas=sigmas)

    torch.manual_seed(42)
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    vs = [torch.randn_like(sample) for _ in range(len(sigmas) - 1)]

    sample_d3 = sample.clone()
    sample_d1 = sample.clone()
    for i, ts in enumerate(sched_d3.timesteps):
        sample_d3 = sched_d3.step(model_output=vs[i], timestep=ts, sample=sample_d3).prev_sample
        sample_d1 = sched_d1.step(model_output=vs[i], timestep=ts, sample=sample_d1).prev_sample

    assert not torch.allclose(sample_d3, sample_d1, atol=1e-4), (
        "order-3 multistep produced same result as order-1 — higher-order branches not engaging"
    )


# --- Stochastic vs deterministic -------------------------------------------------


def test_stochastic_and_deterministic_diverge() -> None:
    """Same seed; stochastic=True vs False must produce visibly different trajectories."""
    sigmas = [0.95, 0.7, 0.5, 0.3, 0.0]
    sched_s = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=2, stochastic=True)
    sched_d = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=2, stochastic=False)
    sched_s.set_timesteps(sigmas=sigmas)
    sched_d.set_timesteps(sigmas=sigmas)

    torch.manual_seed(0)
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    sample_s = sample.clone()
    sample_d = sample.clone()

    torch.manual_seed(123)
    vs = [torch.randn_like(sample) for _ in range(len(sigmas) - 1)]

    gen_s = torch.Generator().manual_seed(7)
    gen_d = torch.Generator().manual_seed(7)
    for i, ts in enumerate(sched_s.timesteps):
        sample_s = sched_s.step(model_output=vs[i], timestep=ts, sample=sample_s, generator=gen_s).prev_sample
        sample_d = sched_d.step(model_output=vs[i], timestep=ts, sample=sample_d, generator=gen_d).prev_sample

    assert not torch.allclose(sample_s, sample_d, atol=1e-5), (
        "stochastic and deterministic runs are identical — noise injection not happening"
    )


# --- Long-run stability ----------------------------------------------------------


def test_30_step_flow_trajectory_no_nan() -> None:
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=3)
    sched.set_timesteps(num_inference_steps=30)
    torch.manual_seed(0)
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    for ts in sched.timesteps:
        # Synthetic "model": small constant velocity so trajectory stays bounded.
        v = torch.randn_like(sample) * 0.1
        out = sched.step(model_output=v, timestep=ts, sample=sample, generator=gen)
        sample = out.prev_sample
        assert torch.isfinite(sample).all(), f"non-finite output at timestep {ts}"


def test_30_step_vp_trajectory_no_nan() -> None:
    sched = ERSDEScheduler(prediction_type="epsilon", use_flow_sigmas=False, solver_order=3)
    sched.set_timesteps(num_inference_steps=30)
    torch.manual_seed(0)
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float32) * sched.init_noise_sigma
    gen = torch.Generator().manual_seed(0)
    for ts in sched.timesteps:
        eps = torch.randn_like(sample) * 0.1
        out = sched.step(model_output=eps, timestep=ts, sample=sample, generator=gen)
        sample = out.prev_sample
        assert torch.isfinite(sample).all(), f"non-finite output at timestep {ts}"


def test_vp_smoke_full_gate() -> None:
    """SD/SDXL VP-mode gate test for the universal-scheduler wiring (Task E).

    Constructs ERSDEScheduler in the SD/SDXL configuration (epsilon prediction,
    VP sigmas, third-order multistep, stochastic), then walks 30 steps with
    synthetic epsilon predictions and asserts:

      1. Every intermediate sample is finite (no NaN/Inf).
      2. L2 norm stays in a sane range — not exploding, not collapsing.
      3. ``lower_order_nums`` ramps 0 -> 1 -> 2 -> 3 (higher-order branches engage).
    """
    sched = ERSDEScheduler(
        prediction_type="epsilon",
        use_flow_sigmas=False,
        solver_order=3,
        stochastic=True,
    )
    sched.set_timesteps(num_inference_steps=30)
    assert sched.num_inference_steps == 30
    assert sched.lower_order_nums == 0

    torch.manual_seed(0)
    # Initial sample at the VP-SDE init scale (sigma_max ~ sqrt(sigma_train_max^2)).
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float32) * float(sched.sigmas[0].item())
    gen = torch.Generator().manual_seed(0)

    # Track ramp + per-step norms.
    ramp = [sched.lower_order_nums]
    norms: list[float] = []

    for i, ts in enumerate(sched.timesteps):
        # Synthetic epsilon prediction — small random tensor.
        eps = torch.randn_like(sample) * 0.1
        out = sched.step(model_output=eps, timestep=ts, sample=sample, generator=gen)
        sample = out.prev_sample
        ramp.append(sched.lower_order_nums)

        norm = float(torch.linalg.vector_norm(sample).item())
        norms.append(norm)

        assert torch.isfinite(sample).all(), f"non-finite sample at step {i} (timestep={ts})"
        # Sanity bounds — won't catch subtle bugs but will catch explosion/collapse.
        assert norm < 1e6, f"norm exploded at step {i}: {norm}"
        assert norm > 1e-6, f"norm collapsed at step {i}: {norm}"

    # Ramp must have hit each multistep order. After 3 steps, lower_order_nums == 3.
    assert ramp[0] == 0
    assert ramp[1] == 1
    assert ramp[2] == 2
    assert ramp[3] == 3
    # And it should saturate at 3 thereafter.
    assert all(n == 3 for n in ramp[3:]), f"lower_order_nums did not saturate at 3: {ramp}"


# --- Misc ------------------------------------------------------------------------


def test_scale_model_input_is_noop() -> None:
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True)
    x = torch.randn(2, 3)
    assert torch.equal(sched.scale_model_input(x, timestep=0), x)


def test_step_advances_step_index() -> None:
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, solver_order=2)
    sched.set_timesteps(sigmas=[0.9, 0.5, 0.0])
    sample = torch.randn(1, 4, 8, 8)
    v = torch.randn_like(sample)
    sched.step(model_output=v, timestep=sched.timesteps[0], sample=sample)
    assert sched.step_index == 1
    sched.step(model_output=v, timestep=sched.timesteps[1], sample=sample)
    assert sched.step_index == 2


def test_add_noise_shape() -> None:
    sched = ERSDEScheduler(prediction_type="flow_prediction", use_flow_sigmas=True)
    sched.set_timesteps(num_inference_steps=10)
    sample = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(sample)
    timesteps = sched.timesteps[:2]
    noisy = sched.add_noise(sample, noise, timesteps)
    assert noisy.shape == sample.shape

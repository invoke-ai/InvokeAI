"""Tests for the ER-SDE rectified-flow stepper.

Reference: Cui et al. (2023), arXiv:2309.06169.
Reference impl: https://github.com/QinpengCui/ER-SDE-Solver
"""

import torch

from invokeai.backend.rectified_flow import er_sde
from invokeai.backend.rectified_flow.er_sde import ErSdeState, er_sde_rf_step


def _make_state() -> ErSdeState:
    return ErSdeState()


def test_er_sde_rf_step_order_1_recovers_deterministic_euler_in_ode_limit(monkeypatch):
    """The load-bearing test: with _fn(x)=x (the func_type=1 ODE case),
    the first-order step must equal deterministic rectified-flow Euler
    x_t + (sigma_next - sigma_curr) * v to ~1e-5."""
    monkeypatch.setattr(er_sde, "_fn", lambda x: float(x))

    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    noise = torch.randn_like(x_t)
    sigma_curr, sigma_next = 0.6, 0.4

    out = er_sde_rf_step(
        x_t=x_t, v=v,
        sigma_curr=sigma_curr, sigma_next=sigma_next,
        state=_make_state(), noise=noise,
    )
    expected = x_t + (sigma_next - sigma_curr) * v
    assert torch.allclose(out, expected, atol=1e-5), (
        f"max abs diff: {(out - expected).abs().max()}"
    )


def test_er_sde_rf_step_sigma_curr_one_uses_closed_form_limit():
    """At sigma_curr = 1.0 (alpha_curr = 0), the limit branch must produce
    x_next = (1 - sigma_next) * x_0 + sigma_next * noise exactly."""
    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)
    sigma_curr, sigma_next = 1.0, 0.95

    out = er_sde_rf_step(
        x_t=x_t, v=v,
        sigma_curr=sigma_curr, sigma_next=sigma_next,
        state=_make_state(), noise=noise,
    )
    x0 = x_t - sigma_curr * v
    expected = (1.0 - sigma_next) * x0 + sigma_next * noise
    assert torch.allclose(out, expected, atol=1e-10)


def test_er_sde_rf_step_sigma_next_zero_terminal_returns_x0():
    """At sigma_next = 0.0, the result must equal the predicted x_0,
    regardless of multistep state."""
    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)

    state = _make_state()
    # Pre-populate state to verify multistep terms are NOT applied at terminal.
    state.old_x0 = torch.randn_like(x_t)
    state.old_d_x0 = torch.randn_like(x_t)
    state.sigma_prev_curr = 0.10
    state.sigma_prev_prev = 0.20

    out = er_sde_rf_step(
        x_t=x_t, v=v,
        sigma_curr=0.05, sigma_next=0.0,
        state=state, noise=noise,
    )
    expected = x_t - 0.05 * v
    # Algebraically exact (not approximate): with sigma_next=0, fn_next=0,
    # so r_fn=0 and noise_std=0, giving x_next = (1-0)*x_0 + 0 = x_0.
    assert torch.equal(out, expected), f"max abs diff: {(out - expected).abs().max()}"


def test_er_sde_rf_step_preserves_shape_and_dtype():
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float32)
    v = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)
    out = er_sde_rf_step(
        x_t=x_t, v=v,
        sigma_curr=0.6, sigma_next=0.4,
        state=_make_state(), noise=noise,
    )
    assert out.shape == x_t.shape
    assert out.dtype == x_t.dtype


def test_er_sde_rf_step_reproducible_with_same_noise():
    """Two calls with the same noise tensor must produce byte-identical output."""
    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)

    out_a = er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.6, sigma_next=0.4,
        state=_make_state(), noise=noise,
    )
    out_b = er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.6, sigma_next=0.4,
        state=_make_state(), noise=noise,
    )
    assert torch.equal(out_a, out_b)


def test_er_sde_rf_step_finite_at_extreme_sigma_pair():
    """Numerical-clip safety: a tight (sigma_curr, sigma_next) pair that lands
    in roundoff territory must still yield finite output (no NaN)."""
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)
    out = er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.5000001, sigma_next=0.5,
        state=_make_state(), noise=noise,
    )
    assert torch.isfinite(out).all()


def test_er_sde_rf_step_state_mutation_after_first_call():
    """After call 1: old_x0 set, old_d_x0 still None, sigma_prev_curr set."""
    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    state = _make_state()
    er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.8, sigma_next=0.6,
        state=state, noise=torch.randn_like(x_t),
    )
    assert state.old_x0 is not None
    assert state.old_d_x0 is None
    assert state.sigma_prev_curr == 0.8
    assert state.sigma_prev_prev is None


def test_er_sde_rf_step_state_after_second_call_has_d_x0():
    """After call 2: old_x0 updated, old_d_x0 set with correct value, sigma_prev_prev set.

    x_t is intentionally re-used across both calls — this test checks state
    threading, not trajectory correctness.
    """
    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    state = _make_state()
    er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.8, sigma_next=0.6,
        state=state, noise=torch.randn_like(x_t),
    )
    er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.6, sigma_next=0.4,
        state=state, noise=torch.randn_like(x_t),
    )
    assert state.old_x0 is not None
    assert state.old_d_x0 is not None
    assert state.sigma_prev_curr == 0.6
    assert state.sigma_prev_prev == 0.8

    # d_x0 = (x0_call2 - x0_call1) / (lambda(0.6) - lambda(0.8))
    # x0 = x_t - sigma * v, so x0_call2 - x0_call1 = (0.8 - 0.6) * v = 0.2 * v.
    # lambda(s) = s/(1-s); lambda(0.6) = 1.5, lambda(0.8) = 4.0.
    expected_d_x0 = (0.2 * v) / (1.5 - 4.0)
    assert torch.allclose(state.old_d_x0, expected_d_x0)


def test_er_sde_rf_step_order_2_engages_after_one_step():
    """When state has one-step history, the 2nd-order Taylor term must
    contribute — output must differ from running 1st-order from empty state."""
    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)

    # Setup: run one step to populate state.old_x0.
    state_2nd = _make_state()
    er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.8, sigma_next=0.6,
        state=state_2nd, noise=torch.randn_like(x_t),
    )

    # 1st-order only (empty state).
    out_1st = er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.6, sigma_next=0.4,
        state=_make_state(), noise=noise,
    )
    # 2nd-order (history available).
    out_2nd = er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.6, sigma_next=0.4,
        state=state_2nd, noise=noise,
    )
    # Order-2 correction coefficient ~0.18 for these sigmas; 1e-3 leaves ample headroom.
    assert not torch.allclose(out_1st, out_2nd, atol=1e-3)


def test_er_sde_rf_step_order_3_engages_after_two_steps():
    """When state has two-step history, the 3rd-order Taylor term must
    contribute — output must differ from a state with one-step history only."""
    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)

    # state_3rd: two prior calls -> both old_x0 and old_d_x0 populated.
    state_3rd = _make_state()
    er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.8, sigma_next=0.7,
        state=state_3rd, noise=torch.randn_like(x_t),
    )
    er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.7, sigma_next=0.6,
        state=state_3rd, noise=torch.randn_like(x_t),
    )

    # state_2nd: one prior call -> only old_x0 populated. Same prior-step inputs
    # so its old_x0 matches state_3rd's old_x0 going into the test step.
    state_2nd = _make_state()
    er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.7, sigma_next=0.6,
        state=state_2nd, noise=torch.randn_like(x_t),
    )

    out_2nd = er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.6, sigma_next=0.5,
        state=state_2nd, noise=noise,
    )
    out_3rd = er_sde_rf_step(
        x_t=x_t, v=v, sigma_curr=0.6, sigma_next=0.5,
        state=state_3rd, noise=noise,
    )
    # 3rd-order correction is ~0.0004 per unit-v element for these sigmas; with
    # 1024 elements and a fixed seed, ~12 elements reliably exceed atol=1e-3.
    assert not torch.allclose(out_2nd, out_3rd, atol=1e-3)


def test_er_sde_rf_step_no_zerodivision_when_prev_sigma_was_one():
    """Regression test: when step 0 goes through the sigma_curr=1 limit branch,
    step 1 must NOT raise ZeroDivisionError.

    Root cause: the sigma=1 boundary path sets state.sigma_prev_curr=1.0.
    On step 1, have_one_back was True and the code called _lambda(1.0) =
    1.0 / (1.0 - 1.0) which is division by zero. The fix adds a guard that
    suppresses have_one_back when sigma_prev_curr is within _SIGMA_ONE_TOLERANCE
    of 1.0, because the finite-difference derivative across that limit step is
    not meaningful.
    """
    torch.manual_seed(0)
    x_t = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64)
    v = torch.randn_like(x_t)
    state = _make_state()

    # Step 0: sigma_curr=1.0 -> closed-form limit branch; sets sigma_prev_curr=1.0.
    x_t = er_sde_rf_step(
        x_t=x_t, v=v,
        sigma_curr=1.0, sigma_next=0.9,
        state=state, noise=torch.randn_like(x_t),
    )
    # Step 1: sigma_prev_curr=1.0 -> _lambda(1.0) was ZeroDivisionError before fix.
    x_t = er_sde_rf_step(
        x_t=x_t, v=v,
        sigma_curr=0.9, sigma_next=0.7,
        state=state, noise=torch.randn_like(x_t),
    )
    assert torch.isfinite(x_t).all()
    # Step 2: verify full sequence completes without error.
    x_t = er_sde_rf_step(
        x_t=x_t, v=v,
        sigma_curr=0.7, sigma_next=0.0,
        state=state, noise=torch.randn_like(x_t),
    )
    assert torch.isfinite(x_t).all()

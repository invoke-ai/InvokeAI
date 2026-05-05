"""ER-SDE (Extended Reverse-time SDE) solver for rectified-flow models.

Implements the multistep Taylor-expansion solver from:

    Cui, Q., Zhang, X., Lu, Z., & Liao, Q. (2023).
    Elucidating the solution space of extended reverse-time SDE
    for diffusion models. arXiv:2309.06169.
    https://arxiv.org/abs/2309.06169

Reference implementation (MIT-licensed):
    https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py

That reference targets VP-/VE-SDE diffusion models with an x_0-prediction
network. This module ports the VP form to rectified flow / flow matching
under the substitution alpha_t = 1 - sigma_t and x_0 = x_t - sigma_t * v
(velocity prediction). The algebra is otherwise unchanged.

Math is re-derived for rectified flow, not copied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Number of sample points for the left Riemann sums approximating the
# Taylor-extension integrals. Matches the reference impl's nums_intergrate=100.
_INTEGRAL_NUM_POINTS = 100

# When 1 - sigma_curr is below this, take the closed-form limit for
# sigma_curr = 1 (alpha_curr = 0, lambda_curr = inf).
_SIGMA_ONE_TOLERANCE = 1e-6


def _fn(x: float) -> float:
    """ER-SDE noise-scale function (paper appendix A.8, "SDE_5").

    Mirrors `customized_func(..., func_type=7)` in the reference impl —
    the variant the paper recommends and tests for fast (~20 NFE) sampling.
    """
    return x * (math.exp(x ** 0.3) + 10.0)


def _integral_one_over_fn(lambda_next: float, lambda_curr: float) -> float:
    """Left Riemann sum of int_{lambda_next}^{lambda_curr} 1/_fn(lam) dlam.

    Precondition: lambda_next > 0. The integrand has a logarithmic singularity
    at lam=0 (_fn(0)=0), so callers must skip this when sigma_next=0 — the
    multistep guard `not_terminal = sigma_next > 0.0` in er_sde_rf_step
    enforces this.
    """
    delta = lambda_curr - lambda_next
    if delta <= 0:
        return 0.0
    step = delta / _INTEGRAL_NUM_POINTS
    total = 0.0
    for k in range(_INTEGRAL_NUM_POINTS):
        lam = lambda_next + k * step
        total += step / _fn(lam)
    return total


def _integral_lam_minus_curr_over_fn(lambda_next: float, lambda_curr: float) -> float:
    """Left Riemann sum of int_{lambda_next}^{lambda_curr} (lam - lambda_curr)/_fn(lam) dlam.

    Precondition: lambda_next > 0. Same singularity at lam=0 as
    _integral_one_over_fn — callers must skip this when sigma_next=0.
    """
    delta = lambda_curr - lambda_next
    if delta <= 0:
        return 0.0
    step = delta / _INTEGRAL_NUM_POINTS
    total = 0.0
    for k in range(_INTEGRAL_NUM_POINTS):
        lam = lambda_next + k * step
        total += step * (lam - lambda_curr) / _fn(lam)
    return total


def _lambda(sigma: float) -> float:
    """lambda = sigma / (1 - sigma). Caller is responsible for sigma < 1 - tolerance."""
    return sigma / (1.0 - sigma)


@dataclass
class ErSdeState:
    """Per-trajectory multistep history. Caller constructs once, mutates per step."""

    old_x0: torch.Tensor | None = None
    old_d_x0: torch.Tensor | None = None
    sigma_prev_curr: float | None = None
    sigma_prev_prev: float | None = None


def er_sde_rf_step(
    x_t: torch.Tensor,
    v: torch.Tensor,
    sigma_curr: float,
    sigma_next: float,
    *,
    state: ErSdeState,
    noise: torch.Tensor,
) -> torch.Tensor:
    """One ER-SDE step for rectified-flow models, auto-warming through orders 1->2->3.

    - Order 1 (used at step 0, when sigma_next == 0, or as multistep warmup):
      ports `vp_1_order` from the reference impl.
    - Order 2 (used from step 1+, except when sigma_next == 0): ports
      `vp_2_order_taylor`.
    - Order 3 (used from step 2+, except when sigma_next == 0): ports
      `vp_3_order_taylor`.

    The sigma_curr = 1.0 boundary (alpha_curr = 0, lambda_curr = inf) is
    handled as a closed-form limit; see the inline comment.

    Args:
        x_t:        Current latents at sigma=sigma_curr.
        v:          Velocity prediction (post-CFG model output), same shape as x_t.
        sigma_curr: sigma at the current step.
        sigma_next: sigma at the next step (target). Equals 0.0 at terminal step.
        state:      Multistep history; mutated in place at the end of every call.
        noise:      Caller-supplied noise tensor with the same shape and dtype as x_t.
                    Sample once per step from a seeded torch.Generator for
                    reproducibility.

    Returns:
        Latents at sigma=sigma_next.
    """
    x0 = x_t - sigma_curr * v

    # sigma_curr = 1.0 boundary: closed-form limit (alpha_curr -> 0, lambda_curr -> inf).
    # Under SDE_5's growth, r_alphas * r_fn -> 0 and noise_std -> sigma_next, giving
    # a clean re-sample around the first x_0 prediction.
    if 1.0 - sigma_curr < _SIGMA_ONE_TOLERANCE:
        x_next = (1.0 - sigma_next) * x0 + sigma_next * noise
        state.sigma_prev_prev = state.sigma_prev_curr
        state.sigma_prev_curr = sigma_curr
        state.old_d_x0 = None  # No d_x0 computable on the very first step.
        state.old_x0 = x0
        return x_next

    alpha_curr = 1.0 - sigma_curr
    alpha_next = 1.0 - sigma_next
    lambda_curr = sigma_curr / alpha_curr
    lambda_next = sigma_next / alpha_next if alpha_next > 0 else 0.0

    fn_curr = _fn(lambda_curr)
    fn_next = _fn(lambda_next)
    r_fn = fn_next / fn_curr
    r_alphas = alpha_next / alpha_curr

    # Order 1 — ports vp_1_order. Numerical clip absorbs roundoff.
    inner = lambda_next**2 - lambda_curr**2 * r_fn**2
    if inner < 0.0:
        inner = 0.0
    noise_std = math.sqrt(inner) * alpha_next
    x_next = r_alphas * r_fn * x_t + alpha_next * (1.0 - r_fn) * x0 + noise_std * noise

    # Multistep extensions: only when we have history AND this is not the terminal step.
    # (Reference's ve_3_order_taylor includes `or sigmas[i+1] == 0` for the same reason;
    # vp_3 omits it but the omission appears to be an oversight.)
    # Also skip when sigma_prev_curr is at the sigma=1 boundary: _lambda(1.0) diverges,
    # and the finite-difference derivative across the limit branch is not meaningful.
    have_one_back = (
        state.old_x0 is not None
        and state.sigma_prev_curr is not None
        and 1.0 - state.sigma_prev_curr >= _SIGMA_ONE_TOLERANCE
    )
    not_terminal = sigma_next > 0.0
    new_d_x0: torch.Tensor | None = None

    if have_one_back and not_terminal:
        # 2nd-order Taylor term — ports vp_2_order_taylor.
        lambda_prev_curr = _lambda(state.sigma_prev_curr)
        d_x0 = (x0 - state.old_x0) / (lambda_curr - lambda_prev_curr)
        s_int = _integral_one_over_fn(lambda_next, lambda_curr)
        x_next = x_next + alpha_next * (
            lambda_next - lambda_curr + s_int * fn_next
        ) * d_x0
        new_d_x0 = d_x0

        # Both conditions are needed: the sigma~=1 boundary path advances
        # sigma_prev_prev while clearing old_d_x0, so a single-field check
        # would be insufficient if a near-unity sigma occurs mid-sequence.
        have_two_back = state.old_d_x0 is not None and state.sigma_prev_prev is not None
        if have_two_back:
            # 3rd-order Taylor term — ports vp_3_order_taylor.
            lambda_prev_prev = _lambda(state.sigma_prev_prev)
            dd_x0 = 2.0 * (d_x0 - state.old_d_x0) / (lambda_curr - lambda_prev_prev)
            s_d_int = _integral_lam_minus_curr_over_fn(lambda_next, lambda_curr)
            x_next = x_next + alpha_next * (
                (lambda_next - lambda_curr) ** 2 / 2.0 + s_d_int * fn_next
            ) * dd_x0

    # State update for the next call.
    state.sigma_prev_prev = state.sigma_prev_curr
    state.sigma_prev_curr = sigma_curr
    state.old_d_x0 = new_d_x0
    state.old_x0 = x0
    return x_next

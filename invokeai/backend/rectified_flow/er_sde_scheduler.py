"""ER-SDE (Extended Reverse-time SDE) ``diffusers`` scheduler.

Implements the multistep Taylor-expansion solver from:

    Cui, Q., Zhang, X., Lu, Z., & Liao, Q. (2023).
    Elucidating the solution space of extended reverse-time SDE
    for diffusion models. arXiv:2309.06169.
    https://arxiv.org/abs/2309.06169

Reference implementation (MIT-licensed):
    https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py

This scheduler unifies two regimes under a single API:

* **VP-SDE** (``use_flow_sigmas=False``) — Stable Diffusion / SDXL style models
  with epsilon, x0, or v prediction. Uses the standard
  ``alpha_t = 1 / sqrt(1 + sigma^2), sigma_t = sigma * alpha_t`` parameterization
  and ports ``vp_*_order_*`` from the reference impl.
* **Rectified flow / flow matching** (``use_flow_sigmas=True``) — FLUX, Z-Image,
  Anima style models with flow_prediction. Uses ``alpha_t = 1 - sigma, sigma_t = sigma``
  and the rectified-flow integral helpers defined locally (``_fn``,
  ``_integral_one_over_fn``, ``_integral_lam_minus_curr_over_fn``).

The rectified-flow integral helpers are kept local so this class is self-contained.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

# Number of sample points for the left Riemann sums approximating the
# Taylor-extension integrals. Matches the reference impl's nums_intergrate=100.
_INTEGRAL_NUM_POINTS = 100


def _fn(x: float) -> float:
    """ER-SDE noise-scale function ``SDE_5`` (paper appendix A.8).

    Mirrors ``customized_func(..., func_type=7)`` in the reference impl —
    the variant the paper recommends and tests for fast (~20 NFE) sampling.
    """
    return x * (math.exp(x**0.3) + 10.0)


def _integral_one_over_fn(lambda_next: float, lambda_curr: float) -> float:
    """Left Riemann sum of int_{lambda_next}^{lambda_curr} 1/_fn(lam) dlam.

    Precondition: ``lambda_next > 0``. The integrand has a logarithmic singularity
    at ``lam = 0`` (``_fn(0) = 0``); callers must skip this when ``sigma_next == 0``.
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

    Precondition: ``lambda_next > 0``. Same singularity at ``lam = 0`` as
    :func:`_integral_one_over_fn`.
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


class ERSDEScheduler(SchedulerMixin, ConfigMixin):
    """``diffusers`` scheduler for the ER-SDE multistep solver.

    See module docstring for paper / reference-impl citations.

    Args:
        num_train_timesteps: Number of diffusion steps used during training.
        beta_start: VP-SDE beta schedule start (ignored when ``use_flow_sigmas=True``).
        beta_end: VP-SDE beta schedule end (ignored when ``use_flow_sigmas=True``).
        beta_schedule: ``"linear"``, ``"scaled_linear"``, or ``"squaredcos_cap_v2"``.
        trained_betas: Override betas with a pre-computed schedule.
        prediction_type: ``"epsilon"``, ``"v_prediction"``, or ``"flow_prediction"``.
        solver_order: Multistep order (1, 2, or 3). The solver auto-warms from order 1.
        use_flow_sigmas: If True, use the rectified-flow parameterization
            (``alpha_t = 1 - sigma``); else VP-SDE.
        flow_shift: Sigma shift applied to the default flow schedule.
        stochastic: If True, inject noise (full ER-SDE). If False, deterministic
            ODE companion — same Taylor expansion with the noise term zeroed.
        sigma_one_tolerance: Boundary tolerance for the ``sigma = 1`` limit
            (rectified-flow only). Numerically paranoid; keep small.
        timestep_spacing: ``"linspace"``, ``"leading"``, or ``"trailing"``.
        steps_offset: Offset added to ``"leading"`` timesteps.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        solver_order: int = 3,
        use_flow_sigmas: bool = False,
        flow_shift: float = 1.0,
        stochastic: bool = True,
        sigma_one_tolerance: float = 1e-6,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        if prediction_type not in ("epsilon", "v_prediction", "flow_prediction"):
            raise ValueError(
                f"prediction_type must be one of 'epsilon', 'v_prediction', 'flow_prediction', got {prediction_type!r}"
            )
        if solver_order not in (1, 2, 3):
            raise ValueError(f"solver_order must be 1, 2, or 3, got {solver_order}")
        if prediction_type == "flow_prediction" and not use_flow_sigmas:
            # Not strictly invalid, but almost certainly a misconfiguration.
            raise ValueError("prediction_type='flow_prediction' requires use_flow_sigmas=True (rectified-flow regime).")

        # VP-SDE noise schedule (only used when use_flow_sigmas=False).
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule.
            betas = []
            for i in range(num_train_timesteps):
                t1 = i / num_train_timesteps
                t2 = (i + 1) / num_train_timesteps
                a1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
                a2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
                betas.append(min(1 - a2 / a1, 0.999))
            self.betas = torch.tensor(betas, dtype=torch.float32)
        else:
            raise NotImplementedError(f"beta_schedule {beta_schedule!r} is not implemented for ERSDEScheduler")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Default sigmas (VP-SDE form). Overwritten in set_timesteps.
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # Standard deviation of initial noise distribution (per Euler convention).
        self.init_noise_sigma = 1.0

        self.num_inference_steps: Optional[int] = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)

        # Multistep history. ``model_outputs`` stores x0 predictions; ``_sigma_history``
        # stores the sigma at which each prediction was made. Both are FIFO with
        # length == solver_order. Slot ``-1`` is the most recent.
        self.model_outputs: List[Optional[torch.Tensor]] = [None] * solver_order
        self._sigma_history: List[Optional[float]] = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None
        self.sigmas = self.sigmas.to("cpu")

    # ---- Index plumbing (mirrors DPM++) ---------------------------------------

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def index_for_timestep(
        self,
        timestep: Union[int, torch.Tensor],
        schedule_timesteps: Optional[torch.Tensor] = None,
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        index_candidates = (schedule_timesteps == timestep).nonzero()
        if len(index_candidates) == 0:
            return len(self.timesteps) - 1
        # On the very first step, prefer the second match if duplicated, so
        # img2img doesn't accidentally skip a sigma.
        if len(index_candidates) > 1:
            return index_candidates[1].item()
        return index_candidates[0].item()

    def _init_step_index(self, timestep: Union[int, torch.Tensor]) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    # ---- Timestep / sigma scheduling ------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
        timesteps: Optional[List[int]] = None,
    ) -> None:
        """Set the discrete timesteps used for inference.

        Exactly one of ``num_inference_steps``, ``timesteps``, or ``sigmas`` must
        be provided. The ``sigmas`` form (mirroring :class:`EulerDiscreteScheduler`)
        lets Anima/FLUX/Z-Image inject pre-shifted sigma schedules directly.
        """
        n_set = sum(x is not None for x in (num_inference_steps, timesteps, sigmas))
        if n_set != 1:
            raise ValueError("Must pass exactly one of `num_inference_steps`, `timesteps`, or `sigmas`.")

        if sigmas is not None:
            if isinstance(sigmas, torch.Tensor):
                sigmas_np = sigmas.detach().cpu().numpy().astype(np.float32)
            else:
                sigmas_np = np.array(sigmas, dtype=np.float32)
            num_inference_steps = len(sigmas_np) - 1
            # Timesteps in the rectified-flow / Anima convention scale sigma to t.
            # For VP-SDE this approximation is wrong but timesteps are only used
            # for indexing; the algebra runs entirely off self.sigmas.
            timesteps_np = (sigmas_np[:-1] * self.config.num_train_timesteps).astype(np.float32)
        elif timesteps is not None:
            timesteps_np = np.array(timesteps, dtype=np.float32)
            num_inference_steps = len(timesteps_np)
            sigmas_np = self._sigmas_for_timesteps(timesteps_np)
        else:
            assert num_inference_steps is not None
            timesteps_np = self._default_timesteps(num_inference_steps)
            sigmas_np = self._sigmas_for_timesteps(timesteps_np)

        self.num_inference_steps = num_inference_steps
        self.sigmas = torch.from_numpy(sigmas_np.astype(np.float32))
        self.timesteps = torch.from_numpy(timesteps_np.astype(np.float32)).to(device=device)

        # Reset multistep state.
        self.model_outputs = [None] * self.config.solver_order
        self._sigma_history = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def _default_timesteps(self, num_inference_steps: int) -> np.ndarray:
        """Standard linspace/leading/trailing schedule (VP-SDE timesteps)."""
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.float32)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // (num_inference_steps + 1)
            timesteps = (
                (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.float32)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.arange(self.config.num_train_timesteps, 0, -step_ratio).round().copy().astype(np.float32)
            timesteps -= 1
        else:
            raise ValueError(
                f"timestep_spacing {self.config.timestep_spacing!r} must be one of 'linspace', 'leading', 'trailing'"
            )
        return timesteps

    def _sigmas_for_timesteps(self, timesteps_np: np.ndarray) -> np.ndarray:
        """Build the sigma schedule (with terminal 0 appended) for given timesteps."""
        if self.config.use_flow_sigmas:
            # Rectified-flow sigmas in [0, 1], time-shifted per Anima/FLUX convention.
            num_inference_steps = len(timesteps_np)
            alphas = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)
            sigmas = 1.0 - alphas
            shift = self.config.flow_shift
            sigmas = np.flip(shift * sigmas / (1 + (shift - 1) * sigmas))[:-1].copy()
            # Terminal sigma is exactly 0.
            return np.concatenate([sigmas, [0.0]]).astype(np.float32)

        # VP-SDE: interpolate against the train sigmas using timestep indexing.
        train_sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps_np, np.arange(0, len(train_sigmas)), train_sigmas)
        return np.concatenate([sigmas, [0.0]]).astype(np.float32)

    # ---- Math helpers ---------------------------------------------------------

    def _sigma_to_alpha_sigma_t(self, sigma: float) -> Tuple[float, float]:
        """Map ``sigma`` to ``(alpha_t, sigma_t)``.

        Rectified flow: ``alpha_t = 1 - sigma, sigma_t = sigma``.
        VP-SDE: ``alpha_t = 1 / sqrt(1 + sigma^2), sigma_t = sigma * alpha_t``.
        """
        if self.config.use_flow_sigmas:
            return 1.0 - sigma, sigma
        alpha_t = 1.0 / math.sqrt(1.0 + sigma * sigma)
        return alpha_t, sigma * alpha_t

    @staticmethod
    def _lambda(alpha_t: float, sigma_t: float) -> float:
        """ER-SDE ``lambda = sigma_t / alpha_t`` — the noise-to-signal ratio.

        This matches the reference impl's ``lambdas = sigmas / alphas`` in both
        VP and rectified-flow regimes (see ``vp_*_order_*`` in
        ``https://github.com/QinpengCui/ER-SDE-Solver``). For VP-SDE this equals
        the stored sigma; for rectified flow it equals ``sigma / (1 - sigma)``.
        Diverges at ``sigma_t = alpha_t = 0`` (rectified flow at sigma=1) — the
        boundary branch in :meth:`_first_order_update` handles that case.
        """
        if alpha_t == 0.0:
            return float("inf")
        return sigma_t / alpha_t

    # ---- Model output conversion ----------------------------------------------

    def _convert_model_output(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Convert raw model output to an ``x0`` prediction at the current sigma."""
        sigma = float(self.sigmas[self.step_index].item())
        if self.config.prediction_type == "flow_prediction":
            # v = (x - x0) / sigma  =>  x0 = x - sigma * v
            return sample - sigma * model_output
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        if self.config.prediction_type == "epsilon":
            return (sample - sigma_t * model_output) / alpha_t
        if self.config.prediction_type == "v_prediction":
            return alpha_t * sample - sigma_t * model_output
        raise ValueError(f"Unsupported prediction_type {self.config.prediction_type!r}")

    # ---- Order-N updates -------------------------------------------------------

    def _first_order_update(
        self,
        x0: torch.Tensor,
        sample: torch.Tensor,
        sigma_curr: float,
        sigma_next: float,
        noise: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Order-1 ER-SDE step (ports ``vp_1_order`` / ``er_sde_rf_step`` order-1 branch)."""
        # Rectified-flow boundary: sigma_curr ~= 1 means alpha_curr ~= 0 so lambda diverges.
        # Closed-form limit (er_sde.py:136-142): x_next = (1 - sigma_next) * x0 + sigma_next * noise.
        if self.config.use_flow_sigmas and 1.0 - sigma_curr < self.config.sigma_one_tolerance:
            x_next = (1.0 - sigma_next) * x0
            if self.config.stochastic and noise is not None and sigma_next > 0.0:
                x_next = x_next + sigma_next * noise
            return x_next

        alpha_curr, sigma_curr_t = self._sigma_to_alpha_sigma_t(sigma_curr)
        alpha_next, sigma_next_t = self._sigma_to_alpha_sigma_t(sigma_next)

        # Reference impl uses lambda = sigma_t / alpha_t in both VP and flow regimes.
        lambda_curr = self._lambda(alpha_curr, sigma_curr_t)
        # At the terminal step, sigma_next == 0 so lambda_next == 0 and fn_next == 0.
        lambda_next = self._lambda(alpha_next, sigma_next_t) if sigma_next_t > 0.0 else 0.0

        fn_curr = _fn(lambda_curr)
        fn_next = _fn(lambda_next)
        r_fn = fn_next / fn_curr if fn_curr != 0.0 else 0.0
        r_alphas = alpha_next / alpha_curr

        # Stochastic noise std (paper appendix eq. for ER-SDE_5 variance).
        # ``inner`` can underflow to tiny negatives by roundoff; clip.
        inner = lambda_next**2 - lambda_curr**2 * r_fn**2
        if inner < 0.0:
            inner = 0.0
        noise_std = math.sqrt(inner) * alpha_next

        x_next = r_alphas * r_fn * sample + alpha_next * (1.0 - r_fn) * x0
        if self.config.stochastic and noise is not None and sigma_next > 0.0:
            x_next = x_next + noise_std * noise
        return x_next

    def _second_order_update(
        self,
        sample: torch.Tensor,
        sigma_curr: float,
        sigma_next: float,
        noise: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Order-2 ER-SDE step (ports ``vp_2_order_taylor``)."""
        x0 = self.model_outputs[-1]
        old_x0 = self.model_outputs[-2]
        sigma_prev_curr = self._sigma_history[-2]
        assert x0 is not None and old_x0 is not None and sigma_prev_curr is not None

        # If the previous step used the sigma=1 closed-form limit, the finite-difference
        # derivative across that boundary is meaningless — fall back to order 1.
        if self.config.use_flow_sigmas and 1.0 - sigma_prev_curr < self.config.sigma_one_tolerance:
            return self._first_order_update(x0, sample, sigma_curr, sigma_next, noise)

        # Order-1 base.
        x_next = self._first_order_update(x0, sample, sigma_curr, sigma_next, noise)

        # Skip the higher-order term at the terminal step — the integral helpers diverge
        # at lambda = 0 (sigma = 0), see _integral_one_over_fn docstring.
        if sigma_next <= 0.0:
            return x_next

        alpha_curr, sigma_curr_t = self._sigma_to_alpha_sigma_t(sigma_curr)
        alpha_next, sigma_next_t = self._sigma_to_alpha_sigma_t(sigma_next)
        alpha_prev, sigma_prev_t = self._sigma_to_alpha_sigma_t(sigma_prev_curr)
        lambda_curr = self._lambda(alpha_curr, sigma_curr_t)
        lambda_next = self._lambda(alpha_next, sigma_next_t)
        lambda_prev = self._lambda(alpha_prev, sigma_prev_t)

        denom = lambda_curr - lambda_prev
        if denom == 0.0:
            return x_next
        d_x0 = (x0 - old_x0) / denom

        fn_next = _fn(lambda_next)
        s_int = _integral_one_over_fn(lambda_next, lambda_curr)
        x_next = x_next + alpha_next * (lambda_next - lambda_curr + s_int * fn_next) * d_x0
        return x_next

    def _third_order_update(
        self,
        sample: torch.Tensor,
        sigma_curr: float,
        sigma_next: float,
        noise: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Order-3 ER-SDE step (ports ``vp_3_order_taylor``)."""
        x0 = self.model_outputs[-1]
        old_x0 = self.model_outputs[-2]
        old_old_x0 = self.model_outputs[-3]
        sigma_prev_curr = self._sigma_history[-2]
        sigma_prev_prev = self._sigma_history[-3]
        assert (
            x0 is not None
            and old_x0 is not None
            and old_old_x0 is not None
            and sigma_prev_curr is not None
            and sigma_prev_prev is not None
        )

        # If any sigma in the lookback hits the boundary, fall back to order 2.
        if self.config.use_flow_sigmas and (
            1.0 - sigma_prev_curr < self.config.sigma_one_tolerance
            or 1.0 - sigma_prev_prev < self.config.sigma_one_tolerance
        ):
            return self._second_order_update(sample, sigma_curr, sigma_next, noise)

        # Order-2 base.
        x_next = self._second_order_update(sample, sigma_curr, sigma_next, noise)

        if sigma_next <= 0.0:
            return x_next

        alpha_curr, sigma_curr_t = self._sigma_to_alpha_sigma_t(sigma_curr)
        alpha_next, sigma_next_t = self._sigma_to_alpha_sigma_t(sigma_next)
        alpha_prev, sigma_prev_t = self._sigma_to_alpha_sigma_t(sigma_prev_curr)
        alpha_pprev, sigma_pprev_t = self._sigma_to_alpha_sigma_t(sigma_prev_prev)
        lambda_curr = self._lambda(alpha_curr, sigma_curr_t)
        lambda_next = self._lambda(alpha_next, sigma_next_t)
        lambda_prev = self._lambda(alpha_prev, sigma_prev_t)
        lambda_pprev = self._lambda(alpha_pprev, sigma_pprev_t)

        denom_d = lambda_curr - lambda_prev
        denom_d_prev = lambda_prev - lambda_pprev
        denom_dd = lambda_curr - lambda_pprev
        if denom_d == 0.0 or denom_d_prev == 0.0 or denom_dd == 0.0:
            return x_next

        d_x0 = (x0 - old_x0) / denom_d
        old_d_x0 = (old_x0 - old_old_x0) / denom_d_prev
        dd_x0 = 2.0 * (d_x0 - old_d_x0) / denom_dd

        fn_next = _fn(lambda_next)
        s_d_int = _integral_lam_minus_curr_over_fn(lambda_next, lambda_curr)
        x_next = x_next + alpha_next * ((lambda_next - lambda_curr) ** 2 / 2.0 + s_d_int * fn_next) * dd_x0
        return x_next

    # ---- Public step ----------------------------------------------------------

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[Union[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """No-op (matches ``FlowMatchEulerDiscreteScheduler``)."""
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample at the next timestep using one ER-SDE step."""
        if self.num_inference_steps is None:
            raise ValueError("num_inference_steps is None — call `set_timesteps` before calling `step`.")
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma_curr = float(self.sigmas[self.step_index].item())
        sigma_next = float(self.sigmas[self.step_index + 1].item())

        # 1. Convert model output to x0 prediction.
        x0 = self._convert_model_output(model_output, sample)

        # 2. FIFO-shift the multistep history. New entry goes in slot -1.
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self._sigma_history[i] = self._sigma_history[i + 1]
        self.model_outputs[-1] = x0
        self._sigma_history[-1] = sigma_curr

        # 3. Sample noise (only when stochastic and not at terminal step).
        if self.config.stochastic and sigma_next > 0.0:
            noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
        else:
            noise = None

        # 4. Dispatch by available history.
        if self.config.solver_order == 1 or self.lower_order_nums < 1:
            prev_sample = self._first_order_update(x0, sample, sigma_curr, sigma_next, noise)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2:
            prev_sample = self._second_order_update(sample, sigma_curr, sigma_next, noise)
        else:
            prev_sample = self._third_order_update(sample, sigma_curr, sigma_next, noise)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # 5. Advance step index.
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    # ---- Forward noising (training / img2img) ---------------------------------

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-noise ``original_samples`` at the given timesteps (img2img style)."""
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        if self.config.use_flow_sigmas:
            alpha_t = 1.0 - sigma
            sigma_t = sigma
        else:
            alpha_t = 1.0 / torch.sqrt(1.0 + sigma * sigma)
            sigma_t = sigma * alpha_t
        return alpha_t * original_samples + sigma_t * noise

    def __len__(self) -> int:
        return self.config.num_train_timesteps

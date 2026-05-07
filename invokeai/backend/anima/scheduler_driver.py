"""Anima scheduler driver.

Encapsulates the per-scheduler API quirks that ``anima_denoise._run_diffusion``
would otherwise have to know about:

* Schedulers that accept ``set_timesteps(sigmas=...)`` get the pre-shifted
  Anima schedule passed directly.
* Schedulers that don't accept ``sigmas=`` use ``set_begin_index()`` over their
  own internal flow-shifted schedule. For Heun, the doubled-array index
  translation (logical step ``k`` → doubled index ``2k``) is handled here.
* SDE-style schedulers receive a seeded ``torch.Generator`` on every step.

The denoise loop iterates :meth:`AnimaSchedulerDriver.iterations` and calls
:meth:`AnimaSchedulerDriver.step` per iteration; the driver yields the
``sigma_prev`` and ``completes_user_step`` flags the caller needs for inpaint
mixing and progress reporting.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterator

import torch
from diffusers import FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from invokeai.backend.flux.schedulers import ANIMA_SCHEDULER_MAP


@dataclass(frozen=True)
class AnimaSchedulerIteration:
    """Per-iteration metadata yielded by :meth:`AnimaSchedulerDriver.iterations`.

    ``sigma_prev`` is the noise level the latents will be at after this iteration's
    :meth:`AnimaSchedulerDriver.step` call. ``completes_user_step`` is True when
    this iteration finishes a user-visible step — for Heun, the second-order
    half of each pair plus the unpaired terminal first-order step; for every
    other scheduler, always True.
    """

    sched_timestep: torch.Tensor
    sigma_curr: float
    sigma_prev: float
    completes_user_step: bool
    order: int


class AnimaSchedulerDriver:
    """Drives a diffusers scheduler over Anima's pre-shifted sigma schedule."""

    def __init__(
        self,
        scheduler_name: str,
        sigmas: list[float],
        steps: int,
        denoising_start: float,
        denoising_end: float,
        device: torch.device,
        seed: int,
    ):
        scheduler_class, scheduler_kwargs = ANIMA_SCHEDULER_MAP[scheduler_name]
        self.scheduler: SchedulerMixin = scheduler_class(num_train_timesteps=1000, **scheduler_kwargs)
        # Heun toggles state_in_first_order during step(); detect by class so we
        # can read it before set_timesteps has run.
        self.is_heun: bool = isinstance(self.scheduler, FlowMatchHeunDiscreteScheduler)
        self._begin_index: int = 0
        self._step_generator = torch.Generator(device=device).manual_seed(seed)

        is_lcm = scheduler_name == "lcm"
        accepts_sigmas = "sigmas" in inspect.signature(self.scheduler.set_timesteps).parameters
        clipped = denoising_start > 0 or denoising_end < 1

        if not is_lcm and accepts_sigmas:
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
            self._num_iterations = len(self.scheduler.timesteps)
        elif not is_lcm and clipped and hasattr(self.scheduler, "set_begin_index"):
            k_start = int(denoising_start * steps)
            k_end = int(denoising_end * steps)
            self.scheduler.set_timesteps(num_inference_steps=steps, device=device)
            if self.is_heun:
                # Heun's timesteps array is 2N-1 entries; logical step k maps to
                # doubled index 2k. min() clamps denoising_end=1.0 to the
                # unpaired terminal first-order step.
                self._begin_index = 2 * k_start
                self._num_iterations = min(
                    2 * (k_end - k_start),
                    len(self.scheduler.timesteps) - self._begin_index,
                )
            else:
                self._begin_index = k_start
                self._num_iterations = k_end - self._begin_index
            self.scheduler.set_begin_index(self._begin_index)
        else:
            self.scheduler.set_timesteps(num_inference_steps=len(sigmas) - 1, device=device)
            self._num_iterations = len(self.scheduler.timesteps)

    @property
    def num_iterations(self) -> int:
        """Total :meth:`step` calls. For Heun this is roughly 2× the user-visible step count."""
        return self._num_iterations

    @property
    def begin_index(self) -> int:
        return self._begin_index

    def iterations(self) -> Iterator[AnimaSchedulerIteration]:
        for i in range(self._num_iterations):
            sched_idx = i + self._begin_index
            sched_timestep = self.scheduler.timesteps[sched_idx]
            sigma_curr = sched_timestep.item() / self.scheduler.config.num_train_timesteps

            # Read state_in_first_order before step (Heun toggles it inside step()).
            in_first_order = self.scheduler.state_in_first_order if self.is_heun else True

            next_idx = sched_idx + 1
            sigma_prev = self.scheduler.sigmas[next_idx].item() if next_idx < len(self.scheduler.sigmas) else 0.0

            # For Heun, a user step completes on the second-order half of each
            # pair AND on the unpaired terminal first-order step (sigma_prev==0).
            is_terminal = sigma_prev == 0.0
            completes_user_step = (not self.is_heun) or (not in_first_order) or is_terminal
            order = 2 if self.is_heun else 1

            yield AnimaSchedulerIteration(
                sched_timestep=sched_timestep,
                sigma_curr=sigma_curr,
                sigma_prev=sigma_prev,
                completes_user_step=completes_user_step,
                order=order,
            )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        step_output = self.scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=self._step_generator,
        )
        return step_output.prev_sample

    @property
    def step_generator(self) -> torch.Generator:
        return self._step_generator

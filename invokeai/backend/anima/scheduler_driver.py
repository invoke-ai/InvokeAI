"""Anima scheduler driver.

Encapsulates the per-scheduler API quirks that ``anima_denoise._run_diffusion``
would otherwise have to know about:

* Schedulers that accept ``set_timesteps(sigmas=...)`` get the pre-shifted
  Anima schedule passed directly. Heun's doubled layout and LCM's terminal sigma
  are set through the helpers in ``invokeai.backend.flux.schedulers``.
* Schedulers that don't accept ``sigmas=`` (DPM++) use ``set_begin_index()`` over
  their own internal flow-shifted schedule.
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

from invokeai.backend.flux.schedulers import (
    ANIMA_SCHEDULER_MAP,
    set_heun_timesteps_from_sigmas,
    set_timesteps_from_sigmas,
)


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

        accepts_sigmas = "sigmas" in inspect.signature(self.scheduler.set_timesteps).parameters
        clipped = denoising_start > 0 or denoising_end < 1

        if self.is_heun:
            # ``sigmas`` is the clipped schedule, so Heun starts at the img2img blend sigma and ends at the
            # requested terminal sigma.
            set_heun_timesteps_from_sigmas(self.scheduler, sigmas[:-1], sigmas[-1], device)
            self._num_iterations = len(self.scheduler.timesteps)
        elif scheduler_name == "lcm":
            # LCM would otherwise build its own linear schedule from 1.0.
            set_timesteps_from_sigmas(self.scheduler, sigmas[:-1], sigmas[-1], device)
            self._num_iterations = len(self.scheduler.timesteps)
        elif accepts_sigmas:
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
            self._num_iterations = len(self.scheduler.timesteps)
        elif clipped and hasattr(self.scheduler, "set_begin_index"):
            # DPM++ takes no sigmas: clip its own flow-shifted schedule by index.
            k_start = int(denoising_start * steps)
            k_end = int(denoising_end * steps)
            self.scheduler.set_timesteps(num_inference_steps=steps, device=device)
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
            # pair AND on the unpaired terminal first-order step, which ends at the requested
            # terminal sigma (not 0 when denoising_end < 1).
            is_terminal = next_idx >= len(self.scheduler.timesteps)
            completes_user_step = (not self.is_heun) or (not in_first_order) or is_terminal

            yield AnimaSchedulerIteration(
                sched_timestep=sched_timestep,
                sigma_curr=sigma_curr,
                sigma_prev=sigma_prev,
                completes_user_step=completes_user_step,
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

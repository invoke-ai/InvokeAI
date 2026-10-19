"""The resolution-dependent shift of a flow-matching sigma schedule.

Larger frames are sampled with more of their steps at high noise. How much more is ``mu``, a straight line through
two (image token count, mu) anchors -- diffusers' ``calculate_shift`` over a scheduler config's
``base_image_seq_len``/``base_shift`` and ``max_image_seq_len``/``max_shift``. A family declares its anchors as a
:class:`ShiftFit`, or reads them from its scheduler config, and takes the result in one of two forms:

* ``mu``, for the exponential time shift ``e^mu / (e^mu + 1/t - 1)``. ``FlowMatchEulerDiscreteScheduler`` applies
  it from ``set_timesteps(mu=...)`` with ``time_shift_type="exponential"``.
* ``shift = e^mu``, for the linear form ``shift / (shift + 1/t - 1)`` (:func:`time_shift`). It is the same schedule,
  written for a family that builds its sigmas by hand or takes a shift from the user.

The line is not evaluated past its upper anchor. Extrapolated, it keeps climbing with the frame's area, the schedule
spends almost every step near pure noise, and an image-to-image pass entered by strength starts so close to noise
that it replaces its input. Each family's anchors record its own numbers. diffusers' pipelines extrapolate.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, TypeVar

import torch

if TYPE_CHECKING:
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

T = TypeVar("T", float, torch.Tensor)


@dataclass(frozen=True)
class ShiftFit:
    """mu along the line through ``(base_seq_len, base_shift)`` and ``(max_seq_len, max_shift)``, held at its upper
    end.

    Below ``base_seq_len`` the line keeps running: mu only sags there, and small frames keep the interpolation they
    were sampled with upstream.
    """

    base_seq_len: int
    max_seq_len: int
    base_shift: float
    max_shift: float

    @classmethod
    def from_scheduler_config(cls, config: Mapping[str, Any]) -> "ShiftFit":
        """The fit a ``FlowMatchEulerDiscreteScheduler`` config declares (see :func:`load_scheduler`)."""
        return cls(
            base_seq_len=config["base_image_seq_len"],
            max_seq_len=config["max_image_seq_len"],
            base_shift=config["base_shift"],
            max_shift=config["max_shift"],
        )

    def mu(self, seq_len: int) -> float:
        # diffusers' `calculate_shift`, in its order of operations, on the held sequence length.
        m = (self.max_shift - self.base_shift) / (self.max_seq_len - self.base_seq_len)
        b = self.base_shift - m * self.base_seq_len
        return min(seq_len, self.max_seq_len) * m + b

    def shift(self, seq_len: int) -> float:
        """``e^mu``, the shift the linear form takes."""
        return math.exp(self.mu(seq_len))


def time_shift(shift: float, t: T) -> T:
    """The linear time shift: ``t`` moved towards 1 by ``shift``, ``shift / (shift + 1/t - 1)``.

    For a positive ``shift``, ``t = 1`` stays 1; a shift of 0 divides zero by zero there. A tensor ``t`` of 0 stays 0;
    a float one divides by zero.
    """
    return shift / (shift + (1 / t - 1))


def load_scheduler(model_path: Path, stock_config: Mapping[str, Any]) -> "FlowMatchEulerDiscreteScheduler":
    """The checkpoint's own ``scheduler/scheduler_config.json``, else a scheduler built from ``stock_config``.

    GGUF and single-file transformers have no ``scheduler/`` folder and get the stock config. A checkpoint's file is
    read the way the reference pipeline reads it: the scheduler registers every init argument, so a key the file
    omits takes ``FlowMatchEulerDiscreteScheduler``'s own default (``max_shift`` 1.15, ``max_image_seq_len`` 4096),
    not the family's stock value.
    """
    # Imported on use: the families that only evaluate a fit do not need diffusers' schedulers.
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    scheduler_path = model_path / "scheduler"
    if scheduler_path.is_dir() and (scheduler_path / "scheduler_config.json").exists():
        return FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)
    return FlowMatchEulerDiscreteScheduler(**stock_config)

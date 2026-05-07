"""Flow Matching scheduler definitions and mapping.

This module provides the scheduler types and mapping for Flow Matching models
(Flux and Z-Image), supporting multiple schedulers from the diffusers library.
"""

from typing import Any, Literal, Type

from diffusers import (
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from invokeai.backend.rectified_flow.er_sde_scheduler import ERSDEScheduler

# Note: FlowMatchLCMScheduler may not be available in all diffusers versions
try:
    from diffusers import FlowMatchLCMScheduler

    _HAS_LCM = True
except ImportError:
    _HAS_LCM = False

# Scheduler name literal type for type checking
FLUX_SCHEDULER_NAME_VALUES = Literal["euler", "heun", "lcm"]

# Human-readable labels for the UI
FLUX_SCHEDULER_LABELS: dict[str, str] = {
    "euler": "Euler",
    "heun": "Heun (2nd order)",
    "lcm": "LCM",
}

# Mapping from scheduler names to scheduler classes
FLUX_SCHEDULER_MAP: dict[str, Type[SchedulerMixin]] = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "heun": FlowMatchHeunDiscreteScheduler,
}

if _HAS_LCM:
    FLUX_SCHEDULER_MAP["lcm"] = FlowMatchLCMScheduler


# Z-Image scheduler types (Flow Matching schedulers)
# Note: Z-Image-Turbo is optimized for ~8 steps with Euler, LCM can also work.
# Z-Image Base (undistilled) should only use Euler or Heun (LCM not supported for undistilled models).
ZIMAGE_SCHEDULER_NAME_VALUES = Literal["euler", "heun", "lcm"]

# Human-readable labels for the UI
ZIMAGE_SCHEDULER_LABELS: dict[str, str] = {
    "euler": "Euler",
    "heun": "Heun (2nd order)",
    "lcm": "LCM",
}

# Mapping from scheduler names to scheduler classes
ZIMAGE_SCHEDULER_MAP: dict[str, Type[SchedulerMixin]] = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "heun": FlowMatchHeunDiscreteScheduler,
}

if _HAS_LCM:
    ZIMAGE_SCHEDULER_MAP["lcm"] = FlowMatchLCMScheduler


# Anima scheduler types.
# Anima uses rectified flow with shift=3.0. The driver passes pre-shifted sigmas via
# set_timesteps(sigmas=...) when the scheduler accepts that signature. For those, the
# entry carries shift=1.0 to avoid double-shifting (the scheduler uses our sigmas verbatim).
# Schedulers that don't accept sigmas= (Heun, DPM++ on diffusers 0.35.1) build their own
# internal schedule, so they need shift=ANIMA_SHIFT/flow_shift=ANIMA_SHIFT in kwargs to match
# Anima's reference loglinear schedule.

# Fixed shift factor for the Anima rectified-flow noise schedule.
ANIMA_SHIFT = 3.0

ANIMA_SCHEDULER_NAME_VALUES = Literal["euler", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde", "lcm"]

ANIMA_SCHEDULER_LABELS: dict[str, str] = {
    "euler": "Euler",
    "heun": "Heun (2nd order)",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "er_sde": "ER-SDE",
    "lcm": "LCM",
}

# When adding a new Anima scheduler: add to all three of NAME_VALUES, LABELS,
# and this MAP. The MAP entry is `(SchedulerClass, scheduler_kwargs)`. For
# rectified-flow schedulers, set `use_flow_sigmas=True` and use
# `prediction_type="flow_prediction"`. If the scheduler accepts set_timesteps(sigmas=...),
# use shift=1.0 (driver passes pre-shifted sigmas); otherwise use shift=ANIMA_SHIFT
# so the scheduler builds the correct internal schedule.
ANIMA_SCHEDULER_MAP: dict[str, tuple[Type[SchedulerMixin], dict[str, Any]]] = {
    "euler": (FlowMatchEulerDiscreteScheduler, {"shift": 1.0}),
    "heun": (FlowMatchHeunDiscreteScheduler, {"shift": ANIMA_SHIFT}),
    "dpmpp_2m": (
        DPMSolverMultistepScheduler,
        {
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "flow_shift": ANIMA_SHIFT,
            "solver_order": 2,
        },
    ),
    "dpmpp_2m_sde": (
        DPMSolverMultistepScheduler,
        {
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "flow_shift": ANIMA_SHIFT,
            "algorithm_type": "sde-dpmsolver++",
            "solver_order": 2,
        },
    ),
    "er_sde": (
        ERSDEScheduler,
        {
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "flow_shift": ANIMA_SHIFT,
            "solver_order": 3,
            "stochastic": True,
        },
    ),
}

if _HAS_LCM:
    ANIMA_SCHEDULER_MAP["lcm"] = (FlowMatchLCMScheduler, {"shift": 1.0})

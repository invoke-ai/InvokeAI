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
# Anima uses rectified flow with shift=3.0. Sigmas are pre-shifted in
# anima_denoise.py via loglinear_timestep_shift, so FlowMatch schedulers
# carry shift=1.0 in their per-entry kwargs to avoid double-shifting.
# DPMSolverMultistepScheduler entries carry flow_shift=3.0 explicitly because
# they do not receive pre-shifted sigmas via set_timesteps(sigmas=...) in the
# same way — the shift is baked into the solver's sigma schedule.
ANIMA_SCHEDULER_NAME_VALUES = Literal["euler", "heun", "dpmpp_2m", "dpmpp_2m_sde", "lcm"]

ANIMA_SCHEDULER_LABELS: dict[str, str] = {
    "euler": "Euler",
    "heun": "Heun (2nd order)",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "lcm": "LCM",
}

ANIMA_SCHEDULER_MAP: dict[str, tuple[Type[SchedulerMixin], dict[str, Any]]] = {
    "euler": (FlowMatchEulerDiscreteScheduler, {"shift": 1.0}),
    "heun": (FlowMatchHeunDiscreteScheduler, {"shift": 1.0}),
    "dpmpp_2m": (
        DPMSolverMultistepScheduler,
        {
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "flow_shift": 3.0,  # matches anima_denoise.ANIMA_SHIFT
            "solver_order": 2,
        },
    ),
    "dpmpp_2m_sde": (
        DPMSolverMultistepScheduler,
        {
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "flow_shift": 3.0,  # matches anima_denoise.ANIMA_SHIFT
            "algorithm_type": "sde-dpmsolver++",
            "solver_order": 2,
        },
    ),
}

if _HAS_LCM:
    ANIMA_SCHEDULER_MAP["lcm"] = (FlowMatchLCMScheduler, {"shift": 1.0})

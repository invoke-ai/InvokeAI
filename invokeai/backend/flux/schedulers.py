"""Flow Matching scheduler definitions and mapping.

This module provides the scheduler types and mapping for Flow Matching models
(Flux and Z-Image), supporting multiple schedulers from the diffusers library.
"""

from typing import Literal, Type

from diffusers import (
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

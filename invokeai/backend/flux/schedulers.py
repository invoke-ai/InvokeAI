"""Flux scheduler definitions and mapping.

This module provides the scheduler types and mapping for Flux models,
supporting multiple Flow Matching schedulers from the diffusers library.
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

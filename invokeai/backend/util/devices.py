from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import autocast
from typing import Union
from invokeai.app.services.config import InvokeAIAppConfig

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")
MPS_DEVICE = torch.device("mps")
config = InvokeAIAppConfig.get_config()


def choose_torch_device() -> torch.device:
    """Convenience routine for guessing which GPU device to run model on"""
    if config.always_use_cpu:
        return CPU_DEVICE
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return CPU_DEVICE


def choose_precision(device: torch.device) -> str:
    """Returns an appropriate precision for the given torch device"""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        if not ("GeForce GTX 1660" in device_name or "GeForce GTX 1650" in device_name):
            return "float16"
    elif device.type == "mps":
        return "float16"
    return "float32"


def torch_dtype(device: torch.device) -> torch.dtype:
    if config.full_precision:
        return torch.float32
    if choose_precision(device) == "float16":
        return torch.float16
    else:
        return torch.float32


def choose_autocast(precision):
    """Returns an autocast context or nullcontext for the given precision string"""
    # float16 currently requires autocast to avoid errors like:
    # 'expected scalar type Half but found Float'
    if precision == "autocast" or precision == "float16":
        return autocast
    return nullcontext


def normalize_device(device: Union[str, torch.device]) -> torch.device:
    """Ensure device has a device index defined, if appropriate."""
    device = torch.device(device)
    if device.index is None:
        # cuda might be the only torch backend that currently uses the device index?
        # I don't see anything like `current_device` for cpu or mps.
        if device.type == "cuda":
            device = torch.device(device.type, torch.cuda.current_device())
    return device

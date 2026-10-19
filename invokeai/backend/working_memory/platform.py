from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import torch

ConvBackend = Literal["cudnn", "miopen", "mps", "xpu", "cpu"]


@dataclass(frozen=True)
class Platform:
    """The traits of the machine an operation runs on, as coefficients are keyed. Each field is a platform key an
    `EstimatorSpec` can name, and holds the true value: what was measured, and what was not, is `UNMEASURED_PRICED_AS`'s
    business."""

    conv_backend: ConvBackend


UNMEASURED_PRICED_AS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        # MPS, XPU and CPU convolutions are unmeasured and take the cuDNN coefficients. On MPS that pairs with a
        # score-matrix term the SDPA probe always charges there, so the total is not thin.
        "conv_backend": MappingProxyType({"mps": "cudnn", "xpu": "cudnn", "cpu": "cudnn"}),
    }
)
"""Per platform key, the values no coefficients are measured for, and the measured value each is priced as. The one
place that borrowing is declared: the resolver applies it and names it in an estimate's `source`, and shipped data
carries entries for measured values only."""


def platform_for(device: torch.device) -> Platform:
    """The platform an operation on `device` runs on.

    A HIP build reports ``device.type == "cuda"``, so the torch build -- not the device string -- is what separates
    MIOpen from cuDNN, and a `cpu_only` model on a ROCm build is not on MIOpen. Any device type other than CUDA, MPS and
    XPU runs torch's CPU kernels, or none at all, and is priced as the CPU.
    """
    if device.type == "cuda":
        return Platform(conv_backend="miopen" if torch.version.hip is not None else "cudnn")
    if device.type == "mps":
        return Platform(conv_backend="mps")
    if device.type == "xpu":
        return Platform(conv_backend="xpu")
    return Platform(conv_backend="cpu")

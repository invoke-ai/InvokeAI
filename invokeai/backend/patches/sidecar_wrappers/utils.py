import torch

from invokeai.backend.flux.modules.layers import RMSNorm
from invokeai.backend.patches.sidecar_wrappers.conv1d_sidecar_wrapper import Conv1dSidecarWrapper
from invokeai.backend.patches.sidecar_wrappers.conv2d_sidecar_wrapper import Conv2dSidecarWrapper
from invokeai.backend.patches.sidecar_wrappers.flux_rms_norm_sidecar_wrapper import FluxRMSNormSidecarWrapper
from invokeai.backend.patches.sidecar_wrappers.linear_sidecar_wrapper import LinearSidecarWrapper


def wrap_module_with_sidecar_wrapper(orig_module: torch.nn.Module) -> torch.nn.Module:
    if isinstance(orig_module, torch.nn.Linear):
        return LinearSidecarWrapper(orig_module)
    elif isinstance(orig_module, torch.nn.Conv1d):
        return Conv1dSidecarWrapper(orig_module)
    elif isinstance(orig_module, torch.nn.Conv2d):
        return Conv2dSidecarWrapper(orig_module)
    elif isinstance(orig_module, RMSNorm):
        return FluxRMSNormSidecarWrapper(orig_module)
    else:
        raise ValueError(f"No sidecar wrapper found for module type: {type(orig_module)}")

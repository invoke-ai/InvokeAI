import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.sidecar_wrappers.conv1d_sidecar_wrapper import Conv1dSidecarWrapper
from invokeai.backend.patches.sidecar_wrappers.conv2d_sidecar_wrapper import Conv2dSidecarWrapper
from invokeai.backend.patches.sidecar_wrappers.linear_sidecar_wrapper import LinearSidecarWrapper


def wrap_module_with_sidecar_wrapper(
    orig_module: torch.nn.Module, patches_and_weights: list[tuple[BaseLayerPatch, float]]
) -> torch.nn.Module:
    if isinstance(orig_module, torch.nn.Linear):
        return LinearSidecarWrapper(orig_module, patches_and_weights)
    elif isinstance(orig_module, torch.nn.Conv1d):
        return Conv1dSidecarWrapper(orig_module, patches_and_weights)
    elif isinstance(orig_module, torch.nn.Conv2d):
        return Conv2dSidecarWrapper(orig_module, patches_and_weights)
    else:
        raise ValueError(f"No sidecar wrapper found for module type: {type(orig_module)}")
